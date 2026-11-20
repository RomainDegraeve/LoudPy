"""
Model layer: name selection, validation, materials, deterministic meshing.
"""
import numpy as np
import pytest

import loudpy as lp
from loudpy.geometry import naming
from support import current_api as api


# ── group selection ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("selector, group, expected", [
    ("membranne", "submeca_membranne", True),
    ("glue", "submeca_glue_1", True),
    ("glue", "submeca_glue_2", True),
    ("coil", "submeca_coil", True),
    ("coil", "submeca_coil_former", False),          # the substring trap of the former API
    ("former", "submeca_coil_former", False),
    ("subacou", "subacou_front", True),
    ("front", "subacou_front", True),                # lp.Air("front")
    ("front", "PML_r+_1_front", False),
    ("interface_constrained", "interface_constrained_rz_2", True),
    ("interface_acou_meca_front", "interface_acou_meca_rear", False),
    ("interface_acou_meca", "interface_acou_meca_rear", True),
    ("PML", "PML_r+_1_front__r0=1.0e-01__t=1.0e-01", True),
    ("PML", "PML_r+_1_front_Corner__z+_r+__r0=0__z0=0__t=0.1", True),
    ("acou_meca", "interface_acou_meca_front", False),
    ("glue_*", "submeca_glue_12", True),
])
def test_selector_rules(selector, group, expected):
    assert naming.matches(selector, group) is expected


def test_names_are_labels_only():
    """A group name selects; it sets nothing physical (components are parameters, normals come from the mesh)."""
    from types import SimpleNamespace
    from loudpy.physics.parts import boundary_part
    group = SimpleNamespace(name="interface_constrained_r_1", node_tags=np.arange(3), node_coords=np.zeros((3, 3)),
                            edges=np.array([[0, 1, 2]]))
    assert boundary_part(lp.Clamped("interface_constrained_r"), group).direction == (0, 1)      # default "rz"
    assert boundary_part(lp.Clamped("interface_constrained_r", dofs="r"), group).direction == (0,)
    assert boundary_part(lp.Coil("interface_forced_r"), group).direction == (1,)                # default "z"
    with pytest.raises(ValueError, match="components"):
        lp.Clamped("x", dofs="xy")
    with pytest.raises(TypeError):
        lp.FluidStructure("interface_acou_meca_front", side="front")              # no side: from the mesh


# ── validation ───────────────────────────────────────────────────────────────
def test_unknown_name_lists_groups():
    model = lp.Model(api.SPEAKER_GEO).add(lp.Solid("membrane", "Paper"))      # typo
    with pytest.raises(ValueError, match="selects no domain group.*submeca_membranne"):
        model.selection()


def test_group_claimed_twice():
    model = lp.Model(api.SPEAKER_GEO).add(lp.Solid("glue", "SolidGlue"), lp.Solid("glue_1", "Paper"))
    with pytest.raises(ValueError, match="selected by both"):
        model.selection()


def test_selection_of_the_speaker():
    sel = api.speaker_model(acoustic=True).selection()
    assert sel["glue"] == ["submeca_glue_1", "submeca_glue_2"]
    assert sel["subacou"] == ["subacou_front", "subacou_rear"]
    assert sel["interface_constrained_rz"] == ["interface_constrained_rz_1", "interface_constrained_rz_2"]
    assert sel["interface_constrained_r"] == ["interface_constrained_r_1"]


# ── materials ────────────────────────────────────────────────────────────────
def test_material_library():
    paper = lp.materials["Paper"]
    assert paper.kind == "solid" and paper.E == 1e9 and paper.temperature == 293.15
    assert paper.rayleigh == (18.0, 4.6e-6) and paper.loss_factor == 0.03
    assert paper.source == "typical value, unverified"
    assert paper.with_(E=2e9).E == 2e9 and paper.E == 1e9
    with pytest.raises(KeyError, match="available"):
        lp.materials["Unobtainium"]


def test_library_formats(tmp_path):
    import json
    from loudpy.errors import LoudpyWarning
    v1 = {"Air": {"type": "acoustic", "rho": 1.204, "c": 343.2, "alpha": 3, "n": 1},
          "Paper": {"type": "elastic", "rho": 1200, "E": 1e9, "nu": 0.3, "eta": 0.03, "alpha_ray": 18.0,
                    "beta_ray": 4.6e-6}}
    (tmp_path / "v1.json").write_text(json.dumps(v1))
    with pytest.warns(LoudpyWarning, match="PML settings"):
        old = lp.load_materials(tmp_path / "v1.json")
    assert old["Air"].kind == "fluid" and old["Air"].c == 343.2
    assert old["Paper"].rayleigh == lp.materials["Paper"].rayleigh
    lib = lp.load_materials()
    again = lp.load_materials(lib.save(tmp_path / "v2.json"))
    assert dict(again) == dict(lib)
    with pytest.raises(ValueError, match="unknown properties"):
        lp.Material.from_dict({"name": "X", "rho": 1.0, "colour": "red"})


def test_rayleigh_from_loss_factor():
    r = lp.Rayleigh.from_loss_factor(0.03, band=(100.0, 1000.0))
    assert np.allclose(r.loss_factor([100.0, 1000.0]), 0.03, rtol=1e-12)
    assert r.loss_factor(np.sqrt(100.0 * 1000.0)) < 0.03                 # lower in between
    glue = lp.materials["SolidGlue"]                                     # epoxy: fit of its own η
    assert np.allclose(lp.Rayleigh(*glue.rayleigh).loss_factor([100.0, 1000.0]), glue.loss_factor, rtol=1e-4)
    for m in lp.materials.values():                    # every library fit follows its own η at 100 Hz and 1 kHz
        if m.kind == "solid":
            eta = lp.Rayleigh(*m.rayleigh).loss_factor([100.0, 1000.0])
            assert np.allclose(eta, m.loss_factor, rtol=0.1), (m.name, eta, m.loss_factor)


def test_air_model():
    """Ideal-gas air vs tabulated values (dry air at 1 atm: 331.3 m/s at 0 °C, ρ 1.204 at 20 °C)."""
    std = lp.materials["Air"]
    assert std == lp.air() and std.temperature == 293.15 and std.humidity == 0.0
    assert abs(std.rho - 1.2041) < 1e-4 and abs(std.c - 343.24) < 0.01
    assert abs(lp.air(273.15).c - 331.3) < 0.05 and abs(lp.air(273.15).rho - 1.2922) < 1e-3
    humid = lp.air(humidity=1.0)                     # water vapour is lighter: ρ down, c up
    assert humid.rho < std.rho and 0.0 < humid.c - std.c < 1.5
    from loudpy.model.materials import C_AIR, RHO_AIR
    assert (RHO_AIR, C_AIR) == (std.rho, std.c)      # one air for 2D, 3D and the toolkit
    with pytest.raises(ValueError):
        lp.air(humidity=1.5)


def test_scene_air():
    import loudpy as lp
    scene = lp.Scene(lambda occ: None, [], air=lp.air(temperature=303.15))
    assert scene.c == lp.air(303.15).c and scene.rho == scene.air.rho
    assert lp.Scene(lambda occ: None, []).c == lp.materials["Air"].c
    with pytest.raises(ValueError, match="not a fluid"):
        lp.Scene(lambda occ: None, [], air="Paper")


def test_missing_property_is_explicit():
    with pytest.raises(ValueError, match="has no E"):
        lp.Material("Mystery", rho=1000.0).require("rho", "E")


def test_mesh_object(mesh_dir):
    mesh = api.plate_model().mesh(mesh_dir / "plate.msh")
    assert mesh.dim == 2 and mesh.names(2) == ["submeca_plate"]
    assert set(mesh.names(1)) == {"interface_constrained_rz_1", "interface_constrained_r_1"}
    assert mesh.coords_of(mesh.groups["submeca_plate"].cells[:2]).shape == (2, 6, 2)


# ── deterministic meshing ────────────────────────────────────────────────────
def test_deterministic_meshing(tmp_path):
    """With deterministic=True the same model gives the same mesh (one gmsh thread)."""
    meshes = []
    for k in range(2):
        model = api.speaker_model()
        mesh = model.mesh(tmp_path / f"m{k}.msh", reuse=False, deterministic=True)
        meshes.append(mesh)
    a, b = meshes
    assert np.array_equal(a.coords, b.coords)
    assert a.names() == b.names()
    for name in a.names():
        assert np.array_equal(a.groups[name].cells, b.groups[name].cells), name


def test_import_has_no_side_effects():
    """import loudpy / loudpy.plot leaves the environment, logging and matplotlib settings alone."""
    import os
    import subprocess
    import sys
    code = ("import os, logging, matplotlib as mpl; mpl.use('Agg'); before = dict(mpl.rcParams);"
            "import loudpy as lp, loudpy.plot, numpy as np;"
            "assert 'KMP_DUPLICATE_LIB_OK' not in os.environ;"
            "assert all(isinstance(h, logging.NullHandler) for h in logging.getLogger('loudpy').handlers);"
            "assert dict(mpl.rcParams) == before;"
            "ax = lp.Data(np.arange(1., 4.), {'f': np.array([10., 100., 1000.])}, 'spl').plot();"
            "assert dict(mpl.rcParams) == before;"
            "assert ax.yaxis.label.get_fontfamily() == ['serif']")
    env = {k: v for k, v in os.environ.items() if k != "KMP_DUPLICATE_LIB_OK"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_set_verbosity_prints_progress(capsys):
    import loudpy as lp
    lp.set_verbosity("info")
    import logging
    logging.getLogger("loudpy.test").info("hello from loudpy")
    lp.set_verbosity("warning")
    assert "hello from loudpy" in capsys.readouterr().err
