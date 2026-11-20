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


def test_group_info():
    info = naming.parse("interface_constrained_rz_2")
    assert (info.index, info.dofs, info.orientation) == (2, (0, 1), None)
    assert naming.parse("interface_acou_meca_rear").orientation == -1.0


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
    assert sel["interface_constrained"] == ["interface_constrained_r_1", "interface_constrained_rz_1",
                                            "interface_constrained_rz_2"]


# ── materials ────────────────────────────────────────────────────────────────
def test_material_library():
    paper = lp.materials["Paper"]
    assert paper.kind == "elastic" and paper.E == 1e9
    assert paper.with_(E=2e9).E == 2e9 and paper.E == 1e9
    with pytest.raises(KeyError, match="available"):
        lp.materials["Unobtainium"]


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
