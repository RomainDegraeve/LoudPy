"""
Selectors (lp.Group, lp.Box, lp.Near, lp.Interface, lp.BoundaryOf): what regions and boundaries cover.

Regions are selected on the geometry before meshing, boundaries on the mesh: the checks compare the
selected edges with the named curve groups drawn in the .geo, and the response of a model whose
boundaries are all selectors with the one of the named model.
"""
import json

import numpy as np
import pytest

import loudpy as lp
from loudpy.errors import ModelError, SelectionError
from loudpy.geometry.selectors import Domains, _codes
from support import current_api as api

SOLIDS = lp.Group(*[n for n, _ in api.SPEAKER_PARTS])


def codes(rows):
    return set(map(int, _codes(rows)))


@pytest.fixture(scope="module")
def speaker(mesh_dir):
    model = api.speaker_model(acoustic=True)
    model.mesh(mesh_dir / "speaker_0.001_fsi.msh")
    return model


@pytest.fixture(scope="module")
def plate(mesh_dir):
    model = api.plate_model()
    model.mesh(mesh_dir / "plate.msh")
    return model


def named(model, label):
    return model._edges.curve_groups[next(n for n in model.meshed.names(1) if lp.geometry.naming.matches(label, n))]


# ── the objects ──────────────────────────────────────────────────────────────
def test_selector_algebra():
    sel = "membranne" | lp.Group("dustcap") - lp.Box(r=0)
    assert repr(sel) == "(Group('membranne') | (Group('dustcap') - Box(r=0)))"      # Python precedence
    assert repr(lp.Box(r=(0, 0.01), z=0.032)) == "Box(r=(0, 0.01), z=0.032)"
    assert repr(lp.Interface(lp.Solid("cone", "Paper"), "subacou")) == "Interface(Solid('cone'), Group('subacou'))"
    assert lp.Group("a").by_groups and (lp.Group("a") | "b").by_groups and not (lp.Group("a") & "b").by_groups
    assert (lp.Group("a") | lp.Box(r=0)).labels() == ["a"] and (lp.Group("a") - lp.Group("b")).labels() == ["a"]
    with pytest.raises(TypeError):
        lp.Group()
    with pytest.raises(TypeError):
        lp.Box()
    with pytest.raises(ValueError):
        lp.Box(r=(1.0, 0.0))
    with pytest.raises(TypeError):
        lp.Solid("x", "Paper", on=3.0).selector


# ── regions: on the geometry, before meshing ─────────────────────────────────
def test_regions_by_selector():
    model = lp.Model(api.SPEAKER_GEO).add(
        lp.Solid("coil_by_point", "Copper", on=lp.Near((0.0058, -0.005))),
        lp.Solid("glue_by_box", "SolidGlue", on=lp.Box(r=(0.044, 0.0448), z=(-0.0018, 0.0003))),
        lp.Solid("cone", "Paper", on=lp.Group("membranne") | "dustcap"),
        lp.Solid("rest", "Rubber", on=lp.Group("submeca*") - "membranne" - "dustcap" - "coil" - "glue_2"),
        lp.Air("front_air", on=lp.Near((0.05, 0.1))))
    assert model.selection() == {
        "coil_by_point": ["submeca_coil"], "glue_by_box": ["submeca_glue_2"],
        "cone": ["submeca_membranne", "submeca_dustcap"],
        "rest": ["submeca_surround", "submeca_former", "submeca_glue_1", "submeca_spider"],
        "front_air": ["subacou_front"]}


def test_region_selector_errors():
    with pytest.raises(SelectionError, match="selects boundary edges, not domains"):
        lp.Model(api.SPEAKER_GEO).add(lp.Solid("x", "Paper", on=lp.Interface("coil", "former"))).selection()
    with pytest.raises(SelectionError, match="'nothing' selects no domain group.*submeca_coil"):
        lp.Model(api.SPEAKER_GEO).add(lp.Solid("y", "Paper", on="nothing")).selection()
    with pytest.raises(SelectionError, match="selected by both"):
        lp.Model(api.SPEAKER_GEO).add(lp.Solid("a", "Paper", on=lp.Near((0.0058, -0.005))),
                                      lp.Solid("b", "Copper", on="coil")).selection()
    with pytest.raises(ModelError, match="already has an item named"):
        lp.Model(api.SPEAKER_GEO).add(lp.Solid("a", "Paper"), lp.Clamped("a"))


# ── boundaries: on the mesh ──────────────────────────────────────────────────
def test_interfaces_from_the_mesh(speaker):
    """The interfaces drawn in the .geo are found from the mesh alone."""
    assert codes(speaker.select(lp.Interface(SOLIDS, "subacou_front"))) == named(speaker, "interface_acou_meca_front")
    # the .geo removed the clamped spider rim from the rear interface by hand; the selector keeps it (its
    # structural DOFs are blocked, so the coupled problem is the same: test_selectors_give_the_same_response)
    rear = codes(speaker.select(lp.Interface(SOLIDS, "subacou_rear")))
    assert rear == named(speaker, "interface_acou_meca_rear") | named(speaker, "interface_constrained_rz_1")
    assert codes(speaker.select(lp.Interface("coil", "former"))) == named(speaker, "interface_forced")
    assert codes(speaker.select(lp.BoundaryOf("dustcap") & lp.Box(r=0))) == named(speaker, "interface_constrained_r")
    both = lp.BoundaryOf("subacou_front") & lp.BoundaryOf(SOLIDS)
    assert codes(speaker.select(both)) == named(speaker, "interface_acou_meca_front")


def test_boundaries_by_position(speaker, plate):
    assert codes(speaker.select(lp.Near((0.04471635, -0.00075)))) == named(speaker, "interface_constrained_rz_1")
    assert codes(speaker.select(lp.Box(z=0.032) & lp.BoundaryOf("surround"))) == \
        named(speaker, "interface_constrained_rz_2")
    assert codes(plate.select(lp.Near((0.1, 0.001)))) == named(plate, "interface_constrained_rz")      # rim
    assert codes(plate.select(lp.Box(r=0))) == named(plate, "interface_constrained_r")                # axis
    faces = plate.select(lp.BoundaryOf("plate") - lp.Box(r=0) - lp.Box(r=0.1))
    assert codes(faces) == codes(plate.select(lp.Box(z=0) | lp.Box(z=0.002)))
    assert len(faces) == len(plate._edges.codes) - 2
    with pytest.raises(SelectionError, match="do not exist in 2D"):
        plate.select(lp.Box(x=0))
    with pytest.raises(SelectionError, match=r"selects domain groups, not boundary edges.*BoundaryOf\('plate'\)"):
        plate.select(lp.Near((0.05, 0.001)) & lp.Group("plate"))


def test_domains_on_the_mesh(speaker):
    assert speaker.select(lp.Near((0.0058, -0.005)), kind="domain") == ["submeca_coil"]
    pml = speaker.select(lp.Near((0.35, 0.6)), kind="domain")          # the PML corner built by the mesher
    assert len(pml) == 1 and pml[0].startswith("PML_r+_1_front_Corner")
    assert lp.Near((0.35, 0.6)).domains(Domains.from_mesh(speaker.meshed)) == set(pml)


def test_selected_boundaries_build_parts(mesh_dir):
    """A model whose boundaries are all selectors: one part per boundary, named by its label, on the same
    mesh as the named model (the saved mesh keeps more curve groups than needed: reused)."""
    path = mesh_dir / "speaker_0.001_fsi.msh"
    before = path.stat().st_mtime_ns
    model = api.speaker_model_by_selectors()
    model.mesh(path)
    assert path.stat().st_mtime_ns == before
    parts = {p.name: p for p in model.parts if p.role not in ("solid", "air", "pml")}
    assert sorted(parts) == ["axis", "drive", "front", "rear", "spider_rim", "surround_rim"]
    assert parts["axis"].direction == (0,) and parts["drive"].direction == (1,)
    assert np.all(np.abs(parts["front"].signs) == 1.0)
    assert set(model.boundary_nodes()) == set(parts)
    assert len(model.boundary_nodes()["front"]["edges"]) == 82


def test_selectors_give_the_same_response(mesh_dir):
    """Coupled structure + air + PML, 1 N on the coil: the model with selectors gives the response of the
    model with the named curve groups (same mesh), to round-off."""
    path = mesh_dir / "speaker_0.001_fsi.msh"
    a = api.speaker_fsi_response(api.speaker_model(acoustic=True), path, [200.0, 1500.0])
    b = api.speaker_fsi_response(api.speaker_model_by_selectors(), path, [200.0, 1500.0])
    for x in ("coords_u", "coords_p"):
        assert np.array_equal(a[x], b[x])
    for x in ("u", "p"):
        assert np.abs(a[x] - b[x]).max() <= 1e-10 * np.abs(a[x]).max()


def test_restrict_field_by_selector(speaker):
    mesh = speaker.meshed
    f = lp.Field("x", mesh, mesh.coords[:, 0], 0.0)
    near = f.restrict(lp.Near((0.0058, -0.005)))
    assert list(near.mesh.groups) == ["submeca_coil"]
    assert np.array_equal(near.values, f.restrict("coil").values)


# ── PML layers: base curves selected on the geometry ─────────────────────────
LEGACY_LAYERS = {   # the committed FSI mesh (written before the layers were stored): read from the names
    "PML_r+_1_front": dict(use_r=True, r0=0.3, tr=0.1, sgn_r=1.0),
    "PML_r+_2_rear": dict(use_r=True, r0=0.2, tr=0.1, sgn_r=1.0),
    "PML_z-_2_rear": dict(use_z=True, z0=-0.169, tz=0.1, sgn_z=-1.0),
    "PML_z+_1_front": dict(use_z=True, z0=0.532, tz=0.1, sgn_z=1.0),
    "PML_r+_1_front_Corner": dict(use_r=True, r0=0.3, tr=0.1, sgn_r=1.0, use_z=True, z0=0.532, tz=0.1, sgn_z=1.0),
    "PML_r+_2_rear_Corner": dict(use_r=True, r0=0.2, tr=0.1, sgn_r=1.0, use_z=True, z0=-0.169, tz=0.1, sgn_z=-1.0),
}


def layers(model):
    from loudpy.geometry.naming import base_name
    return {base_name(p.name): p.layer for p in model.parts if p.role == "pml"}


def same_layer(a, b, rtol=1e-12):
    from dataclasses import asdict
    a, b = asdict(a), asdict(b)
    return all(np.isclose(a[k], b[k], rtol=rtol, atol=1e-15) for k in a)


def test_pml_layers_of_the_committed_mesh(speaker):
    from loudpy.geometry.naming import PMLLayer
    got = layers(speaker)
    assert set(got) == set(LEGACY_LAYERS)
    for name, kw in LEGACY_LAYERS.items():
        assert got[name] == PMLLayer(**kw), name


def test_pml_by_selector(tmp_path):
    """Fresh meshes: the layers of lp.PML("PML") (labels) and of a PML on curves selected by position are the
    ones of the committed mesh; they are stored with the mesh and read back when it is reused."""
    def speaker(pml):
        model = api.speaker_model(acoustic=True)
        model.regions = [r for r in model.regions if not isinstance(r, lp.PML)]
        return model.add(pml)
    kw = dict(thickness=0.1, mesh_size=0.025, tuned_to=1000.0, order=2, strength=1.5)
    by_label = speaker(lp.PML("PML", **kw))
    by_box = speaker(lp.PML("absorber", on=lp.Box(r=0.3) | lp.Box(z=0.532) | lp.Box(r=0.2) | lp.Box(z=-0.169), **kw))
    assert by_box.selection()["absorber"] == ["curve 11", "curve 12", "curve 16", "curve 17"]
    for model, path in [(by_label, tmp_path / "label.msh"), (by_box, tmp_path / "box.msh")]:
        model.mesh(path, deterministic=True)
        expected = sorted(LEGACY_LAYERS.values(), key=repr)
        from loudpy.geometry.naming import PMLLayer
        got = sorted(layers(model).values(), key=lambda L: (L.use_r, L.r0, L.use_z, L.z0))
        exp = sorted((PMLLayer(**kw) for kw in expected), key=lambda L: (L.use_r, L.r0, L.use_z, L.z0))
        assert all(same_layer(a, b) for a, b in zip(got, exp)) and len(got) == len(exp)
        stored = json.loads(path.with_suffix(".msh.json").read_text())["pml_layers"]
        assert len(stored) == 6 and not any("__" in n for n in stored)              # nothing in the names
        again = speaker(model[next(r.name for r in model.regions if isinstance(r, lp.PML))])
        again.mesh(path)                                                              # reused
        assert layers(again) == layers(model)
    assert sorted(layers(by_box)) == ["absorber_1", "absorber_1_Corner", "absorber_2", "absorber_3",
                                      "absorber_3_Corner", "absorber_4"]


BLOCK_GEO = """SetFactory("OpenCASCADE");
Point(1) = {0, 0, 0}; Point(2) = {0.1, 0, 0}; Point(3) = {0.1, 0.1, 0}; Point(4) = {0, 0.1, 0};
Line(1) = {1, 2}; Line(2) = {2, 3}; Line(3) = {3, 4}; Line(4) = {4, 1};
Point(5) = {0, 0.5, 0}; Point(6) = {0.05, 0.5, 0}; Point(7) = {0.05, 0.6, 0}; Point(8) = {0, 0.6, 0};
Line(5) = {5, 6}; Line(6) = {6, 7}; Line(7) = {7, 8}; Line(8) = {8, 5};
Point(9) = {0.2, 0, 0}; Line(9) = {2, 9}; Line(10) = {9, 3};
Physical Curve("subacou_air", 1) = {1, 2, 3, 4};
Physical Curve("submeca_block", 2) = {5, 6, 7, 8};
Physical Curve("subacou_wedge", 3) = {9, 10, 2};
"""


def _block_model(tmp_path, *pml):
    geo = tmp_path / "block.geo"
    geo.write_text(BLOCK_GEO)
    return lp.Model(geo).add(lp.Air("air", mesh_size=0.02), lp.Air("wedge", mesh_size=0.02),
                             lp.Solid("block", "Paper", mesh_size=0.02), *pml)


def test_pml_grows_out_of_the_air(tmp_path):
    """The layer grows along the outward normal of its air, whatever the rest of the drawing: here the
    centroid of the drawing lies above the base (a solid far above), where the former rule would have
    turned the layer into the air."""
    model = _block_model(tmp_path, lp.PML("pml", on=lp.Box(z=0.1) & lp.BoundaryOf("air"), thickness=0.05,
                                          mesh_size=0.02, tuned_to=1000.0))
    model.mesh(tmp_path / "block.msh", deterministic=True)
    (layer,) = layers(model).values()
    assert layer.use_z and not layer.use_r and layer.sgn_z == 1.0 and np.isclose(layer.z0, 0.1)
    z = model.meshed.coords_of(np.unique(model.meshed.groups["pml"].cells))[:, 1]
    assert z.min() > 0.1 - 1e-12 and np.isclose(z.max(), 0.15)


def test_pml_errors(tmp_path):
    from loudpy.errors import MeshError
    kw = dict(thickness=0.05, mesh_size=0.02, tuned_to=1000.0)
    with pytest.raises(MeshError, match="straight line r = const or z = const"):
        _block_model(tmp_path, lp.PML("pml", on=lp.Near((0.15, 0.06)), **kw)).mesh(tmp_path / "a.msh")
    with pytest.raises(MeshError, match="exactly one meshed air domain"):
        _block_model(tmp_path, lp.PML("pml", on=lp.Box(z=0.6), **kw)).mesh(tmp_path / "b.msh")
    with pytest.raises(SelectionError, match="base of both PML"):
        _block_model(tmp_path, lp.PML("a", on=lp.Box(z=0.1), **kw),
                     lp.PML("b", on=lp.Near((0.05, 0.1)), **kw)).selection()
