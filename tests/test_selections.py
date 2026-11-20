"""
Selectors (lp.Group, lp.Box, lp.Near, lp.Interface, lp.BoundaryOf): what regions and boundaries cover.

Regions are selected on the geometry before meshing, boundaries on the mesh: the checks compare the
selected edges with the named curve groups drawn in the .geo, and the response of a model whose
boundaries are all selectors with the one of the named model.
"""
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
