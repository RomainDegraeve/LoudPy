"""
Geometry sources (loudpy.geometry.sources): a .geo of curve loops, a .geo of surfaces, a CAD file (STEP
sketch) and a builder function give the same model.
"""
import numpy as np
import pytest

import loudpy as lp
from loudpy.errors import ModelError, SelectionError
from support import current_api as api

STEEL = lp.Material("Steel", rho=7850.0, E=2.1e11, nu=0.3, rayleigh=(0.0, 0.0))
SURFACE_PLATE = """SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 0.1, 0.002};
Physical Surface("plate") = {1};
Physical Curve("rim") = {2};
"""


def plate(geometry, path):
    model = lp.Model(geometry).add(lp.Solid("plate", STEEL, damping=lp.Undamped(), mesh_size=0.002),
                                   lp.Clamped("rim", on=lp.Box(r=0.1)),
                                   lp.Clamped("axis", on=lp.Box(r=0), dofs="r"))
    model.mesh(path, deterministic=True)
    return model


def rectangle_plate(occ):
    return {"plate": occ.addRectangle(0, 0, 0, 0.1, 0.002)}


def test_plate_from_three_sources(tmp_path):
    """The clamped plate as curve loops (.geo), as a Physical Surface (.geo) and from a builder: the same mesh,
    node for node, and the same modes."""
    (tmp_path / "surface.geo").write_text(SURFACE_PLATE)
    models = [plate(api.PLATE_GEO, tmp_path / "loop.msh"), plate(tmp_path / "surface.geo", tmp_path / "surface.msh"),
              plate(rectangle_plate, tmp_path / "builder.msh")]
    ref = models[0].meshed
    freqs = [lp.Modes(m).solve(f_max=4000.0, f_min=10.0, n_per_shift=10).frequencies for m in models]
    for m, f in zip(models[1:], freqs[1:]):
        assert m.meshed.names(2) == ["plate"]
        assert np.array_equal(m.meshed.coords, ref.coords)
        assert np.array_equal(m.meshed.groups["plate"].cells, ref.groups["submeca_plate"].cells)
        assert np.array_equal(f, freqs[0])
    assert models[1].groups() == ["plate", "rim"] and models[2].groups() == ["plate"]


def test_curve_loops_need_no_prefix(tmp_path):
    """Every closed physical curve group of a .geo is a domain, whatever its name."""
    geo = tmp_path / "loops.geo"
    geo.write_text(api_plate_geo().replace("submeca_plate", "steel_disc"))
    model = lp.Model(geo)
    assert model.groups("domain") == ["steel_disc"]
    assert model.groups("boundary") == ["interface_constrained_rz_1", "interface_constrained_r_1"]
    model.add(lp.Solid("disc", STEEL, on="steel_disc", mesh_size=0.002, damping=lp.Undamped()),
              lp.Clamped("interface_constrained_rz"))
    assert model.mesh(tmp_path / "loops.msh").names(2) == ["steel_disc"]


def api_plate_geo():
    from pathlib import Path
    return Path(api.PLATE_GEO).read_text()


def test_surfaces_named_and_unnamed(tmp_path):
    """Named surface groups keep their names; an unnamed surface ("surface_<tag>") is selected by position and
    its mesh group takes the name of the region; a surface claimed twice through two groups is an error."""
    geo = tmp_path / "two.geo"
    geo.write_text("""SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 0.1, 0.01};
Rectangle(2) = {0, 0.01, 0, 0.1, 0.05};
Physical Surface("disc") = {1};
Physical Surface("everything") = {1, 2};
""")
    model = lp.Model(geo)
    assert model.groups("domain") == ["disc", "everything"]
    model.add(lp.Solid("disc", STEEL, mesh_size=0.005), lp.Solid("all", STEEL, on="everything"))
    with pytest.raises(SelectionError, match=r"surface 1 \(in 'everything'\) is selected by both 'disc' and 'all'"):
        model.selection()

    geo.write_text("""SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 0.1, 0.01};
Rectangle(2) = {0, 0.01, 0, 0.1, 0.05};
Physical Surface("disc") = {1};
""")
    model = lp.Model(geo)
    assert model.groups("domain") == ["disc", "surface_2"]
    model.add(lp.Solid("disc", STEEL, mesh_size=0.005), lp.Air("air", on=lp.Near((0.05, 0.03)), mesh_size=0.01),
              lp.FluidStructure("top", on=lp.Interface("disc", "air")))
    mesh = model.mesh(tmp_path / "two.msh")
    assert mesh.names(2) == ["disc", "air"]                       # the unnamed surface takes the region's name
    top = model.select(model["top"])
    assert np.allclose(mesh.coords_of(top)[..., 1], 0.01)          # the shared edge, found from the mesh
    with pytest.raises(ModelError, match="would hold domains of"):
        lp.Model(geo).add(lp.Solid("cone", STEEL, on="disc"), lp.Air("disc", on=lp.Near((0.05, 0.03)))) \
            .mesh(tmp_path / "clash.msh")


def test_builder_mesh_is_reused_until_it_changes(tmp_path):
    from loudpy.geometry.sources import GeometrySource

    def builder(occ, width=0.1):
        return {"plate": occ.addRectangle(0, 0, 0, width, 0.002)}

    def wider(occ, width=0.12):
        return {"plate": occ.addRectangle(0, 0, 0, width, 0.002)}

    assert GeometrySource(builder).files() != GeometrySource(wider).files()
    path = tmp_path / "builder.msh"
    plate(builder, path)
    stamp = path.stat().st_mtime_ns
    plate(builder, path)
    assert path.stat().st_mtime_ns == stamp                        # reused
    with pytest.raises(ModelError, match="give the mesh file"):
        lp.Model(builder).add(lp.Solid("plate", STEEL)).mesh()


STEP = "tests/data/geometries/Loudspeaker/HPNEW-Sketch.step"
POINTS = {"membranne": (0.00894597, 0.00902379), "dustcap": (0.0089157, 0.01052703),
          "surround": (0.05242295, 0.03701458), "former": (0.00483333, 0.00108668),
          "coil": (0.00583333, -0.00432546), "glue_1": (0.00554515, -0.00106783), "spider": (0.00636, 0.0000095),
          "glue_2": (0.04451774, -0.00018355)}


def p2_area(mesh, groups):
    """Area of the curved P2 triangles of mesh groups (Jacobian integrated exactly)."""
    gp, area = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]]), 0.0
    for g in groups:
        X = mesh.coords_of(mesh.groups[g].cells)
        for x, y in gp:
            l0 = 1 - x - y
            dx = np.array([1 - 4 * l0, 4 * x - 1, 0, 4 * (l0 - x), 4 * y, -4 * y])
            dy = np.array([1 - 4 * l0, 0, 4 * y - 1, -4 * x, 4 * x, 4 * (l0 - y)])
            area += np.abs(np.linalg.det(np.stack([dx @ X, dy @ X], axis=1))).sum() / 6
    return area


def test_speaker_from_the_step_sketch(tmp_path, mesh_dir):
    """The HPNEW STEP sketch (curves only, in mm): its 10 closed regions become the domains, selected by
    position; with the boundaries and the PML given by selectors, the model matches the .geo one: same
    domains (areas within the P2 discretization of the arcs), same straight boundaries, same PML layers."""
    mat = dict(api.SPEAKER_PARTS)
    model = lp.Model(STEP, api.MATERIALS)
    solids = [lp.Solid(n, mat["glue" if n.startswith("glue") else n], mesh_size=1e-3, on=lp.Near(p))
              for n, p in POINTS.items()]
    structure = solids[0]
    for s in solids[1:]:
        structure = structure | s
    model.add(solids, lp.Air("front_air", mesh_size=0.01, on=lp.Near((0.0219, 0.0238))),
              lp.Air("rear_air", mesh_size=0.01, on=lp.Near((0.1, -0.05))),
              lp.PML("pml", on=lp.Box(r=0.3) | lp.Box(z=0.532) | lp.Box(r=0.2) | lp.Box(z=-0.169), thickness=0.1,
                     mesh_size=0.025, tuned_to=1000.0, order=2, strength=1.5),
              lp.Clamped("spider_rim", on=lp.Near((0.04471635, -0.00075))),
              lp.Clamped("axis", on=lp.BoundaryOf("dustcap") & lp.Box(r=0), dofs="r"),
              lp.Coil("drive", on=lp.Interface("coil", "former")),
              lp.FluidStructure("front", on=lp.Interface(structure, "front_air")))
    assert len(model.groups("domain")) == 10 and all(g.startswith("surface_") for g in model.groups("domain"))
    mesh = model.mesh(tmp_path / "step.msh", deterministic=True)
    assert np.ptp(mesh.coords[:, 1]) == pytest.approx(0.532 + 0.169 + 0.2, rel=1e-9)     # metres, PML included

    ref = api.speaker_model(acoustic=True)
    ref.mesh(mesh_dir / "speaker_0.001_fsi.msh")
    for n in POINTS:
        a, b = p2_area(mesh, [n]), p2_area(ref.meshed, [f"submeca_{n}"])
        assert a == pytest.approx(b, rel=1e-3 if n == "spider" else 1e-7), n      # spider: thin arcs, P2 meshes
    for label, named in [("spider_rim", "interface_constrained_rz_1"), ("axis", "interface_constrained_r_1"),
                         ("drive", "interface_forced_z_1")]:
        E = model.boundary_nodes()[label]["edges"]
        F = ref.meshed.groups[named].cells
        lengths = [np.linalg.norm(np.diff(m.coords_of(e[:, :2]), axis=1), axis=-1).sum()
                   for m, e in [(mesh, E), (ref.meshed, F)]]
        assert lengths[0] == pytest.approx(lengths[1], rel=1e-12), label
    from test_selections import LEGACY_LAYERS, same_layer
    from loudpy.geometry.naming import PMLLayer
    key = lambda L: (L.use_r, round(L.r0, 9), L.use_z, round(L.z0, 9))
    got = sorted((p.layer for p in model.parts if p.role == "pml"), key=key)
    exp = sorted((PMLLayer(**kw) for kw in LEGACY_LAYERS.values()), key=key)
    assert len(got) == 6 and all(same_layer(a, b) for a, b in zip(got, exp))


# ── driver wet profiles: curves selected on the geometry ─────────────────────
def test_profiles_from_selectors():
    """The wet profiles of the 3D scenes, selected instead of listed by gmsh curve tags: the same curves in
    the same order, hence the same sampled polylines (bit for bit)."""
    for prof, curves, start in [(api.REAR_PROFILE, api.REAR_CURVES, api.REAR_START),
                                (api.FRONT_PROFILE, api.FRONT_CURVES, api.FRONT_START)]:
        assert prof.resolve(api.SPEAKER_GEO) == (curves, start)
        old = lp.Profile(curves, start).sample(api.SPEAKER_GEO)
        new = prof.sample(api.SPEAKER_GEO)
        assert len(old) == len(new) and all(np.array_equal(a, b) for a, b in zip(old, new))
    # without start: the end of the chain farthest from the axis, exactly
    order, start = lp.Profile(api.FRONT_PROFILE.on).resolve(api.SPEAKER_GEO)
    assert order == api.FRONT_CURVES and start == pytest.approx(api.FRONT_START, abs=2e-7) and start[0] != 0.05735


def test_profile_errors_and_model_names(mesh_dir):
    rear_air = lp.BoundaryOf("subacou_rear")
    with pytest.raises(ModelError, match="do not form one open chain"):
        lp.Profile(rear_air).resolve(api.SPEAKER_GEO)
    with pytest.raises(ModelError, match="branch"):
        lp.Profile(rear_air, start=api.REAR_START).resolve(api.SPEAKER_GEO)
    with pytest.raises(ModelError, match="not an end point"):
        lp.Profile(rear_air, start=(0.1, 0.1), end=api.CONE_FOOT).resolve(api.SPEAKER_GEO)
    with pytest.raises(ModelError, match="needs start"):
        lp.Profile([14, 7])
    # an lp.Model as the geometry: its region names select
    model = lp.Model(api.SPEAKER_GEO).add(lp.Solid("cone", "Paper", on="membranne"),
                                          lp.Air("front_air", on="subacou_front"))
    assert lp.Profile(lp.Interface("cone", "front_air")).resolve(model)[0] == [113]
