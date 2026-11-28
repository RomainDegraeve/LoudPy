"""
Results on a named part (lp.On) and how a plot is shown (lp.plot.options, loudpy.plot.draw): a boundary's nodes in
order along it, its surface of revolution and a region's volume of revolution as the reductions weigh them (against
the geometry), a region's maximum against the field restricted to it; PlotSpec on a named part; the style options
checked and applied by every view (curve, map, polar, table, field, bode, nyquist, balloon).
"""
import matplotlib
import numpy as np
import pytest

import loudpy as lp
from loudpy.plot import options as opts
from loudpy.plot.spec import PlotSpec
from loudpy.results import places
from support import current_api as api

matplotlib.use("Agg")
FREQS = [200.0, 500.0, 1000.0]


@pytest.fixture(scope="module")
def fsi(mesh_dir):
    model = api._meshed(api.speaker_model(acoustic=True), mesh_dir, api._speaker_tag(1e-3, True, False))
    return lp.FrequencyResponse(model, drive=lp.Force(1.0)).solve(FREQS)


def _solid_boundary(res):
    """A stored boundary all of whose edges lie on the solid mesh."""
    mesh = res.blocks[0].meshes["solid"]
    for name, bd in res.boundaries.items():
        rows = mesh.rows_of(np.asarray(bd["edges"]).ravel(), missing="ignore")
        if len(bd["edges"]) > 4 and (rows >= 0).all():
            return name
    pytest.skip("no boundary on the solid mesh")


def test_a_boundary_in_order_and_its_surface(fsi):
    name = _solid_boundary(fsi)
    d = fsi.probe("u", at=lp.On(name))
    n = fsi.nodes_on(name)
    assert d.dims == ("f", "s", "component") and d.shape[:2] == (len(FREQS), len(n["node_tags"]))
    assert set(n["node_tags"]) == set(fsi.boundary(name)) and np.all(np.diff(d.coords["s"]) >= 0)
    np.testing.assert_allclose(d.coords["s"], n["s"])
    area = 0.0
    for k in np.unique(n["piece"]):                                      # each separate piece of the boundary
        on = n["piece"] == k
        xy = np.column_stack([n["r"][on], n["z"][on]])
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        assert step.max() < 0.25 * max(np.ptp(n["s"][on]), 1e-12)       # neighbours along the curve, no jumps
        np.testing.assert_allclose(np.diff(n["s"][on]), step, rtol=1e-12)
        area += 2 * np.pi * np.sum(0.5 * (xy[1:, 0] + xy[:-1, 0]) * step)    # 2π ∫ r ds along the chained nodes
    *_, w, _ = places._place(fsi, fsi.blocks[0], name, fsi.blocks[0].meshes["solid"])
    np.testing.assert_allclose(w.sum(), area, rtol=2e-3)
    u_z = fsi.displacement(at=lp.On(name), component="z")
    mean = fsi.displacement(at=lp.On(name, reduce="mean"), component="z")
    total = fsi.displacement(at=lp.On(name, reduce="integral"), component="z")
    assert mean.dims == ("f",) and total.quantity.unit == "m·m²"
    np.testing.assert_allclose(mean.values, (u_z.values * w).sum(axis=1) / w.sum(), rtol=1e-12)
    np.testing.assert_allclose(total.values, mean.values * w.sum(), rtol=1e-12)


def test_a_region_its_volume_and_maximum(fsi):
    mesh = fsi.blocks[0].meshes["solid"]
    region = mesh.names(2)[0]
    d = fsi.probe("u", at=lp.On(region), component="z")
    assert d.dims == ("f", "node")
    mx = fsi.probe("u", at=lp.On(region, reduce="max"), component="z")
    for k, f in enumerate(FREQS):
        ref = fsi.field("u", f=f).restrict(region).component("z")
        np.testing.assert_allclose(mx.values[k], np.abs(ref.values).max(), rtol=1e-12)
    *_, w, _ = places._place(fsi, fsi.blocks[0], region, mesh)
    cells = np.asarray(mesh.groups[region].cells)[:, :3]
    c = np.asarray(mesh.coords)[:, :2][mesh.rows_of(cells.ravel())].reshape(-1, 3, 2)
    area = 0.5 * np.abs(np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0]))
    volume = 2 * np.pi * np.sum(area * c[:, :, 0].mean(axis=1))          # Pappus on straight-sided triangles
    np.testing.assert_allclose(w.sum(), volume, rtol=5e-3)


def test_refusals(fsi):
    with pytest.raises(lp.errors.ResultsError, match="no boundary or region"):
        fsi.probe("u", at=lp.On("nothing_like_this"))
    with pytest.raises(ValueError, match="reduce"):
        lp.On("x", reduce="sum")
    with pytest.raises(ValueError, match="named part"):
        fsi.field("p", f=500.0).at(lp.On("x"))


def test_plot_spec_on_a_named_part_and_its_table(fsi):
    name = _solid_boundary(fsi)
    out = PlotSpec(quantity="displacement", component="z", where={"kind": "on", "name": name}, axes=["s"],
                   at={"f": 500.0}).resolve(fsi)
    assert out.view == "curve" and out.data.dims == ("s",) and [c.dim for c in out.controls] == ["f"]
    t = PlotSpec(quantity="displacement", component="z", where={"kind": "on", "name": name, "reduce": "mean"},
                 view="table", style={"digits": 3}).resolve(fsi)
    head, rows = t.table()
    assert head[0].startswith("f") and len(rows) == len(FREQS) and all(len(r) == 2 for r in rows)
    spec = PlotSpec(quantity="displacement", where={"kind": "on", "name": name, "reduce": "max"}, style={"marker": "o"})
    assert PlotSpec.from_dict(spec.to_dict()) == spec


def test_style_options_declared_and_checked():
    names = [o.name for o in opts.STYLE_OPTIONS]
    assert len(names) == len(set(names)) and all(set(o.views) <= set(opts.ALL) for o in opts.STYLE_OPTIONS)
    assert {"xscale", "yscale", "line", "marker", "palette"} <= {o.name for o in opts.options("curve")}
    assert {"cmap", "levels", "contours", "range_db"} <= {o.name for o in opts.options("map")}
    assert opts.check({"linewidth": "2", "yscale": "log", "title": ""}) == {"linewidth": 2.0, "yscale": "log"}
    for bad, msg in (({"yscale": "loggy"}, "one of"), ({"colour": "red"}, "no style option"), ({"edges": "yes"}, "true")):
        with pytest.raises(ValueError, match=msg):
            opts.check(bad)


def test_every_view_draws_with_its_options(fsi):
    import matplotlib.pyplot as plt
    arc = {"kind": "arc", "radius": 0.1, "from": 0, "to": 90, "step": 10, "center": [0.0, 0.03]}
    point = {"kind": "point", "point": [0.0, 0.05]}
    cases = [
        (dict(quantity="spl", where=point, style={"yscale": "linear", "line": "dashed", "marker": "o",
                                                  "palette": "Dark2", "grid": "both", "xmin": 150.0}), None),
        (dict(quantity="spl", where=arc, view="map", style={"normalize": "first", "levels": 8, "contours": 2,
                                                            "cmap": "jet", "reverse": True, "mirror": True}), None),
        (dict(quantity="spl", where=arc, view="polar", at={"f": 500.0},
              style={"mirror": True, "range_db": 20.0, "zero": "E", "clockwise": False}), "polar"),
        (dict(quantity="pressure", where=point, view="table", style={"show": "phase", "digits": 4}), None),
        (dict(quantity="displacement", component="z", where=point, view="bode", style={"show": "db"}), None),
        (dict(quantity="displacement", component="z", where=point, view="nyquist", style={"marker": "s"}), None),
        (dict(quantity="spl", where=arc, view="balloon", at={"f": 1000.0}, style={"range_db": 25.0}), "3d"),
        (dict(quantity="field", field="p", view="field", at={"f": 500.0},
              style={"show": "real", "symmetric": True, "edges": True, "mirror": True, "cmap": "coolwarm"}), None),
    ]
    for spec, projection in cases:
        out = PlotSpec(**spec).resolve(fsi)
        fig = plt.figure()
        ax = fig.add_subplot(projection=projection) if projection else None
        drawn = out.plot(ax=ax)
        assert drawn is not None, spec
        plt.close("all")
    shown = PlotSpec(quantity="spl", where=arc, view="map", style={"normalize": "first"}).resolve(fsi).shown()
    np.testing.assert_allclose(shown.isel(angle=0).values, 0.0, atol=1e-12)     # relative to the axis
