"""
Results layer: format 2 round trip, multi-file sweeps, the Results interface and the plots.
"""
import h5py
import numpy as np
import pytest

import loudpy as lp
from support import current_api as api
from test_files import LEGACY, _same_by_coords


def _legacy(name):
    p = LEGACY / name
    if not p.exists():
        pytest.skip("legacy files are written by pytest --update-reference")
    return p


# ── format 2 ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["freq_meca.h5", "eigen_fourier.h5", "time.h5"])
def test_format2_roundtrip(tmp_path, name):
    r = lp.load(_legacy(name))
    path = r.save(tmp_path / "v2.h5")
    back = lp.load(path)
    assert type(back) is type(r) and len(back) == len(r) and back.names == r.names
    np.testing.assert_array_equal(back.values, r.values)
    for b0, b1 in zip(r.blocks, back.blocks):
        assert b0.label == b1.label and b0.meta.get("order") == b1.meta.get("order")
        for n in b0.fields:
            np.testing.assert_array_equal(b0.fields[n], b1.fields[n])
        for phys, m in b0.meshes.items():
            np.testing.assert_array_equal(m.coords, b1.meshes[phys].coords)
            np.testing.assert_array_equal(m.triangles(), b1.meshes[phys].triangles())
        for k in b0.extra:
            np.testing.assert_array_equal(b0.extra[k], b1.extra[k])
    assert back.boundaries.keys() == r.boundaries.keys()


def test_format2_has_no_class_names(tmp_path):
    """Format 2 names the physics by role, not by the Python class of the physics objects."""
    path = lp.load(_legacy("freq_meca.h5")).save(tmp_path / "v2.h5")
    keys = []
    with h5py.File(path, "r") as f:
        f.visit(keys.append)
        assert f.attrs["loudpy_format"] == "2.0" and f.attrs["kind"] == "frequency"
    assert any(k.endswith("meshes/solid") for k in keys)
    assert not any("subdomain" in k.lower() for k in keys)


# ── several files = one sweep ────────────────────────────────────────────────
def test_multi_file_sweep(tmp_path, mesh_dir):
    """Two block files (e.g. one per mesh of a remeshed sweep) load as one sweep, in frequency order."""
    for fs, name in (([1500.0, 3000.0], "b1.h5"), ([100.0, 700.0], "b0.h5")):
        _, study = api.speaker_freq_meca(mesh_dir, fs)
        study.save(tmp_path / name)
    r = lp.load(tmp_path)
    assert list(r.frequencies) == [100.0, 700.0, 1500.0, 3000.0] and len(r.blocks) == 2
    rg = lp.load(str(tmp_path / "b*.h5"))
    np.testing.assert_array_equal(rg.frequencies, r.frequencies)
    rl = lp.load([tmp_path / "b1.h5", tmp_path / "b0.h5"])               # explicit list: kept order
    assert list(rl.frequencies) == [1500.0, 3000.0, 100.0, 700.0]
    assert r.probe("u", at=(0.0, 0.03)).shape == (4, 2)


# ── the Results interface ────────────────────────────────────────────────────
def test_results_interface():
    r = lp.load(_legacy("freq_meca.h5"))
    v = r.field("u", f=900.0)                                    # nearest entry: 1000 Hz
    assert v.coordinate == 1000.0 and v.values.shape == (v.mesh.n_nodes, 2)
    assert r.field("u_meca", index=1).coordinate == 1000.0       # former field names still work
    k = r.mesh("solid").nearest((0.0, 0.03))
    at = r.mesh("solid").coords[k]                               # on a node: interpolation = its value
    p = r.probe("u", at=at)
    assert p.dims == ("f", "component") and p.shape == (len(r), 2)
    np.testing.assert_allclose(p[1], v.values[k], rtol=1e-12)
    np.testing.assert_allclose(r.displacement(at, "z").values, p.values[:, 1], rtol=1e-12)
    np.testing.assert_allclose(r.velocity(at, "z").values, 2j * np.pi * r.frequencies * p.values[:, 1])
    with pytest.raises(KeyError, match="no field 'p'"):
        r.field("p")
    modes = lp.load(_legacy("eigen_fourier.h5"))
    assert modes.orders == [0, 1] and np.all(np.isfinite(modes.zeta))
    assert modes.order(1).mode(0).values.shape[1] == 3            # (U_r, U_z, U_θ)
    t = modes.table()
    assert list(t.columns) == ["set", "order", "f [Hz]", "zeta"] and len(t) == len(modes)
    np.testing.assert_allclose(lp.spl(np.array([20e-6 * np.sqrt(2)])), [0.0], atol=1e-12)


def test_interpolation_is_linear_inside_and_nearest_outside():
    r = lp.load(_legacy("freq_meca.h5"))
    mesh = r.mesh("solid")
    tri = mesh.triangles(refine=True)[10]
    c = mesh.coords[tri].mean(axis=0)                            # centroid of one sub-triangle
    W = mesh.interpolation([c, (5.0, 5.0)])
    np.testing.assert_allclose(W[0].toarray().ravel()[tri], [1 / 3] * 3, rtol=1e-9)
    assert W[1].nnz == 1 and W[1].indices[0] == mesh.nearest((5.0, 5.0))


def test_selections():
    pts = lp.Arc(0.3, [0, 90]).points(2)
    np.testing.assert_allclose(pts, [[0, 0.3], [0.3, 0]], atol=1e-15)
    assert lp.Polar(0.3, 30).dim is None and lp.Polar(0.3, [0, 30]).dim == "angle"
    line = lp.Line((0, 0), (0, 1), n=5)
    np.testing.assert_allclose(line.coord, [0, .25, .5, .75, 1])
    p3 = lp.Arc(1.0, [0, 90], center=(1, 2, 3), axis="z", toward="x").points(3)
    np.testing.assert_allclose(p3, [[1, 2, 4], [2, 2, 3]], atol=1e-15)


def test_data_operations_and_exports(tmp_path):
    import pandas as pd
    from scipy.io import loadmat
    r = lp.load(_legacy("freq_meca.h5"))
    u = r.displacement(at=(0.0, 0.03), component="z")
    assert np.iscomplexobj(u.values) and u.quantity.unit == "m" and u.dims == ("f",)
    np.testing.assert_allclose(u.abs().values, np.abs(u.values))
    np.testing.assert_allclose(u.db().values, 20 * np.log10(np.abs(u.values)))
    assert u.velocity().quantity.name == "v_z" and u.acceleration().quantity.unit == "m/s²"
    assert u.sel(f=950.0).ndim == 0 and complex(np.asarray(u.sel(f=950.0))) == u.values[1]
    two = r.probe("u", at=lp.Points([(0, 0.03), (0.01, 0.03)]), component="z")
    assert two.dims == ("f", "point")
    np.testing.assert_allclose(two.normalize(point=two.coords["point"][0]).values[:, 0], 1.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(u), u.values)                   # NumPy, no copy needed
    s = u.to_pandas()
    assert isinstance(s, pd.Series) and s.index.name == "f [Hz]"
    assert isinstance(two.to_pandas(), pd.DataFrame)
    d = u.abs()
    back = np.loadtxt(d.export(tmp_path / "u.csv"), delimiter=",", comments="#")
    np.testing.assert_allclose(back[:, 0], r.frequencies)
    np.testing.assert_allclose(back[:, 1], d.values, rtol=1e-9)
    assert pd.read_csv(tmp_path / "u.csv", comment="#", header=None).shape == (len(r), 2)
    cplx = np.loadtxt(u.export(tmp_path / "uc.csv"), delimiter=",", comments="#")
    np.testing.assert_allclose(cplx[:, 1] + 1j * cplx[:, 2], u.values, rtol=1e-9)
    m = loadmat(u.export(tmp_path / "u.mat"))
    np.testing.assert_allclose(m["u_z"].ravel(), u.values)
    z = np.load(u.export(tmp_path / "u.npz"))
    np.testing.assert_allclose(z["values"], u.values)
    np.testing.assert_allclose(z["coord_f"], r.frequencies)


def test_field_views_and_exports(tmp_path):
    from scipy.io import loadmat
    r = lp.load(_legacy("freq_meca.h5"))
    u = r.field("u", f=100.0)
    assert u.component("z").values.ndim == 1 and u.component("z").quantity.name == "u_z"
    np.testing.assert_allclose(u.magnitude().values, np.linalg.norm(np.abs(u.values), axis=1))
    k = u.mesh.nearest((0.0, 0.03))
    np.testing.assert_allclose(u.at(u.mesh.coords[k]), u.values[k], rtol=1e-12)
    along = u.at(lp.Line((0.0, 0.03), (0.04, 0.03), n=7))
    assert along.dims == ("s", "component")
    df = u.to_pandas()
    assert list(df.columns[:2]) == ["r", "z"] and len(df) == u.mesh.n_nodes
    u.export(tmp_path / "u.vtu")
    assert (tmp_path / "u.vtu").stat().st_size > 0
    table = np.loadtxt(u.export(tmp_path / "u.csv"), delimiter=",", comments="#")
    assert table.shape == (u.mesh.n_nodes, 6)                              # r, z, u_r re/im, u_z re/im
    assert loadmat(u.export(tmp_path / "u.mat"))["values"].shape == u.values.shape


def test_results_export(tmp_path):
    from scipy.io import loadmat
    r = lp.load(_legacy("freq_meca.h5"))
    m = loadmat(r.export(tmp_path / "all.mat"), simplify_cells=True)
    np.testing.assert_allclose(m["block1"]["frequency"], r.frequencies)
    assert m["block1"]["u"].shape == r.blocks[0].fields["u"].shape
    z = np.load(r.export(tmp_path / "all.npz"))
    np.testing.assert_allclose(z["block0/u"], r.blocks[0].fields["u"])


def test_plots(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    r = lp.load(_legacy("freq_meca.h5"))
    u = r.field("u", f=100.0)
    for ax in (u.plot(), u.plot("z"), u.plot(deformed=True, mirror=True), u.plot("instant"),
               r.displacement(at=(0.0, 0.03), component="z").plot(),
               r.probe("u", at=lp.Points([(0, .03), (.01, .03)]), component="z").plot(),
               r.displacement(at=(0.0, 0.03)).phase(deg=True).plot(),
               lp.plot.boundary(u, r.boundary("interface_acou_meca_front"))):
        assert ax.get_xlabel()
    ax = lp.plot.line([r.displacement(at=(0, .03), component="z").rename(label="a"),
                       r.displacement(at=(.01, .03), component="z").rename(label="b")])
    assert ax.get_xscale() == "log" and ax.get_legend() is not None
    modes = lp.load(_legacy("eigen_fourier.h5"))
    modes.order(1).mode(0).plot("theta")
    lp.plot.modes(modes.order(0), indices=[0, 1])
    with pytest.raises(ValueError, match="pick one mode set"):
        lp.plot.modes(modes)
    trans = lp.load(_legacy("time.h5"))
    d = trans.displacement(at=(0.0, 0.03), component="z")
    assert d.dims == ("t",)
    ax = d.plot()
    assert ax.get_yscale() == "linear"                                      # signed signal
    d.spectrum().plot()
    lp.plot.save(ax.figure, tmp_path / "t.pdf")
    assert (tmp_path / "t.pdf").exists()


def test_region_and_boundary_views():
    r = lp.load(_legacy("freq_meca.h5"))
    v = r.field("u", index=0)
    sur = v.restrict("surround")
    assert 0 < sur.mesh.n_nodes < v.mesh.n_nodes and set(sur.mesh.groups) == {"submeca_surround"}
    k = v.mesh.rows_of(sur.mesh.node_tags[:5])
    np.testing.assert_array_equal(sur.values[:5], v.values[k])
    tags = r.boundary("interface_acou_meca_front")
    coords, vals = v.at_nodes(tags)
    assert len(coords) == len(tags) == len(vals)
    with pytest.raises(KeyError):
        v.restrict("no_such_part")


def test_save_metadata(tmp_path, mesh_dir):
    _, study = api.speaker_freq_meca(mesh_dir, [100.0])
    study.save(tmp_path / "m.h5", case="demo", tones=np.array([1.0, 2.0]))
    meta = lp.load(tmp_path / "m.h5").meta
    assert meta["case"] == "demo" and meta["tones"] == [1.0, 2.0] and meta["study"] == "FrequencyResponse"
