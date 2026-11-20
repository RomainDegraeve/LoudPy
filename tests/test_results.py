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
    assert r.probe((0.0, 0.03), "u").shape == (4, 2)


# ── the Results interface ────────────────────────────────────────────────────
def test_results_interface():
    r = lp.load(_legacy("freq_meca.h5"))
    v = r.field("u", f=900.0)                                    # nearest entry: 1000 Hz
    assert v.coordinate == 1000.0 and v.values.shape == (v.mesh.n_nodes, 2)
    assert r.field("u_meca", index=1).coordinate == 1000.0       # former field names still work
    p = r.probe((0.0, 0.03), "u")
    k = r.mesh("solid").nearest((0.0, 0.03))
    np.testing.assert_array_equal(p[1], v.values[k])
    with pytest.raises(KeyError, match="no field 'p'"):
        r.field("p")
    modes = lp.load(_legacy("eigen_fourier.h5"))
    assert modes.orders == [0, 1] and np.all(np.isfinite(modes.zeta))
    assert modes.order(1).mode(0).values.shape[1] == 3            # (U_r, U_z, U_θ)
    np.testing.assert_allclose(lp.results.spl(np.array([20e-6 * np.sqrt(2)])), [0.0], atol=1e-12)


def test_plots(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    r = lp.load(_legacy("freq_meca.h5"))
    lp.plot.field(r, "u", f=100.0)
    lp.plot.field(r, "u", f=100.0, component="z")
    lp.plot.response(r, at=(0.0, 0.03), name="u", component="z", db=True)
    lp.plot.mode(lp.load(_legacy("eigen_fourier.h5")).order(1), 0, component="theta")
    lp.plot.history(lp.load(_legacy("time.h5")), at=(0.0, 0.03))
    ax = lp.plot.field(r.field("u", index=0))                     # a FieldView works too
    ax.figure.savefig(tmp_path / "f.png")


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


def test_grid_plots_and_save_metadata(tmp_path, mesh_dir):
    import matplotlib
    matplotlib.use("Agg")
    modes = lp.load(_legacy("eigen_fourier.h5"))
    lp.plot.modes_grid(modes.order(0), indices=[0, 1])
    with pytest.raises(ValueError, match="pick one mode set"):
        lp.plot.modes_grid(modes)
    _, study = api.speaker_freq_meca(mesh_dir, [100.0])
    study.save(tmp_path / "m.h5", case="demo", tones=np.array([1.0, 2.0]))
    meta = lp.load(tmp_path / "m.h5").meta
    assert meta["case"] == "demo" and meta["tones"] == [1.0, 2.0] and meta["study"] == "FrequencyResponse"
