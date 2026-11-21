"""Multi-parameter sweeps (grid / zip, item and .geo geometry parameters), sweep results as labelled data, and the
exports for post-processing elsewhere (netCDF for data, VTKHDF for field series)."""
from pathlib import Path

import numpy as np
import pytest

import loudpy as lp

PLATE = """// clamped plate, meridian section: radius a, thickness h (DefineConstant: model parameters may set them)
SetFactory("OpenCASCADE");
DefineConstant[ a = 0.1, h = 0.002 ];
Point(1) = {0, 0, 0}; Point(2) = {a, 0, 0}; Point(3) = {a, h, 0}; Point(4) = {0, h, 0};
Line(1) = {1, 2}; Line(2) = {2, 3}; Line(3) = {3, 4}; Line(4) = {4, 1};
Physical Curve("plate") = {1, 2, 3, 4};
Physical Curve("rim") = {2};
Physical Curve("axis") = {4};
"""


def plate_model(tmp_path, **params):
    geo = tmp_path / "plate.geo"
    geo.write_text(PLATE)
    p = lp.Parameters(**{"E": 2.1e11, "a": 0.1, **params})
    steel = lp.Material("Steel", rho=7850.0, E=2.1e11, nu=0.3)
    model = lp.Model(geo, parameters=p).add(
        lp.Solid("plate", steel, E=lp.P("E"), damping=lp.Undamped(), mesh_size=2e-3),
        lp.Clamped("rim"), lp.Clamped("axis", dofs="r"))
    model.mesh(tmp_path / "plate.msh")
    return model


def test_geo_parameters(tmp_path):
    model = plate_model(tmp_path)
    assert model.source.geo_constants == ["a", "h"] and model.source.geo_values() == {"a": 0.1}
    big = model.with_parameters(a=0.15)
    big.mesh(tmp_path / "big.msh")
    assert np.isclose(big.meshed.coords[:, 0].max(), 0.15) and np.isclose(model.meshed.coords[:, 0].max(), 0.1)
    assert model.meshed.coords[:, 1].max() == pytest.approx(0.002)                 # h: the .geo's own value


def test_grid_and_zip_sweeps(tmp_path, verify):
    model = plate_model(tmp_path)
    E, a = [1.9e11, 2.1e11], [0.08, 0.1]
    grid = lp.Sweep(lp.Modes(model), {"E": E, "a": a}).solve(f_min=10.0, f_max=4000.0)
    assert len(grid) == 4 and grid.combine == "grid" and grid.names == ["E", "a"]
    f = grid.evaluate(lambda r: r.frequencies[:2], name="f")
    assert f.dims == ("E", "a", "axis0") and f.values.shape == (2, 2, 2)
    np.testing.assert_allclose(f.coords["a"], a)
    # physics: same mesh → f ∝ √E exactly; Kirchhoff f ∝ h / a² (here h fixed): ≈ (0.1 / 0.08)² between radii
    np.testing.assert_allclose(f.values[1] / f.values[0], np.sqrt(2.1 / 1.9), rtol=1e-9)
    ratio = f.values[:, 0, 0] / f.values[:, 1, 0]
    verify("Sweeps", "Plate radius sweep through a .geo DefineConstant", "f₁(a = 0.08) / f₁(a = 0.1) − (0.1/0.08)²",
           abs(ratio[1] / 1.5625 - 1), bound=0.01, reference="Kirchhoff plate: f ∝ 1/a²")
    assert np.all(np.abs(ratio / 1.5625 - 1) < 0.01), ratio
    # each run = the model built with those values
    one = lp.Modes(model.with_parameters(E=1.9e11, a=0.08))
    one.model.mesh(tmp_path / "one.msh")
    np.testing.assert_array_equal(grid.at(E=1.9e11, a=0.08).frequencies, one.solve(f_min=10.0, f_max=4000.0).frequencies)
    zipped = lp.Sweep(lp.Modes(model), {"E": E, "a": a}, combine="zip").solve(f_min=10.0, f_max=4000.0)
    assert len(zipped) == 2 and zipped.runs == [{"E": 1.9e11, "a": 0.08}, {"E": 2.1e11, "a": 0.1}]
    fz = zipped.evaluate(lambda r: r.frequencies[0], name="f")
    assert fz.dims == ("run",) and fz.meta["runs"] == zipped.runs
    np.testing.assert_array_equal(fz.values, [f.values[0, 0, 0], f.values[1, 1, 0]])
    assert list(grid.table().columns) == ["E", "a"]
    # one file, read back with its runs
    back = lp.load(grid.save(tmp_path / "grid.h5"))
    assert back.names == ["E", "a"] and back.combine == "grid" and back.runs == grid.runs
    np.testing.assert_array_equal(back.evaluate(lambda r: r.frequencies[:2], name="f").values, f.values)
    # exported for elsewhere and read back
    f.export(tmp_path / "f.nc")
    g = lp.load_data(tmp_path / "f.nc")
    assert g.dims == f.dims and g.meta["runs"] == grid.runs
    np.testing.assert_array_equal(g.values, f.values)
    with pytest.raises(lp.errors.ModelError, match="differ in length"):
        lp.Sweep(lp.Modes(model), {"E": E, "a": [0.1]}, combine="zip")
    with pytest.raises(lp.errors.ModelError, match="no parameter"):
        lp.Sweep(lp.Modes(model), {"rho": [1.0]})


def test_netcdf_round_trip(tmp_path):
    f = np.geomspace(20, 2000, 7)
    ang = np.arange(0, 91, 30.0)
    p = lp.Data(np.exp(1j * f[:, None] / 100) * np.cos(np.radians(ang))[None], {"f": f, "angle": ang}, "p",
                meta={"at": "R = 1 m"})
    back = lp.load_data(p.export(tmp_path / "p.nc"))
    assert back.dims == ("f", "angle") and back.quantity.name == "p" and back.meta == {"at": "R = 1 m"}
    np.testing.assert_array_equal(back.values, p.values)
    u = lp.Data(np.arange(6.0).reshape(2, 3), {"f": [10.0, 20.0], "component": np.array(["r", "z", "theta"])}, "u")
    assert list(lp.load_data(u.export(tmp_path / "u.nc")).coords["component"]) == ["r", "z", "theta"]
    from scipy.io import netcdf_file
    with netcdf_file(str(tmp_path / "p.nc"), "r", mmap=False) as nc:          # what other tools see
        assert set(nc.variables) == {"f", "angle", "p_real", "p_imag"}
        assert nc.variables["f"].units.decode() == "Hz" and nc.Conventions.decode() == "CF-1.8"


def test_vtkhdf_series(tmp_path):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy
    res = lp.load(Path(__file__).parent / "data" / "legacy" / "freq_meca.h5")
    path = res.export(tmp_path / "u.vtkhdf")
    rd = vtk.vtkHDFReader()
    rd.SetFileName(str(path))
    rd.UpdateInformation()
    steps = rd.GetOutputInformation(0).Get(vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS())
    np.testing.assert_allclose(steps, res.frequencies)
    for i in range(len(res)):
        rd.SetStep(i)
        rd.Update()
        out, g = rd.GetOutput(), res.field("u", index=i).to_pyvista()
        assert out.GetNumberOfPoints() == g.n_points
        for name in g.point_data:
            np.testing.assert_array_equal(vtk_to_numpy(out.GetPointData().GetArray(name)), np.asarray(g.point_data[name]))
