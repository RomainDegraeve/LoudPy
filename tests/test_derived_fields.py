"""
Phase G3: derived fields vs analytical cases — strain and stress of uniform and quadratic displacements
(exact for the element orders that contain them), rigid motion, the von Mises maximum over a cycle, particle
velocity and intensity (quadratic pressure exact, plane wave), and the same names through result files.
"""
import h5py
import numpy as np
import pytest

import loudpy as lp
from loudpy.errors import ResultsError
from loudpy.fem.reference import cell_type_name
from loudpy.geometry.mesh import Mesh, MeshGroup
from loudpy.results import derived
from support import current_api as api
from support import fem_meshes as fm

E, NU = 2.0e11, 0.3


def square(n, p, scale=1.0, name="g"):
    d = fm.unit_square(n, p)
    return Mesh(d["node_tags"], d["coords"] * scale, {name: MeshGroup(name, 2, cell_type_name(2, d["cells"].shape[1]),
                                                                       d["cells"])})


@pytest.mark.parametrize("p", [1, 2, 3])
def test_strain_and_stress_are_exact(p):
    """Uniform strain (u_r = a r, u_z = b z; every order) and a quadratic displacement (orders ≥ 2): the
    strain and stress at the nodes equal the analytical ones; ε_θθ on the axis is the limit ∂u_r/∂r."""
    mesh = square(3, p)                                          # 0 ≤ r ≤ 1: nodes on the axis
    r, z = mesh.coords.T
    a, b = 2e-3, -1e-3
    cases = [(np.column_stack([a * r, b * z]), np.column_stack([a + 0 * r, b + 0 * r, 0 * r, a + 0 * r]))]
    if p >= 2:
        cases.append((np.column_stack([r ** 2 + r * z, z ** 2]), np.column_stack([2 * r + z, 2 * z, r, r + z])))
    D = derived.elasticity(E, NU)
    for U, eps in cases:
        u = lp.Field("u", mesh, U.astype(complex), 100.0)
        np.testing.assert_allclose(derived.strain(u).values, eps, atol=1e-12)
        sig = derived.stress(u, {"g": {"E": E, "nu": NU}})
        np.testing.assert_allclose(sig.values, eps @ D.T, atol=1e-12 * E)
        assert sig.component("tt").quantity.unit == "Pa"
    rigid = lp.Field("u", mesh, np.column_stack([0 * r, 1e-3 + 0 * z]).astype(complex), 100.0)
    np.testing.assert_allclose(derived.strain(rigid).values, 0.0, atol=1e-15)


def test_von_mises_of_an_amplitude_is_its_maximum_over_the_cycle():
    mesh = square(1, 1)
    s = np.zeros((mesh.n_nodes, 4), complex)
    s[:, 0] = 1.0 + 1.0j                                         # σ_rr(t) = cos φ − sin φ: amplitude √2
    s[:, 2] = 0.5j                                               # and a shear in quadrature
    sig = lp.Field("stress", mesh, s, 100.0, meta={"components": derived.TENSOR})
    vm = derived.von_mises(sig).values
    phi = np.linspace(0, 2 * np.pi, 20001)
    sr, tr = np.cos(phi) - np.sin(phi), -0.5 * np.sin(phi)
    brute = np.sqrt(sr ** 2 + 3 * tr ** 2).max()                  # σ_vM² = σ_rr² + 3 τ² (uniaxial + shear)
    np.testing.assert_allclose(vm, brute, rtol=1e-7)


def test_particle_velocity_and_intensity():
    c, rho, f = 343.0, 1.2, 500.0
    w, k = 2 * np.pi * f, 2 * np.pi * f / c
    mesh = square(6, 2, 0.5, "air")
    r, z = mesh.coords.T
    p = lp.Field("p", mesh, (1 + r ** 2 - 2 * r * z + 3 * z ** 2).astype(complex), f)        # quadratic: exact
    v = derived.particle_velocity(p, {"air": {"rho": rho}})
    grad = np.column_stack([2 * r - 2 * z, -2 * r + 6 * z])
    np.testing.assert_allclose(v.values, -grad / (1j * w * rho), atol=1e-12)
    fine = square(30, 2, 0.5, "air")
    wave = lp.Field("p", fine, np.exp(-1j * k * fine.coords[:, 1]), f)                         # plane wave along z
    inten = derived.intensity(wave, {"air": {"rho": rho}})
    np.testing.assert_allclose(inten.values[:, 1], 1 / (2 * rho * c), rtol=3e-3)               # |p|² / (2ρc)
    np.testing.assert_allclose(inten.values[:, 0], 0.0, atol=1e-12)


def test_derived_fields_from_a_result_file(tmp_path, mesh_dir):
    """The derived names through a saved and reloaded result: the file holds the material data they need."""
    res = lp.FrequencyResponse(api.speaker_problem(mesh_dir, acoustic=True), drive=lp.Force(1.0)).solve([200.0])
    back = lp.load(res.save(tmp_path / "r.h5"))
    u = back.field("u", f=200.0)
    v = back.field("v", f=200.0)
    np.testing.assert_allclose(v.values, 2j * np.pi * 200.0 * u.values)
    sig = back.field("stress", f=200.0)
    assert sig.values.shape == (u.mesh.n_nodes, 4) and np.isfinite(sig.values).all()
    np.testing.assert_array_equal(sig.values, res.field("stress", f=200.0).values)
    vm = back.field("von_mises", f=200.0)
    assert (vm.values >= 0).all() and vm.values.max() > 0
    inten = back.field("intensity", f=200.0)
    assert inten.values.shape[1] == 2 and inten.component("z").quantity.unit == "W/m²"
    at = sig.at(lp.Points([(0.03, 0.02), (0.04, 0.03)]))
    assert at.dims == ("point", "component") and list(at.coords["component"]) == list(derived.TENSOR)
    with h5py.File(tmp_path / "r.h5", "a") as fh:                 # without the model description: refused
        del fh["model"]
    with pytest.raises(ResultsError, match="physical data of the mesh groups"):
        lp.load(tmp_path / "r.h5").field("stress", f=200.0)
