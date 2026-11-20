"""
Derived fields of the 2D axisymmetric results (azimuthal order 0), computed from the FE fields and the
physical data of the mesh groups (stored with the results, format 3).

    strain(u)                    ε = (ε_rr, ε_zz, γ_rz, ε_θθ): ∂u_r/∂r, ∂u_z/∂z, ∂u_r/∂z + ∂u_z/∂r, u_r/r
    stress(u, groups)            σ = D(E, ν) ε of each element's region (σ_rr, σ_zz, τ_rz, σ_θθ) [Pa]
    von_mises(σ)                 √(½[(σ_rr−σ_zz)² + (σ_zz−σ_θθ)² + (σ_θθ−σ_rr)²] + 3τ_rz²); for a harmonic
                                 amplitude its maximum over the cycle
    particle_velocity(p, groups) v = −∇p / (jωρ) (r, z) [m/s]                         (time factor e^{jωt})
    intensity(p, groups)         active intensity I = ½ Re(p v*) (r, z) [W/m²]

Gradients are those of the isoparametric field at the nodes of each element, averaged over the elements
sharing a node (nodal recovery). On the axis ε_θθ = ∂u_r/∂r (the limit of u_r / r). Inside a PML the air
values are not physical (complex stretched coordinates).
"""
from __future__ import annotations

import numpy as np

from loudpy.errors import ResultsError

TENSOR = ("rr", "zz", "rz", "tt")
VECTOR = ("r", "z")


def _element_node_gradients(mesh, values):
    """
    For every 2D cell group: (group name, rows (n_e, k), field at the element nodes (n_e, k, c), gradients at the
    element nodes (n_e, k, c, 2)). values: (n_nodes,) or (n_nodes, c).
    """
    from loudpy.fem.reference import element_for_cells
    V = values[:, None] if values.ndim == 1 else values
    out = []
    for name, g in mesh.groups.items():
        if g.dim != 2 or not len(g):
            continue
        cells = np.asarray(g.cells)
        k = cells.shape[1]
        elem = element_for_cells(2, k)
        _, dN = elem.tabulate(elem.nodes)                                  # (k at, k basis, 2)
        rows = mesh._rows(cells)
        X = mesh.coords[rows]                                              # (n_e, k, 2)
        A = np.einsum("ebj,abi->eaji", X, dN)                              # ∂x_j/∂ξ_i at each element node
        Ainv = np.linalg.inv(A)
        G = np.einsum("abi,eaij->eabj", dN, Ainv)                          # ∂N_b/∂x_j at node a
        Ve = V[rows]                                                       # (n_e, k, c)
        grads = np.einsum("eabj,ebc->eacj", G, Ve)                         # (n_e, k, c, 2)
        out.append((name, rows, Ve, grads))
    if not out:
        raise ResultsError("derived fields need the 2D cells of the mesh")
    return out


def _average(n_nodes, parts, n_comp):
    """Average over the element nodes sharing each node: parts = [(rows (n_e, k), values (n_e, k, c))]."""
    dtype = np.result_type(*[v.dtype for _, v in parts])
    acc = np.zeros((n_nodes, n_comp), dtype)
    cnt = np.zeros(n_nodes)
    for rows, v in parts:
        np.add.at(acc, rows.ravel(), v.reshape(-1, n_comp))
        np.add.at(cnt, rows.ravel(), 1.0)
    used = cnt > 0
    acc[used] /= cnt[used, None]
    acc[~used] = np.nan
    return acc


def _strain_at_element_nodes(mesh, rows, Ve, grads):
    r = mesh.coords[rows][..., 0]
    ur = Ve[..., 0]
    durdr, durdz = grads[..., 0, 0], grads[..., 0, 1]
    duzdr, duzdz = grads[..., 1, 0], grads[..., 1, 1]
    on_axis = np.abs(r) <= 1e-12 * max(float(np.abs(mesh.coords).max()), 1e-300)
    ett = np.where(on_axis, durdr, ur / np.where(on_axis, 1.0, r))
    return np.stack([durdr, duzdz, durdz + duzdr, ett], axis=-1)       # (n_e, k, 4)


def _check_axisymmetric(field):
    if field.values.ndim != 2 or field.values.shape[1] != 2:
        raise ResultsError(f"strain and stress need an axisymmetric displacement (u_r, u_z); "
                           f"{field.name} has shape {field.values.shape} (azimuthal orders m ≥ 1: not available)")


def strain(field):
    """Strain (ε_rr, ε_zz, γ_rz, ε_θθ) of an axisymmetric displacement Field (see the module docstring)."""
    from loudpy.results.data import Field
    _check_axisymmetric(field)
    parts = [(rows, _strain_at_element_nodes(field.mesh, rows, Ve, gr))
             for _, rows, Ve, gr in _element_node_gradients(field.mesh, field.values)]
    vals = _average(field.mesh.n_nodes, parts, 4)
    return Field("strain", field.mesh, vals, field.coordinate, field.coord, field.extra, "strain",
                 {**field.meta, "components": TENSOR, "derived": "strain of u"})


def elasticity(E: float, nu: float) -> np.ndarray:
    """Axisymmetric elasticity matrix for (ε_rr, ε_zz, γ_rz, ε_θθ)."""
    f = E / ((1 + nu) * (1 - 2 * nu))
    return f * np.array([[1 - nu, nu, 0.0, nu], [nu, 1 - nu, 0.0, nu],
                         [0.0, 0.0, (1 - 2 * nu) / 2, 0.0], [nu, nu, 0.0, 1 - nu]])


def stress(field, groups: dict):
    """Stress (σ_rr, σ_zz, τ_rz, σ_θθ) [Pa] of an axisymmetric displacement Field; groups: {mesh group: {"E",
    "nu", …}} (res.model.groups). Elastic stress (the damping is not included)."""
    from loudpy.results.data import Field
    _check_axisymmetric(field)
    parts = []
    for name, rows, Ve, gr in _element_node_gradients(field.mesh, field.values):
        g = groups.get(name, {})
        if "E" not in g or "nu" not in g:
            raise ResultsError(f"no elastic data (E, ν) for the mesh group {name!r}")
        eps = _strain_at_element_nodes(field.mesh, rows, Ve, gr)
        parts.append((rows, eps @ elasticity(g["E"], g["nu"]).T))
    vals = _average(field.mesh.n_nodes, parts, 4)
    return Field("stress", field.mesh, vals, field.coordinate, field.coord, field.extra, "stress",
                 {**field.meta, "components": TENSOR, "derived": "elastic stress of u"})


def _vm2(s):
    srr, szz, srz, stt = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    return 0.5 * ((srr - szz) ** 2 + (szz - stt) ** 2 + (stt - srr) ** 2) + 3 * srz ** 2


def _vm_bilinear(a, b):
    """The bilinear form of _vm2: _vm2(a + b) = _vm2(a) + 2 B(a, b) + _vm2(b)."""
    return 0.5 * (_vm2(a + b) - _vm2(a) - _vm2(b))


def von_mises(stress_field):
    """von Mises stress [Pa]; for a complex amplitude σ̂ its maximum over the cycle of Re(σ̂ e^{jφ})."""
    s = stress_field.values
    if np.iscomplexobj(s):
        a, b = s.real, s.imag                      # σ(φ) = a cos φ − b sin φ
        qa, qb, bab = _vm2(a), _vm2(b), _vm_bilinear(a, b)
        vm = np.sqrt(np.maximum(0.5 * (qa + qb) + np.sqrt((0.5 * (qa - qb)) ** 2 + bab ** 2), 0.0))
    else:
        vm = np.sqrt(np.maximum(_vm2(s), 0.0))
    return stress_field._scalar(vm, "von_mises")


def particle_velocity(field, groups: dict):
    """Particle velocity v = −∇p / (jωρ) (r, z) [m/s] of a harmonic pressure Field (ρ of each air group)."""
    from loudpy.results.data import Field
    if field.coord != "f" or field.values.ndim != 1:
        raise ResultsError("the particle velocity needs a harmonic pressure field (complex amplitudes over f)")
    w = 2 * np.pi * field.coordinate
    parts = []
    for name, rows, _, gr in _element_node_gradients(field.mesh, field.values):
        rho = groups.get(name, {}).get("rho")
        if rho is None:
            raise ResultsError(f"no density ρ for the mesh group {name!r}")
        parts.append((rows, -gr[:, :, 0, :] / (1j * w * rho)))
    vals = _average(field.mesh.n_nodes, parts, 2)
    return Field("particle_velocity", field.mesh, vals, field.coordinate, field.coord, field.extra,
                 "particle_velocity", {**field.meta, "components": VECTOR, "derived": "−∇p/(jωρ)"})


def intensity(field, groups: dict):
    """Active acoustic intensity ½ Re(p v*) (r, z) [W/m²] of a harmonic pressure Field."""
    from loudpy.results.data import Field
    v = particle_velocity(field, groups).values
    vals = 0.5 * np.real(field.values[:, None] * np.conj(v))
    return Field("intensity", field.mesh, vals, field.coordinate, field.coord, field.extra, "intensity",
                 {**field.meta, "components": VECTOR, "derived": "½ Re(p v*)"})
