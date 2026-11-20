"""
Generic element kernels (Numba) on tabulated Lagrange bases, and the global assembly.

Every kernel loops over the cells of one group in parallel and, at each quadrature point, maps the
reference cell to the cell through its own nodes (isoparametric: J = Σ_a x_a ⊗ ∇N_a), so curved
high-order cells are integrated exactly as gmsh placed them. Inputs:

    ec      (n_e, n, gdim)   cell node coordinates (Space.cell_coords)
    N, dN   (n_q, n), (n_q, n, tdim)   basis tabulated at the rule's points (LagrangeElement.tabulate)
    w       (n_q,)           rule weights
    coef    (n_e, n_q)       coefficient at every quadrature point (real or complex)
    axisym  bool             multiply the measure by 2π r (r = first coordinate)

Volume kernels need gdim = tdim; `boundary_mass` / `boundary_load` integrate on cells of dimension
gdim − 1 (lines in 2D, triangles in 3D) with the measure √det(JᵀJ).

The functions `mass`, `laplace`, `load`, `boundary_mass`, `boundary_load` wrap the kernels for a Space and
a quadrature rule; `Assembler(space)` builds the CSR pattern once and sums element blocks into matrices
and vectors.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from loudpy.fem.indexing import _fill_ij_square
from loudpy.fem.sparse import _SparseAssemblerBase
from loudpy.fem.sparse_kernels import _scatter_add

__all__ = ["Assembler", "mass", "laplace", "load", "boundary_mass", "boundary_load", "quadrature_points",
           "coefficient", "tabulate", "expand_components", "diffusion", "boundary_normal_mass"]

TWO_PI = 2.0 * np.pi


# ------------------------------------------------------------------------------------ geometry (njit)

@njit(cache=True, fastmath=False, inline="always")
def _jacobian(ec_e, dNq, J):
    """J[i, j] = Σ_a x_a[i] ∂N_a/∂ξ_j, by np.dot (the summation order of the former kernels: on flat
    cells det J cancels, and a plain loop differs from them by up to 2e-11 relative)."""
    gdim = ec_e.shape[1]
    tdim = dNq.shape[1]
    for i in range(gdim):
        for j in range(tdim):
            J[i, j] = np.dot(dNq[:, j], ec_e[:, i])


@njit(cache=True, fastmath=False, inline="always")
def _det_inv(J, Ji):
    """Determinant and inverse of a square 1×1, 2×2 or 3×3 Jacobian."""
    d = J.shape[0]
    if d == 1:
        det = J[0, 0]
        Ji[0, 0] = 1.0 / det
    elif d == 2:
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        Ji[0, 0] = J[1, 1] / det
        Ji[0, 1] = -J[0, 1] / det
        Ji[1, 0] = -J[1, 0] / det
        Ji[1, 1] = J[0, 0] / det
    else:
        c00 = J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]
        c01 = J[1, 2] * J[2, 0] - J[1, 0] * J[2, 2]
        c02 = J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]
        det = J[0, 0] * c00 + J[0, 1] * c01 + J[0, 2] * c02
        Ji[0, 0] = c00 / det
        Ji[1, 0] = c01 / det
        Ji[2, 0] = c02 / det
        Ji[0, 1] = (J[0, 2] * J[2, 1] - J[0, 1] * J[2, 2]) / det
        Ji[1, 1] = (J[0, 0] * J[2, 2] - J[0, 2] * J[2, 0]) / det
        Ji[2, 1] = (J[0, 1] * J[2, 0] - J[0, 0] * J[2, 1]) / det
        Ji[0, 2] = (J[0, 1] * J[1, 2] - J[0, 2] * J[1, 1]) / det
        Ji[1, 2] = (J[0, 2] * J[1, 0] - J[0, 0] * J[1, 2]) / det
        Ji[2, 2] = (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]) / det
    return det


@njit(cache=True, fastmath=False, inline="always")
def _gradients(dNq, J, Ji, det, G):
    """Physical gradients G[a, i] = ∂N_a/∂x_i = Σ_j ∂N_a/∂ξ_j (J⁻¹)[j, i]. In 2D the cofactor form
    (J₁₁ ∂_ξ N − J₁₀ ∂_η N) / det of the former kernels (same rounding on flat, ill-conditioned cells)."""
    n, d = G.shape
    if d == 2:
        inv = 1.0 / det
        for a in range(n):
            G[a, 0] = (J[1, 1] * dNq[a, 0] - J[1, 0] * dNq[a, 1]) * inv
            G[a, 1] = (-J[0, 1] * dNq[a, 0] + J[0, 0] * dNq[a, 1]) * inv
    else:
        for a in range(n):
            for i in range(d):
                s = 0.0
                for j in range(d):
                    s += dNq[a, j] * Ji[j, i]
                G[a, i] = s


@njit(cache=True, fastmath=False, inline="always")
def _surface_measure(J):
    """√det(JᵀJ) for a gdim × tdim Jacobian (tdim = gdim − 1: length of a curve, area of a surface)."""
    gdim, tdim = J.shape
    if tdim == 1:
        s = 0.0
        for i in range(gdim):
            s += J[i, 0] * J[i, 0]
        return np.sqrt(s)
    # tdim == 2, gdim == 3: |∂x/∂ξ × ∂x/∂η|
    cx = J[1, 0] * J[2, 1] - J[2, 0] * J[1, 1]
    cy = J[2, 0] * J[0, 1] - J[0, 0] * J[2, 1]
    cz = J[0, 0] * J[1, 1] - J[1, 0] * J[0, 1]
    return np.sqrt(cx * cx + cy * cy + cz * cz)


@njit(cache=True, fastmath=False, inline="always")
def _radius(ec_e, Nq):
    r = 0.0
    for a in range(ec_e.shape[0]):
        r += Nq[a] * ec_e[a, 0]
    return r


# -------------------------------------------------------------------------------------- kernels (njit)

@njit(parallel=True, cache=True, fastmath=False)
def _mass_kernel(ec, N, dN, w, coef, axisym, out):
    n_e, n, gdim = ec.shape
    n_q = N.shape[0]
    for e in prange(n_e):
        J = np.empty((gdim, gdim))
        Ji = np.empty((gdim, gdim))
        Me = np.zeros((n, n), dtype=out.dtype)
        for q in range(n_q):
            _jacobian(ec[e], dN[q], J)
            dx = w[q] * abs(_det_inv(J, Ji))
            if axisym:
                dx *= TWO_PI * _radius(ec[e], N[q])
            c = coef[e, q] * dx
            for a in range(n):
                ca = c * N[q, a]
                for b in range(n):
                    Me[a, b] += ca * N[q, b]
        out[e] = Me.ravel()


@njit(parallel=True, cache=True, fastmath=False)
def _laplace_kernel(ec, N, dN, w, coef, axisym, out):
    n_e, n, gdim = ec.shape
    n_q = N.shape[0]
    for e in prange(n_e):
        J = np.empty((gdim, gdim))
        Ji = np.empty((gdim, gdim))
        G = np.empty((n, gdim))
        Ke = np.zeros((n, n), dtype=out.dtype)
        for q in range(n_q):
            _jacobian(ec[e], dN[q], J)
            det = _det_inv(J, Ji)
            dx = w[q] * abs(det)
            if axisym:
                dx *= TWO_PI * _radius(ec[e], N[q])
            _gradients(dN[q], J, Ji, det, G)                # ∇N = J⁻ᵀ ∇̂N
            c = coef[e, q] * dx
            for a in range(n):
                for b in range(n):
                    s = 0.0
                    for i in range(gdim):
                        s += G[a, i] * G[b, i]
                    Ke[a, b] += c * s
        out[e] = Ke.ravel()


@njit(parallel=True, cache=True, fastmath=False)
def _diffusion_diag_kernel(ec, N, dN, w, coef, axisym, out):
    """Blocks ∫ Σ_i coef_i ∂_i N_a ∂_i N_b dx, coef (n_e, n_q, gdim): a diagonal (e.g. PML) tensor."""
    n_e, n, gdim = ec.shape
    n_q = N.shape[0]
    for e in prange(n_e):
        J = np.empty((gdim, gdim))
        Ji = np.empty((gdim, gdim))
        G = np.empty((n, gdim))
        Ke = np.zeros((n, n), dtype=out.dtype)
        for q in range(n_q):
            _jacobian(ec[e], dN[q], J)
            det = _det_inv(J, Ji)
            dx = w[q] * abs(det)
            if axisym:
                dx *= TWO_PI * _radius(ec[e], N[q])
            _gradients(dN[q], J, Ji, det, G)                # ∇N = J⁻ᵀ ∇̂N
            for i in range(gdim):
                c = coef[e, q, i] * dx
                for a in range(n):
                    ca = c * G[a, i]
                    for b in range(n):
                        Ke[a, b] += ca * G[b, i]
        out[e] = Ke.ravel()


@njit(parallel=True, cache=True, fastmath=False)
def _load_kernel(ec, N, dN, w, coef, axisym, out):
    n_e, n, gdim = ec.shape
    n_q = N.shape[0]
    for e in prange(n_e):
        J = np.empty((gdim, gdim))
        Ji = np.empty((gdim, gdim))
        for a in range(n):
            out[e, a] = 0.0
        for q in range(n_q):
            _jacobian(ec[e], dN[q], J)
            dx = w[q] * abs(_det_inv(J, Ji))
            if axisym:
                dx *= TWO_PI * _radius(ec[e], N[q])
            c = coef[e, q] * dx
            for a in range(n):
                out[e, a] += c * N[q, a]


@njit(parallel=True, cache=True, fastmath=False)
def _boundary_mass_kernel(ec, N, dN, w, coef, axisym, out):
    n_e, n, gdim = ec.shape
    n_q, _, tdim = dN.shape
    for e in prange(n_e):
        J = np.empty((gdim, tdim))
        Me = np.zeros((n, n), dtype=out.dtype)
        for q in range(n_q):
            _jacobian(ec[e], dN[q], J)
            dx = w[q] * _surface_measure(J)
            if axisym:
                dx *= TWO_PI * _radius(ec[e], N[q])
            c = coef[e, q] * dx
            for a in range(n):
                ca = c * N[q, a]
                for b in range(n):
                    Me[a, b] += ca * N[q, b]
        out[e] = Me.ravel()


@njit(parallel=True, cache=True, fastmath=False)
def _boundary_load_kernel(ec, N, dN, w, coef, axisym, out):
    n_e, n, gdim = ec.shape
    n_q, _, tdim = dN.shape
    for e in prange(n_e):
        J = np.empty((gdim, tdim))
        for a in range(n):
            out[e, a] = 0.0
        for q in range(n_q):
            _jacobian(ec[e], dN[q], J)
            dx = w[q] * _surface_measure(J)
            if axisym:
                dx *= TWO_PI * _radius(ec[e], N[q])
            c = coef[e, q] * dx
            for a in range(n):
                out[e, a] += c * N[q, a]


@njit(parallel=True, cache=True, fastmath=False)
def _boundary_normal_mass_kernel(ec, N, dN, w, axisym, out):
    """Blocks ∫_Γ N_a N_b n_i (2π r) ds for every component i of the (unnormalised) normal n ds:
    (∂z/∂ξ, −∂r/∂ξ) on lines in 2D (the right of the direction of travel), ∂x/∂ξ × ∂x/∂η on
    triangles in 3D. out (n_e, gdim, n²)."""
    n_e, n, gdim = ec.shape
    n_q, _, tdim = dN.shape
    for e in prange(n_e):
        J = np.empty((gdim, tdim))
        nds = np.empty(gdim)
        Me = np.zeros((gdim, n, n))
        for q in range(n_q):
            _jacobian(ec[e], dN[q], J)
            if tdim == 1:
                nds[0] = J[1, 0]
                nds[1] = -J[0, 0]
            else:
                nds[0] = J[1, 0] * J[2, 1] - J[2, 0] * J[1, 1]
                nds[1] = J[2, 0] * J[0, 1] - J[0, 0] * J[2, 1]
                nds[2] = J[0, 0] * J[1, 1] - J[1, 0] * J[0, 1]
            common = w[q]
            if axisym:
                common = TWO_PI * _radius(ec[e], N[q]) * w[q]
            for i in range(gdim):
                c = common * nds[i]
                for a in range(n):
                    ca = c * N[q, a]
                    for b in range(n):
                        Me[i, a, b] += ca * N[q, b]
        for i in range(gdim):
            out[e, i] = Me[i].ravel()


# ------------------------------------------------------------------------------------- Python wrappers

def tabulate(element, rule):
    """(N, dN, w) of `element` at the points of `rule`, contiguous, for the kernels."""
    N, dN = element.tabulate(rule.points)
    return np.ascontiguousarray(N), np.ascontiguousarray(dN), np.ascontiguousarray(rule.weights, dtype=float)


def expand_components(blocks, components: int) -> np.ndarray:
    """Scalar blocks (n_e, n²) → blocks (n_e, (c·n)²) with the same block on every component,
    DOFs interleaved per node (c·a + i)."""
    b = np.asarray(blocks)
    n_e = b.shape[0]
    n = int(round(np.sqrt(b.shape[1])))
    c = int(components)
    out = np.zeros((n_e, n, c, n, c), dtype=b.dtype)
    s = b.reshape(n_e, n, n)
    for i in range(c):
        out[:, :, i, :, i] = s
    return out.reshape(n_e, (c * n) ** 2)


def quadrature_points(ec, N) -> np.ndarray:
    """Physical coordinates of the quadrature points, (n_e, n_q, gdim)."""
    return np.einsum("qa,ead->eqd", N, ec)


def coefficient(value, ec, N) -> np.ndarray:
    """A coefficient at every quadrature point (n_e, n_q): a number, an array, or f(points (m, gdim))."""
    n_e, n_q = ec.shape[0], N.shape[0]
    if callable(value):
        x = quadrature_points(ec, N)
        out = np.asarray(value(x.reshape(-1, ec.shape[2])))
        return np.ascontiguousarray(out.reshape(n_e, n_q))
    out = np.asarray(value)
    dtype = complex if np.iscomplexobj(out) else float
    return np.ascontiguousarray(np.broadcast_to(out, (n_e, n_q)), dtype=dtype)


def _blocks(kernel, space, rule, value, axisymmetric, square):
    """Element blocks of every group, concatenated in the space's cell order."""
    N, dN = space.element.tabulate(rule.points)
    w = np.ascontiguousarray(rule.weights)
    n = space.element.n_nodes
    out_all = []
    for ec in space.cell_coords:
        c = coefficient(value, ec, N)
        out = np.empty((len(ec), n * n if square else n), dtype=c.dtype)
        if len(ec):
            kernel(ec, N, dN, w, c, bool(axisymmetric), out)
        out_all.append(out)
    return np.concatenate(out_all) if out_all else np.zeros((0, n * n if square else n))


def _check_volume(space):
    if space.element.dim != space.gdim:
        raise ValueError(f"volume kernel on {space.element.dim}D cells in {space.gdim}D: use boundary_*")


def _check_boundary(space):
    if space.element.dim != space.gdim - 1:
        raise ValueError(f"boundary kernel needs cells of dimension {space.gdim - 1}")


def mass(space, rule, coef=1.0, axisymmetric=False) -> np.ndarray:
    """Blocks ∫ coef N_a N_b (2π r) dx, (n_cells, n²) (scalar space: one component)."""
    _check_volume(space)
    return _blocks(_mass_kernel, space, rule, coef, axisymmetric, True)


def laplace(space, rule, coef=1.0, axisymmetric=False) -> np.ndarray:
    """Blocks ∫ coef ∇N_a · ∇N_b (2π r) dx, (n_cells, n²)."""
    _check_volume(space)
    return _blocks(_laplace_kernel, space, rule, coef, axisymmetric, True)


def diffusion(space, rule, coef, axisymmetric=False) -> np.ndarray:
    """Blocks ∫ Σ_i coef_i ∂_i N_a ∂_i N_b (2π r) dx with a diagonal tensor coef (n_e, n_q, gdim) per group
    (one array, or a list of arrays: one per group)."""
    _check_volume(space)
    N, dN, w = tabulate(space.element, rule)
    coefs = coef if isinstance(coef, (list, tuple)) else [coef]
    n = space.element.n_nodes
    out_all = []
    for ec, c in zip(space.cell_coords, coefs):
        c = np.ascontiguousarray(c)
        out = np.empty((len(ec), n * n), dtype=c.dtype)
        if len(ec):
            _diffusion_diag_kernel(ec, N, dN, w, c, bool(axisymmetric), out)
        out_all.append(out)
    return np.concatenate(out_all)


def load(space, rule, f, axisymmetric=False) -> np.ndarray:
    """Blocks ∫ f N_a (2π r) dx, (n_cells, n)."""
    _check_volume(space)
    return _blocks(_load_kernel, space, rule, f, axisymmetric, False)


def boundary_mass(space, rule, coef=1.0, axisymmetric=False) -> np.ndarray:
    """Blocks ∫_Γ coef N_a N_b (2π r) ds on boundary cells, (n_cells, n²)."""
    _check_boundary(space)
    return _blocks(_boundary_mass_kernel, space, rule, coef, axisymmetric, True)


def boundary_normal_mass(space, rule, axisymmetric=False) -> np.ndarray:
    """Blocks ∫_Γ N_a N_b n_i (2π r) ds per normal component, (n_cells, gdim, n²); n points to the right
    of the direction of travel along a line (2D), along ∂x/∂ξ × ∂x/∂η on a triangle (3D)."""
    _check_boundary(space)
    N, dN, w = tabulate(space.element, rule)
    n = space.element.n_nodes
    out_all = []
    for ec in space.cell_coords:
        out = np.empty((len(ec), space.gdim, n * n))
        if len(ec):
            _boundary_normal_mass_kernel(ec, N, dN, w, bool(axisymmetric), out)
        out_all.append(out)
    return np.concatenate(out_all)


def boundary_load(space, rule, f, axisymmetric=False) -> np.ndarray:
    """Blocks ∫_Γ f N_a (2π r) ds on boundary cells, (n_cells, n)."""
    _check_boundary(space)
    return _blocks(_boundary_load_kernel, space, rule, f, axisymmetric, False)


class Assembler(_SparseAssemblerBase):
    """
    Global assembly over a Space: the CSR pattern of all cells is built once; `matrix(blocks)` sums
    element blocks (n_cells, m²) and `vector(blocks)` element vectors (n_cells, m), m = DOFs per cell, in
    the space's cell order (groups concatenated).
    """

    def __init__(self, space):
        self.space = space
        self.dofs = np.ascontiguousarray(np.concatenate(space.cell_dofs)) if space.cell_dofs else \
            np.zeros((0, 0), np.int32)
        n_e, m = self.dofs.shape
        size = n_e * m * m
        I = np.empty(size, dtype=np.int32)
        J = np.empty(size, dtype=np.int32)
        if n_e:
            _fill_ij_square(self.dofs, I, J, 0)
        self._finalize_pattern(I, J, size, (space.n_dofs, space.n_dofs))

    def matrix(self, blocks):
        """CSR matrix from element blocks (real or complex)."""
        V = np.ascontiguousarray(np.asarray(blocks).reshape(-1))
        if V.size != self.V_size:
            raise ValueError(f"{V.size} block values for a pattern of {self.V_size}")
        return self._to_csr(V)

    def vector(self, blocks) -> np.ndarray:
        """Global vector from element vectors."""
        b = np.asarray(blocks)
        if b.shape != self.dofs.shape:
            raise ValueError(f"element vectors {b.shape}, expected {self.dofs.shape}")
        idx = self.dofs.ravel()
        if np.iscomplexobj(b):
            out = np.zeros(self.space.n_dofs, complex)
            _scatter_add(np.ascontiguousarray(b.real.ravel()), idx, out.real)
            _scatter_add(np.ascontiguousarray(b.imag.ravel()), idx, out.imag)
            return out
        out = np.zeros(self.space.n_dofs)
        _scatter_add(np.ascontiguousarray(b.ravel(), dtype=float), idx, out)
        return out
