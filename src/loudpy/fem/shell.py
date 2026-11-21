"""
loudpy.fem.shell — flat Reissner–Mindlin shell elements in 3D: MITC3 (3-node triangle) and MITC4 (4-node
quadrilateral), six degrees of freedom per node (u_x, u_y, u_z, θ_x, θ_y, θ_z, global axes).

Each cell has its own local frame (e1, e2 in its plane, e3 its normal; a slightly warped quadrilateral is
projected on its mean plane). In that frame, with rotations θ_1, θ_2 about e1, e2 and z the coordinate
across the thickness h,

    u_1 = u + z θ_2,    u_2 = v − z θ_1,    u_3 = w

    membrane   ε = (∂u/∂x, ∂v/∂y, ∂u/∂y + ∂v/∂x)                   D_m = E h / (1 − ν²) [1 ν 0; ν 1 0; 0 0 (1−ν)/2]
    bending    κ = (∂θ_2/∂x, −∂θ_1/∂y, ∂θ_2/∂y − ∂θ_1/∂x)          D_b = h² / 12 · D_m
    shear      γ = (∂w/∂x + θ_2, ∂w/∂y − θ_1)                      D_s = k G h I,  k = 5/6
    drilling   θ_3 tied to the in-plane rotation ω = ½(∂v/∂x − ∂u/∂y) by a penalty α G h ∫ (θ_3 − ω)²
               (Hughes–Brezzi, full integration; rigid rotations stay exact), α = 1e-3 by default

The transverse shear strains are the MITC ones (no shear locking in thin shells):

    MITC4 (Bathe & Dvorkin 1985): covariant e_ξz tied at the mid-points of the edges η = ±1, e_ηz at ξ = ±1,
          interpolated linearly across; 2 × 2 Gauss points;
    MITC3 (Lee & Bathe 2004): e_rt = e_rt⁽¹⁾ + c s, e_st = e_st⁽²⁾ − c r with c = e_rt⁽³⁾ − e_rt⁽¹⁾ − e_st⁽³⁾ + e_st⁽²⁾,
          tying points (½, 0), (0, ½), (½, ½); constant membrane and bending strains.

Mass: consistent, ρh for the translations, ρh³/12 for θ_1, θ_2 and α ρh³/12 for the drilling rotation, which
puts the drilling modes at the thickness-shear frequencies √(12 G / ρ) / h (beyond any shell model's range;
with ρh³/12 they would fall at √(12 α G / ρ) / h, e.g. 550 Hz for a 10 cm aluminium plate).

    K, M = shell_matrices(nodes, {"quad4": quads, "tri3": tris}, E=…, nu=…, rho=…, thickness=…)

nodes (n, 3); cells as node rows (gmsh order: counter-clockwise); E, nu, rho, thickness are numbers or one value
per cell (per cell type: a dict like the cells). K and M are sparse (6n × 6n), DOF 6·a + i.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from loudpy.errors import MeshError

NDOF = 6
SHEAR_FACTOR = 5.0 / 6.0
DRILL = 1e-3

_G = 1.0 / np.sqrt(3.0)
_QUAD_GAUSS = (np.array([[-_G, -_G], [_G, -_G], [_G, _G], [-_G, _G]]), np.ones(4))
_TRI3 = (np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]]), np.full(3, 1 / 6))     # degree 2


# ───────────────────────────────────────────────────────────────── shape functions (gmsh order)
def _shape(cell, pts):
    """N (n_q, n) and reference gradients dN (n_q, n, 2) of the linear triangle / bilinear quadrilateral."""
    pts = np.atleast_2d(pts)
    if cell == "tri3":
        r, s = pts[:, 0], pts[:, 1]
        N = np.column_stack([1 - r - s, r, s])
        dN = np.broadcast_to(np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]), (len(pts), 3, 2)).copy()
        return N, dN
    xi, eta = pts[:, 0], pts[:, 1]
    sx = np.array([-1.0, 1.0, 1.0, -1.0])
    se = np.array([-1.0, -1.0, 1.0, 1.0])
    N = 0.25 * (1 + xi[:, None] * sx) * (1 + eta[:, None] * se)
    dN = np.stack([0.25 * sx * (1 + eta[:, None] * se), 0.25 * se * (1 + xi[:, None] * sx)], axis=2)
    return N, dN


def local_frames(nodes, cells):
    """
    Local frames of flat cells: rotation R (n_e, 3, 3) whose rows are e1, e2, e3 (e3 the unit normal,
    right-handed with the node order), and the local in-plane coordinates xl (n_e, n, 2) of the nodes
    (origin at the cell centre).
    """
    X = np.asarray(nodes, float)[cells]                                   # (n_e, n, 3)
    if cells.shape[1] == 3:
        n = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
        e1 = X[:, 1] - X[:, 0]
    else:
        n = np.cross(X[:, 2] - X[:, 0], X[:, 3] - X[:, 1])                # diagonals: mean plane
        e1 = (X[:, 1] + X[:, 2]) - (X[:, 0] + X[:, 3])
    area2 = np.linalg.norm(n, axis=1)
    if np.any(area2 <= 0):
        raise MeshError(f"{int(np.sum(area2 <= 0))} degenerate shell cells (zero area)")
    e3 = n / area2[:, None]
    e1 = e1 - np.einsum("ij,ij->i", e1, e3)[:, None] * e3
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(e3, e1)
    R = np.stack([e1, e2, e3], axis=1)
    xc = X.mean(axis=1, keepdims=True)
    xl = np.einsum("eij,enj->eni", R[:, :2], X - xc)
    return R, xl


def warp_offsets(nodes, cells, R):
    """Distance of each node from its cell's mean plane (n_e, n): zero for triangles."""
    X = np.asarray(nodes, float)[cells]
    return np.einsum("ej,enj->en", R[:, 2], X - X.mean(axis=1, keepdims=True))


def _per_cell(v, n_e, name):
    a = np.broadcast_to(np.asarray(v, float), (n_e,)).copy()
    if np.any(~np.isfinite(a)):
        raise ValueError(f"shell {name}: non-finite value")
    return a


# ───────────────────────────────────────────────────────────────── B matrices in the local frame
# local DOF order per node: u v w θ1 θ2 θ3 (index 6a + i)
def cell_jacobian(dN, xl):
    """J (n_e, 2, 2) = ∂(x, y)/∂(ξ, η) at one point (dN (n, 2), xl (n_e, n, 2)), its inverse and det."""
    J = np.einsum("na,enb->eab", dN, xl)
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    inv = np.stack([np.stack([J[:, 1, 1], -J[:, 0, 1]], -1), np.stack([-J[:, 1, 0], J[:, 0, 0]], -1)], 1) / det[:, None, None]
    return J, inv, det


def _cart(dN, inv):
    """Cartesian gradients (n_e, n, 2) from reference ones (n, 2) and J⁻¹ (n_e, 2, 2)."""
    return np.einsum("na,eba->enb", dN, inv)          # ∂N/∂x_b = Σ_a ∂N/∂ξ_a (J⁻¹)_{b a}


def _membrane_bending(N, g, n):
    """B_m, B_b (n_e, 3, 6n) and the drilling row (n_e, 6n) at one point (N (n,), g (n_e, n, 2))."""
    n_e = g.shape[0]
    Bm = np.zeros((n_e, 3, NDOF * n))
    Bb = np.zeros((n_e, 3, NDOF * n))
    Bd = np.zeros((n_e, NDOF * n))
    for a in range(n):
        gx, gy = g[:, a, 0], g[:, a, 1]
        c = NDOF * a
        Bm[:, 0, c + 0] = gx
        Bm[:, 1, c + 1] = gy
        Bm[:, 2, c + 0] = gy
        Bm[:, 2, c + 1] = gx
        Bb[:, 0, c + 4] = gx                     # κx = ∂θ2/∂x
        Bb[:, 1, c + 3] = -gy                    # κy = −∂θ1/∂y
        Bb[:, 2, c + 4] = gy                     # κxy = ∂θ2/∂y − ∂θ1/∂x
        Bb[:, 2, c + 3] = -gx
        Bd[:, c + 5] = N[a]                      # θ3 − ½(∂v/∂x − ∂u/∂y)
        Bd[:, c + 1] = -0.5 * gx
        Bd[:, c + 0] = 0.5 * gy
    return Bm, Bb, Bd


def _shear_cart(N, g, n):
    """Cartesian transverse shear B_s (n_e, 2, 6n): γxz = ∂w/∂x + θ2, γyz = ∂w/∂y − θ1."""
    n_e = g.shape[0]
    Bs = np.zeros((n_e, 2, NDOF * n))
    for a in range(n):
        c = NDOF * a
        Bs[:, 0, c + 2] = g[:, a, 0]
        Bs[:, 0, c + 4] = N[a]
        Bs[:, 1, c + 2] = g[:, a, 1]
        Bs[:, 1, c + 3] = -N[a]
    return Bs


def _covariant_shear(cell, pt, xl):
    """Covariant shear rows (e_ξz, e_ηz) (n_e, 2, 6n) at a reference point: J · B_s (exact, no tying)."""
    N, dN = _shape(cell, pt)
    J, inv, _ = cell_jacobian(dN[0], xl)
    Bs = _shear_cart(N[0], _cart(dN[0], inv), xl.shape[1])
    return np.einsum("eab,ebk->eak", J, Bs)


def _mitc_shear(cell, xl):
    """Assumed covariant shear rows as a function of the reference point: returns f(pt) → (n_e, 2, 6n)."""
    if cell == "quad4":
        A = _covariant_shear(cell, [0.0, 1.0], xl)[:, 0]      # e_ξz at η = +1
        C = _covariant_shear(cell, [0.0, -1.0], xl)[:, 0]     # e_ξz at η = −1
        D = _covariant_shear(cell, [1.0, 0.0], xl)[:, 1]      # e_ηz at ξ = +1
        B = _covariant_shear(cell, [-1.0, 0.0], xl)[:, 1]     # e_ηz at ξ = −1

        def f(pt):
            xi, eta = pt
            return np.stack([0.5 * (1 + eta) * A + 0.5 * (1 - eta) * C,
                             0.5 * (1 + xi) * D + 0.5 * (1 - xi) * B], axis=1)
        return f
    e1 = _covariant_shear(cell, [0.5, 0.0], xl)
    e2 = _covariant_shear(cell, [0.0, 0.5], xl)
    e3 = _covariant_shear(cell, [0.5, 0.5], xl)
    ert1, est2 = e1[:, 0], e2[:, 1]
    c = e3[:, 0] - ert1 - e3[:, 1] + est2

    def f(pt):
        r, s = pt
        return np.stack([ert1 + c * s, est2 - c * r], axis=1)
    return f


# ───────────────────────────────────────────────────────────────── element matrices
def _material(E, nu):
    """Plane-stress matrices E / (1 − ν²) [1 ν 0; ν 1 0; 0 0 (1 − ν)/2] (n_e, 3, 3)."""
    D = np.zeros((len(E), 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = 1.0
    D[:, 0, 1] = D[:, 1, 0] = nu
    D[:, 2, 2] = 0.5 * (1 - nu)
    return D * (E / (1 - nu ** 2))[:, None, None]


def element_matrices(nodes, cells, E, nu, rho, thickness, shear_factor=SHEAR_FACTOR, drill=DRILL):
    """
    Local-to-global element stiffness and mass (n_e, 6n, 6n) of one cell type (3 or 4 nodes), in the global
    axes. E, nu, rho, thickness: numbers or per-cell arrays.
    """
    cells = np.asarray(cells, np.int64)
    n = cells.shape[1]
    cell = {3: "tri3", 4: "quad4"}.get(n)
    if cell is None:
        raise MeshError(f"shell cells have 3 or 4 nodes, not {n}")
    n_e = len(cells)
    E, nu, rho, h = (_per_cell(v, n_e, nm) for v, nm in ((E, "E"), (nu, "nu"), (rho, "rho"), (thickness, "thickness")))
    R, xl = local_frames(nodes, cells)
    Dm = _material(E, nu) * h[:, None, None]
    Db = Dm * (h ** 2 / 12.0)[:, None, None]
    G = E / (2 * (1 + nu))
    Ds = shear_factor * G * h
    nd = NDOF * n
    K = np.zeros((n_e, nd, nd))
    M = np.zeros((n_e, nd, nd))
    shear = _mitc_shear(cell, xl)
    pts, wts = _QUAD_GAUSS if cell == "quad4" else _TRI3
    for pt, w in zip(pts, wts):
        N, dN = _shape(cell, pt)
        J, inv, det = cell_jacobian(dN[0], xl)
        if np.any(det <= 0):
            raise MeshError(f"{int(np.sum(det <= 0))} inverted shell cells")
        g = _cart(dN[0], inv)
        Bm, Bb, Bd = _membrane_bending(N[0], g, n)
        Bs = np.einsum("eab,ebk->eak", inv, shear(pt))      # Cartesian γ = J⁻¹ · covariant
        dA = (w * det)[:, None, None]
        tr = lambda B: np.swapaxes(B, 1, 2)
        K += dA * (tr(Bm) @ (Dm @ Bm) + tr(Bb) @ (Db @ Bb) + Ds[:, None, None] * (tr(Bs) @ Bs)
                   + (drill * G * h)[:, None, None] * (Bd[:, :, None] * Bd[:, None, :]))
        # mass: ρh (translations), ρh³/12 (rotations)
        NN = np.outer(N[0], N[0])
        for i in range(NDOF):
            m = rho * h if i < 3 else rho * h ** 3 / 12.0 * (drill if i == 5 else 1.0)
            M[:, i::NDOF, i::NDOF] += (w * det * m)[:, None, None] * NN
    # local → global: per node, translations and rotations both rotate with R (rows e1, e2, e3); a warped
    # quadrilateral's nodes are linked rigidly to their projections on the mean plane (offset z_a along e3):
    # u_p = u_a − z_a θ_2,  v_p = v_a + z_a θ_1, which keeps rigid motions exact
    T = np.zeros((n_e, nd, nd))
    z = warp_offsets(nodes, cells, R)
    for a in range(n):
        c = NDOF * a
        T[:, c:c + 3, c:c + 3] = R
        T[:, c + 3:c + 6, c + 3:c + 6] = R
        T[:, c + 0, c + 3:c + 6] = -z[:, a, None] * R[:, 1]      # − z θ_2 (θ_2 = e2 · θ)
        T[:, c + 1, c + 3:c + 6] = z[:, a, None] * R[:, 0]       # + z θ_1
    Tt = np.swapaxes(T, 1, 2)
    K = Tt @ K @ T
    M = Tt @ M @ T
    return K, M


def cell_sets(cells):
    """Cells as a dict {"tri3": …, "quad4": …} of node-row arrays (an array of 3- or 4-node cells is put
    under its type; empty sets are dropped)."""
    if isinstance(cells, dict):
        return {k: np.asarray(v, np.int64) for k, v in cells.items() if len(v)}
    cells = np.asarray(cells, np.int64)
    return {{3: "tri3", 4: "quad4"}[cells.shape[1]]: cells}


def _value(v, key):
    return v[key] if isinstance(v, dict) else v


def shell_matrices(nodes, cells, E, nu, rho, thickness, shear_factor=SHEAR_FACTOR, drill=DRILL):
    """
    Global sparse stiffness K and mass M (6n × 6n, csr) of a shell mesh. cells: an array of triangles or
    quadrilaterals, or a dict {"tri3": …, "quad4": …}; E, nu, rho, thickness: numbers, per-cell arrays, or
    dicts keyed like the cells.
    """
    nodes = np.asarray(nodes, float)
    n = len(nodes)
    rows, cols, kv, mv = [], [], [], []
    for key, c in cell_sets(cells).items():
        Ke, Me = element_matrices(nodes, c, _value(E, key), _value(nu, key), _value(rho, key),
                                  _value(thickness, key), shear_factor, drill)
        dofs = (NDOF * c[:, :, None] + np.arange(NDOF)).reshape(len(c), -1)
        nd = dofs.shape[1]
        rows.append(np.repeat(dofs, nd, axis=1).ravel())
        cols.append(np.tile(dofs, (1, nd)).ravel())
        kv.append(Ke.ravel())
        mv.append(Me.ravel())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    K = sp.csr_matrix((np.concatenate(kv), (rows, cols)), shape=(NDOF * n, NDOF * n))
    M = sp.csr_matrix((np.concatenate(mv), (rows, cols)), shape=(NDOF * n, NDOF * n))
    K.sum_duplicates()
    M.sum_duplicates()
    return K, M


def cell_normals(nodes, cells):
    """Unit normals (n_e, 3) of the cells (right-handed with the node order) and their areas."""
    R, xl = local_frames(nodes, cells)
    if cells.shape[1] == 3:
        a = 0.5 * np.abs(np.cross(np.c_[xl[:, 1] - xl[:, 0], np.zeros(len(xl))],
                                  np.c_[xl[:, 2] - xl[:, 0], np.zeros(len(xl))])[:, 2])
    else:
        d1, d2 = xl[:, 2] - xl[:, 0], xl[:, 3] - xl[:, 1]
        a = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    return R[:, 2], a


def fixed_dofs(node_rows, components="all"):
    """DOF indices of the nodes `node_rows` for the components ("all", "translations", or a string of
    'xyz' (translations) and 'abc' (rotations about x, y, z), e.g. "z" or "xyzab")."""
    node_rows = np.asarray(node_rows, np.int64)
    if components == "all":
        comp = range(NDOF)
    elif components == "translations":
        comp = range(3)
    else:
        comp = ["xyzabc".index(ch) for ch in components]
    return np.unique((NDOF * node_rows[:, None] + np.asarray(list(comp))[None, :]).ravel())


def feature_nodes(nodes, cells, angle=30.0):
    """
    Node rows on the feature lines of a shell surface: its free boundary and the edges where two cells meet
    at more than `angle` degrees (panel edges of a box), as a sorted array.
    """
    edges, normals = [], []
    for c in cell_sets(cells).values():
        k = c.shape[1]
        nrm, _ = cell_normals(nodes, c)
        for a in range(k):
            edges.append(np.sort(np.column_stack([c[:, a], c[:, (a + 1) % k]]), axis=1))
            normals.append(nrm)
    E, Nr = np.vstack(edges), np.vstack(normals)
    key, inv, cnt = np.unique(E, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    first = order[start]                                         # one cell normal per edge
    cosang = np.abs(np.einsum("ij,ij->i", Nr, Nr[first[inv]]))
    cos_min = np.ones(len(key))
    np.minimum.at(cos_min, inv, cosang)
    sel = (cnt == 1) | (cnt > 2) | (cos_min < np.cos(np.radians(angle)))   # free edge, T-junction, fold
    return np.unique(key[sel].ravel())


def symmetric_lu(A):
    """Sparse LU of a symmetric matrix with diagonal pivots (SuperLU symmetric mode, minimum degree on A + Aᵀ).
    The default partial pivoting fills shell matrices badly (the small drilling and rotary terms make it swap
    rows: minutes instead of a fraction of a second on 20k DOFs)."""
    import scipy.sparse.linalg as spla
    return spla.splu(sp.csc_matrix(A), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                     options=dict(SymmetricMode=True))


def modes(K, M, n, fixed=(), sigma=0.0):
    """
    Lowest `n` undamped modes above the shift `sigma` [rad²/s²] of K φ = ω² M φ with the DOFs `fixed` removed:
    frequencies [Hz] (ascending) and mass-normalised vectors on all DOFs (zero on the fixed ones).
    """
    import scipy.sparse.linalg as spla
    from loudpy.solvers.eigen import start_vector
    N = K.shape[0]
    free = np.setdiff1d(np.arange(N), np.asarray(fixed, np.int64))
    Kf = sp.csc_matrix(K)[free][:, free]
    Mf = sp.csc_matrix(M)[free][:, free]
    lu = symmetric_lu(Kf - sigma * Mf)
    op = spla.LinearOperator(Kf.shape, matvec=lu.solve, dtype=float)
    lam, vec = spla.eigsh(Kf, k=min(n, len(free) - 2), M=Mf, sigma=sigma, which="LM", OPinv=op,
                          v0=start_vector(len(free), complex_=False))
    o = np.argsort(lam)
    lam, vec = lam[o], vec[:, o]
    vec /= np.sqrt(np.einsum("ik,ik->k", vec, Mf @ vec))
    full = np.zeros((N, len(lam)))
    full[free] = vec
    return np.sqrt(np.maximum(lam, 0.0)) / (2 * np.pi), full
