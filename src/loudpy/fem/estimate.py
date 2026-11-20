"""
A-posteriori error estimation on a Lagrange space (2D: triangles, quadrilaterals; any order).

    G = recover_gradient(space, U)                 nodal gradients by polynomial-preserving recovery (PPR):
                                                   around each node a least-squares fit of a polynomial of
                                                   degree p + 1 (2p on quadrilaterals) to the nodal values of
                                                   a patch of cells; its gradient at the node (a patch that
                                                   cannot determine that degree, e.g. one cell across a thin
                                                   layer, fits the highest degree it can)
    eta = recovery_indicator(space, U)             per cell ∫ |G*u_h − ∇u_h|² (×2πr axisymmetric)
    eta = elastic_indicator(space, U, D)           per cell ∫ (ε* − ε_h)ᵀ D (ε* − ε_h) 2πr (axisymmetric solid)
    e = gradient_energy(space, U) / elastic_energy(space, U, D)    per cell ∫|∇u_h|² / ∫ε_hᴴDε_h 2πr (the norms)
    eta = residual_indicator(space, U, f, k2)      per cell h_K² ‖f + Δu_h + k² u_h‖² + ½ Σ_E h_E ‖[∂u_h/∂n]‖²
                                                   (Poisson / Helmholtz −Δu − k²u = f; interior edges)

Indicators are squared and per cell, in the order of space.cell_nodes (groups concatenated); the estimate of
the energy-norm error is √Σ η_K. The recovery estimate is asymptotically exact on smooth solutions
(effectivity measured on manufactured solutions: P1 1.03, P2 1.005, P3 → 1, Q1 1.01, Q2 1.00); the residual
indicator converges at the error's rate with an unknown constant (it ranks cells, it does not measure).
The Hessians of the residual are those of straight cells (exact for straight cells, approximate on curved).
"""
from __future__ import annotations

import numpy as np

from loudpy.fem.quadrature import quadrature

__all__ = ["recovery_operator", "recover_gradient", "recovery_indicator", "elastic_indicator", "residual_indicator", "cell_sizes",
           "gradient_energy", "elastic_energy"]


def _fit_degree(element) -> int:
    return element.order + 1 if element.cell.simplex else 2 * element.order


def _node_cells(space):
    cells = np.vstack(space.cell_nodes)
    node_cells = [[] for _ in range(space.n_nodes)]
    for c, row in enumerate(cells):
        for a in row:
            node_cells[a].append(c)
    return cells, node_cells


def recovery_operator(space):
    """The PPR gradient recovery as a sparse matrix R (2 n_nodes, n_nodes): row 2z + d is ∂/∂x_d at node z of
    the fit around z. It depends on the mesh only (built once per space, kept on it)."""
    R = getattr(space, "_ppr", None)
    if R is not None:
        return R
    import scipy.sparse as sps
    if space.gdim != 2:
        raise NotImplementedError("gradient recovery: 2D spaces (triangles, quadrilaterals)")
    top = _fit_degree(space.element)
    bases = {g: [(i, s - i) for s in range(g + 1) for i in range(s + 1)] for g in range(1, top + 1)}
    need = len(bases[top])
    X = space.node_coords
    cells, node_cells = _node_cells(space)
    nv = len(space.element.cell.vertices)
    rows, cols, vals = [], [], []
    for z in range(space.n_nodes):
        cs = np.asarray(node_cells[z])
        pts = np.unique(cells[cs])
        for _ in range(4):                                       # grow the patch through the cells' vertices
            if len(pts) >= 1.5 * need:
                break
            vs = np.unique(cells[cs][:, :nv])
            cs = np.unique(np.concatenate([node_cells[v] for v in vs]))
            pts = np.unique(cells[cs])
        d = X[pts] - X[z]
        h = np.abs(d).max()
        d = d / h
        for deg in range(top, 0, -1):                            # a patch that cannot resolve degree p + 1 (one
            exps = bases[deg]                                    # cell across a thin layer: 3 levels of nodes)
            A = np.column_stack([d[:, 0] ** i * d[:, 1] ** j for i, j in exps])   # fits the highest it can
            sv = np.linalg.svd(A, compute_uv=False)
            if len(sv) == len(exps) and sv[-1] > 1e-8 * sv[0]:
                break
        i10, i01 = exps.index((1, 0)), exps.index((0, 1))
        Ap = np.linalg.pinv(A)                                   # least squares: coefficients = Ap @ values
        rows += [np.full(len(pts), 2 * z), np.full(len(pts), 2 * z + 1)]
        cols += [pts, pts]
        vals += [Ap[i10] / h, Ap[i01] / h]
    R = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(2 * space.n_nodes, space.n_nodes))
    space._ppr = R
    return R


def recover_gradient(space, U) -> np.ndarray:
    """Recovered nodal gradients (n_nodes, gdim) of a scalar field, or (n_nodes, components, gdim)."""
    U = np.asarray(U).reshape(space.n_nodes, space.components)
    G = (recovery_operator(space) @ U).reshape(space.n_nodes, 2, space.components).transpose(0, 2, 1)
    return G[:, 0] if space.components == 1 else G


def _cell_data(space, rule):
    """Per group: (nodes, N, ∇N in x (n_e, n_q, k, 2), |J| w (n_e, n_q), quadrature points (n_e, n_q, 2))."""
    elem = space.element
    N, dN = elem.tabulate(rule.points)
    for ec, nodes in zip(space.cell_coords, space.cell_nodes):
        J = np.einsum("eai,qaj->eqij", ec, dN)                    # ∂x_i / ∂ξ_j
        Ji = np.linalg.inv(J)
        G = np.einsum("qaj,eqji->eqai", dN, Ji)
        dx = np.abs(np.linalg.det(J)) * rule.weights
        x = np.einsum("qa,eai->eqi", N, ec)
        yield nodes, N, G, dx, x


def recovery_indicator(space, U, axisymmetric: bool = False, G=None) -> np.ndarray:
    """Squared recovery indicators per cell of a scalar field: ∫_K |G*u_h − ∇u_h|² (2πr dA axisymmetric)."""
    U = np.asarray(U).ravel()
    G = recover_gradient(space, U) if G is None else G
    rule = quadrature(space.element.cell.name, 2 * space.element.degree + 2)
    out = []
    for nodes, N, Gx, dx, x in _cell_data(space, rule):
        gh = np.einsum("eqai,ea->eqi", Gx, U[nodes])
        gs = np.einsum("qa,eai->eqi", N, G[nodes])
        w = dx * (2 * np.pi * x[..., 0] if axisymmetric else 1.0)
        out.append(np.sum(w * np.sum(np.abs(gs - gh) ** 2, axis=-1), axis=1))
    return np.concatenate(out)


def _strain(grad, ur, r, durdr):
    """(…, 4) strains (ε_rr, ε_zz, γ_rz, ε_θθ) from the gradient (…, 2 comp, 2 dir) and u_r / r (axis: ∂u_r/∂r)."""
    on_axis = np.abs(r) < 1e-12
    ett = np.where(on_axis, durdr, ur / np.where(on_axis, 1.0, r))
    return np.stack([grad[..., 0, 0], grad[..., 1, 1], grad[..., 0, 1] + grad[..., 1, 0], ett], axis=-1)


def elastic_indicator(space, U, D) -> np.ndarray:
    """Squared energy-norm recovery indicators per cell of an axisymmetric displacement (2 components):
    ∫_K (ε* − ε_h)ᵀ D (ε* − ε_h) 2πr dA. D: one (4, 4) matrix, or one per cell group of the space."""
    if space.components != 2:
        raise ValueError("elastic_indicator: an axisymmetric displacement space (2 components)")
    U = np.asarray(U).reshape(space.n_nodes, 2)
    G = recover_gradient(space, U)                                # (n_nodes, 2, 2)
    r_n = space.node_coords[:, 0]
    eps_n = _strain(G, U[:, 0], r_n, G[:, 0, 0])                  # recovered nodal strains
    Ds = D if isinstance(D, (list, tuple)) else [D] * len(space.cell_nodes)
    rule = quadrature(space.element.cell.name, 2 * space.element.degree + 2)
    out = []
    for (nodes, N, Gx, dx, x), Dg in zip(_cell_data(space, rule), Ds):
        grad = np.einsum("eqai,eac->eqci", Gx, U[nodes])            # (n_e, n_q, comp, dir)
        ur = np.einsum("qa,ea->eq", N, U[nodes][..., 0])
        eh = _strain(grad, ur, x[..., 0], grad[..., 0, 0])
        es = np.einsum("qa,eak->eqk", N, eps_n[nodes])
        de = es - eh
        e2 = np.real(np.einsum("eqk,kl,eql->eq", np.conj(de), np.asarray(Dg), de))
        out.append(np.sum(dx * 2 * np.pi * x[..., 0] * e2, axis=1))
    return np.concatenate(out)


def gradient_energy(space, U, axisymmetric: bool = False) -> np.ndarray:
    """Per cell ∫_K |∇u_h|² (2πr dA axisymmetric) of a scalar field: the norm the recovery indicator compares to."""
    U = np.asarray(U).ravel()
    rule = quadrature(space.element.cell.name, 2 * space.element.degree)
    out = []
    for nodes, N, Gx, dx, x in _cell_data(space, rule):
        gh = np.einsum("eqai,ea->eqi", Gx, U[nodes])
        w = dx * (2 * np.pi * x[..., 0] if axisymmetric else 1.0)
        out.append(np.sum(w * np.sum(np.abs(gh) ** 2, axis=-1), axis=1))
    return np.concatenate(out)


def elastic_energy(space, U, D) -> np.ndarray:
    """Per cell ∫_K ε_hᴴ D ε_h 2πr dA of an axisymmetric displacement (twice the strain energy of a real field)."""
    U = np.asarray(U).reshape(space.n_nodes, 2)
    Ds = D if isinstance(D, (list, tuple)) else [D] * len(space.cell_nodes)
    rule = quadrature(space.element.cell.name, 2 * space.element.degree)
    out = []
    for (nodes, N, Gx, dx, x), Dg in zip(_cell_data(space, rule), Ds):
        grad = np.einsum("eqai,eac->eqci", Gx, U[nodes])
        eh = _strain(grad, np.einsum("qa,ea->eq", N, U[nodes][..., 0]), x[..., 0], grad[..., 0, 0])
        e2 = np.real(np.einsum("eqk,kl,eql->eq", np.conj(eh), np.asarray(Dg), eh))
        out.append(np.sum(dx * 2 * np.pi * x[..., 0] * e2, axis=1))
    return np.concatenate(out)


def cell_sizes(space) -> np.ndarray:
    """Diameter of every cell (largest distance between its corners), groups concatenated."""
    nv = len(space.element.cell.vertices)
    out = []
    for ec in space.cell_coords:
        c = ec[:, :nv]
        d = np.linalg.norm(c[:, :, None, :] - c[:, None, :, :], axis=-1)
        out.append(d.max(axis=(1, 2)))
    return np.concatenate(out)


def _hessians(element, points, h=1e-3):
    """Reference Hessians of the basis (n_q, k, 2, 2) by central differences of the gradients (exact for
    the polynomial gradients of order ≤ 3 elements up to round-off)."""
    out = np.zeros((len(points), element.n_nodes, 2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        _, gp = element.tabulate(points + e)
        _, gm = element.tabulate(points - e)
        out[..., j] = (gp - gm) / (2 * h)
    return out


def residual_indicator(space, U, f, k2: float = 0.0, axisymmetric: bool = False) -> np.ndarray:
    """Squared explicit residual indicators per cell for −Δu − k²u = f (f: points (n, 2) → values): interior
    residual h_K² ‖f + Δu_h + k²u_h‖²_K and half the jumps of the normal derivative on the interior edges
    (Dirichlet boundaries assumed; on the axis the 2πr weight cancels the natural condition)."""
    if space.gdim != 2 or space.components != 1:
        raise NotImplementedError("residual indicator: scalar fields on 2D spaces")
    U = np.asarray(U).ravel()
    elem = space.element
    rule = quadrature(elem.cell.name, 2 * elem.degree + 2)
    Hr = _hessians(elem, rule.points)
    hK = cell_sizes(space)
    out, k0 = [], 0
    N, dN = elem.tabulate(rule.points)
    for ec, nodes in zip(space.cell_coords, space.cell_nodes):
        J = np.einsum("eai,qaj->eqij", ec, dN)
        Ji = np.linalg.inv(J)                                      # ∂ξ_j / ∂x_i (indices [j, i])
        Gx = np.einsum("qaj,eqji->eqai", dN, Ji)
        Hx = np.einsum("eqji,qajl,eqlm->eqaim", Ji, Hr, Ji)          # straight-cell Hessians in x
        x = np.einsum("qa,eai->eqi", N, ec)
        ue = U[nodes]
        lap = np.einsum("eqaii,ea->eq", Hx, ue)
        if axisymmetric:
            lap = lap + np.einsum("eqa,ea->eq", Gx[..., 0], ue) / x[..., 0]
        uh = np.einsum("qa,ea->eq", N, ue)
        res = f(x.reshape(-1, 2)).reshape(uh.shape) + lap + k2 * uh
        dx = np.abs(np.linalg.det(J)) * rule.weights * (2 * np.pi * x[..., 0] if axisymmetric else 1.0)
        out.append(hK[k0:k0 + len(ec)] ** 2 * np.sum(dx * res ** 2, axis=1))
        k0 += len(ec)
    eta = np.concatenate(out)
    eta += _jumps(space, U, axisymmetric)
    return eta


def _jumps(space, U, axisymmetric):
    """½ h_E ∫_E [∂u_h/∂n]² on the interior edges, given to each of the two cells."""
    elem = space.element
    ref = elem.cell
    xg, wg = np.polynomial.legendre.leggauss(elem.degree + 2)
    t = 0.5 * (xg + 1.0)
    cells = np.vstack(space.cell_nodes)
    coords = np.vstack(space.cell_coords)
    owner = {}
    for c, row in enumerate(cells):
        for f, (a, b) in enumerate(ref.edges):
            key = (min(row[a], row[b]), max(row[a], row[b]))
            owner.setdefault(key, []).append((c, f, row[a]))
    eta = np.zeros(len(cells))

    def grad_at(c, f, start_node):
        a, b = ref.edges[f]
        va, vb = ref.vertices[a], ref.vertices[b]
        if cells[c][a] != start_node:                             # walk the edge from the same physical end
            va, vb = vb, va
        xi = va + t[:, None] * (vb - va)
        N, dN = elem.tabulate(xi)
        J = np.einsum("ai,qaj->qij", coords[c], dN)
        Gx = np.einsum("qaj,qji->qai", dN, np.linalg.inv(J))
        return np.einsum("qai,a->qi", Gx, U[cells[c]]), np.einsum("qa,ai->qi", N, coords[c])

    for key, sides in owner.items():
        if len(sides) != 2:
            continue
        (c1, f1, s1), (c2, f2, _) = sides
        g1, x = grad_at(c1, f1, s1)
        g2, _ = grad_at(c2, f2, s1)
        p0, p1 = coords[c1][list(ref.edges[f1])][:, :2]
        hE = float(np.linalg.norm(p1 - p0))
        n = np.array([p1[1] - p0[1], -(p1[0] - p0[0])]) / hE
        jump = (g1 - g2) @ n
        w = 0.5 * hE * wg * (2 * np.pi * x[:, 0] if axisymmetric else 1.0)
        val = 0.5 * hE * np.sum(w * jump ** 2)
        eta[c1] += val
        eta[c2] += val
    return eta
