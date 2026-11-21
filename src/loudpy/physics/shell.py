"""
loudpy.physics.shell — shell structures (MITC3 / MITC4, loudpy.fem.shell) and their direct coupling to a 3D air
domain of P2 tetrahedra.

    walls = wall_shells(cavity.nodes, cavity.tets, cavity.face_set("walls"))      # shells on the air's faces
    shell = ShellStructure(walls.nodes, walls.cells, "MDF", thickness=0.018, fixed=walls.feature_nodes())
    A = walls.coupling(n_air=len(cavity.nodes))                                  # 6 n_s × n_air

Conforming coupling. The shell cells are the corner triangles of the air mesh's TRI6 boundary faces (one MITC3
cell per face, its normal n pointing into the air), so the shell nodes are air nodes and the coupling is exact
face by face:

    A[6a + i, j] = ∫_S N_a n_i N_j dS        (N_a: linear shell basis, N_j: P2 pressure basis)

Equations (time convention e^{st}, s = jω; d shell DOFs, p air pressure; n out of the structure into the air):

    (K + s C + s² M) d + A p = F                     the air pushes on the shell with −p n
    (H + s² Q) p − ρ s² Aᵀ d = 0                     ∂p/∂n_air,out = ρ s² d·n on the wall

with H = ∫∇N∇N, Q = ∫N N / c². A closed cavity at low frequency gives p = ρc² ΔV / V (ΔV = ∫ d·n dS > 0 when the
wall moves into the air). `direct_builder` puts this on a SystemBuilder (fields "d", "p"); `direct_modes` gives the
coupled modes (unsymmetric eigenproblem, shift-invert); `direct_response` a frequency sweep (one sparse LU per
frequency).

The shells are placed on the air's boundary (the wall's mid-surface is taken at its inner face: no offset).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from loudpy.errors import MeshError, ModelError
from loudpy.fem import shell as fe


# ════════════════════════════════════════════════════════════════════════════
#  Shell structure
# ════════════════════════════════════════════════════════════════════════════
class ShellStructure:
    """
    A shell structure: linear cells (MITC3 triangles / MITC4 quadrilaterals), one material and thickness (or one
    per cell), damping, fixed DOFs.

        ShellStructure(nodes, cells, "MDF", thickness=0.018, damping=lp.Rayleigh(), fixed=rows)

    material: a library name or an lp.Material (solid); damping: lp.Rayleigh() (default, the material's fit),
    lp.Hysteretic() (complex stiffness K (1 + iη)), lp.Undamped(); fixed: node rows clamped on every DOF, or
    DOF indices with fixed_dofs=…
    """

    def __init__(self, nodes, cells, material, thickness, damping=None, fixed=(), fixed_dofs=None,
                 shear_factor=fe.SHEAR_FACTOR, drill=fe.DRILL):
        from loudpy.checks import check
        from loudpy.model.damping import Hysteretic, Rayleigh, Undamped
        from loudpy.model.materials import Material, materials
        self.nodes = np.asarray(nodes, float)
        self.cells = fe.cell_sets(cells)
        mat = material if isinstance(material, Material) else materials[material]
        if mat.kind != "solid":
            raise ModelError(f"shell material {mat.name!r} is a {mat.kind}, not a solid")
        mat.require("rho", "E", "nu")
        self.material = mat
        for h in np.atleast_1d(np.concatenate([np.ravel(v) for v in thickness.values()])
                               if isinstance(thickness, dict) else thickness):
            check(float(h), "length", "shell thickness")
        self.thickness = thickness
        self.damping = Rayleigh() if damping is None else damping
        if not isinstance(self.damping, (Rayleigh, Hysteretic, Undamped)):
            raise ModelError(f"shell damping: lp.Rayleigh, lp.Hysteretic or lp.Undamped, not {self.damping!r}")
        dofs = [fe.fixed_dofs(np.asarray(fixed, np.int64))] if len(fixed) else []
        if fixed_dofs is not None:
            dofs.append(np.asarray(fixed_dofs, np.int64))
        self.fixed = np.unique(np.concatenate(dofs)) if dofs else np.zeros(0, np.int64)
        self.free = np.setdiff1d(np.arange(fe.NDOF * len(self.nodes)), self.fixed)
        self._K, self._M = fe.shell_matrices(self.nodes, self.cells, mat.E, mat.nu, mat.rho, thickness,
                                             shear_factor, drill)

    @property
    def n_dofs(self) -> int:
        return fe.NDOF * len(self.nodes)

    @property
    def K(self):
        """Stiffness (real; complex K (1 + iη) with lp.Hysteretic)."""
        from loudpy.model.damping import Hysteretic
        if isinstance(self.damping, Hysteretic):
            eta = self.damping.eta if self.damping.eta is not None else self.material.loss_factor
            if eta is None:
                raise ModelError(f"{self.material.name}: no loss factor for lp.Hysteretic()")
            return self._K * (1 + 1j * eta)
        return self._K

    @property
    def M(self):
        return self._M

    @property
    def rayleigh(self) -> tuple[float, float]:
        """(α, β) of the viscous damping C = α M + β K ((0, 0) unless lp.Rayleigh)."""
        from loudpy.model.damping import Rayleigh
        if not isinstance(self.damping, Rayleigh):
            return 0.0, 0.0
        fit = self.material.rayleigh or (None, None)
        a = self.damping.alpha if self.damping.alpha is not None else fit[0]
        b = self.damping.beta if self.damping.beta is not None else fit[1]
        if a is None or b is None:
            raise ModelError(f"{self.material.name}: no Rayleigh fit for lp.Rayleigh()")
        return float(a), float(b)

    @property
    def C(self):
        a, b = self.rayleigh
        return a * self._M + b * self._K

    def modes(self, n=None, f_max=None, sigma=0.0):
        """
        In-vacuo modes: frequencies [Hz], mass-normalised vectors (n_dofs × k, zero on the fixed DOFs) and modal
        damping ratios (Rayleigh: ζ = α/2ω + βω/2; hysteretic: η/2; undamped: 0). Give n, or f_max (all modes up
        to f_max, found by growing the count).
        """
        if n is None and f_max is None:
            raise ModelError("ShellStructure.modes: give n or f_max")
        k = n or 20
        while True:
            f, phi = fe.modes(self._K, self._M, k, self.fixed, sigma)
            if f_max is None or f[-1] > f_max or k >= len(self.free) - 2:
                break
            k = int(k * 1.6) + 10
        if f_max is not None:
            keep = f <= f_max
            f, phi = f[keep], phi[:, keep]
        return f, phi, self.modal_damping(f)

    def modal_damping(self, f):
        from loudpy.model.damping import Hysteretic, Rayleigh
        w = 2 * np.pi * np.maximum(np.asarray(f, float), 1e-12)
        if isinstance(self.damping, Rayleigh):
            a, b = self.rayleigh
            return a / (2 * w) + b * w / 2
        if isinstance(self.damping, Hysteretic):
            eta = self.damping.eta if self.damping.eta is not None else self.material.loss_factor
            return np.full(len(w), 0.5 * eta)
        return np.zeros(len(w))


# ════════════════════════════════════════════════════════════════════════════
#  Shells on the faces of a P2 air mesh
# ════════════════════════════════════════════════════════════════════════════
_FLIP6 = np.array([0, 2, 1, 5, 4, 3])          # TRI6 with the corners 1 and 2 swapped (mid-edge nodes follow)


@dataclass
class WallShells:
    """Shell cells laid on TRI6 boundary faces of an air mesh: `rows` (air node row of each shell node),
    `nodes` (their coordinates), `cells` (tri3, shell node indices, normals into the air), `faces` (the TRI6
    faces, air rows, oriented like the cells)."""
    rows: np.ndarray
    nodes: np.ndarray
    cells: np.ndarray
    faces: np.ndarray

    def feature_nodes(self, angle=30.0):
        """Shell nodes on the free boundary of the walls and on their folds (panel edges)."""
        return fe.feature_nodes(self.nodes, self.cells, angle)

    def coupling(self, n_air):
        """A (6 n_shell × n_air, csr): ∫ N_a n_i N_j dS (see the module docstring)."""
        return shell_air_coupling(self, n_air)

    def area(self) -> float:
        return float(fe.cell_normals(self.nodes, self.cells)[1].sum())


def wall_shells(nodes, tets, faces) -> WallShells:
    """
    Shells on the TRI6 faces `faces` (air rows) of a P2 tetrahedral mesh (nodes, tets): one MITC3 cell per face,
    oriented with its normal into the air (towards the tetrahedron that owns the face).
    """
    nodes = np.asarray(nodes, float)
    tets = np.asarray(tets, np.int64)
    faces = np.asarray(faces, np.int64).copy()
    if len(faces) == 0:
        raise MeshError("no wall faces")
    corners = tets[:, :4]
    local = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    tf = np.sort(corners[:, local].reshape(-1, 3), axis=1)
    owner = np.repeat(np.arange(len(tets)), 4)
    key_t = tf[:, 0] * (len(nodes) ** 2) + tf[:, 1] * len(nodes) + tf[:, 2]
    fs = np.sort(faces[:, :3], axis=1)
    key_f = fs[:, 0] * (len(nodes) ** 2) + fs[:, 1] * len(nodes) + fs[:, 2]
    order = np.argsort(key_t)
    pos = np.searchsorted(key_t[order], key_f)
    pos = np.minimum(pos, len(order) - 1)
    hit = key_t[order][pos] == key_f
    if not np.all(hit):
        raise MeshError(f"{int(np.sum(~hit))} wall faces are not faces of the tetrahedra")
    t = owner[order][pos]
    X = nodes[faces[:, :3]]
    n = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
    inward = np.einsum("ij,ij->i", n, nodes[corners[t]].mean(axis=1) - X.mean(axis=1)) > 0
    faces[~inward] = faces[~inward][:, _FLIP6]
    rows, cells = np.unique(faces[:, :3], return_inverse=True)
    return WallShells(rows, nodes[rows], cells.reshape(-1, 3), faces)


def shell_air_coupling(walls: WallShells, n_air: int):
    """A[6a + i, j] = ∫ N_a n_i N_j dS on every wall face (flat triangle, P1 shell × P2 pressure, exact)."""
    from loudpy.fem.quadrature import quadrature
    from loudpy.fem.reference import lagrange
    rule = quadrature("triangle", 4)
    N2, _ = lagrange("triangle", 2).tabulate(rule.points)              # (q, 6)
    N1, _ = lagrange("triangle", 1).tabulate(rule.points)              # (q, 3)
    nrm, area = fe.cell_normals(walls.nodes, walls.cells)
    w = rule.weights / rule.weights.sum()                              # Σ w = 1 → × area
    local = np.einsum("q,qa,qj->aj", w, N1, N2)                        # (3, 6) ∫ N_a N_j / area
    vals = area[:, None, None, None] * local[None, :, None, :] * nrm[:, None, :, None]   # (f, a, i, j)
    rows = fe.NDOF * walls.cells[:, :, None, None] + np.arange(3)[None, None, :, None]
    rows = np.broadcast_to(rows, vals.shape)
    cols = np.broadcast_to(walls.faces[:, None, None, :], vals.shape)
    A = sp.csr_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=(fe.NDOF * len(walls.nodes), n_air))
    A.sum_duplicates()
    return A


# ════════════════════════════════════════════════════════════════════════════
#  Direct (monolithic) shell + air system
# ════════════════════════════════════════════════════════════════════════════
def direct_builder(shell: ShellStructure, A, H, Mair, rho, c, F=None):
    """
    SystemBuilder with fields "d" (free shell DOFs) and "p" (air pressure, all nodes) for the equations of the
    module docstring; F: force on all shell DOFs (or None). H, Mair: ∫∇N∇N, ∫N N of the air (assemble_p2).
    """
    from loudpy.physics.builder import Field, Load, SystemBuilder, Term
    fr = shell.free
    Af = sp.csr_matrix(A)[fr]
    K, M = shell.K[fr][:, fr], shell.M[fr][:, fr]
    n_s = len(fr)

    def expand(x):
        out = np.zeros(x.shape[:-1] + (shell.n_dofs,), dtype=x.dtype)
        out[..., fr] = x
        return out
    b = SystemBuilder()
    b.add_field(Field("d", n_s, expand))
    b.add_field(Field("p", H.shape[0]))
    b.add(Term("d", "d", sp.csr_matrix(K)), Term("d", "d", sp.csr_matrix(shell.C[fr][:, fr]), power=1),
          Term("d", "d", sp.csr_matrix(M), power=2), Term("d", "p", Af),
          Term("p", "p", sp.csr_matrix(H)), Term("p", "p", sp.csr_matrix(Mair) / c ** 2, power=2),
          Term("p", "d", sp.csr_matrix(-rho * Af.T), power=2))
    if F is not None:
        b.add_load(Load("d", np.asarray(F)[fr]))
    return b


def _lu(A):
    return spla.splu(sp.csc_matrix(A), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.1)


def direct_modes(shell, A, H, Mair, rho, c, n, f_shift=0.0):
    """
    Undamped coupled modes of the shell + air system nearest above f_shift: frequencies [Hz] (ascending) and
    vectors split into (d on all shell DOFs, p). Unsymmetric eigenproblem K x = ω² M x, shift-invert.
    """
    b = direct_builder(shell, A, H, Mair, rho, c)
    K0, _, M2 = b.polynomial()
    K0 = sp.csc_matrix(K0.real)
    M2 = sp.csc_matrix(M2.real)                          # A(s) = K0 + s² M2, s² = −ω² → K0 x = ω² M2 x
    sigma = (2 * np.pi * f_shift) ** 2 - 1.0
    lu = _lu(K0 - sigma * M2)
    # standard eigenproblem of (K0 − σ M2)⁻¹ M2 (ARPACK's generalized mode needs a symmetric M): ν = 1 / (λ − σ)
    op = spla.LinearOperator(K0.shape, matvec=lambda v: lu.solve(np.asarray(M2 @ v)), dtype=float)
    from loudpy.solvers.eigen import start_vector
    nu, vec = spla.eigs(op, k=n + 1, which="LM", v0=start_vector(K0.shape[0], complex_=False))
    lam = sigma + 1.0 / nu
    lam, vec = lam.real, vec.real
    o = np.argsort(lam)
    lam, vec = lam[o], vec[:, o]
    # the u–p form has a static solution at ω = 0 (uniform pressure balanced by the walls' static deflection,
    # K0 is singular through H): not a vibration mode
    keep = lam > 1e-6 * max(lam.max(), 1.0)
    lam, vec = lam[keep][:n], vec[:, keep][:, :n]
    parts = b.split(vec.T)
    d = b.fields["d"].expand(parts["d"])
    return np.sqrt(np.maximum(lam, 0.0)) / (2 * np.pi), d, parts["p"]


def direct_response(shell, A, H, Mair, rho, c, freqs, F):
    """Forced response: d (n_f, n_dofs) and p (n_f, n_air) for the force F on the shell DOFs."""
    b = direct_builder(shell, A, H, Mair, rho, c, F)
    d = np.zeros((len(freqs), shell.n_dofs), complex)
    p = np.zeros((len(freqs), H.shape[0]), complex)
    for i, f in enumerate(np.atleast_1d(freqs)):
        s = 2j * np.pi * f
        x = _lu(b.at(s)).solve(b.rhs(s))
        parts = b.split(x)
        d[i] = b.fields["d"].expand(parts["d"])
        p[i] = parts["p"]
    return d, p


# ════════════════════════════════════════════════════════════════════════════
#  Elastic cavity walls by modes (the walls of lp.Scene)
# ════════════════════════════════════════════════════════════════════════════
class WallModes:
    """
    In-vacuo modes of shell walls laid on faces of a cavity mesh, as extra modal coordinates of the cavity
    coupling (physics.modal): freq [Hz], zeta, phi (6 n_shell × k, mass-normalised) and the coupling columns
    B[i, k] = ∫ N_i (φ_k·n) dS on the cavity nodes `rows` (n into the air: the definition of the drivers' Bc).
    """

    def __init__(self, walls: WallShells, shell: ShellStructure, freq, phi, zeta, n_air):
        self.walls, self.shell = walls, shell
        self.freq, self.phi, self.zeta = np.asarray(freq), np.asarray(phi), np.asarray(zeta)
        A = walls.coupling(n_air)
        self.rows = np.unique(walls.faces)
        self.B = np.asarray((A.T @ self.phi))[self.rows]
        self.volume_disp = self.B.sum(axis=0)                     # ∫ φ_k·n dS (into the air)

    def __len__(self):
        return len(self.freq)

    def translations(self):
        """Mode shapes as nodal translations (k, n_shell, 3)."""
        return self.phi.reshape(len(self.walls.nodes), 6, -1)[:, :3].transpose(2, 0, 1)

    def normal(self):
        """Mode shapes as normal displacements into the air at the shell nodes (k, n_shell): cell normals
        averaged at the nodes, weighted by area."""
        n, a = fe.cell_normals(self.walls.nodes, self.walls.cells)
        acc = np.zeros((len(self.walls.nodes), 3))
        np.add.at(acc, self.walls.cells.ravel(), np.repeat(n * a[:, None], 3, axis=0))
        acc /= np.linalg.norm(acc, axis=1, keepdims=True)
        return np.einsum("knd,nd->kn", self.translations(), acc)


def cavity_walls(nodes, tets, faces, material, thickness, damping=None, fixed="edges", f_max=1000.0,
                 angle=30.0) -> WallModes:
    """Shells on the wall faces of a cavity (`faces`: TRI6, cavity rows), their in-vacuo modes up to f_max and
    the coupling columns. fixed: "edges" (folds sharper than `angle` and the free boundary of the walls, e.g.
    the rims of the driver cut-outs) or a function of the node coordinates (n, 3) → mask of clamped nodes."""
    walls = wall_shells(nodes, tets, faces)
    if isinstance(fixed, str):
        if fixed != "edges":
            raise ModelError(f"walls fixed= 'edges' or a function of the node coordinates, not {fixed!r}")
        rows = walls.feature_nodes(angle)
    elif callable(fixed):
        rows = np.where(np.asarray(fixed(walls.nodes), bool))[0]
    else:
        raise ModelError("walls fixed= 'edges' or a function of the node coordinates")
    if len(rows) == 0:
        raise ModelError("the walls are not held anywhere (fixed selects no node): their rigid motions "
                         "would be free")
    shell = ShellStructure(walls.nodes, walls.cells, material, thickness, damping, fixed=rows)
    freq, phi, zeta = shell.modes(f_max=f_max)
    return WallModes(walls, shell, freq, phi, zeta, len(nodes))
