"""
Cavity_coupling.py — 3D acoustic cavity of any shape coupled to the azimuthal-order (Fourier)
modes of an axisymmetric driver, by modal superposition.

Ingredients
-----------
  - driver : in-vacuo modes of orders m = 0, 1, 2, ... (EigenStudy.solve_meca_eigen_fourier),
             made real and mass-normalised, with both azimuthal families for m ≥ 1
                 cos family : u_r = U_r cos mθ,  u_z = U_z cos mθ,  u_θ =  U_θ sin mθ
                 sin family : u_r = U_r sin mθ,  u_z = U_z sin mθ,  u_θ = −U_θ cos mθ
  - cavity : rigid-wall acoustic modes ψ_n of a 3D cavity meshed with P2 tetrahedra
             (K ψ = κ² M ψ, ψᵀMψ = 1, κ = ω/c), the driver cut out of it; its wet surface
             is the physical surface "interface_driver"
  - coupling: Bc[i, k] = ∫_S N_i (φ_k·n) dS  over the wet surface (n out of the structure,
             into the air), evaluated on the cavity surface: each point is mapped to (r, θ, z)
             in the driver frame and φ_k·n is taken on the nearest wet edge of the meridian
             mesh;  S = Bcᵀ ψ  (structure modes × cavity modes)

Coupled equations (e^{iωt}, q structural modal amplitudes, P cavity modal pressures)
-------------------------------------------------------------------------------------
    (ω_k² + 2iζ_k ω_k ω − ω²) q_k − ρω² (R q)_k + Σ_n S_kn P_n = F_k
    (ω_n² + iη ω_n ω − ω²) P_n + ρc²ω² Σ_k S_kn q_k = 0

R = Bcᵀ K⁺ Bc − Σ_{n≥1} S_n S_nᵀ / κ_n² is the static correction of the truncated cavity modes
(added mass of the air near the wet surface). Eliminating P gives the structure loaded by the
cavity modal impedance; the same matrices give the coupled modes (quadratic eigenproblem).
Only the cavity side is loaded: the front of the driver is in vacuo.

Several drivers
---------------
Each driver has its own frame (Frame: origin + rotation, any orientation) and its own
wet patch ("interface_driver_0", "interface_driver_1", ...). MultiDriver stacks their modal
coordinates (driver 0 first), so q, S, R and the solvers are unchanged: the drivers interact
through the off-diagonal blocks of S D Sᵀ + R, i.e. through the air.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from loudpy.solvers.eigen import start_vector

# Geometry (frames, cut-outs, meshing, P2 mesh reading) lives in loudpy.geometry; the names
# are re-exported here so that existing scripts keep working.
from loudpy.geometry.frame import Frame, as_frame, _is_single  # noqa: F401
from loudpy.geometry.frame import Frame as DriverFrame  # noqa: F401  (former name)
from loudpy.geometry.gmsh_session import gmsh_module as _gmsh  # noqa: F401
from loudpy.geometry.meshing3d import (sample_profile, add_driver_cutout, carve_drivers,  # noqa: F401
                                       classify_driver_faces, _local_extent, mesh_cavity,
                                       read_p2_mesh)

RHO_AIR, C_AIR = 1.2, 343.0


# ════════════════════════════════════════════════════════════════════════════
#  P2 tetrahedron (straight edges): exact reference integrals
# ════════════════════════════════════════════════════════════════════════════
_TET_EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)]      # gmsh TET10 node order


def _tet10_tables():
    from math import factorial
    Q = np.zeros((10, 4, 4))                       # N_p = λᵀ Q_p λ (homogeneous in λ)
    one = np.ones(4)
    for i in range(4):
        e = np.eye(4)[i]
        Q[i] = 2 * np.outer(e, e) - 0.5 * (np.outer(e, one) + np.outer(one, e))
    for p, (a, b) in enumerate(_TET_EDGES, start=4):
        Q[p, a, b] = Q[p, b, a] = 2.0
    L2 = (np.ones((4, 4)) + np.eye(4)) / 20.0      # ∫ λ_f λ_g dV / V
    A = 2 * Q                                      # ∂N_p/∂λ_e = Σ_f A[p,e,f] λ_f
    T = np.einsum("pef,qhg,fg->pqeh", A, A, L2)    # ∫ c_pe c_qh dV / V
    I4 = np.zeros((4, 4, 4, 4))                    # ∫ λ_a λ_b λ_c λ_d dV / V
    for idx in np.ndindex(4, 4, 4, 4):
        cnt = np.bincount(idx, minlength=4)
        I4[idx] = 6.0 * np.prod([factorial(c) for c in cnt]) / factorial(7)
    Mref = np.einsum("pab,qcd,abcd->pq", Q, Q, I4)
    return T, Mref


_T10, _M10 = _tet10_tables()

# 6-point Dunavant rule on triangles (degree 4), barycentric (l1, l2), weights sum to 1
_TRI_QP = np.array([[0.445948490915965, 0.445948490915965], [0.108103018168070, 0.445948490915965],
                    [0.445948490915965, 0.108103018168070], [0.091576213509771, 0.091576213509771],
                    [0.816847572980459, 0.091576213509771], [0.091576213509771, 0.816847572980459]])
_TRI_W = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)


def _tri6_shape(l1, l2):
    """TRI6 shape functions (gmsh order: corners 0, 1, 2, mid-edges 01, 12, 20)."""
    l0 = 1.0 - l1 - l2
    return np.array([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1),
                     4 * l0 * l1, 4 * l1 * l2, 4 * l2 * l0])


def assemble_p2(nodes, tets):
    """Stiffness ∫∇N·∇N, mass ∫N N (sparse, straight P2 tetrahedra) and the volume."""
    X = nodes[tets[:, :4]]                                     # (n_e, 4, 3)
    J = np.transpose(X[:, 1:] - X[:, :1], (0, 2, 1))           # columns X_i − X_0
    det = np.linalg.det(J)
    Ji = np.linalg.inv(J)                                      # rows: ∇λ_1..3
    g = np.concatenate([-Ji.sum(axis=1, keepdims=True), Ji], axis=1)   # (n_e, 4, 3)
    G = np.einsum("nai,nbi->nab", g, g)
    V = np.abs(det) / 6.0
    Ke = V[:, None, None] * np.einsum("pqeh,neh->npq", _T10, G)
    Me = V[:, None, None] * _M10[None]
    rows = np.repeat(tets, 10, axis=1).ravel()
    cols = np.tile(tets, (1, 10)).ravel()
    n = len(nodes)
    K = sp.csc_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))
    M = sp.csc_matrix((Me.ravel(), (rows, cols)), shape=(n, n))
    return K, M, float(V.sum())


# ════════════════════════════════════════════════════════════════════════════
#  Cavity modes (3D P2 acoustic FE)
# ════════════════════════════════════════════════════════════════════════════
class CavityModes:
    """Rigid-wall acoustic modes of a 3D cavity (P2 tetrahedra)."""

    def __init__(self, nodes, tets, patch_tris, patches=None):
        self.nodes = np.asarray(nodes, float)            # (n, 3)
        self.tets = np.asarray(tets, np.int64)           # (n_e, 10)
        self.patch_tris = np.asarray(patch_tris, np.int64)  # (n_f, 6) wet surface (all drivers)
        self.patches = dict(patches or {})               # wet surface of each driver, by name
        self.surfaces = {}                               # other named TRI6 surfaces (e.g. port mouth)
        self.kappa2 = None                               # (n_modes,) κ_n² = (ω_n / c)²
        self.psi = None                                  # (n, n_modes), ψᵀMψ = I
        self.c = C_AIR
        self._K = self._M = self._lu = None

    # ── input ──────────────────────────────────────────────────────────────
    @classmethod
    def from_msh(cls, msh_path, patch="interface_driver", surfaces=()):
        """
        patch: name (or list of names) of the wet surface(s). With a single name, the per-driver
        surfaces "<patch>_0", "<patch>_1", … of a multi-driver mesh are read too.
        """
        names = [patch, f"{patch}_*"] if isinstance(patch, str) else list(patch)
        nodes, tets, faces = read_p2_mesh(msh_path, [*names, *surfaces])
        patches = {k: v for k, v in faces.items() if k not in surfaces and len(v)}
        tris = np.vstack(list(patches.values())) if patches else np.zeros((0, 6), np.int64)
        obj = cls(nodes, tets, tris, patches)
        obj.surfaces = {s: faces[s] for s in surfaces}
        return obj

    def face_set(self, name):
        """TRI6 faces of a named wet patch or surface."""
        if name in self.patches:
            return self.patches[name]
        if name in self.surfaces:
            return self.surfaces[name]
        raise KeyError(f"no surface {name!r} (patches: {list(self.patches)}, "
                       f"surfaces: {list(self.surfaces)})")

    # ── FE ─────────────────────────────────────────────────────────────────
    def assemble(self):
        self._K, self._M, self.volume = assemble_p2(self.nodes, self.tets)
        return self._K, self._M

    def solve(self, f_max, c=C_AIR, n_guess=None):
        """Modes up to f_max (the constant mode, κ = 0, first)."""
        self.c = c
        if self._K is None:
            self.assemble()
        k_max2 = (2 * np.pi * f_max / c) ** 2
        n = n_guess or int(1.4 * self.volume * (2 * np.pi * f_max / c) ** 3 / (6 * np.pi ** 2)) + 30
        while True:
            n = min(n, len(self.nodes) - 2)
            lam, vec = spla.eigsh(self._K, k=n, M=self._M, sigma=-1e-4 * k_max2, which="LM",
                                  v0=start_vector(len(self.nodes), complex_=False))   # reproducible
            o = np.argsort(lam)
            lam, vec = lam[o], vec[:, o]
            if lam[-1] >= k_max2 or n >= len(self.nodes) - 2:
                break
            n = int(n * 1.5)
        keep = lam <= k_max2
        lam, vec = lam[keep], vec[:, keep]
        lam[0] = 0.0
        vec[:, 0] = 1.0 / np.sqrt(self.volume)                 # exact constant mode
        self.kappa2, self.psi = lam, vec
        return self.freqs

    @property
    def freqs(self):
        return self.c * np.sqrt(np.maximum(self.kappa2, 0.0)) / (2 * np.pi)

    def static_inverse(self, B):
        """K⁺ B on the mean-free subspace (columns of B are loads)."""
        if self._K is None:
            self.assemble()
        psi0 = np.full(len(self.nodes), 1.0 / np.sqrt(self.volume))
        if getattr(self, "_lu", None) is None:
            eps = 1e-8 * self.kappa2[1]
            self._lu = spla.splu((self._K + eps * self._M).tocsc())
        lu = self._lu
        Bm = B - np.outer(self._M @ psi0, psi0 @ B)
        X = lu.solve(np.asarray(Bm))
        return X - np.outer(psi0, psi0 @ (self._M @ X))

    # ── storage ────────────────────────────────────────────────────────────
    def save(self, path):
        import h5py
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f["nodes"], f["tets"], f["patch_tris"] = self.nodes, self.tets, self.patch_tris
            f["kappa2"], f["psi"] = self.kappa2, self.psi.astype(np.float32)
            f.attrs["c"], f.attrs["volume"] = self.c, self.volume
            for k, v in self.surfaces.items():
                f[f"surfaces/{k}"] = v
            for k, v in self.patches.items():
                f[f"patches/{k}"] = v

    @classmethod
    def load(cls, path):
        import h5py
        with h5py.File(path, "r") as f:
            obj = cls(f["nodes"][()], f["tets"][()], f["patch_tris"][()])
            obj.kappa2, obj.psi = f["kappa2"][()], f["psi"][()].astype(float)
            obj.c, obj.volume = float(f.attrs["c"]), float(f.attrs["volume"])
            if "surfaces" in f:
                obj.surfaces = {k: f[f"surfaces/{k}"][()] for k in f["surfaces"]}
            if "patches" in f:
                obj.patches = {k: f[f"patches/{k}"][()] for k in f["patches"]}
        return obj

    def nearest_node(self, x):
        return int(np.argmin(np.linalg.norm(self.nodes - np.asarray(x, float), axis=1)))


# ════════════════════════════════════════════════════════════════════════════
#  Driver modes (real, mass-normalised, both azimuthal families)
# ════════════════════════════════════════════════════════════════════════════
class DriverModes:
    """
    Modes of an axisymmetric driver, one column per (order m, family, mode):
      m[k], family[k] ('c' or 's'), freq[k] (undamped, Hz), zeta[k], U[k] (n_nodes, 3).
    The complex QEP-normalised modes (φᵀ(2sM + C)φ = 1) are turned into real ones with unit
    modal mass: φ = Re(U √(2iω_d)) after phase alignment (exact for proportional damping).
    """

    def __init__(self, xy, tris, m, family, freq, zeta, U):
        self.xy, self.tris = np.asarray(xy, float), np.asarray(tris, np.int64)
        self.m, self.family = np.asarray(m, int), np.asarray(family)
        self.freq, self.zeta = np.asarray(freq, float), np.asarray(zeta, float)
        self.U = np.asarray(U, float)                          # (n_modes, n_nodes, 3)

    @classmethod
    def from_eigen_h5(cls, path, orders, f_max=None):
        """
        Modes of the azimuthal orders `orders` from an eigen result file: format 2 (lp.Modes,
        read with lp.load) or format 1 (former EigenStudy.save, read with the former reader).
        """
        import h5py
        with h5py.File(path, "r") as f:
            fmt2 = str(f.attrs.get("loudpy_format", "1.0")).startswith("2")
        m_, fam, fr, ze, Us = [], [], [], [], []

        def add(m, freq, zeta, U):
            if f_max is not None and freq > f_max:
                return
            U = np.asarray(U)
            if U.shape[1] == 2:
                U = np.column_stack([U, np.zeros(len(U))])
            w = 2 * np.pi * freq
            v = U * np.sqrt(2j * w * np.sqrt(max(1 - zeta ** 2, 0.0)))
            v = (v * np.exp(-1j * np.angle(v.ravel()[np.argmax(np.abs(v))]))).real
            for f_ in (("c", "s") if m > 0 else ("c",)):
                m_.append(m), fam.append(f_), fr.append(freq), ze.append(zeta), Us.append(v)

        if fmt2:
            from loudpy.results import load
            res = load(path)
            for m in orders:
                b = res.order(m).blocks[0]
                for k in range(len(b)):
                    add(m, float(b.values[k]), float(b.extra["zeta"][k]), b.fields["u"][k])
                mesh = b.mesh_of("u")
                xy, tris, tags = np.asarray(mesh.coords), mesh.triangles(), np.asarray(mesh.node_tags)
        else:
            from loudpy.Files_Loader import EigenReader
            with EigenReader(str(path)) as r:
                for m in orders:
                    snaps, mesh, _ = r.load_fourier(m)
                    for s in snaps:
                        add(m, s.freq, s.zeta, s.fields["u_meca"])
                    xy, tris, tags = np.asarray(mesh.coords), np.asarray(mesh.tris)[:, :3], np.asarray(mesh.tags)
        out = cls(xy, tris, m_, fam, fr, ze, np.array(Us))
        out.node_tags = tags                    # node tag of each row of xy (for boundaries of the file)
        return out

    def __len__(self):
        return len(self.freq)

    def angular(self, theta):
        """(n_modes, n_theta) factors of the (u_r, u_z) components: cos mθ or sin mθ."""
        th = np.atleast_1d(theta)[None, :]
        mm = self.m[:, None]
        return np.where((self.family == "s")[:, None], np.sin(mm * th), np.cos(mm * th))

    def displacement(self, q, nodes, theta):
        """3D displacement (u_r, u_θ, u_z) at meridian nodes × angles for modal amplitudes q."""
        q = np.asarray(q)
        th = np.atleast_1d(theta)
        a = self.angular(th)                                                   # (k, t)
        b = np.where((self.family == "s")[:, None], -np.cos(self.m[:, None] * th[None]),
                     np.sin(self.m[:, None] * th[None]))
        b[self.m == 0] = 1.0
        U = self.U[:, nodes, :]                                                # (k, n, 3)
        ur = np.einsum("k,kn,kt->nt", q, U[..., 0], a)
        uz = np.einsum("k,kn,kt->nt", q, U[..., 1], a)
        ut = np.einsum("k,kn,kt->nt", q, U[..., 2], b)
        return ur, ut, uz


class MultiDriver:
    """
    Several axisymmetric drivers placed in one 3D model, each with its modes (DriverModes) and
    its frame (DriverFrame, or a 2D center for an axis along global z). Their modal coordinates
    are stacked, driver 0 first: m, family, freq, zeta, len() describe the whole set, like
    DriverModes, so DriverCavityCoupling, ExteriorFE and the solvers take it in place of one
    driver. Identical drivers can share one DriverModes object (pass it once).
    """

    def __init__(self, modes, frames, names=None):
        self.frames = [as_frame(f) for f in frames]
        n_d = len(self.frames)
        self.modes = [modes] * n_d if isinstance(modes, DriverModes) else list(modes)
        if len(self.modes) != n_d:
            raise ValueError(f"{len(self.modes)} DriverModes for {n_d} frames")
        self.names = list(names) if names is not None else [f"driver_{j}" for j in range(n_d)]
        n = [len(d) for d in self.modes]
        self.offsets = np.concatenate([[0], np.cumsum(n)]).astype(int)
        self.m = np.concatenate([d.m for d in self.modes])
        self.family = np.concatenate([d.family for d in self.modes])
        self.freq = np.concatenate([d.freq for d in self.modes])
        self.zeta = np.concatenate([d.zeta for d in self.modes])
        self.driver_index = np.repeat(np.arange(n_d), n)        # driver of each modal column

    def __len__(self):
        return int(self.offsets[-1])

    @property
    def n_drivers(self):
        return len(self.frames)

    def slice(self, j):
        """Columns of driver j in the stacked coordinates."""
        return slice(self.offsets[j], self.offsets[j + 1])

    def split(self, q):
        """Per-driver modal amplitudes q[..., slice(j)] (extra columns after the modes are ignored)."""
        q = np.asarray(q)
        return [q[..., self.slice(j)] for j in range(self.n_drivers)]

    def coil_force(self, j, coil_nodes, total=1.0):
        """
        Modal force of an axial force `total` [N] (complex: amplitude and phase) on the coil
        nodes of driver j (meridian node indices, axisymmetric force along its axis). Zero on the
        other drivers' columns.
        """
        d, sl = self.modes[j], self.slice(j)
        F = np.zeros(len(self), np.result_type(total, float))
        F[sl] = total * d.U[:, coil_nodes, 1].mean(axis=1) * (d.m == 0)
        return F

    def coil_forces(self, coil_nodes, amplitudes=None):
        """
        Unit coil forces of all drivers, one column per driver (n_modes × n_drivers), or their
        combination F = Σ_j a_j F_j for complex amplitudes a_j. coil_nodes: one array for all
        drivers, or one per driver.
        """
        per = (isinstance(coil_nodes, (list, tuple)) and len(coil_nodes) == self.n_drivers
               and np.ndim(coil_nodes[0]) == 1)
        Fs = np.column_stack([self.coil_force(j, coil_nodes[j] if per else coil_nodes)
                              for j in range(self.n_drivers)])
        return Fs if amplitudes is None else Fs @ np.asarray(amplitudes)

    def surface(self, j, q, **kwargs):
        """Deformed surface of driver j in its frame (driver_surface) from the stacked amplitudes q."""
        return driver_surface(self.modes[j], np.asarray(q)[self.slice(j)], self.frames[j], **kwargs)

    def __repr__(self):
        return (f"MultiDriver({self.n_drivers} drivers, {len(self)} modes: "
                + ", ".join(f"{nm} {len(d)} @ {fr.origin.round(4).tolist()}"
                            for nm, d, fr in zip(self.names, self.modes, self.frames)) + ")")


def as_drivers(driver, profile=None, center=(0.0, 0.0)):
    """(MultiDriver, per-driver profiles, single?) from one DriverModes or a MultiDriver."""
    if isinstance(driver, MultiDriver):
        if profile is None or len(profile) != driver.n_drivers:
            raise ValueError("with a MultiDriver, profile is a list with one wet profile per driver")
        return driver, list(profile), False
    return MultiDriver([driver], [center]), [profile], True


def wet_edges(driver: DriverModes, profile, tol=2.5e-4):
    """
    Boundary edges of the meridian mesh lying on the wet profile (chained polylines), with their
    outward normals. A tuple of profiles (e.g. (front, rear)) gives the edges of all of them.
    """
    if isinstance(profile, tuple):
        parts = [wet_edges(driver, p, tol) for p in profile]
        e, n = np.vstack([a for a, _ in parts]), np.vstack([b for _, b in parts])
        _, keep = np.unique(np.sort(e, axis=1), axis=0, return_index=True)
        return e[keep], n[keep]
    t = driver.tris
    e = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    opp = np.concatenate([t[:, 2], t[:, 0], t[:, 1]])
    key = np.sort(e, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bnd = cnt[inv.ravel()] == 1
    e, opp = e[bnd], opp[bnd]
    xy = driver.xy
    mid = 0.5 * (xy[e[:, 0]] + xy[e[:, 1]])
    poly = np.vstack(profile) if isinstance(profile, (list, tuple)) else np.asarray(profile)
    dense = np.vstack([np.linspace(poly[i], poly[i + 1], 20) for i in range(len(poly) - 1)])
    d, _ = cKDTree(dense).query(mid)
    sel = d < tol
    e, opp = e[sel], opp[sel]
    tvec = xy[e[:, 1]] - xy[e[:, 0]]
    n = np.column_stack([tvec[:, 1], -tvec[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    away = np.einsum("ij,ij->i", n, 0.5 * (xy[e[:, 0]] + xy[e[:, 1]]) - xy[opp]) > 0
    n[~away] *= -1
    return e, n


# ════════════════════════════════════════════════════════════════════════════
#  Coupling
# ════════════════════════════════════════════════════════════════════════════
def solve_forced(freqs, F, w_s, zeta, S, R, kappa2, rho=RHO_AIR, c=C_AIR, eta_air=0.0):
    """
    Forced response of the coupled system from its modal data (see the module docstring):
    q (n_f, n_s) driver modal amplitudes, P (n_f, n_a) cavity modal pressures.
    S = None gives the driver in vacuo (P = 0).
    F may be (n_s, n_rhs), e.g. one unit coil force per driver (MultiDriver.coil_forces): then
    q is (n_f, n_s, n_rhs) and P (n_f, n_a, n_rhs), one factorisation per frequency.
    """
    F = np.asarray(F)
    rhs = F.shape[1:]
    q = np.zeros((len(freqs), len(w_s)) + rhs, complex)
    P = np.zeros((len(freqs), len(kappa2)) + rhs, complex)
    for i, f in enumerate(freqs):
        w = 2 * np.pi * f
        A = np.diag(w_s ** 2 + 2j * zeta * w_s * w - w ** 2)
        if S is None:
            q[i] = np.linalg.solve(A, F)
            continue
        D = 1.0 / (kappa2 - (w / c) ** 2 + 1j * eta_air * np.sqrt(kappa2) * w / c)
        A = A - rho * w ** 2 * ((S * D) @ S.T + R)
        q[i] = np.linalg.solve(A, F)
        P[i] = -rho * w ** 2 * D.reshape((-1,) + (1,) * len(rhs)) * (S.T @ q[i])
    return q, P

def surface_quadrature(nodes, tri):
    """
    Quadrature on a TRI6 surface (straight faces, 6 points per face): points xq (n_q, 3),
    weights wq (n_q,), the surface nodes and the sparse interpolation W (n_q × n_surface_nodes).
    ∫_S N_i f dS = (Wᵀ (wq f))_i.
    """
    Xc = nodes[tri[:, :3]]
    area = 0.5 * np.linalg.norm(np.cross(Xc[:, 1] - Xc[:, 0], Xc[:, 2] - Xc[:, 0]), axis=1)
    Nq = np.array([_tri6_shape(l1, l2) for l1, l2 in _TRI_QP])                        # (6q, 6n)
    L = np.column_stack([1 - _TRI_QP.sum(1), _TRI_QP])                                # (6q, 3)
    xq = np.einsum("qa,fad->fqd", L, Xc).reshape(-1, 3)
    wq = (area[:, None] * _TRI_W[None]).ravel()
    surf_nodes, loc = np.unique(tri, return_inverse=True)
    loc = loc.reshape(tri.shape)
    W = sp.csr_matrix((np.tile(Nq, (len(tri), 1)).ravel(),
                       (np.repeat(np.arange(len(xq)), 6), np.tile(loc, (1, 6)).reshape(-1))),
                      shape=(len(xq), len(surf_nodes)))
    return xq, wq, surf_nodes, W


def surface_coupling(nodes, tri, drv, edges, edge_n, center, map_tol=5e-4):
    """
    Bc[i, k] = ∫_S N_i (φ_k·n) dS on a TRI6 wet surface of a 3D air mesh (n out of the
    structure). Each quadrature point is mapped to (r, θ, z) in the driver frame (`center`: a
    DriverFrame, or a 2D center for an axis along global z); φ_k·n is interpolated on the
    nearest wet edge of the meridian mesh (points farther than map_tol from it are rigid).
    Returns (surface nodes, Bc, wet area).
    """
    xq, wq, surf_nodes, W = surface_quadrature(nodes, tri)
    xl = as_frame(center).to_local(xq)
    dx, dy = xl[:, 0], xl[:, 1]
    r, th, z = np.hypot(dx, dy), np.arctan2(dy, dx), xl[:, 2]
    xy = drv.xy
    A, B = xy[edges[:, 0]], xy[edges[:, 1]]
    _, cand = cKDTree(0.5 * (A + B)).query(np.column_stack([r, z]), k=min(8, len(A)))
    P = np.column_stack([r, z])[:, None, :]
    AB = (B - A)[cand]
    t = np.clip(np.einsum("qkd,qkd->qk", P - A[cand], AB) / np.einsum("qkd,qkd->qk", AB, AB), 0, 1)
    d = np.linalg.norm(P - (A[cand] + t[..., None] * AB), axis=2)
    j = np.argmin(d, axis=1)
    rows = np.arange(len(r))
    best, tb = cand[rows, j], t[rows, j]
    wet = d[rows, j] < map_tol                                   # off the wet surface → rigid
    ia, ib = edges[best, 0], edges[best, 1]
    nr, nz = edge_n[best, 0], edge_n[best, 1]
    U = drv.U
    Un = ((1 - tb) * (U[:, ia, 0] * nr + U[:, ia, 1] * nz)
          + tb * (U[:, ib, 0] * nr + U[:, ib, 1] * nz))                                   # (k, q)
    vn = Un * drv.angular(th) * wet[None]                                                # φ_k·n
    Bc = np.asarray(W.T @ (wq[:, None] * vn.T))                                          # (n_surf, k)
    return surf_nodes, Bc, float(wq[wet].sum())


def add_pistons(nodes, surf_nodes, B, pistons):
    """
    Append columns to a surface coupling matrix B (rows: surf_nodes): for each piston
    (TRI6 faces, scale) the column scale·∫_S N_i dS, for None a zero column. Returns the union of
    the surface nodes and the extended matrix.
    """
    cols = [(surf_nodes, B)]
    for pst in pistons:
        if pst is None:
            cols.append((np.zeros(0, np.int64), np.zeros((0, 1))))
            continue
        tris, scale = pst
        xq, wq, sn, W = surface_quadrature(nodes, tris)
        cols.append((sn, scale * np.asarray(W.T @ wq).reshape(-1, 1)))
    return stack_columns(cols)


def stack_columns(blocks):
    """
    Side-by-side surface coupling matrices on different (possibly overlapping) node sets:
    blocks [(nodes_j, B_j (len(nodes_j) × k_j)), …] → (union of the nodes, B (n × Σ k_j)).
    """
    allnodes = np.unique(np.concatenate([np.asarray(sn, np.int64) for sn, _ in blocks]))
    out = np.zeros((len(allnodes), sum(B.shape[1] for _, B in blocks)))
    c = 0
    for sn, B in blocks:
        if len(sn):
            out[np.searchsorted(allnodes, sn), c:c + B.shape[1]] += B
        c += B.shape[1]
    return allnodes, out


def couple_drivers(nodes, face_sets, drivers: MultiDriver, profiles, map_tol=5e-4):
    """
    Surface coupling of every driver with its own wet faces of one air mesh, stacked
    (surface_coupling per driver, then stack_columns). face_sets: TRI6 faces per driver.
    Returns (surface nodes, B, wet area per driver, (edges, edge normals) per driver).
    """
    blocks, areas, wet = [], [], []
    for drv, prof, fr, tris in zip(drivers.modes, profiles, drivers.frames, face_sets):
        e, en = wet_edges(drv, prof)
        sn, B, a = surface_coupling(nodes, tris, drv, e, en, fr, map_tol)
        blocks.append((sn, B))
        areas.append(a)
        wet.append((e, en))
    sn, B = stack_columns(blocks)
    return sn, B, areas, wet


class DriverCavityCoupling:
    """
    Modal coupling of a cavity (CavityModes) and an axisymmetric driver (DriverModes), or
    several drivers (MultiDriver), each on its own wet patch.
    """

    def __init__(self, cavity: CavityModes, driver: DriverModes | MultiDriver, profile,
                 center=(0.0, 0.0), rho=RHO_AIR, c=None, eta_air=0.0, map_tol=5e-4,
                 static_correction=True, pistons=(), patch=None):
        """
        driver, profile, center: one DriverModes, its wet profile and its center or DriverFrame;
            or a MultiDriver (frames inside, center ignored) and a list with one wet profile per
            driver.
        patch: wet patch name(s) in the cavity. Default: the whole wet surface for one driver,
            "interface_driver_<j>" for driver j of a MultiDriver.
        pistons: extra generalised coordinates appended after the driver modes, each a uniform
            normal displacement of a cavity surface: (TRI6 faces, displacement into the air per
            unit coordinate), or None for a coordinate that does not touch the cavity.
        """
        self.cav, self.drv = cavity, driver
        self.drivers, profiles, single = as_drivers(driver, profile, center)
        self.center = np.asarray(center, float) if single else None
        self.rho, self.c = rho, (c or cavity.c)
        self.eta_air = eta_air
        if single:
            faces = [cavity.patch_tris if patch is None else cavity.face_set(patch)]
        else:
            names = patch or [f"interface_driver_{j}" for j in range(self.drivers.n_drivers)]
            faces = [cavity.face_set(nm) for nm in names]
        self._build_Bc(faces, profiles, map_tol)
        if pistons:
            self.patch_nodes, self.Bc = add_pistons(cavity.nodes, self.patch_nodes, self.Bc, pistons)
            self.volume_disp = self.Bc.sum(axis=0)
        self.S = self.Bc.T @ cavity.psi[self.patch_nodes]                     # (n_y, n_a)
        self.R = self._static_correction() if static_correction else np.zeros((self.Bc.shape[1],) * 2)
        self.w_s = 2 * np.pi * self.drivers.freq
        self.w_a = self.c * np.sqrt(np.maximum(cavity.kappa2, 0.0))

    # ── Bc = ∫_S N_i (φ_k·n) dS on the cavity side ─────────────────────────
    def _build_Bc(self, faces, profiles, map_tol):
        self.patch_nodes, self.Bc, self.wet_areas, self.wet = couple_drivers(
            self.cav.nodes, faces, self.drivers, profiles, map_tol)
        self.wet_area = float(sum(self.wet_areas))
        self.edges, self.edge_n = self.wet[0]                    # first driver (single-driver API)
        self.volume_disp = self.Bc.sum(axis=0)                                          # ∫ φ·n dS

    def _static_correction(self):
        cav = self.cav
        n = len(cav.nodes)
        R = np.zeros((self.Bc.shape[1],) * 2)
        Bfull = sp.csr_matrix((self.Bc.ravel(), (np.repeat(self.patch_nodes, self.Bc.shape[1]),
                                                 np.tile(np.arange(self.Bc.shape[1]), len(self.patch_nodes)))),
                              shape=(n, self.Bc.shape[1]))
        for s0 in range(0, self.Bc.shape[1], 64):
            cols = slice(s0, min(s0 + 64, self.Bc.shape[1]))
            X = cav.static_inverse(Bfull[:, cols].toarray())
            R[:, cols] = self.Bc.T @ X[self.patch_nodes]
        S1, k1 = self.S[:, 1:], cav.kappa2[1:]
        R -= (S1 / k1) @ S1.T
        return 0.5 * (R + R.T)

    # ── forced response ────────────────────────────────────────────────────
    def coil_force(self, coil_nodes, total=1.0, driver=0):
        """
        Modal force of an axial force `total` [N] (complex allowed: amplitude and phase) shared
        by the coil nodes of driver `driver` (axisymmetric); zero on the other drivers.
        """
        return self.drivers.coil_force(driver, coil_nodes, total)

    def coil_forces(self, coil_nodes, amplitudes=None):
        """Unit coil force of each driver (columns), or Σ_j a_j F_j (MultiDriver.coil_forces)."""
        return self.drivers.coil_forces(coil_nodes, amplitudes)

    def forced(self, freqs, F, loaded=True):
        """
        Modal amplitudes q (n_f, n_s) and cavity modal pressures P (n_f, n_a); with F of shape
        (n_s, n_rhs) (e.g. coil_forces), one response per column: the transfer matrix.
        """
        return solve_forced(freqs, F, self.w_s, self.drivers.zeta, self.S if loaded else None,
                            self.R, self.cav.kappa2, self.rho, self.c, self.eta_air)

    # ── coupled modes ──────────────────────────────────────────────────────
    def coupled_modes(self, f_max=None):
        """Coupled eigenvalues s = −ζω + iω_d and right vectors x = (q, P)."""
        ns, na = len(self.drivers), len(self.w_a)
        I_s, I_a = np.eye(ns), np.eye(na)
        K = np.block([[np.diag(self.w_s ** 2), self.S], [np.zeros((na, ns)), np.diag(self.w_a ** 2)]])
        C = np.block([[np.diag(2 * self.drivers.zeta * self.w_s), np.zeros((ns, na))],
                      [np.zeros((na, ns)), np.diag(self.eta_air * self.w_a)]])
        M = np.block([[I_s + self.rho * self.R, np.zeros((ns, na))],
                      [-self.rho * self.c ** 2 * self.S.T, I_a]])
        n = ns + na
        Z, I = np.zeros((n, n)), np.eye(n)
        s, X = sla.eig(np.block([[Z, I], [-K, -C]]), np.block([[I, Z], [Z, M]]))
        keep = (s.imag > 1.0) & np.isfinite(s)
        s, X = s[keep], X[:n, keep]
        o = np.argsort(s.imag)
        s, X = s[o], X[:, o]
        if f_max is not None:
            k = s.imag / (2 * np.pi) <= f_max
            s, X = s[k], X[:, k]
        return s, X[:ns], X[ns:]


# ════════════════════════════════════════════════════════════════════════════
#  3D views (pyvista)
# ════════════════════════════════════════════════════════════════════════════
def driver_surface(drv: DriverModes, q, center=(0.0, 0.0, 0.0), theta=(0.0, 2 * np.pi),
                   n_theta=73, deformation=0.10, phase=None, u_max=None):
    """
    Deformed driver (boundary of the meridian mesh revolved over `theta`, placed at `center`:
    a translation (x0, y0[, z0]), or a DriverFrame for any orientation; θ is measured from its
    x_l axis) for the modal amplitudes q (complex, all orders summed): real part at `phase`
    (default: phase of the largest displacement), max |u| = deformation × driver diameter.
    u_max: displacement [m] mapped to that deformation and to |u| = 1 (default: the max of this
    surface); pass a common value (and phase) to draw several drivers at one scale and instant.
    Returns (pv.PolyData with point data "|u|" (normalised) and "|u| [m]", phase).
    """
    import pyvista as pv
    t = drv.tris
    e = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    edges = uniq[cnt == 1]
    nodes = np.unique(edges)
    loc = np.full(len(drv.xy), -1)
    loc[nodes] = np.arange(len(nodes))
    th = np.linspace(theta[0], theta[1], n_theta)
    ur, ut, uz = drv.displacement(q, nodes, th)
    if phase is None:
        amp = np.abs(ur) ** 2 + np.abs(ut) ** 2 + np.abs(uz) ** 2
        j = np.argmax(amp.ravel())
        big = max((ur.ravel()[j], ut.ravel()[j], uz.ravel()[j]), key=abs)
        phase = -np.angle(big)
    ur, ut, uz = (np.real(x * np.exp(1j * phase)) for x in (ur, ut, uz))
    mag = np.sqrt(ur ** 2 + ut ** 2 + uz ** 2)
    ref = max(mag.max() if u_max is None else u_max, 1e-30)
    sc = deformation * 2 * drv.xy[:, 0].max() / ref
    r, z = drv.xy[nodes, 0][:, None], drv.xy[nodes, 1][:, None]
    rr = r + sc * ur
    pts = np.column_stack([(rr * np.cos(th) - sc * ut * np.sin(th)).ravel(),
                           (rr * np.sin(th) + sc * ut * np.cos(th)).ravel(),
                           (z + sc * uz).ravel()])
    if isinstance(center, DriverFrame):
        pts = center.to_global(pts)
    else:
        pts = pts + np.array([center[0], center[1], center[2] if len(center) > 2 else 0.0])
    ia, ib = loc[edges[:, 0]], loc[edges[:, 1]]
    k = np.arange(n_theta - 1)
    quads = np.stack([np.full((len(edges), n_theta - 1), 4), ia[:, None] * n_theta + k,
                      ib[:, None] * n_theta + k, ib[:, None] * n_theta + k + 1,
                      ia[:, None] * n_theta + k + 1], axis=2)
    surf = pv.PolyData(pts, quads.reshape(-1))
    surf["|u|"] = (mag / ref).ravel()
    surf["|u| [m]"] = mag.ravel()
    return surf, phase


def cavity_grid(cav: CavityModes):
    """Linear-tetrahedron pyvista grid of the cavity (P2 corner nodes), point data "node"."""
    import pyvista as pv
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: cav.tets[:, :4]}, cav.nodes)
    grid["node"] = np.arange(len(cav.nodes))
    return grid
