"""
loudpy.physics.exterior — 3D exterior acoustics above a rigid ground, with the exact DtN (Dirichlet-to-
Neumann) radiation condition on a hemisphere, coupled to an axisymmetric driver (front side)
and to its enclosure (rear side, loudpy.physics.modal).

Half-space DtN
--------------
The ground (plane through the hemisphere centre, normal `up`) is perfectly reflective: the field
is that of the structure and its mirror image in free field, so outside the hemisphere Γ (r ≥ R)

    p(r, θ, φ) = Σ_h c_h  h_n(kr)/h_n(kR)  Y_h(θ, φ),   h = (n, m) with n + |m| even,

θ measured from the ground normal, Y_h real spherical harmonics orthonormal on the unit sphere
(n + |m| even ⇔ even about the ground plane), h_n = h_n^(2) (outgoing for e^{iωt}).
On Γ:  c_h = (2/R²) ∫_Γ p Y_h dS  and  ∂p/∂r = Σ_h γ_n c_h Y_h,  γ_n = k h_n'(kR)/h_n(kR).
In the P2 FE system this gives the low-rank boundary operator

    D = G diag(2γ_n/R²) Gᵀ,     G[i, h] = ∫_Γ N_i Y_h dS,
    (K − k²M − D) p = f,        f = −ρω² Bf q  (front wet surface, n out of the structure).

Coupled sweep (driver + enclosure + exterior), per frequency
------------------------------------------------------------
    H(ω) q + Bfᵀ p = F,   H = diag(ω_k² + 2iζ_kω_kω − ω²) − ρω²(S Dcav Sᵀ + R)   (physics.modal)
    (K − k²M − D) p + ρω² Bf q = 0
q is eliminated (H is small and dense) and the exterior equation is solved by GMRES,
preconditioned by a sparse LU of K − k_c²M + (ik_c + 1/R) M_Γ (local absorbing condition)
factorised once per chunk of frequencies around k_c.

Several drivers: mesh_exterior with lists of frames writes "interface_front_<j>", and
ExteriorFE.couple_driver takes a MultiDriver (physics.modal); q stacks all the drivers' modes.
coupled_sweep takes one force vector: combine the drivers' coil forces with their complex
amplitudes (MultiDriver.coil_forces(coil_nodes, amplitudes)), or run it once per driver.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import spherical_jn, spherical_yn, sph_harm_y

from loudpy.errors import ConvergenceWarning
from loudpy.fem.tet10 import assemble_p2, surface_quadrature
from loudpy.geometry.meshing3d import read_p2_mesh
from loudpy.model.materials import C_AIR, RHO_AIR
from loudpy.physics.modal import add_pistons, as_drivers, couple_drivers
from loudpy.errors import SelectionError, StateError


# ════════════════════════════════════════════════════════════════════════════
#  Half-space DtN on a hemisphere
# ════════════════════════════════════════════════════════════════════════════
def _h2(n, x, derivative=False):
    return spherical_jn(n, x, derivative) - 1j * spherical_yn(n, x, derivative)


class HalfSpaceDtN:
    """DtN operator of the half-space above a rigid plane, on a hemisphere (centre on the plane)."""

    def __init__(self, center, radius, up=1, n_extra=10):
        self.center = np.asarray(center, float)
        self.R = float(radius)
        self.up = int(up)
        self.n_extra = n_extra
        ax = [i for i in range(3) if i != self.up]
        self._ax = ax                                         # (x', y') in the ground plane

    def n_max(self, k):
        """Highest spherical-harmonic degree kept at wavenumber k: ceil(kR) + n_extra."""
        return int(np.ceil(k * self.R)) + self.n_extra

    @staticmethod
    def harmonics(n_max):
        """(n, m) with n + |m| even, ordered by n: (n_max + 1)(n_max + 2)/2 of them."""
        return np.array([(n, m) for n in range(n_max + 1) for m in range(-n, n + 1) if (n + m) % 2 == 0])

    @staticmethod
    def n_harm(n_max):
        """Number of harmonics (n, m) with n + |m| even up to degree n_max."""
        return (n_max + 1) * (n_max + 2) // 2

    def local(self, x):
        """Spherical coordinates (r, θ from the ground normal, φ) of points about the centre."""
        d = np.atleast_2d(x) - self.center
        r = np.linalg.norm(d, axis=1)
        theta = np.arccos(np.clip(d[:, self.up] / np.maximum(r, 1e-300), -1, 1))
        phi = np.arctan2(d[:, self._ax[1]], d[:, self._ax[0]])
        return r, theta, phi

    def Y(self, theta, phi, n_max):
        """Real orthonormal harmonics (n_points, n_harm)."""
        out = np.empty((len(theta), self.n_harm(n_max)))
        for j, (n, m) in enumerate(self.harmonics(n_max)):
            y = sph_harm_y(n, abs(m), theta, phi)
            out[:, j] = y.real if m == 0 else np.sqrt(2) * (-1.0) ** abs(m) * (y.real if m > 0 else y.imag)
        return out

    def gamma(self, k, n_max):
        """2γ_n/R² for every harmonic up to n_max."""
        n = self.harmonics(n_max)[:, 0]
        x = k * self.R
        return 2.0 * k * _h2(n, x, True) / _h2(n, x) / self.R ** 2

    def G(self, nodes, tris, n_max):
        """G[i, h] = ∫_Γ N_i Y_h dS on the TRI6 faces of Γ: (Γ nodes, G, boundary mass M_Γ)."""
        xq, wq, gnodes, W = surface_quadrature(nodes, tris)
        _, th, ph = self.local(xq)
        G = np.asarray(W.T @ (wq[:, None] * self.Y(th, ph, n_max)))
        MG = (W.T @ sp.diags(wq) @ W).tocsc()
        return gnodes, G, MG

    def pressure(self, points, c, k):
        """Pressure at points with r ≥ R from the coefficients c (length n_harm(n) for some n)."""
        n_max = int(round((np.sqrt(8 * len(c) + 1) - 3) / 2))
        r, th, ph = self.local(points)
        n = self.harmonics(n_max)[:, 0]
        ratio = _h2(n[None, :], k * r[:, None]) / _h2(n, k * self.R)[None, :]
        return np.sum(self.Y(th, ph, n_max) * ratio * np.asarray(c)[None, :], axis=1)


# ════════════════════════════════════════════════════════════════════════════
#  Geometry: hemisphere over the ground, minus the enclosure and the driver front
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
#  Exterior FE on one mesh (one frequency band)
# ════════════════════════════════════════════════════════════════════════════
class ExteriorFE:
    """P2 exterior air: K, M, the DtN pieces on Γ and (optionally) the front coupling Bf."""

    def __init__(self, msh_path, dtn: HalfSpaceDtN, f_max, front="interface_front", c=C_AIR, surfaces=()):
        """
        front: name of the wet surface; with a single name the per-driver surfaces "<front>_0",
        "<front>_1", … of a multi-driver mesh are read too. A list of names is also accepted.
        """
        fronts = [front, f"{front}_*"] if isinstance(front, str) else list(front)
        self.nodes, self.tets, self.faces = read_p2_mesh(msh_path, ["dtn", "ground", *fronts, "walls",
                                                                    *surfaces])
        self.dtn, self.c = dtn, c
        self.K, self.M, self.volume = assemble_p2(self.nodes, self.tets)
        self.n_max_band = dtn.n_max(2 * np.pi * f_max / c)
        self.gnodes, self.G, MG = dtn.G(self.nodes, self.faces["dtn"], self.n_max_band)
        n = len(self.nodes)
        Pg = sp.csr_matrix((np.ones(len(self.gnodes)), (self.gnodes, np.arange(len(self.gnodes)))),
                           shape=(n, len(self.gnodes)))
        self.MG = (Pg @ MG @ Pg.T).tocsc()
        self.front = front
        self.pnodes = self.Bf = None

    def couple_driver(self, drv, profile, center=(0.0, 0.0), map_tol=5e-4, pistons=(), patch=None):
        """
        Coupling of the wet surface with the driver modes (profile: one wet profile, or a tuple
        of them, e.g. (front, rear) when one air domain wets both faces), plus extra piston
        coordinates appended after the modes (see add_pistons).

        Several drivers: drv is a MultiDriver (frames inside, center ignored), profile a list
        with one profile (or tuple of profiles) per driver, and patch the wet surface of each
        driver (default: the names given as `front` if it was a list, else "<front>_<j>").
        """
        drivers, profiles, single = as_drivers(drv, profile, center)
        if patch is not None:
            names = [patch] if isinstance(patch, str) else list(patch)
        elif single:
            names = [self.front if isinstance(self.front, str) else self.front[0]]
        elif isinstance(self.front, str):
            names = [f"{self.front}_{j}" for j in range(drivers.n_drivers)]
        else:
            names = list(self.front)
        missing = [nm for nm in names if nm not in self.faces or not len(self.faces[nm])]
        if missing:
            raise SelectionError(f"no faces for {missing} in the mesh (read: {list(self.faces)})")
        self.pnodes, self.Bf, self.wet_areas, _ = couple_drivers(
            self.nodes, [self.faces[nm] for nm in names], drivers, profiles, map_tol)
        self.wet_area = float(sum(self.wet_areas))
        if pistons:
            self.pnodes, self.Bf = add_pistons(self.nodes, self.pnodes, self.Bf, pistons)
        return self

    def couple_walls(self, walls, gap=None):
        """
        Elastic walls (physics.shell.WallModes) coupled to this air through every wall face of the mesh ("walls",
        "openings": the cabinet's outer faces, and in a merged air domain its inner faces and the boards' two sides):
        each point moves with the shell at its nearest point (rigid through the thickness: thin walls); faces
        farther than `gap` from every shell stay rigid (physics.shell.wall_face_coupling). The wall modes' columns
        are appended to Bf, after the driver modes (couple_driver first).
        """
        from loudpy.physics.modal import stack_columns
        from loudpy.physics.shell import wall_face_coupling
        if self.Bf is None:
            raise StateError("couple_driver first: the wall modes come after the driver modes")
        faces = [self.faces[n] for n in ("walls", "openings") if n in self.faces and len(self.faces[n])]
        if not faces:
            raise SelectionError("no wall faces ('walls', 'openings') in the air mesh")
        rows, Bw, dist = wall_face_coupling(walls, self.nodes, self.tets, np.vstack(faces), gap)
        n_drv = self.Bf.shape[1]
        self.pnodes, self.Bf = stack_columns([(self.pnodes, self.Bf), (rows, Bw)])
        self.wall_rows, self.wall_columns = rows, np.arange(n_drv, n_drv + Bw.shape[1])
        self.wall_volume_disp = Bw.sum(axis=0)                       # ∫ u_k·n dS into this air, per wall mode
        self.wall_distance = dist                                    # of each coupled row to the shells [m]
        return self

    def preconditioner(self, k):
        """LU of K − k²M + (ik + 1/R) M_Γ (local absorbing condition) as a LinearOperator: the GMRES
        preconditioner around wavenumber k."""
        A = (self.K - k * k * self.M + (1j * k + 1.0 / self.dtn.R) * self.MG).tocsc()
        lu = spla.splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                       options=dict(SymmetricMode=True))
        return spla.LinearOperator(A.shape, matvec=lu.solve, dtype=complex)

    def coefficients(self, p, k):
        """DtN expansion coefficients c_h = (2/R²) ∫_Γ p Y_h dS of an exterior field p at wavenumber k."""
        nh = self.dtn.n_harm(self.dtn.n_max(k))
        return 2.0 / self.dtn.R ** 2 * (self.G[:, :nh].T @ p[self.gnodes])

    def solve(self, k, f, precond=None, x0=None, rtol=1e-8):
        """Exterior field for the load f (no structure), on the builder: GMRES on K − k²M − D."""
        from loudpy.physics.builder import BlockSolver, Field, Load, SystemBuilder
        b = SystemBuilder()
        b.add_field(Field("p", len(self.nodes)))
        add_exterior(b, self, self.c)
        b.add_load(Load("p", np.asarray(f, complex)))
        x, it, info = BlockSolver(b, small=[], big="p", rtol=rtol, restart=150, maxiter=20).solve(
            1j * k * self.c, x0=x0, preconditioner=precond or self.preconditioner(k))
        return x["p"], it, info


def add_exterior(b, ext, c, field="p"):
    """Terms of the exterior FE air on `field`: (K + s² M/c²) p − G γ(k) Gᵀ p on Γ (half-space DtN, LowRank)."""
    from loudpy.physics.builder import LowRank, Term
    n = len(ext.nodes)
    b.add(Term(field, field, ext.K), Term(field, field, ext.M / (c * c), power=2))

    def dtn(z):
        k = z.imag / c
        n_max = ext.dtn.n_max(k)
        nh = ext.dtn.n_harm(n_max)
        return LowRank((n, n), ext.gnodes, ext.G[:, :nh], -ext.dtn.gamma(k, n_max))
    b.add(Term(field, field, dtn))


# ════════════════════════════════════════════════════════════════════════════
#  Coupled sweep: driver + enclosure (modal) + exterior (FE + DtN)
# ════════════════════════════════════════════════════════════════════════════
def scene_builder(ext, F, w_s, zeta, S, R, kappa2, eta_air=0.0, rho=RHO_AIR, c=C_AIR, H_extra=None, coils=None):
    """
    The 3D scene system on a SystemBuilder (time convention e^{st}, s = jω, k = ω/c):

      q row (driver modes + extra coordinates):
          [diag(ω_s²) + s diag(2ζω_s) + s² I] q  + H_extra(ω) x  + ρ s² (S D(s) Sᵀ + R) q  + Bfᵀ p[Γ_f] = F
          D(s) = 1/(κ² − k² + iη κ k): the cavity eliminated through its modes (S = 0: no cavity)
      p row (exterior FE):
          (K + s² M/c²) p − G γ(k) Gᵀ p (on Γ, the half-space DtN: a LowRank term)  − ρ s² Bf q = 0

    ext: an ExteriorFE with couple_driver done (pnodes, Bf). coils: a voltage drive, one current per driver
    (modal.add_coils: dict(Fc, Bl, Re, Le, n, U)); F is then the force on the modes besides the coils (zero).
    """
    from loudpy.physics.builder import Field, Load, SystemBuilder, Term
    from loudpy.physics.modal import add_driver_modes
    n_x = 0 if H_extra is None else H_extra(1.0).shape[0]
    n_y, n = len(w_s) + n_x, len(ext.nodes)
    b = SystemBuilder()
    b.add_field(Field("q", n_y))
    b.add_field(Field("p", n))
    add_driver_modes(b, w_s, zeta, S, R, kappa2, rho, c, eta_air, H_extra)
    add_exterior(b, ext, c)
    # wet surface: structure ↔ exterior air
    pn, Bf = ext.pnodes, ext.Bf
    Bq = sp.csr_matrix((Bf.ravel(), (np.repeat(pn, n_y), np.tile(np.arange(n_y), len(pn)))), shape=(n, n_y))
    b.add(Term("q", "p", Bq.T.tocsr()), Term("p", "q", -rho * Bq, power=2))
    b.add_load(Load("q", np.asarray(F, complex)))
    if coils is not None:
        from loudpy.physics.modal import add_coils
        add_coils(b, **coils)
    return b


def coupled_sweep(bands, freqs, F, w_s, zeta, S, R, kappa2, eta_air=0.0, rho=RHO_AIR, c=C_AIR,
                  chunk_hz=50.0, rtol=1e-7, on_frequency=None, log=None, H_extra=None, coils=None):
    """
    bands: list of (f_lo, f_hi, ExteriorFE with couple_driver done); each frequency is solved on
    the band containing it. Returns q (n_f, n_y) and, per frequency, the DtN coefficients c_h
    (list) — on_frequency(i, band_index, p, c_h, q) receives the full exterior field.
    H_extra(ω): dynamic matrix of extra generalised coordinates appended after the driver modes
    (e.g. the two mouths of a port); S, R, Bf and F then have n_y = n_modes + n_extra columns.
    Without a cavity, pass S of shape (n_y, 0), R = 0 and an empty kappa2. coils (a voltage drive: one current
    per driver, modal.add_coils): the currents are eliminated with q, and returned too — (q, coefs, iters, I).

    The system is the scene builder (scene_builder) solved by the generic BlockSolver: the modal field q
    eliminated exactly at each frequency, GMRES on the exterior field p with an LU preconditioner of
    K − k_c²M + (ik_c + 1/R) M_Γ refreshed every chunk_hz.
    """
    from loudpy.physics.builder import BlockSolver
    if log is None:                                   # progress messages: logger "loudpy"
        from loudpy._log import info as log
    n_x = 0 if H_extra is None else H_extra(1.0).shape[0]
    q_all = np.zeros((len(freqs), len(w_s) + n_x), complex)
    I_all = None if coils is None else np.zeros((len(freqs), np.shape(coils["Fc"])[1]), complex)
    coefs, iters = [None] * len(freqs), np.zeros(len(freqs), int)
    for b, (f_lo, f_hi, ext) in enumerate(bands):
        idx = [i for i, f in enumerate(freqs) if f_lo <= f <= f_hi and coefs[i] is None]
        if not idx:
            continue
        log(f"  band {f_lo:.0f}–{f_hi:.0f} Hz: {len(ext.nodes)} nodes, {len(ext.gnodes)} on Γ, "
            f"{len(ext.pnodes)} on the driver, n ≤ {ext.n_max_band}")
        solver = BlockSolver(scene_builder(ext, F, w_s, zeta, S, R, kappa2, eta_air, rho, c, H_extra, coils),
                             small="q" if coils is None else ["q", "I"], big="p", rtol=rtol, restart=150,
                             maxiter=20)
        p_prev, f_c, Mpre = None, None, None
        for i in idx:
            f = freqs[i]
            w, k = 2 * np.pi * f, 2 * np.pi * f / c
            if f_c is None or abs(f - f_c) > chunk_hz / 2:
                f_c = min(f + chunk_hz / 2, f_hi)
                t0 = time.time()
                Mpre = None                                   # free the previous LU first
                Mpre = ext.preconditioner(2 * np.pi * f_c / c)
                log(f"    LU at {f_c:.0f} Hz ({time.time() - t0:.0f} s)", flush=True)
            x, it, info = solver.solve(1j * w, x0=p_prev, preconditioner=Mpre)
            p = x["p"]
            if info != 0:
                warnings.warn(f"GMRES did not reach rtol = {rtol:g} at {f:.0f} Hz (info {info}); the result is kept",
                              ConvergenceWarning, stacklevel=2)
            q_all[i] = x["q"]
            if I_all is not None:
                I_all[i] = x["I"]
            coefs[i], iters[i], p_prev = ext.coefficients(p, k), it, p
            if on_frequency is not None:
                on_frequency(i, b, p, coefs[i], q_all[i])
    return (q_all, coefs, iters) if I_all is None else (q_all, coefs, iters, I_all)


def port_block(length, radius, rho=RHO_AIR, c=C_AIR, losses=False, nu=1.5e-5, gamma=1.4, prandtl=0.71):
    """
    Dynamic matrix of a straight port (plane waves) for the coordinates X1, X2 = volume
    displacements entering at the inner mouth and leaving at the outer mouth:
        [p̄1, −p̄2] = H(ω) [X1, X2],   H = (Z_c ω / sin kL) [[cos kL, −1], [−1, cos kL]],
    Z_c = ρc/S (no end correction: the 3D fields at both mouths provide it). losses: visco-thermal
    boundary-layer attenuation (Kirchhoff, wide tube).
    """
    S = np.pi * radius ** 2
    Zc = rho * c / S

    def H(w):
        k = w / c
        if losses:
            k = k * (1 + (1 - 1j) * np.sqrt(nu / (2 * w)) / radius * (1 + (gamma - 1) / np.sqrt(prandtl)))
        s, co = np.sin(k * length), np.cos(k * length)
        return Zc * w / s * np.array([[co, -1.0], [-1.0, co]])
    return H

