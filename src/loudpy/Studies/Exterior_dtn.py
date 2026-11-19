"""
Exterior_dtn.py — 3D exterior acoustics above a rigid ground, with the exact DtN (Dirichlet-to-
Neumann) radiation condition on a hemisphere, coupled to an axisymmetric driver (front side)
and to its enclosure (rear side, Cavity_coupling).

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
    H(ω) q + Bfᵀ p = F,   H = diag(ω_k² + 2iζ_kω_kω − ω²) − ρω²(S Dcav Sᵀ + R)   (Cavity_coupling)
    (K − k²M − D) p + ρω² Bf q = 0
q is eliminated (H is small and dense) and the exterior equation is solved by GMRES,
preconditioned by a sparse LU of K − k_c²M + (ik_c + 1/R) M_Γ (local absorbing condition)
factorised once per chunk of frequencies around k_c.

Several drivers: mesh_exterior with lists of frames writes "interface_front_<j>", and
ExteriorFE.couple_driver takes a MultiDriver (Cavity_coupling); q stacks all the drivers' modes.
coupled_sweep takes one force vector: combine the drivers' coil forces with their complex
amplitudes (MultiDriver.coil_forces(coil_nodes, amplitudes)), or run it once per driver.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import spherical_jn, spherical_yn, sph_harm_y

from loudpy.Studies.Cavity_coupling import (_gmsh, assemble_p2, read_p2_mesh, surface_quadrature,
                                            surface_coupling, wet_edges, add_pistons, RHO_AIR, C_AIR,
                                            as_frame, as_drivers, couple_drivers, _is_single,
                                            classify_driver_faces)


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
        return int(np.ceil(k * self.R)) + self.n_extra

    @staticmethod
    def harmonics(n_max):
        """(n, m) with n + |m| even, ordered by n: (n_max + 1)(n_max + 2)/2 of them."""
        return np.array([(n, m) for n in range(n_max + 1) for m in range(-n, n + 1) if (n + m) % 2 == 0])

    @staticmethod
    def n_harm(n_max):
        return (n_max + 1) * (n_max + 2) // 2

    def local(self, x):
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
def add_driver_front(segments, center, z_low):
    """
    In the current gmsh OCC model, the solid under the wet front profile of the driver
    (revolved about z through `center`) between z_low and the profile, and the cylinder
    r ≤ R_f, z_low ≤ z ≤ z_front that it replaces in the box. segments: chained (r, z)
    polylines from the box front face (first point, r = R_f, z = z_front) to the axis.
    `center` may be a DriverFrame: then everything is built in its local frame (z_low, R_f,
    z_front are local). Returns (rev, cyl, R_f, z_front).
    """
    gmsh = _gmsh()
    occ = gmsh.model.occ
    frame = as_frame(center)
    x0, y0, z0 = frame.origin if frame.is_identity else (0.0, 0.0, 0.0)
    r_f, z_f = segments[0][0]
    P = lambda r, z: occ.addPoint(x0 + float(r), y0, z0 + float(z))
    lines, first, cur = [], None, None
    for s in segments:
        tags = [cur if (cur is not None and k == 0) else P(*q) for k, q in enumerate(s)]
        first = first or tags[0]
        ok = np.allclose(np.cross(s[-1] - s[0], s - s[0]), 0.0, atol=1e-12)
        lines.append(occ.addLine(tags[0], tags[-1]) if ok else occ.addSpline(tags))
        cur = tags[-1]
    a, b = P(0.0, z_low), P(r_f, z_low)
    lines += [occ.addLine(cur, a), occ.addLine(a, b), occ.addLine(b, first)]
    face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    rev = [t for d, t in occ.revolve([(2, face)], x0, y0, z0, 0, 0, 1, 2 * np.pi) if d == 3][0]
    cyl = occ.addCylinder(x0, y0, z0 + z_low, 0, 0, z_f - z_low, r_f)
    if not frame.is_identity:
        frame.place([(3, rev), (3, cyl)])
    return rev, cyl, float(r_f), float(z_f)


def mesh_exterior(msh_path, volume, dtn: HalfSpaceDtN, center, r_f, z_front, h_near, h_far,
                  d_near=0.004, d_far=0.08):
    """
    Mesh the exterior (P2, straight edges) with the physical surfaces "dtn" (hemisphere),
    "ground", "interface_front" (wet front of the driver) and "walls" (box).

    Several drivers: center, r_f and z_front are lists (one center or DriverFrame per driver;
    r_f and z_front may be scalars); the wet fronts are "interface_front_0", "interface_front_1", …
    """
    gmsh = _gmsh()
    gmsh.model.occ.synchronize()
    single = _is_single(center)
    frames = [center] if single else list(center)
    fronts = ["interface_front"] if single else [f"interface_front_{j}" for j in range(len(frames))]
    groups = {"dtn": [], "ground": [], **{nm: [] for nm in fronts}, "walls": []}
    rest = []
    for _, t in gmsh.model.getBoundary([(3, volume)], oriented=False):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, t)
        lo, hi = (xmin, ymin, zmin), (xmax, ymax, zmax)
        if gmsh.model.getType(2, t) == "Sphere":
            groups["dtn"].append(t)
        elif abs(hi[dtn.up] - lo[dtn.up]) < 1e-6 and abs(hi[dtn.up] - dtn.center[dtn.up]) < 1e-6:
            groups["ground"].append(t)
        else:
            rest.append(t)
    owner = classify_driver_faces(rest, frames, r_f, z_front, atol=1e-6)
    for t, o in zip(rest, owner):
        groups[fronts[o] if o >= 0 else "walls"].append(t)
    for nm in fronts:
        if not groups[nm]:
            raise ValueError(f"no wet surface found for {nm}")
    gmsh.model.addPhysicalGroup(3, [volume], 1, name="air")
    for k, (name, ts) in enumerate(groups.items(), start=2):
        gmsh.model.addPhysicalGroup(2, ts, k, name=name)
    f = gmsh.model.mesh.field
    fd = f.add("Distance")
    f.setNumbers(fd, "SurfacesList", [t for nm in fronts for t in groups[nm]])
    f.setNumber(fd, "Sampling", 60)
    ft = f.add("Threshold")
    f.setNumber(ft, "InField", fd)
    for k, v in (("SizeMin", h_near), ("SizeMax", h_far), ("DistMin", d_near), ("DistMax", d_far)):
        f.setNumber(ft, k, v)
    f.setAsBackgroundMesh(ft)
    for k, v in (("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeFromPoints", 0),
                 ("Mesh.MeshSizeFromCurvature", 0), ("Mesh.ElementOrder", 2),
                 ("Mesh.SecondOrderLinear", 1), ("Mesh.Algorithm3D", 10)):
        gmsh.option.setNumber(k, v)
    gmsh.model.mesh.generate(3)
    Path(msh_path).parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(msh_path))
    return {k: len(v) for k, v in groups.items()}


def mesh_volume(msh_path, volume, classify, refine, h_far):
    """
    Mesh one air volume (P2, straight edges). classify(surface tag) → physical surface name;
    refine: {name: (h, d_near, d_far)} size ramps from the surfaces of that name up to h_far.
    Returns {name: number of surfaces}.
    """
    gmsh = _gmsh()
    gmsh.model.occ.synchronize()
    groups = {}
    for _, t in gmsh.model.getBoundary([(3, volume)], oriented=False):
        groups.setdefault(classify(t), []).append(t)
    gmsh.model.addPhysicalGroup(3, [volume], 1, name="air")
    for k, (name, ts) in enumerate(sorted(groups.items()), start=2):
        gmsh.model.addPhysicalGroup(2, ts, k, name=name)
    f = gmsh.model.mesh.field
    fields = []
    for name, (h, d0, d1) in refine.items():
        if name not in groups:
            continue
        fd = f.add("Distance")
        f.setNumbers(fd, "SurfacesList", groups[name])
        f.setNumber(fd, "Sampling", 60)
        ft = f.add("Threshold")
        f.setNumber(ft, "InField", fd)
        for k, v in (("SizeMin", h), ("SizeMax", h_far), ("DistMin", d0), ("DistMax", d1)):
            f.setNumber(ft, k, v)
        fields.append(ft)
    fm = f.add("Min")
    f.setNumbers(fm, "FieldsList", fields)
    f.setAsBackgroundMesh(fm)
    for k, v in (("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeFromPoints", 0),
                 ("Mesh.MeshSizeFromCurvature", 0), ("Mesh.MeshSizeMax", h_far),
                 ("Mesh.ElementOrder", 2), ("Mesh.SecondOrderLinear", 1), ("Mesh.Algorithm3D", 10)):
        gmsh.option.setNumber(k, v)
    gmsh.model.mesh.generate(3)
    Path(msh_path).parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(msh_path))
    return {k: len(v) for k, v in groups.items()}


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
            raise KeyError(f"no faces for {missing} in the mesh (read: {list(self.faces)})")
        self.pnodes, self.Bf, self.wet_areas, _ = couple_drivers(
            self.nodes, [self.faces[nm] for nm in names], drivers, profiles, map_tol)
        self.wet_area = float(sum(self.wet_areas))
        if pistons:
            self.pnodes, self.Bf = add_pistons(self.nodes, self.pnodes, self.Bf, pistons)
        return self

    def preconditioner(self, k):
        A = (self.K - k * k * self.M + (1j * k + 1.0 / self.dtn.R) * self.MG).tocsc()
        lu = spla.splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                       options=dict(SymmetricMode=True))
        return spla.LinearOperator(A.shape, matvec=lu.solve, dtype=complex)

    def operator(self, k, extra=None):
        """K − k²M − D (+ extra(v), a low-rank term) as a LinearOperator."""
        nh = self.dtn.n_harm(self.dtn.n_max(k))
        G, gam, gn = self.G[:, :nh], self.dtn.gamma(k, self.dtn.n_max(k)), self.gnodes
        A0 = (self.K - k * k * self.M).tocsr()

        def mv(v):
            out = A0 @ v
            out[gn] -= G @ (gam * (G.T @ v[gn]))
            if extra is not None:
                out += extra(v)
            return out
        return spla.LinearOperator(A0.shape, matvec=mv, dtype=complex)

    def coefficients(self, p, k):
        nh = self.dtn.n_harm(self.dtn.n_max(k))
        return 2.0 / self.dtn.R ** 2 * (self.G[:, :nh].T @ p[self.gnodes])

    def solve(self, k, f, precond=None, x0=None, rtol=1e-8, extra=None):
        """Exterior field for the load f (no structure unless `extra`)."""
        M = precond or self.preconditioner(k)
        it = [0]
        p, info = spla.gmres(self.operator(k, extra), f, x0=x0, M=M, rtol=rtol, restart=150,
                             maxiter=20, callback=lambda r: it.__setitem__(0, it[0] + 1),
                             callback_type="pr_norm")
        return p, it[0], info


# ════════════════════════════════════════════════════════════════════════════
#  Coupled sweep: driver + enclosure (modal) + exterior (FE + DtN)
# ════════════════════════════════════════════════════════════════════════════
def coupled_sweep(bands, freqs, F, w_s, zeta, S, R, kappa2, eta_air=0.0, rho=RHO_AIR, c=C_AIR,
                  chunk_hz=50.0, rtol=1e-7, on_frequency=None, log=print, H_extra=None):
    """
    bands: list of (f_lo, f_hi, ExteriorFE with couple_driver done); each frequency is solved on
    the band containing it. Returns q (n_f, n_y) and, per frequency, the DtN coefficients c_h
    (list) — on_frequency(i, band_index, p, c_h, q) receives the full exterior field.
    H_extra(ω): dynamic matrix of extra generalised coordinates appended after the driver modes
    (e.g. the two mouths of a port); S, R, Bf and F then have n_y = n_modes + n_extra columns.
    Without a cavity, pass S of shape (n_y, 0), R = 0 and an empty kappa2.
    """
    n_x = 0 if H_extra is None else H_extra(1.0).shape[0]
    q_all = np.zeros((len(freqs), len(w_s) + n_x), complex)
    coefs, iters = [None] * len(freqs), np.zeros(len(freqs), int)
    for b, (f_lo, f_hi, ext) in enumerate(bands):
        idx = [i for i, f in enumerate(freqs) if f_lo <= f <= f_hi and coefs[i] is None]
        if not idx:
            continue
        log(f"  band {f_lo:.0f}–{f_hi:.0f} Hz: {len(ext.nodes)} nodes, {len(ext.gnodes)} on Γ, "
            f"{len(ext.pnodes)} on the driver, n ≤ {ext.n_max_band}")
        pn, Bf = ext.pnodes, ext.Bf
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
            D = 1.0 / (kappa2 - k * k + 1j * eta_air * np.sqrt(kappa2) * k)
            H = np.zeros((len(w_s) + n_x,) * 2, complex)
            H[:len(w_s), :len(w_s)] = np.diag(w_s ** 2 + 2j * zeta * w_s * w - w ** 2)
            if n_x:
                H[len(w_s):, len(w_s):] = H_extra(w)
            H -= rho * w ** 2 * ((S * D) @ S.T + R)
            Hlu = sla.lu_factor(H)
            n = len(ext.nodes)

            def extra(v):
                out = np.zeros(n, complex)
                out[pn] = -rho * w ** 2 * (Bf @ sla.lu_solve(Hlu, Bf.T @ v[pn]))
                return out
            rhs = np.zeros(n, complex)
            rhs[pn] = -rho * w ** 2 * (Bf @ sla.lu_solve(Hlu, F))
            p, it, info = ext.solve(k, rhs, Mpre, p_prev, rtol, extra)
            if info != 0:
                log(f"    ! GMRES did not converge at {f:.0f} Hz (info {info})")
            q_all[i] = sla.lu_solve(Hlu, F - Bf.T @ p[pn])
            coefs[i], iters[i], p_prev = ext.coefficients(p, k), it, p
            if on_frequency is not None:
                on_frequency(i, b, p, coefs[i], q_all[i])
    return q_all, coefs, iters


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

