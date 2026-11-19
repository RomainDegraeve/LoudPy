import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, splu, LinearOperator, ArpackNoConvergence
from loudpy.Studies.fast_linalg import factorize, QEPShiftInvert  # noqa: F401
from loudpy.solvers import eigen as _eig
from loudpy.solvers.eigen import (ZETA_REAL, has_imaginary_part as _has_imaginary_part,  # noqa: F401
                                  to_csc_complex as _to_csc_complex, shift_invert_operator as _shift_invert_op,
                                  arpack_eigs as _run_arpack, real_modes as _real_modes,
                                  weighted_mac as _weighted_mac, mac_deduplicate as _mac_deduplicate)
import warnings
import sys


from loudpy.physics.structural import MecaAssembler
from loudpy.physics.registry import select
from scipy.sparse.linalg import eigsh
from loudpy.Studies.Study_class import Problem
from loudpy.Studies.DofMapMixin import DofMapMixin

# ═══════════════════════════════════════════════════════════════════════════ #
#  Helpers                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #


# ═══════════════════════════════════════════════════════════════════════════ #
#  EigenStudy                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class EigenStudy(Problem, DofMapMixin):
    """
    Modal analysis — four cases handled transparently:

    ┌──────────────────┬───────┬──────────────────────────────────────────┐
    │ K                │  C    │ Solver                                   │
    ├──────────────────┼───────┼──────────────────────────────────────────┤
    │ real             │  no   │ GEP  (K−λM)φ=0,  σ = +ω²  (real)       │
    │ real             │  yes  │ QEP state-space,  σ = jω   (complex)    │
    │ complex (hyster) │  no   │ GEP  (K−λM)φ=0,  σ = −ω²  (complex)    │
    │ complex (hyster) │  yes  │ QEP state-space,  σ = jω   (complex)    │
    └──────────────────┴───────┴──────────────────────────────────────────┘

    Loss factor η and damping ratio ζ are extracted uniformly from the
    s-plane eigenvalue:  s = j√λ,  ζ = −Re(s)/|s|,  η = 2ζ.
    """

    def __init__(self, problem):
        super().__init__(geo_path=problem.geo_path,
                        msh_path=problem.msh_path,
                        mat_path=problem.mat_path,
                        subdomains_key=problem.subdomains_key)
        self.fem_objects = problem.fem_objects
        self.nodes       = problem.nodes              # ← keep for ResultStore.save_mesh_from_problem
        self.specs_dom   = problem.specs_dom          # ← so specs_dict() works
        self._assemblers: dict = {}
        self._results: list[dict] = []   
        self.specs_interf  = problem.specs_interf
        self._problem      = problem

         # ── collect interface node tags ────────────────────────────────────
        self._interface_meta = {}
        for spec in self.specs_interf:
            tags = self._get_interface_node_tags(spec)
            self._interface_meta[spec.name] = {
                "kind":      spec.__class__.__name__,
                "node_tags": tags,
                "edges":     self._get_interface_edges(spec),

            }

    


    # ------------------------------------------------------------------ #
    #  Assembly                                                            #
    # ------------------------------------------------------------------ #
    def _assemble(self):
        self.meca = MecaAssembler(self.fem_objects)
        reduced = self.meca.apply_blocked_boundaries(
                    self.fem_objects, *self.meca.assemble())

        if len(reduced) == 2:
            K, M = reduced;  C = None
        elif len(reduced) == 3:
            K, M, C = reduced
        else:
            raise ValueError(f"Expected 2 or 3 matrices, got {len(reduced)}")

        K, M = _to_csc_complex(K, M)
        if C is not None:
            [C] = _to_csc_complex(C)
            if np.abs(C).sum() < 1e-30:
                print("  C is numerically zero → treated as undamped.")
                C = None

        # register for DofMapMixin (after BCs so `_free` is set)
        self._assemblers["u_meca"] = self.meca

        return K, M, C

    # ------------------------------------------------------------------ #
    #  GEP solver  —  (K − λM)φ = 0,  λ = ω²                             #
    # ------------------------------------------------------------------ #

    
    def solve_meca_eigen_ARPAC(
        self,
        n_modes: int   = 30,
        f_min:   float = 20.0,
        f_max:   float = 20_000.0,
        f_target: float = 100.0,
    ) -> tuple:
        """
        Returns
        -------
        frequencies  : (n,)   natural frequencies in Hz
        zeta         : (n,)   modal damping ratios
        mode_shapes  : (n, N) complex mode shapes (normalised)
        """
        K, M, C = self._assemble()
 
        hysteretic  = _has_imaginary_part(K)   # complex K → hysteretic damping
        viscous     = C is not None             # explicit C matrix present
 
        print(f"\n  K={'complex' if hysteretic else 'real'}, "
              f"viscous damping={'yes' if viscous else 'no'}")
 
        # ── Solve (raw eigenpairs, no filtering) ───────────────────────
        if viscous:
            s_vals, phi = self._solve_qep(K, M, C, n_modes, f_target)
        else:
            s_vals, phi = self._solve_gep(K, M, n_modes, f_target, hysteretic)
 
        # ── Remove numerical duplicates ────────────────────────────────
        s_vals, phi = _mac_deduplicate(s_vals, phi,
                                       freq_tol_rad=1.0, mac_tol=0.99999)
 
        # ── Post-process: half-plane, ζ/real, growing, band, sort ──────
        frequencies, zeta, loss_factors, s_vals, phi = \
            self._postprocess(s_vals, phi, f_min, f_max,
                              filter_halfplane=(hysteretic or viscous))
 
        # ── Normalise mode shapes ──────────────────────────────────────
        reduced_mode_shapes = self._normalise(
            phi, s_vals, M, C, hysteretic, viscous
        )
 
        # ── Expand to full DOF size ────────────────────────────────────
        full_modes_list = [self.meca.expand(reduced_mode_shapes[:, i])
                           for i in range(reduced_mode_shapes.shape[1])]
        mode_shapes_full = np.array(full_modes_list)
 
        # ── Print and Return ───────────────────────────────────────────
        self._print_summary(frequencies, zeta, loss_factors, hysteretic, viscous)
 
        self._results.append({
            "kind":    "meca",
            "dof_key": "u_meca",
            "freqs":   frequencies,
            "zeta":    zeta,
            "modes":   mode_shapes_full,           # already full-DOF
            "solver_params": {
                "n_modes":  n_modes,
                "f_min":    f_min,
                "f_max":    f_max,
                "f_target": f_target,
            },
            "interfaces": self._interface_meta,
        })
 
        return frequencies, zeta, mode_shapes_full
 
    # ------------------------------------------------------------------ #
    #  QEP solver  —  (s²M + sC + K)φ = 0  via state-space               #
    #                                                                      #
    #   A = [ 0   I ]    B = [ I  0 ]    σ = j·ω_t                       #
    #       [−K  −C ]        [ 0  M ]                                     #
    #                                                                      #
    #  Returns RAW eigenpairs (both halves of each conjugate pair, real   #
    #  eigenvalues included). Filtering is done in _postprocess.          #
    # ------------------------------------------------------------------ #
 
    # ------------------------------------------------------------------ #
    #  Post-processing  —  the ONLY place where eigenvalues are filtered  #
    # ------------------------------------------------------------------ #
 
    # ------------------------------------------------------------------ #
    #  Multi-shift core  (in-vacuo solve)                                 #
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    #  Public entry point  (multi-shift)                                  #
    # ------------------------------------------------------------------ #
    def solve_meca_eigen_multishift(
        self,
        f_targets: list[float] = None,
        n_modes_per_shift: int = 20,
        f_min: float = 20.0,
        f_max: float = 6000.0,
        dedup_freq_tol_hz: float = 0.5,
        mac_tol: float = 0.99,
    ) -> tuple:
        """
        Multi-shift modal analysis: runs the QEP/GEP solver at several spectral
        shifts spread across [f_min, f_max], then merges + deduplicates the
        results. Covers a wide band uniformly without a huge single n_modes.
 
        Parameters
        ----------
        f_targets : list[float]
            Shift frequencies [Hz]. If None, log-spaced across the band.
        n_modes_per_shift : int
            Modes requested per shift (kept modest — each shift is local).
        dedup_freq_tol_hz : float
            Two modes closer than this (Hz) and MAC-similar are merged.
        """
        K, M, C = self._assemble()
        hysteretic = _has_imaginary_part(K)
        viscous    = C is not None
 
        print(f"\n  K={'complex' if hysteretic else 'real'}, "
              f"viscous damping={'yes' if viscous else 'no'}")
 
        if f_targets is None:
            f_targets = self._default_shifts(f_min, f_max)

        s_vals, phi = self._multishift_core(K, M, C, f_targets,
                                            n_modes_per_shift, hysteretic)

        # ── global dedup across shifts (freq + MAC) ────────────────────────
        freq_tol_rad = 2.0 * np.pi * dedup_freq_tol_hz
        s_vals, phi  = _mac_deduplicate(
            s_vals, phi,
            freq_tol_rad=freq_tol_rad,
            mac_tol=mac_tol,
        )
 
        # ── post-process: half-plane, ζ/real, growing, band, sort ──────────
        frequencies, zeta, loss_factors, s_vals, phi = \
            self._postprocess(s_vals, phi, f_min, f_max,
                              filter_halfplane=(hysteretic or viscous))
 
        # ── normalise + expand ─────────────────────────────────────────────
        reduced = self._normalise(phi, s_vals, M, C, hysteretic, viscous)
        full_modes_list = [self.meca.expand(reduced[:, i])
                           for i in range(reduced.shape[1])]
        mode_shapes_full = np.array(full_modes_list)
 
        self._print_summary(frequencies, zeta, loss_factors, hysteretic, viscous)
 
        self._results.append({
            "kind":    "meca",
            "dof_key": "u_meca",
            "freqs":   frequencies,
            "zeta":    zeta,
            "modes":   mode_shapes_full,
            "solver_params": {
                "f_targets":         list(f_targets),
                "n_modes_per_shift": n_modes_per_shift,
                "f_min":             f_min,
                "f_max":             f_max,
            },
            "interfaces": self._interface_meta,
        })
 
        return frequencies, zeta, mode_shapes_full

    # ------------------------------------------------------------------ #
    #  Non-axisymmetric modes: azimuthal (Fourier) order m                #
    # ------------------------------------------------------------------ #
    def solve_meca_eigen_fourier(
        self,
        m: int,
        f_targets: list[float] = None,
        n_modes_per_shift: int = 20,
        f_min: float = 20.0,
        f_max: float = 6000.0,
        dedup_freq_tol_hz: float = 0.5,
        mac_tol: float = 0.99,
    ) -> tuple:
        """
        In-vacuo modes of azimuthal order m of the axisymmetric structure (multishift,
        same solver, damping and normalisation as solve_meca_eigen_multishift).

        The 3D mode is  u_r = U_r(r,z) cos mθ,  u_z = U_z(r,z) cos mθ,  u_θ = U_θ(r,z) sin mθ
        (each mode of order m ≥ 1 has a twin rotated by π/(2m): same frequency).
        m = 0 calls solve_meca_eigen_multishift (axisymmetric modes, 2 DOF per node,
        saved as "meca"); m ≥ 1 uses MecaAssemblerFourier (3 DOF per node, saved as
        "meca_m<m>", fields u_meca = (U_r, U_z, U_θ)). The modes are normalised over the
        full 3D body (θ-integral π for m ≥ 1, 2π for m = 0).
        """
        m = int(m)
        if m == 0:
            return self.solve_meca_eigen_multishift(
                f_targets=f_targets, n_modes_per_shift=n_modes_per_shift, f_min=f_min,
                f_max=f_max, dedup_freq_tol_hz=dedup_freq_tol_hz, mac_tol=mac_tol)

        from loudpy.physics.structural import MecaAssemblerFourier
        asm = MecaAssemblerFourier(self.fem_objects, m)
        K, M, C = asm.apply_blocked_boundaries(self.fem_objects, *asm.assemble())
        K, M = _to_csc_complex(K, M)
        [C] = _to_csc_complex(C)
        if np.abs(C).sum() < 1e-30:
            C = None
        hysteretic = _has_imaginary_part(K)
        viscous = C is not None
        key = f"u_meca_m{m}"
        self._assemblers[key] = asm
        print(f"\n  Azimuthal order m = {m} | {asm.matrix_size} DOF (3 per node), "
              f"{K.shape[0]} after constraints | K={'complex' if hysteretic else 'real'}, "
              f"viscous damping={'yes' if viscous else 'no'}")

        if f_targets is None:
            f_targets = self._default_shifts(f_min, f_max)
        s_vals, phi = self._multishift_core(K, M, C, f_targets, n_modes_per_shift, hysteretic)
        s_vals, phi = _mac_deduplicate(s_vals, phi, freq_tol_rad=2.0 * np.pi * dedup_freq_tol_hz,
                                       mac_tol=mac_tol)
        frequencies, zeta, loss_factors, s_vals, phi = \
            self._postprocess(s_vals, phi, f_min, f_max, filter_halfplane=(hysteretic or viscous))
        reduced = self._normalise(phi, s_vals, M, C, hysteretic, viscous)
        modes = np.array([asm.expand(reduced[:, i]) for i in range(reduced.shape[1])])
        self._print_summary(frequencies, zeta, loss_factors, hysteretic, viscous)

        self._results.append({
            "kind":    f"meca_m{m}",
            "dof_key": key,
            "freqs":   frequencies,
            "zeta":    zeta,
            "modes":   modes,
            "solver_params": {
                "azimuthal_order":   m,
                "f_targets":         list(f_targets),
                "n_modes_per_shift": n_modes_per_shift,
                "f_min":             f_min,
                "f_max":             f_max,
            },
            "interfaces": self._interface_meta,
        })
        return frequencies, zeta, modes

    # ------------------------------------------------------------------ #
    #  Coupled fluid–structure eigen study (exact DtN radiation condition)#
    # ------------------------------------------------------------------ #
    def _assemble_fsi_blocks(self):
        """Structural (K, M, R), acoustic (H, Q, ρ) and coupling (Cam) blocks."""
        from loudpy.physics.acoustic import AcouAssembler
        from loudpy.physics.coupling import FsiAssembler

        fo   = self.fem_objects
        meca = MecaAssembler(fo)
        K, M, R = meca.assemble()
        acou = AcouAssembler(fo)
        H, Q, rho = acou.assemble()
        Cam  = FsiAssembler(fo, meca, acou).assemble()        # (n_acou, n_meca)
        K, M, R, Cam = meca.apply_blocked_boundaries(fo, K, M, R, Cam)

        self.meca, self.acou = meca, acou
        self._assemblers["u_meca"] = meca
        self._assemblers["p_acou"] = acou
        if np.abs(R).sum() < 1e-30:
            print("  R is numerically zero → treated as undamped.")
            R = None
        print(f"\n  Coupled FSI | n_meca = {K.shape[0]} (free), n_acou = {H.shape[0]}, ρ = {rho}")
        return K, M, R, Cam, H, Q, rho

    def _assemble_dtn_system(self, center, f_max, half=True, n_max=None):
        """
        Coupled system with the exact DtN radiation condition (no PML), in the
        SYMMETRIC u–ψ–φ form (p = −sψ, φ = auxiliary DtN unknowns):

            (s² Ms + s Cs + Ks) [u; ψ; φ] = [F; 0; 0]

          Ms = diag(M, −Q/ρ, 0)
          Cs = [[R, Camᵀ, 0], [Cam, −Σ σβ_n b_n b_nᵀ/(ρc), 0], [0, 0, −σβ/(ρc) I]]
          Ks = [[K, 0, 0], [0, −H/ρ − Σ σβ_n b_n b_nᵀ/(ρR), G], [0, Gᵀ, diag(σβ z_i/(ρR))]]
          G[:, i] = σ β_n √z_i b_n/(ρR)  for the i-th root z_i of order n

        One DtN operator per InterfaceDtN arc (e.g. a front and a rear
        hemisphere, each closing its own half-space): every arc has its own
        centre, radius, orders and auxiliary unknowns. σ = ±1 is the arc's
        orientation ("front"/"rear" in its name), the sign of the acoustic
        sub-domain it closes (loudpy multiplies the rear acoustic matrices by −1).

        center: (0, z0) for every arc, {arc name: (0, z0)}, or None — then each
        arc's centre is (0, z of its node farthest from the axis), i.e. the
        centre of a hemisphere standing on a baffle plane.

        (see loudpy.physics.radiation for D_n(s) and its exact partial fractions).
        Complex symmetric (z_i complex), spectrum symmetric under conjugation.
        """
        from loudpy.physics.radiation import dtn_boundary_vectors, dtn_orders, reverse_bessel_roots

        K, M, R, Cam, H, Q, rho = self._assemble_fsi_blocks()
        nm, na = K.shape[0], H.shape[0]
        c = self._acoustic_c()

        arcs = select(self.fem_objects, "radiation")
        if not arcs:
            raise ValueError("No InterfaceDtN boundary in the problem "
                             "(add InterfaceSpecDtN('dtn_boundary')).")

        Dk = sp.csr_matrix((na, na)); Dc = sp.csr_matrix((na, na))
        G_rows, G_cols, G_vals = [], [], []
        z_aux, beta_aux, sign_aux, R_aux = [], [], [], []
        info = []
        for bd in arcs:
            xy = np.asarray(bd.node_coords).reshape(len(bd.node_tags), -1)[:, :2]
            if isinstance(center, dict) and bd.name in center:
                ctr = tuple(center[bd.name])
            elif center is not None and not isinstance(center, dict):
                ctr = tuple(center)
            else:
                ctr = (0.0, float(xy[np.argmax(xy[:, 0]), 1]))
            sign = float(getattr(bd, "orientation", 1.0))
            # radius first (order N depends on it)
            _, R_b = dtn_boundary_vectors(bd.edges, bd.node_tags, bd.node_coords,
                                          self.acou.tag_to_local, na, ctr, [0])
            orders, n_top = dtn_orders(R_b, f_max, c, half, n_max)
            Bvec, R_b = dtn_boundary_vectors(bd.edges, bd.node_tags, bd.node_coords,
                                             self.acou.tag_to_local, na, ctr, orders)
            bnd = np.flatnonzero(np.any(Bvec != 0, axis=0))           # boundary DOFs
            dk = np.zeros((bnd.size, bnd.size)); dc = np.zeros_like(dk)
            n_aux0 = len(z_aux)
            for j, n in enumerate(orders):
                beta = (2 * n + 1) / ((2.0 if half else 4.0) * np.pi * R_b ** 2)
                b = Bvec[j, bnd]
                dk -= sign * beta / (rho * R_b) * np.outer(b, b)
                dc -= sign * beta / (rho * c) * np.outer(b, b)
                for zi in reverse_bessel_roots(n):
                    G_rows.append(bnd); G_cols.append(np.full(bnd.size, len(z_aux)))
                    G_vals.append(sign * beta * np.sqrt(zi) / (rho * R_b) * b)
                    z_aux.append(zi); beta_aux.append(beta); sign_aux.append(sign); R_aux.append(R_b)
            ii, jj = np.meshgrid(bnd, bnd, indexing="ij")
            Dk = Dk + sp.csr_matrix((dk.ravel(), (ii.ravel(), jj.ravel())), shape=(na, na))
            Dc = Dc + sp.csr_matrix((dc.ravel(), (ii.ravel(), jj.ravel())), shape=(na, na))
            info.append(dict(name=bd.name, R=R_b, center=ctr, orders=orders, sign=sign,
                             n_aux=len(z_aux) - n_aux0, n_bnd=bnd.size))
            print(f"\n  DtN arc {bd.name} | R = {R_b:.4f} m, centre {ctr}, orientation {sign:+.0f}, "
                  f"orders {orders[0]}..{orders[-1]} "
                  f"({'even, baffled hemisphere' if half else 'all, sphere'}), "
                  f"{len(z_aux) - n_aux0} auxiliary unknowns, {bnd.size} boundary nodes")

        n_aux = len(z_aux)
        z_aux, beta_aux = np.array(z_aux), np.array(beta_aux)
        sign_aux, R_aux = np.array(sign_aux), np.array(R_aux)
        G = (sp.csr_matrix((np.concatenate(G_vals), (np.concatenate(G_rows), np.concatenate(G_cols))),
                           shape=(na, n_aux)) if n_aux else sp.csr_matrix((na, 0)))
        Raux = sp.csr_matrix((nm, nm)) if R is None else R
        Zaa = sp.diags(sign_aux * beta_aux * z_aux / (rho * R_aux)) if n_aux else None
        Caa = sp.diags(-sign_aux * beta_aux / (rho * c)) if n_aux else None

        Ms = sp.bmat([[M, None, None], [None, -Q / rho, None],
                      [None, None, sp.csr_matrix((n_aux, n_aux))]])
        Cs = sp.bmat([[Raux, Cam.T, None], [Cam, Dc, None], [None, None, Caa]])
        Ks = sp.bmat([[K, None, None], [None, -H / rho + Dk, G], [None, G.T, Zaa]])
        Ks, Ms, Cs = _to_csc_complex(Ks, Ms, Cs)

        print(f"\n  Coupled FSI + DtN | n_meca = {nm}, n_acou = {na}, {len(arcs)} DtN arc(s), "
              f"{n_aux} auxiliary unknowns in total")
        self._dtn_system = dict(Ks=Ks, Ms=Ms, Cs=Cs, nm=nm, na=na, n_aux=n_aux, arcs=info, c=c)
        return Ks, Ms, Cs, nm, na

    def _make_goal_check(self, Ks, Ms, Cs, nm, na, f_min, f_max, goal):
        """Build the goal-oriented stop function for _covering_core (DtN form)."""
        from scipy.sparse.linalg import spsolve
        from loudpy.physics.forcing import ForceAssembler

        fo = self.fem_objects
        F_full = ForceAssembler(fo, self.meca).assemble() * float(goal.get("force", 1.0))
        F = F_full[self.meca._free]
        rhs = np.zeros(Ks.shape[0], dtype=complex); rhs[:nm] = F

        xy = {}
        for o in fo:
            if hasattr(o, "node_tags") and hasattr(o, "node_coords"):
                cc = np.asarray(o.node_coords).reshape(len(o.node_tags), -1)
                for t, c_ in zip(o.node_tags, cc):
                    xy[int(t)] = c_[:2]
        axy = np.array([xy[int(t)] for t in self.acou.unique_tags])
        rec = np.array([int(np.argmin(np.sum((axy - np.asarray(q)) ** 2, axis=1)))
                        for q in goal["receivers"]])
        freqs = np.asarray(goal.get("freqs", np.geomspace(max(20.0, f_min), f_max / 1.6, 30)))
        tol = float(goal.get("tol_db", 1.0))

        p_ex = np.empty((freqs.size, rec.size), dtype=complex)
        for i, f in enumerate(freqs):
            s = 2j * np.pi * f
            x = factorize(s * s * Ms + s * Cs + Ks).solve(rhs)
            p_ex[i] = -s * x[nm:nm + na][rec]
        print(f"  Goal | exact pressure at {rec.size} receivers, {freqs.size} frequencies "
              f"({freqs.min():.0f}–{freqs.max():.0f} Hz), tolerance {tol:g} dB")

        jw = 2j * np.pi * freqs[:, None]
        MsT, CsT = Ms, Cs
        cache = {}                                       # index → (s, a_k, p_k at receivers)

        def stop_fn(s_list, phi_list):
            for k in range(len(cache), len(s_list)):
                (sk,), (real,) = _real_modes([s_list[k]])
                x = phi_list[k]
                keep = (sk.real < 0) and (real or sk.imag > 0)
                if not keep:
                    cache[k] = None; continue
                norm = x @ (2.0 * sk * (MsT @ x) + CsT @ x)
                x = x / np.sqrt(norm)
                # weight ½ for a real mode: with its "partner" term below it counts once
                cache[k] = (sk, (0.5 if real else 1.0) * (x[:nm] @ F),
                            -sk * x[nm:nm + na][rec])
            items = [v for v in cache.values() if v is not None]
            if not items:
                return False, "no mode yet"
            S = np.array([v[0] for v in items]); A = np.array([v[1] for v in items])
            P = np.array([v[2] for v in items])
            p = (((jw / S) * A / (jw - S)) @ P
                 + ((jw / np.conj(S)) * np.conj(A) / (jw - np.conj(S))) @ np.conj(P))
            e = 20 * np.log10(np.abs(p) / np.abs(p_ex))
            emax = float(np.abs(e).max())
            return emax <= tol, (f"max error {emax:.2f} dB, rms {np.sqrt(np.mean(e**2)):.2f} dB "
                                 f"({len(items)} modes)")

        return stop_fn

    def _acoustic_c(self) -> float:
        """Speed of sound of the acoustic sub-domains."""
        cs = {float(sd.c) for sd in self.acou.subdomains if getattr(sd, "c", None) is not None}
        if len(cs) != 1:
            raise ValueError(f"DtN needs one speed of sound in the air, found {cs}")
        return cs.pop()


    def solve_fsi_eigen(
        self,
        f_min: float = 20.0,
        f_max: float = 5000.0,
        n_modes_per_shift: int = 30,
        dedup_freq_tol_hz: float = 0.5,
        dedup_rel_tol: float = 0.002,
        mac_tol: float = 0.99,
        cover_zeta_max: float = 0.999,
        cover_overlap: float = 0.85,
        max_shifts: int = 60,
        dtn_center: tuple | None = None,
        dtn_half: bool = True,
        dtn_order: int | None = None,
        cover_goal: dict | None = None,
        cover_avoid_frac: float = 0.25,
        checkpoint_path: str | None = None,
        checkpoint_every: int = 3,
    ) -> tuple:
        """
        Coupled structural–acoustic modes of the loudspeaker radiating in free
        space, with the EXACT non-reflecting DtN condition on the circular
        arc(s) of the problem (InterfaceSpecDtN; one arc per half-space, e.g.
        "dtn_boundary_front" and "dtn_boundary_rear", oriented by their names).
        dtn_center: centre (0, z0) of every arc, {arc name: (0, z0)}, or None
        (centre of each hemisphere found from its foot on the baffle plane).
        dtn_half=True: hemispheres on a rigid baffle through their centres
        (even orders only); False: full sphere. dtn_order: truncation N
        (default ⌈k_max R⌉ + 10).

        The modes are found by a gap-free covering of the s-plane (see
        _covering_core): every mode with f in [f_min, f_max] and
        ζ ≤ cover_zeta_max, including the heavily damped radiation modes that
        the modal sum of the pressure needs (keep cover_zeta_max ≈ 0.999).
        Those radiation modes depend on the DtN radius; individually they
        have no physical meaning, only their sum does.

        The real (overdamped, ζ = 1) eigenvalues are searched and kept too,
        stored with Im(s) = 0: the pressure needs them as well (e.g. two
        rear-air modes near −1.6 and −1.9 kHz carry ~10 % of the front
        pressure). A real mode is its own conjugate partner, so it enters the
        modal sum once:
            p(jω) = Σ_oscillating [(jω/s)p a/(jω − s) + conjugate] + Σ_real (jω/s)p a/(jω − s)

        n_modes_per_shift : eigenvalues requested per ARPACK solve.
        max_shifts        : cap on the number of ARPACK solves.
        cover_overlap     : a grid point is covered within overlap·radius of a shift.
        dedup_freq_tol_hz, dedup_rel_tol, mac_tol : duplicate removal between
                            overlapping shifts. Two eigenvalues closer than
                            max(dedup_freq_tol_hz, dedup_rel_tol·f) are compared,
                            and are the same mode if their mass-weighted MAC
                            exceeds mac_tol (the MAC decides, the window only
                            limits the comparisons).

        cover_avoid_frac: Rayleigh damping (C = αM + βK) sends every
            structural mode above f = 1/(πβ) to ζ > 1, and their eigenvalues
            pile up on the real axis at s = −1/β (thousands of them for a fine
            mesh). The covering skips a disc of radius cover_avoid_frac·|1/β|
            around each such point inside the region (one per material β), and
            shifts placed near a pile solve a filtered operator that keeps the
            pile out of ARPACK's wanted set (see _solve_qep_sigma), so ARPACK
            never stalls there. Set 0 to disable.

        checkpoint_path: if given, the modes found so far are normalised and
            saved to this file every checkpoint_every shifts (same format as
            save(); written to a temporary file first, then renamed, so the
            file is never left half-written). If the run stops or blocks, the
            last checkpoint is usable as is. Usually the same path as the final
            study.save(...).

        cover_goal: goal-oriented stop. The exact forced pressure at given
        receivers is computed first (direct solves); after every shift the
        modal sum of the modes found so far is compared with it, and the
        covering stops as soon as the error is ≤ tol_db everywhere. Keys:
            receivers : list of (r, z) points [m], inside the DtN sphere (required)
            freqs     : check frequencies [Hz]  (default: 30 log-spaced, 20 Hz–f_max/1.6)
            tol_db    : max |SPL_modal − SPL_exact| [dB]  (default 1.0)
            force     : amplitude [N] of the interface_forced pattern (default 1.0)

        Returns
        -------
        frequencies : (n,)            Hz
        zeta        : (n,)            damping ratios (structural + radiation)
        u_modes     : (n, n_dof_meca) complex, full mechanical DOF size
        p_modes     : (n, n_dof_acou) complex
        s_vals      : (n,)            complex eigenvalues s_k (also saved)
        Normalisation: x_kᵀ(2 s_k Ms + Cs) x_k = 1 on the symmetric u–ψ–φ form
        (p = −sψ, see _assemble_dtn_system), so that for a structural force F
            u(jω) = Σ_k u_k (u_kᵀF)/(jω − s_k) + conj partners
            p(jω) = Σ_k (jω/s_k) p_k (u_kᵀF)/(jω − s_k) + conj partners
        """
        if not select(self.fem_objects, "radiation"):
            raise ValueError("solve_fsi_eigen needs a DtN radiation boundary "
                             "(add InterfaceSpecDtN('dtn_boundary') to the problem).")
        Ks, Ms, Cs, nm, na = self._assemble_dtn_system(dtn_center, f_max, dtn_half, dtn_order)
        if np.abs(Cs).sum() == 0:
            raise ValueError("The coupled DtN system has no damping matrix (C = 0).")

        stop_fn = None
        if cover_goal is not None:
            stop_fn = self._make_goal_check(Ks, Ms, Cs, nm, na, f_min, f_max, cover_goal)
        avoid = []
        if cover_avoid_frac > 0:
            betas = sorted({float(getattr(o, "beta_ray", 0.0) or 0.0)
                            for o in self.fem_objects} - {0.0})
            for b in betas:
                if 1.0 / b <= 1.3 * 2 * np.pi * f_max:          # pile within reach
                    avoid.append((-1.0 / b + 0j, cover_avoid_frac / b))
        ds = self._dtn_system

        def build_result(s_raw, phi_raw, shifts, quiet=False):
            """Filter, normalise (symmetric u–ψ–φ form, p = −sψ) and pack the modes."""
            import contextlib, io
            with warnings.catch_warnings(), \
                    contextlib.redirect_stdout(io.StringIO() if quiet else sys.stdout):
                if quiet:
                    warnings.simplefilter("ignore")
                frequencies, zeta, loss_factors, s_vals, phi = \
                    self._postprocess(s_raw, phi_raw, f_min, f_max, filter_halfplane=True,
                                      keep_real=True)
            phi = self._normalise_sym(phi, s_vals, Ms, Cs)
            u_modes = np.array([self.meca.expand(phi[:nm, k]) for k in range(phi.shape[1])])
            p_modes = np.ascontiguousarray((-s_vals[None, :] * phi[nm:nm + na]).T)
            if not quiet:
                self._print_summary(frequencies, zeta, loss_factors, True, True)
            result = {
                "kind":   "fsi",
                "freqs":  frequencies,
                "zeta":   zeta,
                "fields": {"u_meca": u_modes, "p_acou": p_modes},
                "extras": {"s": s_vals},
                "solver_params": {
                    "f_targets": [float(abs(sg) / (2 * np.pi)) for sg, _ in shifts],   # |σ|/2π [Hz]
                    "n_modes_per_shift": n_modes_per_shift,
                    "f_min": f_min, "f_max": f_max, "cover_zeta_max": cover_zeta_max,
                    "exterior": "dtn",
                    "dtn_arcs": ", ".join(arc["name"] for arc in ds["arcs"]),
                    "dtn_R": [float(arc["R"]) for arc in ds["arcs"]],
                    "dtn_orders": len(ds["arcs"][0]["orders"]),
                    "dtn_half": dtn_half,
                },
                "interfaces": self._interface_meta,
            }
            return result, frequencies, zeta, u_modes, p_modes, s_vals

        def checkpoint(s_list, phi_list, shifts):
            """Save the modes found so far to checkpoint_path (atomic replace)."""
            import contextlib, io, os
            result = build_result(np.array(s_list), np.column_stack(phi_list), shifts, quiet=True)[0]
            previous, self._results = self._results, self._results + [result]
            tmp = f"{checkpoint_path}.partial"
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.save(tmp, case="fsi_eigen_dtn", checkpoint_shifts=len(shifts))
                os.replace(tmp, checkpoint_path)
                print(f"             checkpoint: {len(result['freqs'])} modes after "
                      f"{len(shifts)} shifts → {checkpoint_path}")
            except OSError as e:                     # e.g. iCloud holding the file
                print(f"             checkpoint skipped ({e.__class__.__name__}: {e})")
            finally:
                self._results = previous

        s_vals, phi, shifts = self._covering_core(
            Ks, Ms, Cs, f_min, f_max, n_modes_per_shift,
            zeta_cover=cover_zeta_max, overlap=cover_overlap, max_shifts=max_shifts,
            freq_tol_rad=2.0 * np.pi * dedup_freq_tol_hz, mac_tol=mac_tol,
            rel_tol=dedup_rel_tol, stop_fn=stop_fn, avoid=avoid,
            progress_fn=checkpoint if checkpoint_path else None,
            progress_every=checkpoint_every)

        result, frequencies, zeta, u_modes, p_modes, s_vals = build_result(s_vals, phi, shifts)
        self._results.append(result)
        return frequencies, zeta, u_modes, p_modes, s_vals

    def solve_acoustic_eigen_ARPAC(
        self,
        n_modes: int   = 30,
        f_min:   float = 20.0,
        f_max:   float = 20_000.0,
        f_target: float = 100.0,
    ) -> tuple:
        """
        Solves the acoustic eigenvalue problem: (H - ω²Q) p = 0
        Returns
        -------
        frequencies  : (n,)   natural frequencies in Hz
        mode_shapes  : (n, N) real pressure mode shapes
        """
        # 1. Assemble Acoustic matrices (assuming you have an AcouAssembler)
        from loudpy.physics.acoustic import AcouAssembler
        acou = AcouAssembler(self.fem_objects)
        self.acou = acou
        H, Q, _ = acou.assemble()
 
        self._assemblers["p_acou"] = acou
        # Ensure they are real CSC formats
        H = H.tocsc().astype(float)
        Q = Q.tocsc().astype(float)
        
        n_dof = H.shape[0]
        print(f"\n  Acoustic GEP | Real symmetric H & Q | DOFs: {n_dof}")
 
        # 2. Shift-invert solve using eigsh (Real Symmetric)
        omega_t = 2.0 * np.pi * f_target
        sigma = float(omega_t**2)
        
        k_req = min(n_modes, n_dof - 2)
        
        # eigsh handles the shift-invert natively if sigma is provided
        eigenvalues, phi_reduced = eigsh(
            H, M=Q, k=k_req, sigma=sigma, which='LM', tol=1e-10,
            v0=_eig.start_vector(H.shape[0], complex_=False),
        )
        
        # 3. Post-process (λ = ω²)
        # Clean up tiny negative eigenvalues (numerical noise)
        eigenvalues = np.maximum(eigenvalues, 0.0) 
        frequencies = np.sqrt(eigenvalues) / (2.0 * np.pi)
        
        # 4. Filter by frequency bounds
        mask = (frequencies >= f_min) & (frequencies <= f_max)
        frequencies = frequencies[mask]
        phi_reduced = phi_reduced[:, mask]
        
        # Sort
        idx = np.argsort(frequencies)
        frequencies = frequencies[idx]
        phi_reduced = phi_reduced[:, idx]
        
        # 5. Reconstruct full mode shapes (handle blocked DOFs)
        free_dofs = np.arange(n_dof) # Update this if AcouAssembler drops DOFs
        mode_shapes = np.zeros((len(frequencies), n_dof), dtype=float)
        mode_shapes[:, free_dofs] = phi_reduced.T
        
        # 6. Mass normalize: p^T Q p = 1
        for k in range(len(frequencies)):
            v = mode_shapes[k]
            norm = v @ (Q @ v)
            if norm > 1e-30:
                mode_shapes[k] /= np.sqrt(norm)

        self._print_acou_summary(frequencies)
        self._results.append({
            "kind":    "acou",
            "dof_key": "p_acou",
            "freqs":   frequencies,
            "zeta":    None,
            "modes":   mode_shapes,                # shape (n_modes, n_dof)
            "solver_params": {
                "n_modes":  n_modes,
                "f_min":    f_min,
                "f_max":    f_max,
                "f_target": f_target,
            },
            "interfaces": self._interface_meta,
        })

        return frequencies, mode_shapes
    

        
    # ------------------------------------------------------------------ #
    #  Console summary                                                     #
    # ------------------------------------------------------------------ #


    def save(self, path, **metadata):
        """
        One-shot persistence: mesh + specs + all solved modes + metadata.

        Parameters
        ----------
        path : str | Path
            Output .h5 file.
        **metadata
            Extra free-form metadata (case name, notes, …).
        """
        from loudpy.Files_Saver import ResultStore     # local import avoids cycles

        if not self._results:
            raise RuntimeError("Nothing to save — run a solve_* method first.")

        with ResultStore(path) as store:
            # ── metadata (auto + user) ─────────────────────────────────────
            auto = {
                "geo_path": self.geo_path,
                "msh_path": self.msh_path,
                "mat_path": self.mat_path,
            }
            for r in self._results:
                for k, v in r["solver_params"].items():
                    auto[f"{r['kind']}_{k}"] = v
            store.set_metadata(**auto, **metadata)

            # ── specs ──────────────────────────────────────────────────────
            store.save_specs(self.specs_dict())

            # ── mesh ───────────────────────────────────────────────────────
            mesh_id = store.save_mesh_from_problem(self._problem)

            # ── modes (one entry per solve_* call) ─────────────────────────
            from loudpy.Files_Saver.Study_Saver import StudySaver
            StudySaver(self._problem, store).save_eigen_modes(
                self._results, self.dof_maps(), mesh_id)

        print(f"✓ saved → {path}")




# The algorithms live in loudpy.solvers.eigen; the study keeps its former method names.
EigenStudy._solve_gep = staticmethod(_eig.solve_gep)
EigenStudy._solve_qep = staticmethod(_eig.solve_qep)
EigenStudy._solve_qep_sigma = staticmethod(_eig.solve_qep_at)
EigenStudy._postprocess = staticmethod(_eig.filter_modes)
EigenStudy._default_shifts = staticmethod(_eig.default_shifts)
EigenStudy._multishift_core = staticmethod(_eig.multishift)
EigenStudy._covering_core = staticmethod(_eig.covering)
EigenStudy._normalise_sym = staticmethod(_eig.normalise_symmetric)
EigenStudy._normalise = staticmethod(_eig.normalise_modes)
EigenStudy._print_summary = staticmethod(_eig.print_mode_summary)
EigenStudy._print_acou_summary = staticmethod(_eig.print_acoustic_summary)
