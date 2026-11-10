import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, splu, LinearOperator
from scipy.linalg import eig as dense_eig
import warnings


from loudpy.assembly import MecaAssembler
from scipy.sparse.linalg import eigsh
from loudpy.Studies.Study_class import Problem
from loudpy.Studies.DofMapMixin import DofMapMixin

# ═══════════════════════════════════════════════════════════════════════════ #
#  Helpers                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

def _has_imaginary_part(A: sp.spmatrix) -> bool:
    """True when A carries a non-trivial imaginary part (e.g. hysteretic K)."""
    return (np.issubdtype(A.dtype, np.complexfloating)
            and np.any(np.imag(A.data) != 0))


def _to_csc_complex(*matrices):
    """Cast every matrix to complex CSC in one call."""
    return [A.tocsc().astype(complex) for A in matrices]


def _shift_invert_op(A_shifted: sp.spmatrix, B_matvec, size: int) -> LinearOperator:
    """
    Returns the ARPACK operator  (A − σB)⁻¹ B  as a LinearOperator.
    `B_matvec` is a callable x → B @ x.
    """
    lu = splu(A_shifted.tocsc())
    return LinearOperator(
        (size, size),
        matvec=lambda x: lu.solve(B_matvec(x)),
        dtype=complex,
    )


def _run_arpack(op: LinearOperator, k: int) -> tuple:
    """Run eigs with sensible defaults; returns (mu, vecs)."""
    ncv = min(max(4 * k, 40), op.shape[0] - 1)
    return eigs(op, k=k, which="LM", ncv=ncv, maxiter=30_000, tol=1e-7)


def _mac_deduplicate(s_vals: np.ndarray, phi: np.ndarray,
                     freq_tol_rad: float = 1.0,
                     mac_tol: float = 0.99) -> tuple:
    """
    Remove numerical duplicates (from ARPACK, or modes found by several
    shifts) using the MAC, while preserving genuine degeneracies
    (modes close in frequency but with different shapes → low MAC).
 
    No physical filtering here: the half-plane / damping / frequency-band
    selection is done once, in EigenStudy._postprocess.
 
    Parameters
    ----------
    s_vals : (n,) complex
        Complex eigenvalues s = -ζω ± jω√(1-ζ²).
    phi : (ndof, n) complex
        Associated mode shapes (as columns).
    freq_tol_rad : float
        Tolerance in rad/s for considering two modes "close".
    mac_tol : float
        MAC threshold above which two close modes are considered identical.
    """
    keep = []
    for i, si in enumerate(s_vals):
        is_dup = False
        for j in keep:
            if abs(si - s_vals[j]) < freq_tol_rad:
                ni = np.linalg.norm(phi[:, i])
                nj = np.linalg.norm(phi[:, j])
                if ni > 0 and nj > 0:
                    mac = abs(phi[:, i].conj() @ phi[:, j])**2 / (ni * nj)**2
                    if mac > mac_tol:
                        is_dup = True
                        break
        if not is_dup:
            keep.append(i)
 
    idx = np.array(keep, dtype=int)
    print(f"  Dedup | {len(s_vals)} candidates → {len(idx)} after MAC dedup")
    return s_vals[idx], phi[:, idx]


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
    def _solve_gep(self, K, M, n_modes, f_target, hysteretic):
        omega_t = 2.0 * np.pi * f_target
         # Shift at the target eigenvalue λ = ω_t² on the positive real axis.
        # Real K    → eigenvalues λ_k = ω_k² (real, positive)
        # Complex K → eigenvalues λ_k = ω_k²(1 + jη_k), close to the positive real
        #             axis, so the same shift finds the modes nearest to f_target.
        #             K − σM stays non-singular because the λ_k are off the real axis.
        sigma = complex(omega_t**2) if hysteretic else float(omega_t**2)
        print(f"  GEP | {'complex' if hysteretic else 'real'} K | σ = {sigma:.4e}")

        op    = _shift_invert_op(K - sigma * M, lambda x: M @ x, K.shape[0])
        k_req = min(n_modes, K.shape[0] - 2)
        mu, eigvecs = _run_arpack(op, k_req)

        eigenvalues = sigma + 1.0 / mu          # undo shift: λ = σ + 1/μ
        s_vals      = 1j * np.sqrt(eigenvalues + 0j)   # upper-half-plane root

        return s_vals, eigvecs

    
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
    def _solve_qep(self, K, M, C, n_modes, f_target):
        n     = K.shape[0]
        sigma = 1j * 2.0 * np.pi * f_target
        print(f"  QEP | σ = {sigma:.4e}")
 
        In  = sp.eye(n, format="csc", dtype=complex)
        # (A − σB) assembled as 2×2 block matrix
        AmSB = sp.bmat([
            [-sigma * In,       In            ],
            [-K,          -(C + sigma * M)    ],
        ], format="csc")
 
        def B_matvec(x):
            return np.concatenate([x[:n], M @ x[n:]])
 
        op    = _shift_invert_op(AmSB, B_matvec, 2 * n)
        k_req = min(2 * n_modes, 2 * n - 2)
        mu, eigvecs = _run_arpack(op, k_req)
 
        s_vals = sigma + 1.0 / mu
        phi    = eigvecs[:n, :]                # displacement block only
        return s_vals, phi
 
    # ------------------------------------------------------------------ #
    #  Post-processing  —  the ONLY place where eigenvalues are filtered  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _postprocess(s_vals, phi, f_min, f_max, filter_halfplane,
                     rtol=1e-8, zeta_max=0.999999):
        """
        Compute modal quantities, filter the eigenvalues and sort them by frequency.
 
        filter_halfplane=True keeps only underdamped modes (they oscillate and decay):
          - one eigenvalue per conjugate pair, the one with Im(s) > 0
          - real or near-real eigenvalues (overdamped, rigid-body, or spurious
            ζ = 1 modes) are discarded, with a warning if any lie in [f_min, f_max]
          - growing eigenvalues (Re(s) > 0, unphysical) are discarded with a warning
 
        In modal superposition, every kept mode must then get its conjugate term:
            phi phi^T / (jw - s)  +  conj(phi) conj(phi)^T / (jw - conj(s))
 
        rtol     : relative tolerance deciding that an imaginary or real part is
                   only numerical noise (scaled by the largest |s|).
        zeta_max : eigenvalues with ζ >= zeta_max are treated as real.
        """
        omega_n      = np.abs(s_vals)
        frequencies  = omega_n / (2.0 * np.pi)
        zeta         = -np.real(s_vals) / np.where(omega_n > 0, omega_n, 1.0)
        loss_factors = 2.0 * zeta
 
        mask = np.logical_and(frequencies >= f_min, frequencies <= f_max)
 
        if filter_halfplane and s_vals.size > 0:
            noise = rtol * np.max(np.abs(s_vals))
 
            near_real   = np.logical_or(np.abs(np.imag(s_vals)) <= noise,
                                        zeta >= zeta_max)       # no oscillation
            oscillating = np.logical_and(np.imag(s_vals) > noise,
                                         np.logical_not(near_real))  # upper member of a pair
            growing     = np.real(s_vals) > noise               # unphysical
 
            n_real = int(np.sum(np.logical_and(mask, near_real)))
            if n_real > 0:
                warnings.warn(
                    f"{n_real} real or near-real (overdamped / spurious ζ≈1) "
                    f"eigenvalue(s) between {f_min} and {f_max} Hz ignored "
                    f"in the modal basis.",
                    stacklevel=2,
                )
 
            n_growing = int(np.sum(np.logical_and(mask, growing)))
            if n_growing > 0:
                warnings.warn(
                    f"{n_growing} eigenvalue(s) with Re(s) > 0 (growing in time) "
                    f"between {f_min} and {f_max} Hz ignored. "
                    f"Check the signs of M, C and K.",
                    stacklevel=2,
                )

            lower_half = np.imag(s_vals) < -noise
            print(f"  Filter | {s_vals.size} eigenvalues: "
                  f"{int(np.sum(lower_half))} lower half-plane, "
                  f"{int(np.sum(near_real))} near-real "
                  f"({n_real} in band), "
                  f"{int(np.sum(growing))} growing, "
                  f"{int(np.sum(np.logical_not(mask)))} outside "
                  f"{f_min}–{f_max} Hz")
            keep = np.logical_and(oscillating, np.logical_not(growing))
            mask = np.logical_and(mask, keep)
 
        s_vals, phi  = s_vals[mask], phi[:, mask]
        frequencies  = frequencies[mask]
        zeta         = zeta[mask]
        loss_factors = loss_factors[mask]
 
        idx = np.argsort(frequencies)
        return (frequencies[idx], zeta[idx], loss_factors[idx],
                s_vals[idx], phi[:, idx])
 
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
            # log-spaced shifts; ~1 shift per octave is usually plenty
            n_shifts  = max(2, int(np.ceil(np.log2(f_max / max(f_min, 1.0)))))
            f_targets = np.geomspace(max(f_min, 20.0), f_max, n_shifts).tolist()
 
        print(f"\n  Multi-shift | {len(f_targets)} shifts: "
              f"{[f'{f:.0f}' for f in f_targets]} Hz")
 
        # ── collect raw (s, phi) from every shift ──────────────────────────
        s_all, phi_all = [], []
        for ft in f_targets:
            try:
                if viscous:
                    s_v, phi_v = self._solve_qep(K, M, C, n_modes_per_shift, ft)
                else:
                    s_v, phi_v = self._solve_gep(K, M, n_modes_per_shift, ft, hysteretic)
                s_all.append(s_v)
                phi_all.append(phi_v)
                print(f"    shift {ft:7.1f} Hz → {len(s_v)} raw eigenvalues")
            except Exception as e:
                print(f"    shift {ft:7.1f} Hz → FAILED ({type(e).__name__}: {e})")
 
        if not s_all:
            raise RuntimeError("All shifts failed — no modes collected.")
 
        s_vals = np.concatenate(s_all)
        phi    = np.concatenate(phi_all, axis=1)
 
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
        from loudpy.assembly import AcouAssembler
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
            H, M=Q, k=k_req, sigma=sigma, which='LM', tol=1e-10
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
    
    @staticmethod
    def _print_acou_summary(frequencies):
        print(f"\n  Acoustic modes (Undamped):")
        print(f"  {'#':>4}  {'f [Hz]':>12}")
        print("  " + "─" * 20)
        for i, f in enumerate(frequencies, 1):
            print(f"  {i:4d}  {f:12.4f}")
        print(f"\n  {len(frequencies)} modes found.")

        
    # ------------------------------------------------------------------ #
    #  Console summary                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _print_summary(frequencies, zeta, loss_factors, hysteretic, viscous):
        kind = ("viscous-damped"        if viscous
                else "hysteretic"       if hysteretic
                else "undamped")
        print(f"\n  Structural modes ({kind}):")
        print(f"  {'#':>4}  {'f [Hz]':>12}  {'ζ':>10}  {'η = 2ζ':>10}")
        print("  " + "─" * 44)
        for i, (f, z, eta) in enumerate(zip(frequencies, zeta, loss_factors), 1):
            print(f"  {i:4d}  {f:12.4f}  {z:10.4f}  {eta:10.4f}")
        print(f"\n  {len(frequencies)} modes found.")


    @staticmethod
    def _normalise(phi, s_vals, M, C, hysteretic, viscous):
        """
        Exact normalisation per solver type:
        • viscous (QEP)         : φᵀ (2 s M + C) φ = 1
        • hysteretic (complex K): φᵀ M φ        = 1   (complex, un-conjugated)
        • undamped real         : φᵀ M φ        = 1   (real)
        """
        n_found = phi.shape[1]
        normalized_phi = np.zeros_like(phi)
 
        for k in range(n_found):
            v = phi[:, k]
 
            if viscous:
                # Works for Rayleigh, element-viscous, or any mix — depends only on C_global
                norm = v @ ((2.0 * s_vals[k]) * (M @ v) + C @ v)
            else:
                # GEP case (C = 0): hysteretic complex K or purely real K
                norm = v @ (M @ v)         # un-conjugated bilinear form
                if not hysteretic:
                    norm = np.real(norm)   # guarantee real for undamped
 
            scaling = np.sqrt(norm) if np.abs(norm) > 1e-30 else 1.0
            normalized_phi[:, k] = v / scaling
 
        return normalized_phi

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


