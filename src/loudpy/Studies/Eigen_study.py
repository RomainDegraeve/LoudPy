import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, splu, LinearOperator, ArpackNoConvergence
from loudpy.Studies.fast_linalg import factorize, QEPShiftInvert
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
    lu = factorize(A_shifted)
    return LinearOperator(
        (size, size),
        matvec=lambda x: lu.solve(B_matvec(x)),
        dtype=complex,
    )


def _run_arpack(op: LinearOperator, k: int, maxiter: int = 30_000,
                seed: int | None = None) -> tuple:
    """
    Run eigs with sensible defaults; returns (mu, vecs).

    seed: fixed start vector (reproducible runs). Without it ARPACK starts
    from a random vector, so two identical runs can differ (and one can fail
    to converge where the other did not).
    """
    v0 = None
    if seed is not None:
        rng = np.random.default_rng(seed)
        n = op.shape[0]
        v0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    # Krylov space 2.5k (ARPACK's cost per restart grows as ncv²: 2.4× faster than 4k on the
    # HPNEW multishift, same eigenpairs); the former 4k is the fallback if it does not converge
    from loudpy.Studies.fast_linalg import ENABLED
    try:
        ncv = min(max(int((2.5 if ENABLED else 4) * k), 40), op.shape[0] - 1)
        return eigs(op, k=k, which="LM", ncv=ncv, maxiter=maxiter, tol=1e-7, v0=v0)
    except ArpackNoConvergence:
        ncv = min(max(4 * k, 40), op.shape[0] - 1)
        return eigs(op, k=k, which="LM", ncv=ncv, maxiter=maxiter, tol=1e-7, v0=v0)


ZETA_REAL = 0.999999   # ζ at or above this: a real (overdamped, non-oscillating) eigenvalue


def _real_modes(s_vals):
    """
    (s, real): the eigenvalues with ζ ≥ ZETA_REAL put exactly on the real axis,
    and the mask of those real ones. A real mode is its own conjugate partner:
    it enters the modal sum once, (jω/s) p a / (jω − s), with no partner term.
    """
    s = np.asarray(s_vals, dtype=complex)
    zeta = -s.real / np.where(np.abs(s) > 0, np.abs(s), 1.0)
    real = zeta >= ZETA_REAL
    return np.where(real, s.real + 0j, s), real


def _weighted_mac(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """
    MAC of two mode shapes with fixed weights w ≥ 0 (the same for every mode):
    |aᴴWb|² / (aᴴWa · bᴴWb), W = diag(w).

    With w = |diag(M)| each part of a coupled vector counts by its share of
    the mass-weighted norm: mixed units (m for u, Pa·s for ψ) need no
    rescaling, and a part that is only round-off for a mode (the displacement
    of a pure air mode) does not count. Scaling each part to unit norm per
    mode instead would give that round-off half of the MAC, and two copies of
    the same mode found by two shifts would not be recognised.
    """
    wb = w * b
    den = (a.conj() @ (w * a)).real * (b.conj() @ wb).real
    return abs(a.conj() @ wb) ** 2 / max(den, 1e-300)


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
    cmp = phi
    keep = []
    for i, si in enumerate(s_vals):
        is_dup = False
        for j in keep:
            if abs(si - s_vals[j]) < freq_tol_rad:
                ni = np.linalg.norm(cmp[:, i])
                nj = np.linalg.norm(cmp[:, j])
                if ni > 0 and nj > 0:
                    mac = abs(cmp[:, i].conj() @ cmp[:, j])**2 / (ni * nj)**2
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
 
        # (A − σB)⁻¹B from an LU of Q(σ) = K + σC + σ²M (n×n) — see fast_linalg.QEPShiftInvert
        S     = QEPShiftInvert(K, M, C, sigma)
        op    = LinearOperator((2 * n, 2 * n), matvec=S, dtype=complex)
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
                     rtol=1e-8, zeta_max=ZETA_REAL, keep_real=False):
        """
        Compute modal quantities, filter the eigenvalues and sort them by frequency.

        filter_halfplane=True keeps only underdamped modes (they oscillate and decay):
          - one eigenvalue per conjugate pair, the one with Im(s) > 0
          - real or near-real eigenvalues (overdamped, rigid-body, or spurious
            ζ = 1 modes) are discarded, with a warning if any lie in [f_min, f_max]
          - growing eigenvalues (Re(s) > 0, unphysical) are discarded with a warning

        In modal superposition, every kept mode must then get its conjugate term:
            phi phi^T / (jw - s)  +  conj(phi) conj(phi)^T / (jw - conj(s))

        keep_real=True (coupled DtN modes): the real eigenvalues with Re(s) < 0
        are kept too, put exactly on the real axis (Im(s) = 0). They are
        overdamped modes, not spurious ones, and the modal sum of the pressure
        needs them; each one enters the sum ONCE (it is its own partner).

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
            real_kept   = np.zeros(s_vals.size, bool)
            if keep_real:
                real_kept = near_real & np.logical_not(growing) & (np.real(s_vals) < 0)
                s_vals = np.where(real_kept, np.real(s_vals) + 0j, s_vals)
                zeta = np.where(real_kept, 1.0, zeta)
                loss_factors = 2.0 * zeta
                n_kept = int(np.sum(np.logical_and(mask, real_kept)))
                if n_kept:
                    print(f"  Filter | {n_kept} real (overdamped) mode(s) kept, "
                          f"counted once in the modal sum")

            n_real = int(np.sum(mask & near_real & np.logical_not(real_kept)))
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
                  f"({n_real} dropped in band), "
                  f"{int(np.sum(growing))} growing, "
                  f"{int(np.sum(np.logical_not(mask)))} outside "
                  f"{f_min}–{f_max} Hz")
            keep = np.logical_and(oscillating | real_kept, np.logical_not(growing))
            mask = np.logical_and(mask, keep)
 
        s_vals, phi  = s_vals[mask], phi[:, mask]
        frequencies  = frequencies[mask]
        zeta         = zeta[mask]
        loss_factors = loss_factors[mask]
 
        idx = np.argsort(frequencies)
        return (frequencies[idx], zeta[idx], loss_factors[idx],
                s_vals[idx], phi[:, idx])
 
    # ------------------------------------------------------------------ #
    #  Multi-shift core  (in-vacuo solve)                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_shifts(f_min: float, f_max: float) -> list[float]:
        """Log-spaced shifts, ~1 per octave, across [f_min, f_max]."""
        n_shifts = max(2, int(np.ceil(np.log2(f_max / max(f_min, 1.0)))))
        return np.geomspace(max(f_min, 20.0), f_max, n_shifts).tolist()

    def _multishift_core(self, K, M, C, f_targets, n_modes_per_shift,
                         hysteretic) -> tuple:
        """
        Run the QEP (C given) or GEP (C None) solver at every shift and
        return the concatenated RAW eigenpairs (s_vals, phi) — no dedup,
        no filtering.
        """
        print(f"\n  Multi-shift | {len(f_targets)} shifts: "
              f"{[f'{f:.0f}' for f in f_targets]} Hz")

        s_all, phi_all = [], []
        for ft in f_targets:
            try:
                if C is not None:
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

        return np.concatenate(s_all), np.concatenate(phi_all, axis=1)

    def _covering_core(self, K, M, C, f_min, f_max, n_modes_per_shift,
                       zeta_cover=0.95, overlap=0.85, max_shifts=60,
                       freq_tol_rad=2.0 * np.pi * 0.5, mac_tol=0.99,
                       rel_tol=0.002, stop_fn=None, avoid=None,
                       progress_fn=None, progress_every=3) -> tuple:
        """
        Gap-free multi-shift QEP solve: every eigenvalue s (upper half-plane)
        with f = |s|/2π in [f_min, f_max] and ζ ≤ zeta_cover is found, and
        also the real (overdamped, ζ = 1) eigenvalues on the negative real axis.

        Shift-invert ARPACK returns the k eigenvalues CLOSEST to the shift σ,
        so one solve finds every eigenvalue inside the disc |s − σ| < r, with
        r = distance to the farthest one returned. The target region of the
        s-plane is sampled on a polar grid; shifts are placed greedily at the
        uncovered grid point of smallest |s| (complex σ, off the imaginary
        axis when needed to reach the heavily damped radiation modes), and a grid
        point counts as covered once it lies within overlap·r of a shift.
        The margin (1 − overlap)·r absorbs the grid spacing.

        Duplicates between overlapping discs are removed on the fly, so memory
        stays at the size of the final mode set: a new eigenvalue is compared
        with the kept ones closer than max(freq_tol_rad, rel_tol·|s|), and is a
        duplicate if the MAC weighted by |diag(M)| exceeds mac_tol (see
        _weighted_mac). The relative window matters for the radiation modes
        near the DtN poles: their eigenvalues are so ill-conditioned that two
        shifts can return the same mode 1–2 Hz apart near 2 kHz.

        stop_fn(s_list, phi_list) -> (stop, message), called after every
        shift, ends the covering early (goal-oriented stop, see
        solve_fsi_eigen(cover_goal=...)).

        progress_fn(s_list, phi_list, shifts), called every progress_every
        shifts, lets the caller save what has been found so far (checkpoint).

        avoid: list of (centre, radius) discs (rad/s) excluded from the
        region — the points where overdamped eigenvalues pile up on the real
        axis (s = −1/β with Rayleigh damping). Thousands of nearly equal real
        eigenvalues sit there: a shift placed nearby stalls ARPACK and covers
        almost nothing, and those eigenvalues are not used anyway (ζ > 1).

        Returns the raw, deduplicated (s_vals, phi) — no band/half-plane
        filtering — plus the shift list [(σ, r)] (σ, r in rad/s).
        """
        w_lo, w_hi = 2 * np.pi * f_min, 2 * np.pi * f_max
        theta_max = np.arcsin(min(zeta_cover, 0.999))
        n_w, n_th = 300, 80
        # polar grid rows: ζ ≤ zeta_cover, plus the thin wedge up to the real
        # axis (ζ = 1), where the overdamped real modes are (the modal sum needs them)
        thetas = np.r_[np.linspace(0.0, theta_max, n_th),
                       np.linspace(theta_max, np.pi / 2, 4)[1:]]
        W, TH = np.meshgrid(np.linspace(w_lo, w_hi, n_w), thetas, indexing="ij")
        grid = (W * (-np.sin(TH) + 1j * np.cos(TH))).ravel()     # s = ω(−ζ + j√(1−ζ²))
        order = np.lexsort((TH.ravel(), np.abs(grid)))           # small |s| first
        covered = np.zeros(grid.size, bool)
        # fixed MAC weights: |M| (u), |Q|/ρ (ψ), 0 (DtN auxiliary unknowns, no mass)
        w_mac = np.abs(M.diagonal())
        for c_av, r_av in (avoid or []):
            hit = np.abs(grid - c_av) <= r_av
            covered |= hit
            print(f"  Covering | skipping the overdamped pile at s/2π = {c_av.real/(2*np.pi):.0f} Hz "
                  f"(disc of {r_av/(2*np.pi):.0f} Hz, {hit.mean():.1%} of the region)")

        print(f"\n  Covering | f = {f_min:g}–{f_max:g} Hz, ζ ≤ {zeta_cover:g}, "
              f"{n_modes_per_shift} modes/shift, overlap {overlap:g}, "
              f"max {max_shifts} shifts")

        s_keep, phi_keep, shifts, skipped = [], [], [], []
        while not covered.all() and len(shifts) + len(skipped) < max_shifts:
            sigma0 = sigma = grid[order[~covered[order]][0]]
            s_v = None
            k_full = 2 * n_modes_per_shift
            # piles close to σ: filter them out of ARPACK's wanted set
            near_piles = [c_av for c_av, _ in (avoid or [])
                          if abs(sigma - c_av) < 0.6 * abs(c_av)]
            for attempt in range(3):                     # σ on an eigenvalue, or slow ARPACK
                try:
                    # each attempt already retries the same σ with k, k/2, k/4
                    s_v, phi_v, k_used, tau = self._solve_qep_sigma(
                        K, M, C, n_modes_per_shift, sigma, seed=attempt, piles=near_piles)
                    if k_used < k_full:
                        print(f"    σ/2π = {sigma/(2*np.pi):.1f} Hz: slow convergence, "
                              f"solved with {k_used // 2} modes instead of "
                              f"{n_modes_per_shift} (smaller disc, still complete)")
                    break
                except Exception as e:
                    print(f"    σ/2π = {sigma/(2*np.pi):.1f} Hz failed "
                          f"({type(e).__name__}), moving the shift")
                    sigma = sigma * (1.03 * np.exp(-0.03j))  # slightly higher |σ|, less damped
            if s_v is None:
                # give up on this spot: mark a small neighbourhood as done so the
                # covering moves on; the modes there may be missing (reported)
                r_skip = 0.5 * (min(r for _, r in shifts) if shifts else 0.05 * abs(sigma0))
                covered |= np.abs(grid - sigma0) <= r_skip
                skipped.append(sigma0)
                warnings.warn(f"ARPACK failed around σ/2π = {sigma0/(2*np.pi):.1f} Hz "
                              f"(ζ = {-sigma0.real/abs(sigma0):.2f}); skipped, modes within "
                              f"{r_skip/(2*np.pi):.0f} Hz of it may be missing.", stacklevel=2)
                continue
            r = np.max(np.abs(s_v - sigma))
            shifts.append((sigma, r))
            if near_piles:
                # covered: g(z) = Π|z − p_i| / |z − σ|^m above tau (margin: overlap)
                m = len(near_piles)
                gz = np.prod([np.abs(grid - p) for p in near_piles], axis=0) \
                    / np.maximum(np.abs(grid - sigma), 1e-30) ** m
                covered |= gz >= tau / overlap ** m
            else:
                covered |= np.abs(grid - sigma) <= overlap * r

            n_new = 0
            for i, si in enumerate(s_v):
                window = max(freq_tol_rad, rel_tol * abs(si))
                near = (np.flatnonzero(np.abs(np.asarray(s_keep) - si) < window)
                        if s_keep else [])
                dup = any(_weighted_mac(phi_v[:, i], phi_keep[j], w_mac) > mac_tol
                          for j in near)
                if not dup:
                    s_keep.append(si); phi_keep.append(phi_v[:, i])
                    n_new += 1
            print(f"    shift {len(shifts):3d}: σ/2π = {sigma.imag/(2*np.pi):7.1f} Hz, "
                  f"ζ(σ) = {-sigma.real/abs(sigma):.2f}, radius {r/(2*np.pi):6.1f} Hz → "
                  f"{n_new:3d} new ({len(s_keep)} total), "
                  f"{covered.mean():6.1%} of the region covered"
                  + (f" [pile{'s' if len(near_piles) > 1 else ''} filtered]" if near_piles else ""))
            if progress_fn is not None and len(shifts) % progress_every == 0:
                progress_fn(s_keep, phi_keep, shifts)
            if stop_fn is not None:
                stop, msg = stop_fn(s_keep, phi_keep)
                print(f"             goal check: {msg}")
                if stop:
                    print(f"  Covering | goal reached after {len(shifts)} shifts, "
                          f"{len(s_keep)} distinct eigenvalues "
                          f"({covered.mean():.1%} of the region covered)")
                    return np.array(s_keep), np.column_stack(phi_keep), shifts

        if not covered.all():
            miss = grid[~covered]
            warnings.warn(
                f"Covering stopped at max_shifts = {max_shifts} with "
                f"{(~covered).mean():.1%} of the region uncovered "
                f"(f from {np.abs(miss).min()/(2*np.pi):.0f} Hz, "
                f"ζ up to {(-miss.real/np.abs(miss)).max():.2f}). "
                f"Raise max_shifts or n_modes_per_shift.", stacklevel=2)
        else:
            print(f"  Covering | complete with {len(shifts)} shifts, "
                  f"{len(s_keep)} distinct eigenvalues")
        return np.array(s_keep), np.column_stack(phi_keep), shifts

    def _solve_qep_sigma(self, K, M, C, n_modes, sigma, maxiter=60, seed=0, piles=()):
        """
        _solve_qep at an arbitrary complex shift σ (rad/s), quiet.

        Robust to slow ARPACK convergence: the factorisation is done once and,
        if the k wanted eigenvalues do not converge within `maxiter`
        restarts, the SAME σ is retried asking for fewer (k/2, then k/4).
        The start vector is fixed, so a run is reproducible.

        piles: points p_i (rad/s) where thousands of eigenvalues pile up
        (s = −1/β, overdamped structural modes of Rayleigh-damped parts).
        With S = (A − σB)⁻¹B, ARPACK then works on the filtered operator
            Π_i (I + (σ − p_i) S),   eigenvalue  g(λ) = Π_i (λ − p_i) / (λ − σ)^m
        so the wanted eigenvalues are the k largest |g| — close to σ AND far
        from the piles. The pile eigenvalues get |g| ≈ 0: they never enter the
        wanted set and cannot stall the convergence. λ is recovered from the
        Ritz vector (λ = σ + 1/μ, μ the Rayleigh quotient of S).

        Returns (s, phi, k_used, tau): tau = smallest |g| among the returned
        eigenvalues — every eigenvalue with |g| > tau has been found (for
        piles = (), |g| = 1/|λ − σ| and this is the usual disc). Raises
        ArpackNoConvergence if even the smallest k fails.
        """
        n = K.shape[0]
        S = QEPShiftInvert(K, M, C, sigma)       # (A − σB)⁻¹B from an n×n LU of Q(σ)
        ds = [sigma - p for p in piles]
        if ds:
            def mv(x):
                y = x
                for d in ds:
                    y = y + d * S(y)
                return y
        else:
            mv = S
        op = LinearOperator((2 * n, 2 * n), matvec=mv, dtype=complex)
        k0 = min(2 * n_modes, 2 * n - 2)
        ks = [k0] + [k for k in (k0 // 2, k0 // 4) if k >= 8]
        err = None
        for k in ks:
            try:
                g, vecs = _run_arpack(op, k, maxiter=maxiter, seed=seed)
            except ArpackNoConvergence as e:
                err = e
                continue
            if ds:
                mu = np.array([(v.conj() @ S(v)) / (v.conj() @ v) for v in vecs.T])
                lam = sigma + 1.0 / mu
                g_abs = (np.prod([np.abs(lam - p) for p in piles], axis=0)
                         / np.abs(lam - sigma) ** len(ds))
            else:
                lam = sigma + 1.0 / g
                g_abs = np.abs(g)
            return lam, vecs[:n, :], k, float(np.min(g_abs))
        raise err

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

    @staticmethod
    def _normalise_sym(phi, s_vals, Ms, Cs):
        """Generic modal normalisation of a symmetric QEP: xᵀ(2sMs + Cs)x = 1."""
        out = np.empty_like(phi)
        MX, CX = Ms @ phi, Cs @ phi
        for k in range(phi.shape[1]):
            norm = phi[:, k] @ (2.0 * s_vals[k] * MX[:, k] + CX[:, k])
            out[:, k] = phi[:, k] / (np.sqrt(norm) if abs(norm) > 1e-300 else 1.0)
        return out

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
                if not hysteretic:
                    # undamped: the mode is real up to an arbitrary complex phase from the
                    # (complex) eigensolver — remove it, otherwise Re(vᵀMv) can be ≤ 0
                    v = (v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))).real.astype(v.dtype)
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


