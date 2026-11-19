"""
Eigenvalue solvers on assembled matrices (no physics, no files).

  GEP  (K − λM) φ = 0, λ = ω²            solve_gep         (undamped / hysteretic: complex K)
  QEP  (s²M + sC + K) φ = 0              solve_qep, solve_qep_at (shift-invert on the
                                         linearisation, LU of Q(σ) = K + σC + σ²M)
  several shifts over a band             multishift (fixed shifts), covering (gap-free:
                                         shifts placed until the band is covered)
  merging the shifts                     mac_deduplicate, weighted_mac
  physical selection, the only filter    filter_modes (half-plane, damping, band)
  normalisation                          normalise_modes, normalise_symmetric
  ARPACK                                 arpack_eigs (start vector from a seed: reproducible)
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

from loudpy.solvers.linear import QEPShiftInvert, factorize


def has_imaginary_part(A: sp.spmatrix) -> bool:
    """True when A carries a non-trivial imaginary part (e.g. hysteretic K)."""
    return (np.issubdtype(A.dtype, np.complexfloating)
            and np.any(np.imag(A.data) != 0))

def to_csc_complex(*matrices):
    """Cast every matrix to complex CSC in one call."""
    return [A.tocsc().astype(complex) for A in matrices]

def shift_invert_operator(A_shifted: sp.spmatrix, B_matvec, size: int) -> LinearOperator:
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

def start_vector(n: int, seed: int = 0, complex_: bool = True) -> np.ndarray:
    """Fixed ARPACK start vector (pseudo-random from `seed`): the same run gives the same modes."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    return v + 1j * rng.standard_normal(n) if complex_ else v


def arpack_eigs(op: LinearOperator, k: int, maxiter: int = 30_000,
                seed: int | None = 0) -> tuple:
    """
    Run eigs with sensible defaults; returns (mu, vecs).

    seed: the start vector comes from it, so two identical runs give identical modes
    (default 0). seed=None: ARPACK's own random start (runs differ at solver precision,
    and one can fail to converge where another did not).
    """
    v0 = None if seed is None else start_vector(op.shape[0], seed)
    # Krylov space 2.5k (ARPACK's cost per restart grows as ncv²: 2.4× faster than 4k on the
    # HPNEW multishift, same eigenpairs); the former 4k is the fallback if it does not converge
    from loudpy.solvers.linear import ENABLED
    try:
        ncv = min(max(int((2.5 if ENABLED else 4) * k), 40), op.shape[0] - 1)
        return eigs(op, k=k, which="LM", ncv=ncv, maxiter=maxiter, tol=1e-7, v0=v0)
    except ArpackNoConvergence:
        ncv = min(max(4 * k, 40), op.shape[0] - 1)
        return eigs(op, k=k, which="LM", ncv=ncv, maxiter=maxiter, tol=1e-7, v0=v0)

ZETA_REAL = 0.999999   # ζ at or above this: a real (overdamped, non-oscillating) eigenvalue

def real_modes(s_vals):
    """
    (s, real): the eigenvalues with ζ ≥ ZETA_REAL put exactly on the real axis,
    and the mask of those real ones. A real mode is its own conjugate partner:
    it enters the modal sum once, (jω/s) p a / (jω − s), with no partner term.
    """
    s = np.asarray(s_vals, dtype=complex)
    zeta = -s.real / np.where(np.abs(s) > 0, np.abs(s), 1.0)
    real = zeta >= ZETA_REAL
    return np.where(real, s.real + 0j, s), real

def weighted_mac(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
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

def mac_deduplicate(s_vals: np.ndarray, phi: np.ndarray,
                    freq_tol_rad: float = 1.0,
                    mac_tol: float = 0.99) -> tuple:
    """
    Remove numerical duplicates (from ARPACK, or modes found by several
    shifts) using the MAC, while preserving genuine degeneracies
    (modes close in frequency but with different shapes → low MAC).
 
    No physical filtering here: the half-plane / damping / frequency-band
    selection is done once, in filter_modes.
 
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

def solve_gep(K, M, n_modes, f_target, hysteretic):
    omega_t = 2.0 * np.pi * f_target
     # Shift at the target eigenvalue λ = ω_t² on the positive real axis.
    # Real K    → eigenvalues λ_k = ω_k² (real, positive)
    # Complex K → eigenvalues λ_k = ω_k²(1 + jη_k), close to the positive real
    #             axis, so the same shift finds the modes nearest to f_target.
    #             K − σM stays non-singular because the λ_k are off the real axis.
    sigma = complex(omega_t**2) if hysteretic else float(omega_t**2)
    print(f"  GEP | {'complex' if hysteretic else 'real'} K | σ = {sigma:.4e}")

    op    = shift_invert_operator(K - sigma * M, lambda x: M @ x, K.shape[0])
    k_req = min(n_modes, K.shape[0] - 2)
    mu, eigvecs = arpack_eigs(op, k_req)

    eigenvalues = sigma + 1.0 / mu          # undo shift: λ = σ + 1/μ
    s_vals      = 1j * np.sqrt(eigenvalues + 0j)   # upper-half-plane root

    return s_vals, eigvecs

def solve_qep(K, M, C, n_modes, f_target):
    n     = K.shape[0]
    sigma = 1j * 2.0 * np.pi * f_target
    print(f"  QEP | σ = {sigma:.4e}")

    # (A − σB)⁻¹B from an LU of Q(σ) = K + σC + σ²M (n×n) — see fast_linalg.QEPShiftInvert
    S     = QEPShiftInvert(K, M, C, sigma)
    op    = LinearOperator((2 * n, 2 * n), matvec=S, dtype=complex)
    k_req = min(2 * n_modes, 2 * n - 2)
    mu, eigvecs = arpack_eigs(op, k_req)

    s_vals = sigma + 1.0 / mu
    phi    = eigvecs[:n, :]                # displacement block only
    return s_vals, phi

def filter_modes(s_vals, phi, f_min, f_max, filter_halfplane,
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

def default_shifts(f_min: float, f_max: float) -> list[float]:
    """Log-spaced shifts, ~1 per octave, across [f_min, f_max]."""
    n_shifts = max(2, int(np.ceil(np.log2(f_max / max(f_min, 1.0)))))
    return np.geomspace(max(f_min, 20.0), f_max, n_shifts).tolist()

def multishift(K, M, C, f_targets, n_modes_per_shift,
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
                s_v, phi_v = solve_qep(K, M, C, n_modes_per_shift, ft)
            else:
                s_v, phi_v = solve_gep(K, M, n_modes_per_shift, ft, hysteretic)
            s_all.append(s_v)
            phi_all.append(phi_v)
            print(f"    shift {ft:7.1f} Hz → {len(s_v)} raw eigenvalues")
        except Exception as e:
            print(f"    shift {ft:7.1f} Hz → FAILED ({type(e).__name__}: {e})")

    if not s_all:
        raise RuntimeError("All shifts failed — no modes collected.")

    return np.concatenate(s_all), np.concatenate(phi_all, axis=1)

def covering(K, M, C, f_min, f_max, n_modes_per_shift,
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
    weighted_mac). The relative window matters for the radiation modes
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
                s_v, phi_v, k_used, tau = solve_qep_at(
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
            dup = any(weighted_mac(phi_v[:, i], phi_keep[j], w_mac) > mac_tol
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

def solve_qep_at(K, M, C, n_modes, sigma, maxiter=60, seed=0, piles=()):
    """
    solve_qep at an arbitrary complex shift σ (rad/s), quiet.

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
            g, vecs = arpack_eigs(op, k, maxiter=maxiter, seed=seed)
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

def normalise_symmetric(phi, s_vals, Ms, Cs):
    """Generic modal normalisation of a symmetric QEP: xᵀ(2sMs + Cs)x = 1."""
    out = np.empty_like(phi)
    MX, CX = Ms @ phi, Cs @ phi
    for k in range(phi.shape[1]):
        norm = phi[:, k] @ (2.0 * s_vals[k] * MX[:, k] + CX[:, k])
        out[:, k] = phi[:, k] / (np.sqrt(norm) if abs(norm) > 1e-300 else 1.0)
    return out

def print_acoustic_summary(frequencies):
    print(f"\n  Acoustic modes (Undamped):")
    print(f"  {'#':>4}  {'f [Hz]':>12}")
    print("  " + "─" * 20)
    for i, f in enumerate(frequencies, 1):
        print(f"  {i:4d}  {f:12.4f}")
    print(f"\n  {len(frequencies)} modes found.")

def print_mode_summary(frequencies, zeta, loss_factors, hysteretic, viscous):
    kind = ("viscous-damped"        if viscous
            else "hysteretic"       if hysteretic
            else "undamped")
    print(f"\n  Structural modes ({kind}):")
    print(f"  {'#':>4}  {'f [Hz]':>12}  {'ζ':>10}  {'η = 2ζ':>10}")
    print("  " + "─" * 44)
    for i, (f, z, eta) in enumerate(zip(frequencies, zeta, loss_factors), 1):
        print(f"  {i:4d}  {f:12.4f}  {z:10.4f}  {eta:10.4f}")
    print(f"\n  {len(frequencies)} modes found.")

def normalise_modes(phi, s_vals, M, C, hysteretic, viscous):
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
