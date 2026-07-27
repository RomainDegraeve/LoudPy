"""
1-D signal plotting — all functions take plain numpy arrays.

No Snapshot, Mesh, Reader or study objects are imported here.
"""
from __future__ import annotations

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

# ── global style (applied at import) ──────────────────────────────────────────
plt.rc("lines",  linewidth=2)
plt.rc("font",   size=14)
plt.rc("axes",   linewidth=1.5, labelsize=14)
plt.rc("legend", fontsize=14)
plt.rcParams["font.family"]                  = "serif"
plt.rcParams["axes.formatter.use_mathtext"]  = True
plt.rcParams["mathtext.fontset"]             = "cm"


# ── SPL / phase frequency response ────────────────────────────────────────────

def plot_spl_sweep(freqs: np.ndarray, p_complex: np.ndarray, *,
                   p_ref: float = 20e-6, title: str = "") -> plt.Figure:
    """
    Parameters
    ----------
    freqs     : (n_freq,)  frequency axis [Hz]
    p_complex : (n_freq,)  complex pressure at probe
    """
    spl   = 20 * np.log10(np.maximum(np.abs(p_complex), 1e-30) / p_ref)
    phase = np.degrees(np.unwrap(np.angle(p_complex)))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    a1.semilogx(freqs, spl,'o',markersize = 0.5, color="C0")
    a1.set_ylabel(r"SPL  [dB re 20\,$\mu$Pa]")
    a1.set_title(title or "SPL and Phase Frequency Response")
    a1.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)

    a2.semilogx(freqs, phase, 'o',markersize = 0.5, color="C1")
    a2.set_ylabel(r"Phase  [$^\circ$]")
    a2.set_xlabel("Frequency  [Hz]")
    a2.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)

    fig.tight_layout()
    return fig



# ── SPL / phase frequency response, several probes on one figure ─────────────

def plot_spl_sweep_multi(freqs: np.ndarray, curves: dict, *,
                         p_ref: float = 20e-6, title: str = "",
                         wrap_phase: bool = True) -> plt.Figure:
    """
    SPL and phase vs frequency for several probes.

    Parameters
    ----------
    freqs      : (n_freq,)  frequency axis [Hz]
    curves     : {label: p_complex (n_freq,)}  one entry per probe
    wrap_phase : True  -> phase wrapped to [-180, 180] deg (np.angle, no unwrap)
                 False -> continuous unwrapped phase
    """
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for i, (label, p) in enumerate(curves.items()):
        p     = np.asarray(p)
        spl   = 20 * np.log10(np.maximum(np.abs(p), 1e-30) / p_ref)
        # np.angle already returns (-pi, pi] -> degrees gives (-180, 180]
        phase = np.angle(p) if wrap_phase else np.unwrap(np.angle(p))
        phase = np.degrees(phase)

        color = f"C{i % 10}"
        a1.semilogx(freqs, spl,   lw=1.2, color=color, label=label)
        a2.semilogx(freqs, phase, lw=1.2, color=color, label=label)

    a1.set_ylabel(r"SPL  [dB re 20 $\mu$Pa]")
    a1.set_title(title or "SPL and Phase Frequency Response")
    a1.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
    a1.legend(fontsize=10, ncol=2)

    a2.set_ylabel(r"Phase  [$^\circ$]")
    a2.set_xlabel("Frequency  [Hz]")
    a2.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
    if wrap_phase:
        a2.set_ylim(-180, 180)
        a2.set_yticks([-180, -90, 0, 90, 180])

    fig.tight_layout()
    return fig


# ── directivity map: frequency x angle -> SPL ────────────────────────────────

def plot_spl_angle_map(freqs: np.ndarray, angles_deg: np.ndarray,
                       p_complex: np.ndarray, *,
                       p_ref: float = 20e-6, title: str = "",
                       cmap: str = "viridis",
                       normalize: bool = False,
                       vmin=None, vmax=None,
                       dynamic_range: float | None = None,
                       levels=None,
                       color_levels=None,
                       contour_color: str = "k",
                       contour_labels: bool = True) -> plt.Figure:
    """
    2-D directivity map: frequency (x, log) vs angle (y) coloured by SPL.

    Parameters
    ----------
    freqs         : (n_freq,)          frequency axis [Hz]
    angles_deg    : (n_angle,)         angle from the z axis [deg]
    p_complex     : (n_angle, n_freq)  complex pressure at each angle/frequency
    normalize     : if True, subtract the on-axis (smallest |angle|) SPL at each
                    frequency -> relative directivity in dB (0 dB on axis)
    vmin, vmax    : colour limits [dB]. By default vmin is the smallest value in
                    the data and vmax the largest — or exactly 0 dB when
                    `normalize` is True.
    dynamic_range : optional override; if given (and vmin is None), vmin is
                    clamped to `vmax - dynamic_range` instead of the data floor.
    levels        : contour LINES. int -> that many lines evenly spaced over
                    [vmin, vmax]; sequence -> explicit dB values
                    (e.g. [-18, -12, -6, -3]). None -> no lines.
    color_levels  : COLOUR steps, independent of `levels`. int -> that many
                    discrete colour bands over [vmin, vmax]; sequence -> explicit
                    band boundaries. None -> smooth continuous colour map.
    contour_color : colour of the contour lines.
    contour_labels: annotate each contour line with its dB value.
    """
    freqs      = np.asarray(freqs)
    angles_deg = np.asarray(angles_deg)
    p_complex  = np.asarray(p_complex)

    if p_complex.shape != (angles_deg.size, freqs.size):
        raise ValueError(
            f"p_complex must have shape (n_angle, n_freq) = "
            f"({angles_deg.size}, {freqs.size}), got {p_complex.shape}."
        )

    spl = 20 * np.log10(np.maximum(np.abs(p_complex), 1e-30) / p_ref)

    if normalize:
        on_axis = spl[int(np.argmin(np.abs(angles_deg))), :]   # per-frequency ref
        spl     = spl - on_axis[None, :]
        cbar_lbl = r"Relative SPL  [dB re on-axis]"
    else:
        cbar_lbl = r"SPL  [dB re 20 $\mu$Pa]"

    # Full data range by default; 0 dB is the natural ceiling once normalised.
    if vmax is None:
        vmax = 0.0 if normalize else float(np.nanmax(spl))
    if vmin is None:
        vmin = (vmax - dynamic_range) if dynamic_range is not None \
               else float(np.nanmin(spl))

    fig, ax = plt.subplots(figsize=(9, 5))

    if levels is None:
        # Smooth, continuous colour map.
        mesh = ax.pcolormesh(freqs, angles_deg, spl, cmap=cmap,
                             shading="gouraud", vmin=vmin, vmax=vmax,
                             rasterized=True)
    else:
        # Discrete colour map: one colour band per contour step.
        # An int N -> N bands (N+1 boundaries); a sequence is used as boundaries.
        lv = (np.linspace(vmin, vmax, int(levels) + 1)
              if np.isscalar(levels) else np.asarray(levels, dtype=float))
        n_bands  = len(lv) - 1
        cmap_obj = plt.get_cmap(cmap, n_bands)
        norm     = BoundaryNorm(lv, n_bands)
        # 'gouraud' would interpolate across the bands and wash them out.
        # rasterized + edgecolors='face' + no AA: otherwise every quad is a
        # separate vector polygon in the PDF and the seams look like a grid.
        mesh = ax.pcolormesh(freqs, angles_deg, spl, cmap=cmap_obj,
                             norm=norm, shading="gouraud",
                             edgecolors="face", linewidth=0,
                             antialiased=False, rasterized=True)
        cs = ax.contour(freqs, angles_deg, spl, levels=lv,
                        colors=contour_color, linewidths=0.8, alpha=0.8)
        if contour_labels:
            ax.clabel(cs, inline=True, fontsize=8, fmt="%g")

    ax.set_xlabel("Frequency  [Hz]")
    ax.set_ylabel(r"Angle from $z$ axis  [$^\circ$]")
    ax.set_title(title or "Directivity map")
    cbar = fig.colorbar(mesh, ax=ax, label=cbar_lbl)
    if cbar.solids is not None:          # kill the same seam artifact in the bar
        cbar.solids.set_edgecolor("face")
        cbar.solids.set_rasterized(True)

    fig.tight_layout()
    return fig


# ── impulse response (IR from a complex frequency response) ──────────────────

def plot_impulse_response(freqs: np.ndarray, H: np.ndarray, *,
                          fs: float | None = None,
                          n_fft: int | None = None,
                          df: float = 1.0,
                          taper: bool = True,
                          t_max: float | None = None,
                          db: bool = False,
                          title: str = "") -> tuple[plt.Figure, np.ndarray, np.ndarray]:
    """
    Impulse response obtained by inverse-FFT of a one-sided frequency response.

    The measured/simulated response is defined on an arbitrary (possibly
    log-spaced) `freqs` grid, so it is first interpolated — magnitude and
    unwrapped phase separately — onto a *uniform* grid 0 … fs/2 before the
    irfft.  Bins outside the measured band are zeroed.

    The uniform-grid resolution must be fine enough that the synthesised IR
    fully decays within the reconstruction window ``T = 1/df``; otherwise the
    still-ringing resonances fold back onto themselves (circular / time-domain
    aliasing) and the IR looks like a waveform that never decays.  The grid is
    therefore sized from a *target frequency resolution* ``df`` (default 1 Hz →
    ~1 s window), not from the number of input points.

    Parameters
    ----------
    freqs : (n_freq,)  frequency axis [Hz], increasing
    H     : (n_freq,)  complex frequency response (e.g. p, u, v or a)
    fs    : sampling rate of the synthesised IR [Hz]; default 2*freqs[-1]
    df    : target frequency resolution of the uniform grid [Hz]; the window
            length is T = 1/df.  Make it a few times smaller than the narrowest
            resonance bandwidth so the IR rings down inside the window.
    n_fft : explicit FFT length (overrides ``df``).  The uniform one-sided grid
            has n_fft//2+1 bins; default: next power of two >= fs/df.
    taper : raised-cosine roll-off at the band edges to suppress the Gibbs
            ringing caused by the hard 0 / band-edge truncation.
    t_max : if given, truncate the displayed IR to [0, t_max] s
    db    : plot 20*log10|h| (normalised) instead of the linear waveform

    Returns
    -------
    (fig, t, h) : figure, time axis [s], real impulse response
    """
    freqs = np.asarray(freqs, dtype=float)
    H     = np.asarray(H,     dtype=complex)

    if fs is None:
        fs = 2.0 * freqs[-1]
    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(max(fs / df, 256))))

    # Interpolate magnitude and *unwrapped* phase separately — interpolating
    # real/imag directly mangles a response whose phase rotates between samples.
    f_uni   = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    mag_i   = np.interp(f_uni, freqs, np.abs(H),              left=0.0, right=0.0)
    phase_i = np.interp(f_uni, freqs, np.unwrap(np.angle(H)))
    in_band = (f_uni >= freqs[0]) & (f_uni <= freqs[-1])

    H_uni = np.zeros_like(f_uni, dtype=complex)
    H_uni[in_band] = (mag_i * np.exp(1j * phase_i))[in_band]

    if taper:
        # raised-cosine roll-off over the first/last ~5 % of the in-band region
        idx = np.flatnonzero(in_band)
        nb  = idx.size
        e   = max(1, nb // 20)
        w   = np.ones(nb)
        edge = 0.5 * (1.0 - np.cos(np.pi * np.arange(e) / e))
        w[:e]  = edge
        w[-e:] = edge[::-1]
        H_uni[idx] *= w

    h = np.fft.irfft(H_uni, n=n_fft)
    t = np.arange(n_fft) / fs

    if t_max is not None:
        keep = t <= t_max
        t, h = t[keep], h[keep]

    fig, ax = plt.subplots(figsize=(10, 4))
    if db:
        env = np.abs(h)
        y   = 20.0 * np.log10(np.maximum(env / max(env.max(), 1e-30), 1e-6))
        ax.plot(t * 1e3, y, lw=1.0, color="firebrick")
        ax.set_ylabel("Normalised IR [dB]")
        ax.set_ylim(-120, 5)
    else:
        ax.plot(t * 1e3, h, lw=1.0, color="firebrick")
        ax.set_ylabel("Impulse response [a.u.]")

    ax.set_xlabel("Time [ms]")
    ax.set_title(title or "Impulse response")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, t, h


# ── directivity ───────────────────────────────────────────────────────────────

def plot_directivity(theta: np.ndarray, p: np.ndarray, *,
                     normalize: bool = True, db: bool = True,
                     ax=None, label: str = "") -> plt.Axes:
    """
    Parameters
    ----------
    theta : (n_angles,)  angle array [rad]
    p     : (n_angles,)  complex pressure
    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(7, 7))

    mag = np.ma.masked_invalid(np.abs(np.asarray(p, dtype=complex)))
    if normalize and mag.max() > 0:
        mag = mag / mag.max()
    if db:
        val, vmin = 20 * np.log10(np.maximum(mag, 1e-6)), -40
    else:
        val, vmin = mag, 0

    ax.plot( theta - np.pi,  val, lw=1.8, color="blue", label=label)
    ax.plot(-theta + np.pi,  val, lw=1.8, color="blue")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(135)
    ax.set_ylim(vmin, val[np.isfinite(val)].max() if not db else 0)
    ax.set_thetamin(-180); ax.set_thetamax(180)
    if label:
        ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.05))
    return ax


# ── time-domain UVA signals ───────────────────────────────────────────────────

def plot_uva_time(t: np.ndarray,
                  u: np.ndarray, v: np.ndarray, a: np.ndarray, *,
                  title: str = "",
                  t_ramp: float | None = None) -> plt.Figure:
    """
    Parameters
    ----------
    t, u, v, a : (n_t,)  time axis and displacement / velocity / acceleration
    t_ramp     : float   if given, draw a vertical line at the ramp-end time
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for ax, sig, lbl, col in zip(
            axs,
            [u, v, a],
            [r"Displacement [m]", r"Velocity [m/s]", r"Acceleration [m/s$^2$]"],
            ["steelblue", "darkorange", "forestgreen"]):
        ax.plot(t, sig, lw=1.2, color=col)
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
        if t_ramp is not None:
            ax.axvline(t_ramp, color="gray", ls="--", lw=0.8, label="ramp end")

    if title:
        axs[0].set_title(title)
    if t_ramp is not None:
        axs[0].legend()
    axs[2].set_xlabel("Time [s]")
    fig.tight_layout()
    return fig


# ── FFT of UVA ────────────────────────────────────────────────────────────────

class UvaSpectra(NamedTuple):
    """Result of :func:`plot_uva_fft`.

    Supports both attribute access (``res.fig``) and tuple indexing/unpacking
    (``res[0]``).  The first five fields are always present; the rest are filled
    only when ``excitation_freqs`` is given, and are ``None`` otherwise.
    """
    fig:        plt.Figure               # the 3-panel spectrum figure
    freqs:      np.ndarray               # (n_freq,) frequency axis [Hz]
    u_mag:      np.ndarray               # (n_freq,) displacement magnitude spectrum
    v_mag:      np.ndarray               # (n_freq,) velocity magnitude spectrum
    a_mag:      np.ndarray               # (n_freq,) acceleration magnitude spectrum
    noise_mask: np.ndarray | None = None  # (n_freq,) True on non-excitation bins
    u_complex:  np.ndarray | None = None  # raw complex rfft of displacement
    v_complex:  np.ndarray | None = None  # raw complex rfft of velocity
    a_complex:  np.ndarray | None = None  # raw complex rfft of acceleration
    n_samples:  int | None        = None  # samples in the steady-state block
    ss_start:   int | None        = None  # index where the steady-state block starts
    t_block:    np.ndarray | None = None  # (n_samples,) time axis of the block [s]
    u_block:    np.ndarray | None = None  # (n_samples,) steady-state displacement
    v_block:    np.ndarray | None = None  # (n_samples,) steady-state velocity
    a_block:    np.ndarray | None = None  # (n_samples,) steady-state acceleration


def plot_uva_fft(t: np.ndarray,
                 u: np.ndarray, v: np.ndarray, a: np.ndarray, *,
                 t_start: float | None = None,
                 excitation_freqs: np.ndarray | None = None,
                 separate_tones: bool = True,      # split excitation vs remnant
                 plot_vars: str | tuple = ("u", "v", "a"),
                 line_style: str = "-",
                 line_width: float = 1.0,
                 color: str = "steelblue",
                 title: str = "") -> tuple:
    """
    FFT of displacement, velocity, acceleration.

    Parameters
    ----------
    t, u, v, a       : (n_t,)
    t_start          : start of steady-state window (default: t[n//4])
    excitation_freqs : if given, use a rectangular window; the FFT bins at the
                       excitation frequencies are identified exactly.
    separate_tones   : when excitation_freqs is given, plot the excitation bins
                       and the remnant (noise + nonlinearities) as two series.
                       If False, plot a single continuous spectrum.
                       Ignored when excitation_freqs is None.
    plot_vars        : which panels to draw: any subset/order of ("u","v","a"),
                       or a single string such as "a".
    line_style       : matplotlib linestyle for the spectrum ("-", "--", ":").
    line_width, color: line appearance.

    Returns
    -------
    UvaSpectra  (always .fig, .freqs, .u_mag, .v_mag, .a_mag)
    """
    if isinstance(plot_vars, str):
        plot_vars = (plot_vars,)
    plot_vars = tuple(plot_vars)
    if not set(plot_vars) <= {"u", "v", "a"}:
        raise ValueError('plot_vars entries must be "u", "v" or "a"')
    if len(plot_vars) == 0:
        raise ValueError("plot_vars is empty")

    dt = t[1] - t[0]
    if t_start is None:
        t_start = t[len(t) // 4]
    idx_ss = int(np.searchsorted(t, t_start))

    def _fft(sig):
        s = sig[idx_ss:]
        N = len(s)
        if excitation_freqs is not None:
            S_raw = np.fft.rfft(s)                 # rectangular window
            S_mag = 2.0 / N * np.abs(S_raw)
        else:
            w     = np.hanning(N)
            S_raw = np.fft.rfft(s * w)
            S_mag = 2.0 / N * np.abs(S_raw) / w.mean()
        f  = np.fft.rfftfreq(N, d=dt)
        db = 20.0 * np.log10(S_mag + 1e-20)
        return f, db, S_mag, S_raw, N

    fu, u_db, u_lin, U_c, Nu = _fft(u)
    fv, v_db, v_lin, V_c, Nv = _fft(v)
    fa, a_db, a_lin, A_c, Na = _fft(a)

    panels = {
        "u": (fu, u_db, r"Disp. [dB ref 1 m]"),
        "v": (fv, v_db, r"Vel. [dB ref 1 m/s]"),
        "a": (fa, a_db, r"Acc. [dB ref 1 m/s$^2$]"),
    }

    n = len(plot_vars)
    fig, axs = plt.subplots(n, 1, figsize=(8, 2.4 * n + 0.8), sharex=True,
                            squeeze=False)
    axs = axs[:, 0]

    split = separate_tones and (excitation_freqs is not None)
    mask_remnant = None

    for k, key in enumerate(plot_vars):
        ax = axs[k]
        f_, db_, lbl = panels[key]

        if split:
            idx_exc = np.unique([int(np.argmin(np.abs(f_ - fe)))
                                 for fe in excitation_freqs])
            mask_remnant = np.ones(len(f_), dtype=bool)
            mask_remnant[idx_exc] = False
            ax.semilogx(f_[mask_remnant], db_[mask_remnant],
                        color="gray", alpha=0.35, lw=0.6, ls=line_style,
                        label="Noise / nonlinear")
            ax.semilogx(f_[idx_exc], db_[idx_exc], "o", ms=3.0,
                        color=color, label="Excitation")
            if k == 0:
                ax.legend(loc="upper right", fontsize="x-small")
        else:
            if excitation_freqs is not None:
                # still record the mask for the caller, but draw one series
                idx_exc = np.unique([int(np.argmin(np.abs(f_ - fe)))
                                     for fe in excitation_freqs])
                mask_remnant = np.ones(len(f_), dtype=bool)
                mask_remnant[idx_exc] = False
            ax.semilogx(f_, db_, color=color, alpha=0.85,
                        lw=line_width, ls=line_style)
            ax.semilogx(f_[idx_exc], db_[idx_exc], "o", ms=3.0,
                        color="red", label="Excitation")
            ax.legend(loc="upper right", fontsize="x-small")

        ax.set_ylabel(lbl)
        ax.grid(True, which="both", alpha=0.2)

    win_type = "Rectangular" if excitation_freqs is not None else "Hanning"
    axs[0].set_title(title if title else f"FFT Analysis — {win_type} window",
                     pad=10.0)
    axs[-1].set_xlabel("Frequency [Hz]")

    fig.tight_layout()

    t_ss = t[idx_ss: idx_ss + Nu]
    if excitation_freqs is not None:
        return UvaSpectra(fig, fu, u_lin, v_lin, a_lin, mask_remnant,
                          U_c, V_c, A_c, Nu, idx_ss, t_ss,
                          u[idx_ss:], v[idx_ss:], a[idx_ss:])
    return UvaSpectra(fig, fu, u_lin, v_lin, a_lin)

# ── mechanical sweep (U/V/A vs frequency, in dB) ─────────────────────────────

def plot_meca_sweep(freqs: np.ndarray,
                    u: np.ndarray, v: np.ndarray, a: np.ndarray, *,
                    title: str = "") -> plt.Figure:
    """
    Parameters
    ----------
    freqs, u, v, a : (n_freq,)  frequency axis and complex amplitude at probe
    """
    def _db(q):
        mag = np.abs(q)
        return 20.0 * np.log10(np.where(mag > 0, mag, np.finfo(float).tiny))

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, sig, lbl, col in zip(
            axs,
            [u, v, a],
            [r"Disp. [dB ref 1 m]", r"Vel. [dB ref 1 m/s]", r"Acc. [dB ref 1 m/s$^2$]"],
            ["steelblue", "darkorange", "forestgreen"]):
        ax.semilogx(freqs, _db(sig), 'o',markersize = 0.5,color=col)
        ax.set_ylabel(lbl)
        ax.grid(True, which="both", alpha=0.3)

    if title:
        axs[0].set_title(title)
    axs[2].set_xlabel("Frequency [Hz]")
    fig.tight_layout()
    return fig
