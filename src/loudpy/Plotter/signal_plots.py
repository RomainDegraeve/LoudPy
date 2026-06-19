"""
1-D signal plotting — all functions take plain numpy arrays.

No Snapshot, Mesh, Reader or study objects are imported here.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

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
    a1.set_title(title or "SPL & Phase Frequency Response")
    a1.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)

    a2.semilogx(freqs, phase, 'o',markersize = 0.5, color="C1")
    a2.set_ylabel(r"Phase  [$^\circ$]")
    a2.set_xlabel("Frequency  [Hz]")
    a2.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)

    fig.tight_layout()
    return fig


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

def plot_uva_fft(t: np.ndarray,
                 u: np.ndarray, v: np.ndarray, a: np.ndarray, *,
                 t_start: float | None = None,
                 excitation_freqs: np.ndarray | None = None,
                 title: str = "") -> tuple:
    """
    FFT of displacement, velocity, acceleration.

    Parameters
    ----------
    t, u, v, a        : (n_t,)
    t_start           : float  start of steady-state window (default: t[n//4])
    excitation_freqs  : array  if given, use rectangular window and highlight
                               excitation peaks vs noise/nonlinearities

    Returns
    -------
    excitation_freqs is None  → fig, freqs, u_lin, v_lin, a_lin
    excitation_freqs provided → fig, freqs, u_lin, v_lin, a_lin,
                                mask_remnant, U_c, V_c, A_c, N_ss, idx_ss,
                                t_ss, u_ss, v_ss, a_ss
    """
    dt = t[1] - t[0]
    if t_start is None:
        t_start = t[len(t) // 4]
    idx_ss = int(np.searchsorted(t, t_start))

    def _fft(sig):
        s = sig[idx_ss:]
        N = len(s)
        if excitation_freqs is not None:
            S_raw = np.fft.rfft(s)
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

    fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    labels   = [r"Disp. [dB ref 1 m]", r"Vel.  [dB ref 1 m/s]", r"Acc.  [dB ref 1 m/s$^2$]"]

    mask_remnant = None
    for ax, (f_, db_), lbl in zip(axs, [(fu, u_db), (fv, v_db), (fa, a_db)], labels):
        if excitation_freqs is not None:
            idx_exc = np.unique([np.argmin(np.abs(f_ - fe)) for fe in excitation_freqs])
            mask_remnant = np.ones(len(f_), dtype=bool)
            mask_remnant[idx_exc] = False
            ax.semilogx(f_[mask_remnant], db_[mask_remnant],
                        color="gray", alpha=0.35, lw=0.6, label="Noise / nonlinear")
            ax.semilogx(f_[idx_exc], db_[idx_exc], "o", ms=0.5,
                        color="steelblue", label="Excitation")
            if ax is axs[0]:
                ax.legend(loc="upper right", fontsize="x-small")
        else:
            ax.semilogx(f_, db_, color="steelblue", alpha=0.8, lw=1.0)
        ax.set_ylabel(lbl)
        ax.grid(True, which="both", alpha=0.2)

    win_type = "Rectangular" if excitation_freqs is not None else "Hanning"
    axs[0].set_title(title or f"FFT Analysis - {win_type} window")
    axs[2].set_xlabel("Frequency [Hz]")
    fig.tight_layout()

    t_ss = t[idx_ss: idx_ss + Nu]
    if excitation_freqs is not None:
        return (fig, fu, u_lin, v_lin, a_lin, mask_remnant,
                U_c, V_c, A_c, Nu, idx_ss, t_ss,
                u[idx_ss:], v[idx_ss:], a[idx_ss:])
    return fig, fu, u_lin, v_lin, a_lin


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
