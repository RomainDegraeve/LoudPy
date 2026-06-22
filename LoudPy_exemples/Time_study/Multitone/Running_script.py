"""
Running_script.py — Nonlinear time-domain simulation with a multitone excitation.

Signal design — zero-leakage multitone
----------------------------------------
The input force is a sum of sinusoids whose frequencies all fall exactly on
FFT bin centres (integer multiples of df = 1/T_block).  This guarantees
*zero spectral leakage*: the steady-state block can be FFT-analysed without
any windowing function, making it possible to read nonlinear distortion
products (harmonics, intermodulation) directly from the spectrum.

Construction steps:
  1. Generate n_tones_wanted target frequencies on a logarithmic scale.
  2. Snap each to the nearest FFT bin (round to the nearest multiple of df).
  3. Remove duplicates (at low frequencies, consecutive log steps can share
     the same bin).
  4. Assign random phases (seeded for reproducibility).
  5. Concatenate two identical blocks: the first is faded in with a
     raised-cosine ramp so the structure reaches steady state; the second
     is the clean periodic block used for post-processing.

Solver
------
The time integration uses the Generalised-α method (controlled by ρ∞).
Geometric or material nonlinearity is handled by Newton-Raphson iterations
at each time step.

What this produces
------------------
    Results/time_study_large_signal.h5    — full time history (U, V, A)
    Results/signal_preview.pdf            — signal and its FFT for verification

Typical use
-----------
Run once.  Then use Loading_script.py to visualise the results.
"""
import numpy as np
import matplotlib.pyplot as plt

from loudpy.Studies import Problem, TimeStudy
from loudpy import (
    DomainSpecMecaRayleigh,
    InterfaceSpecClamped,
    InterfaceSpecForced,
    InterfaceSpecAcouMeca,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
geo_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
msh_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.msh"
mat_path = "src/loudpy/Materials_Bank/materials.json"
out_path = "LoudPy_exemples/Time_study/Multitone/Results/time_study_large_signal.h5"

# ── Signal parameters ──────────────────────────────────────────────────────────
f_min           = 20        # lowest excitation frequency [Hz]
f_max           = 1000      # highest excitation frequency [Hz]
fs              = 10_000    # sampling rate [Hz]  (must be > 2 · f_max)
t_block         = 1.0       # duration of one periodic block [s]
n_tones_wanted  = 90        # target number of tones before de-duplication
force_amplitude = 1e-4     # peak force [N] — small signal (large signal)
seed            = 42        # random seed for reproducible phase randomisation

# ── Step 1 — build a zero-leakage frequency grid ──────────────────────────────
df_FFT    = 1.0 / t_block                                # FFT bin width [Hz]
f_log     = np.geomspace(f_min, f_max, n_tones_wanted)  # ideal log spacing
f_snapped = np.round(f_log / df_FFT) * df_FFT           # snap to nearest bin
freqs     = np.unique(f_snapped)                         # remove bin collisions
freqs     = freqs[(freqs >= f_min) & (freqs <= f_max)]  # enforce band limits
n_tones   = len(freqs)

print(f"Tones requested:  {n_tones_wanted}")
print(f"Tones after snap: {n_tones}  (duplicates removed)")
print(f"FFT resolution:   df = {df_FFT} Hz  → zero leakage guaranteed")

# ── Step 2 — one periodic block with random phases ────────────────────────────
n_per_block = int(t_block * fs)
t_block_arr = np.arange(n_per_block) / fs

rng    = np.random.default_rng(seed)
phases = rng.uniform(-np.pi, np.pi, size=n_tones)

# Sum of unit-amplitude sinusoids at the chosen frequencies
single_block = sum(
    np.sin(2.0 * np.pi * f * t_block_arr + phi)
    for f, phi in zip(freqs, phases)
)

# ── Step 3 — ramp-up prefix + clean steady-state block ────────────────────────
# Playing the block twice lets transients die out during the first pass.
# A raised-cosine (Hann) ramp avoids a velocity discontinuity at t = 0.
signal_total = np.concatenate([single_block, single_block])
t_total      = np.arange(len(signal_total)) / fs

ramp = 0.5 * (1.0 - np.cos(np.pi * t_block_arr / t_block))   # 0 → 1 over t_block
full_ramp = np.ones_like(t_total)
full_ramp[:n_per_block] = ramp

force_signal = signal_total * full_ramp
n_steps      = len(force_signal) - 1   # number of time integration steps

# ── Step 4 — verify zero leakage on the steady-state block ────────────────────
steady = force_signal[n_per_block:]           # second block (pure periodic)
S      = np.fft.rfft(steady) / len(steady)
f_axis = np.fft.rfftfreq(len(steady), 1.0 / fs)
mag_db = 20 * np.log10(2 * np.abs(S) + 1e-100)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), tight_layout=True)
axes[0].plot(t_total, force_signal, lw=0.8)
axes[0].axvline(t_block, color="r", ls="--", label="end of ramp / start of steady state")
axes[0].set(xlabel="time [s]", ylabel="Amplitude [normalised]",
            title=f"Multitone signal — {n_tones} tones, {t_block*2:.1f} s total")
axes[0].legend()
axes[0].grid(True, which="both", alpha=0.3)
axes[1].semilogx(f_axis, mag_db, lw=0.9)
axes[1].set(xlabel="frequency [Hz]", ylabel="magnitude [dB]",
            title="FFT of steady-state block (zero leakage — all energy at bin centres)")
axes[1].grid(True, which="both", alpha=0.3)
fig.savefig("LoudPy_exemples/Time_study/Multitone/Results/signal_preview.pdf",
            bbox_inches="tight")
plt.show()

# ── Problem definition ─────────────────────────────────────────────────────────
problem = Problem(geo_path=geo_path, msh_path=msh_path, mat_path=mat_path)

problem.add_sub_domain(
    DomainSpecMecaRayleigh("membranne",  material="Paper",         size=0.0005),
    DomainSpecMecaRayleigh("coil",       material="Copper",        size=0.0005),
    DomainSpecMecaRayleigh("surround",   material="Rubber",        size=0.0005),
    DomainSpecMecaRayleigh("spider",     material="PhenolicCloth", size=0.0005),
    DomainSpecMecaRayleigh("former",     material="Kapton",        size=0.0005),
    DomainSpecMecaRayleigh("glue",       material="SolidGlue",     size=0.0005),
    DomainSpecMecaRayleigh("dustcap",    material="Polypropylene", size=0.0005),
)

problem.add_interface(
    InterfaceSpecClamped("interface_constrained"),
    InterfaceSpecForced("interface_forced"),
    InterfaceSpecAcouMeca("interface_acou_meca"),
)

problem.mesh(show_mesh_gui=False)

# ── Time-domain solve ──────────────────────────────────────────────────────────
# Generalised-α method:
#   ρ∞ = 1.0 → no algorithmic damping (energy-conserving)
#   ρ∞ < 1.0 → damps spurious high-frequency modes; 0.9 is a mild setting
# Newton-Raphson (nr_*) handles geometric / material nonlinearity.
study = TimeStudy(problem)
study.assemble_init()

study.solve_time_domain_rayleigh_nl(
    force_amplitude = force_amplitude,  # N
    force_direction = "_z_",            # axial direction
    force_signal    = force_signal,
    dt              = 1.0 / fs,         # time step [s]
    n_steps         = n_steps,
    nr_max_iter     = 10,               # Newton-Raphson: max iterations per step
    nr_tol          = 1e-10,             # Newton-Raphson: convergence tolerance
    rho_inf         = 0.7,             # generalised-α spectral radius
)

# ── Save ───────────────────────────────────────────────────────────────────────
study.save(
    path            = out_path,
    case            = "multitone_small_signal",
    f_min           = f_min,
    f_max           = f_max,
    freqs_tones     = freqs,
    t_total         = t_block * 2,
    t_ramp          = t_block,
    seed            = seed,
    force_amplitude = force_amplitude,
)
print(f"Saved → {out_path}")
