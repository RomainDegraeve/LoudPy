import numpy as np
import matplotlib.pyplot as plt

plt.rc("lines",  linewidth=2)
plt.rc("font",   size=14)
plt.rc("axes",   linewidth=1.5, labelsize=14)
plt.rc("legend", fontsize=14)
plt.rcParams["font.family"]                  = "serif"
plt.rcParams["font.serif"]                   = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"]  = True
plt.rcParams["mathtext.fontset"]             = "stix"

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
out_path = "LoudPy_exemples/Time_study/Multitone/Results/time_study_1N_bis.h5"

# ── Signal parameters ──────────────────────────────────────────────────────────
f_min           = 20
f_max           = 1000
fs              = 48_000
t_block = 0.5   # analyse window AND periodic block [s]  →  df = 2 Hz
t_ramp  = 0.5   # ramp [s]  →  total signal = 1.0 s
n_tones_wanted  = 90
force_amplitude = 1
seed            = 42

# ── Grille fréquentielle zero-leakage ─────────────────────────────────────────
df_FFT    = 1.0 / t_block
f_snapped = np.round(np.geomspace(f_min, f_max, n_tones_wanted) / df_FFT) * df_FFT
freqs     = np.unique(f_snapped)
freqs     = freqs[(freqs >= f_min) & (freqs <= f_max)]
n_tones   = len(freqs)
print(f"Tones: {n_tones}  |  df = {df_FFT} Hz")

# ── Bloc périodique (1.0 s) ───────────────────────────────────────────────────
n_per_block = int(t_block * fs)          # 24 000 samples
t_block_arr = np.arange(n_per_block) / fs

rng    = np.random.default_rng(seed)
phases = rng.uniform(-np.pi, np.pi, size=n_tones)

single_block = sum(
    np.sin(2.0 * np.pi * f * t_block_arr + phi)
    for f, phi in zip(freqs, phases)
)

# ── Signal total : ramp 0.5 s  +  multitone 1.0 s  =  1.5 s ─────────────────
n_ramp      = int(t_ramp * fs)           # 12 000 samples
ramp_window = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_ramp) / n_ramp))

ramp_part    = single_block[:n_ramp] * ramp_window   # 0 → 1 sur 0.5 s
force_signal = np.concatenate([ramp_part, single_block])   # 1.5 s au total
t_total      = np.arange(len(force_signal)) / fs
n_steps      = len(force_signal) - 1

# ── Vérification zero-leakage sur le bloc steady-state ────────────────────────
steady = force_signal[n_ramp:]           # dernière 1.0 s = single_block
S      = np.fft.rfft(steady) / len(steady)
f_axis = np.fft.rfftfreq(len(steady), 1.0 / fs)

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7), tight_layout=True)

# Time-domain signal
axes[0].plot(t_total, force_signal/abs(force_signal).max(), lw=0.8,  label="Force signal")

# Ramp envelope (rising, over the ramp duration) plotted in grey
axes[0].plot(t_total[:n_ramp], ramp_window, lw=1.2, color="grey", label="Ramp envelope")
# Mirror envelope on the negative side to show ± bounds
axes[0].plot(t_total[:n_ramp], -ramp_window, lw=1.2, color="grey")

# Steady-state part is unwindowed (envelope = ±1)
axes[0].hlines(1, t_ramp, t_total[-1], color="grey", lw=1.2, ls=":")
axes[0].hlines(-1, t_ramp, t_total[-1], color="grey", lw=1.2, ls=":")

axes[0].axvline(t_ramp, color="red", ls="--", label=f"Ramp end / steady start ({t_ramp} s)")
axes[0].set(
    xlabel="Time [s]",
    ylabel="Amplitude",
    title=f"{n_tones} tones, ramp {t_ramp} s + signal {t_block} s"
)
axes[0].legend()
axes[0].grid(True, which="both", alpha=0.3)

# Frequency-domain (FFT of steady-state block)
axes[1].semilogx(f_axis, 20 * np.log10(2 * np.abs(S) + 1e-100), lw=0.9)
axes[1].set(
    xlabel="Frequency [Hz]",
    ylabel="Amplitude [dB]",
    title="FFT of steady-state block"
)
axes[1].grid(True, which="both", alpha=0.3)

plt.show()

# ── Problem ───────────────────────────────────────────────────────────────────
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
    InterfaceSpecAcouMeca("interface_acou_meca_front"),
)
problem.mesh(show_mesh_gui=False)

# ── Solve ─────────────────────────────────────────────────────────────────────
study = TimeStudy(problem)
study.assemble_init()
study.solve_time_domain_rayleigh_nl(
    force_amplitude = force_amplitude,
    force_direction = "_z_",
    force_signal    = force_signal,
    dt              = 1.0 / fs,
    n_steps         = n_steps,
    nr_max_iter     = 10,
    nr_tol          = 1e-6,
    rho_inf         = 0.98,
)

# ── Save ──────────────────────────────────────────────────────────────────────
study.save(
    path            = out_path,
    case            = "multitone_small_signal",
    f_min           = f_min,
    f_max           = f_max,
    freqs_tones     = freqs,
    t_total         = t_ramp + t_block,   # 1.5 s
    t_ramp          = t_ramp,             # 0.5 s — steady state commence ici
    seed            = seed,
    force_amplitude = force_amplitude,
)
print(f"Saved → {out_path}")