import numpy as np
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt

from loudpy.Studies import Problem, TimeStudy
from loudpy import DomainSpecMecaRayleigh, InterfaceSpecClamped, InterfaceSpecForced, InterfaceSpecAcouMeca
from loudpy import DomainSpecMeca, DomainSpecMecaRayleigh, InterfaceSpecAcouMeca, InterfaceSpecClamped, InterfaceSpecForced

# ── Paths ────────────────────────────────────────────────────────────────────
mat_path = "src/loudpy/Materials_Bank/materials.json"
out_path = "LoudPy_exemples/Time_study/Multitone/Results/time_study_small_signal.h5"
geo_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
msh_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.msh"

import numpy as np
import matplotlib.pyplot as plt


# ── Paramètres ───────────────────────────────────────────────────────────────
f_min   = 20
f_max   = 1000
t_block = 1            # Durée d'un bloc (période fondamentale)
fs      = 10000           # Fréquence d'échantillonnage
df_FFT  = 1.0 / t_block    # 2.0 Hz : Résolution minimale pour zéro leakage
n_tones_wanted = 90


# 1. Générer des fréquences logarithmiques idéales (théoriques)
f_log_ideal = np.geomspace(f_min, f_max, n_tones_wanted)

# 2. "Aimantage" sur la grille FFT (multiples de df_FFT)
# La formule : round(f / df) * df
freqs_snapped = np.round(f_log_ideal / df_FFT) * df_FFT

# 3. Supprimer les doublons 
# (À basse fréquence, plusieurs points log peuvent tomber sur le même bin FFT)
freqs = np.unique(freqs_snapped)

# 4. Vérifier que l'on reste dans les bornes
freqs = freqs[(freqs >= f_min) & (freqs <= f_max)]

n_tones = len(freqs)
print(f"Nombre de tons demandés : {n_tones_wanted}")
print(f"Nombre de tons après alignement et unicité : {n_tones}")



# ── 2. Génération d'UN SEUL bloc de  ─────────────────────────────────────
n_steps_block = int(t_block * fs)
# On utilise arange pour avoir exactement n_steps_block points
t_single_block = np.arange(n_steps_block) / fs 

seed = 42                                  # any integer
rng  = np.random.default_rng(seed)


k = np.arange(1, n_tones + 1)
# CORRECT: Now it uses the seeded generator
phases = rng.uniform(-np.pi, np.pi, size=n_tones)


# Construction du bloc de base
single_block = np.zeros(n_steps_block)
for f, phi in zip(freqs, phases):
    single_block += np.sin(2.0 * np.pi * f * t_single_block + phi)

# ── 3. Duplication et Rampe ──────────────────────────────────────────────────
# On joue le signal deux fois (Total = 1.0s)
signal_total = np.concatenate([single_block, single_block])
t_total_axis = np.arange(len(signal_total)) / fs

# Création de la rampe sur la durée du premier bloc (0.5s)
# La rampe s'arrête exactement à la fin du premier bloc
ramp_values = 0.5 * (1.0 - np.cos(np.pi * t_single_block / t_block))
full_ramp = np.ones_like(t_total_axis)
full_ramp[:n_steps_block] = ramp_values

# Application de la rampe
force_signal  = signal_total * full_ramp 
n_steps      = len(force_signal) - 1

# ── 4. Analyse FFT (sur le 2ème bloc uniquement) ─────────────────────────────
seg_fft = force_signal[n_steps_block:]  # Le bloc pur de 0.5s à 1.0s
N = len(seg_fft)

S_fft = np.fft.rfft(seg_fft) / N
f_axis = np.fft.rfftfreq(N, 1.0 / fs)
magnitudes_db = 20 * np.log10(2 * np.abs(S_fft))

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Time plot
axes[0].plot(t_total_axis, force_signal, color='steelblue', lw=0.8)
axes[0].axvline(t_block, color='red', linestyle='--', label='Fin du 1er bloc (Rampe)')

axes[0].set_title(f"Signal : 2 blocs de {t_block}s (Total 1s) | {n_tones} tones")
axes[0].set_ylabel("Amplitude")
axes[0].legend()

# FFT plot
axes[1].semilogx(f_axis, magnitudes_db, color='crimson', lw=1)
axes[1].set_xlim(0, f_max + 100)

axes[1].set_title("Spectre FFT du 2ème bloc (Zéro Leakage)")
axes[1].set_xlabel("Fréquence [Hz]")
axes[1].set_ylabel("Magnitude [dB]")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Nombre de fréquences : {n_tones}")
print(f"Résolution FFT (df)  : {df_FFT} Hz")
print(f"Le 2ème bloc est une réplique parfaite : {np.allclose(single_block, seg_fft)}")

force_amplitude = 1e-6

# ── Problem setup ─────────────────────────────────────────────────────────────
problem = Problem(geo_path=geo_path, msh_path=msh_path, mat_path=mat_path)
problem.add_sub_domain(
    DomainSpecMecaRayleigh("membranne",  material="Paper",           size=0.0005),
    DomainSpecMecaRayleigh("coil",       material="Copper",          size=0.0005),
    DomainSpecMecaRayleigh("surround",   material="Rubber",          size=0.0005),
    DomainSpecMecaRayleigh("spider",     material="PhenolicCloth",   size=0.0005),
    DomainSpecMecaRayleigh("former",     material="Kapton",          size=0.0005),
    DomainSpecMecaRayleigh("glue",       material="SolidGlue",       size=0.0005),
    DomainSpecMecaRayleigh("dustcap",    material="Polypropylene",   size=0.0005),
   
)

problem.add_interface(InterfaceSpecClamped("interface_constrained"))
problem.add_interface(InterfaceSpecForced("interface_forced"))
problem.add_interface(InterfaceSpecAcouMeca("interface_acou_meca"))

problem.mesh(show_mesh_gui=False)

# ── Time study ────────────────────────────────────────────────────────────────
study = TimeStudy(problem)
study.assemble_init()

time_array, results_u, results_v, results_a = study.solve_time_domain_rayleigh_nl(
    force_amplitude = force_amplitude,          # N
    force_direction = '_z_',        # axial (z) excitation
    force_signal    = force_signal,
    dt              = t_total_axis[1]-t_total_axis[0],
    n_steps         = n_steps,
    nr_max_iter     = 10,
    nr_tol          = 1e-5,
    rho_inf         = 0.90,
)

# ── Save ──────────────────────────────────────────────────────────────────────
study.save(
    path              = out_path,
    case              = "membrane_chirp",
    f_min             = f_min,
    f_max             = f_max,
    freqs_tones        = freqs,
    t_total           = t_block*2,
    t_ramp            = t_block,
    seed              = 42,
    force_amplitude = force_amplitude,  
)
