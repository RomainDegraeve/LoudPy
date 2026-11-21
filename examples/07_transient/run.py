"""
run.py — nonlinear time response of the loudspeaker to a multitone force.

Signal: N_TONES sines on a zero-leakage grid (multiples of 1 / T_BLOCK) with random phases,
one periodic block of T_BLOCK, preceded by a raised-cosine ramp of T_RAMP from rest. The FFT of
the steady-state block (after the ramp) then has no leakage.

What this produces
------------------
    Results/Files/time.h5   — u, v, a at the cone tip at every step (full rate, a probe) and u on the mesh every
                        STORE_EVERY steps (loudpy result format 3), with the signal parameters in its metadata
                        (t_ramp, freqs_tones, …). Storing u, v, a at every node and step would take 8.7 GB.

Then run plot.py.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import loudpy as lp

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

lp.set_verbosity("info")                 # progress messages (loudpy is silent by default)

lp.plot.style.use()                       # loudpy figure style (fonts, colours, SPL in dB RMS)


# ── Paths ──────────────────────────────────────────────────────────────────────
geometry = str(EXAMPLES / "Geometries/Loudspeaker/HPNEW-Sketch.geo")
mesh_path = str(HERE / "Results/Files/mesh.msh")      # reused on reruns
out_path = str(HERE / "Results/Files/time.h5")

# ── Signal parameters ──────────────────────────────────────────────────────────
F_MIN, F_MAX = 20, 1000        # band of the tones [Hz]
FS = 48_000                    # sampling rate [Hz]
T_BLOCK = 0.5                  # analysis window = periodic block [s] → df = 2 Hz
T_RAMP = 0.5                   # ramp [s] → total signal 1.0 s
N_TONES_WANTED = 90
FORCE_AMPLITUDE = 1.0          # [N] (scales the sum of unit sines)
SEED = 42

# ── Zero-leakage frequency grid ────────────────────────────────────────────────
df = 1.0 / T_BLOCK
freqs = np.unique(np.round(np.geomspace(F_MIN, F_MAX, N_TONES_WANTED) / df) * df)
freqs = freqs[(freqs >= F_MIN) & (freqs <= F_MAX)]
print(f"Tones: {len(freqs)}  |  df = {df} Hz")

# ── One periodic block, then the ramp in front of it ──────────────────────────
n_block = int(T_BLOCK * FS)
t_block = np.arange(n_block) / FS
phases = np.random.default_rng(SEED).uniform(-np.pi, np.pi, size=len(freqs))
block = sum(np.sin(2.0 * np.pi * f * t_block + phi) for f, phi in zip(freqs, phases))

n_ramp = int(T_RAMP * FS)
ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_ramp) / n_ramp))
signal = FORCE_AMPLITUDE * np.concatenate([block[:n_ramp] * ramp, block])
t = np.arange(len(signal)) / FS

# ── Check: spectrum of the steady-state block ──────────────────────────────────
steady = signal[n_ramp:]
S = np.fft.rfft(steady) / len(steady)
f_axis = np.fft.rfftfreq(len(steady), 1.0 / FS)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), tight_layout=True)
axes[0].plot(t, signal / abs(signal).max(), lw=0.8, label="Force signal")
axes[0].plot(t[:n_ramp], ramp, lw=1.2, color="grey", label="Ramp envelope")
axes[0].plot(t[:n_ramp], -ramp, lw=1.2, color="grey")
axes[0].hlines([1, -1], T_RAMP, t[-1], color="grey", lw=1.2, ls=":")
axes[0].axvline(T_RAMP, color="red", ls="--", label=f"Ramp end / steady start ({T_RAMP} s)")
axes[0].set(xlabel="Time [s]", ylabel="Amplitude", title=f"{len(freqs)} tones, ramp {T_RAMP} s + signal {T_BLOCK} s")
axes[0].legend()
axes[0].grid(True, which="both", alpha=0.3)
axes[1].semilogx(f_axis, 20 * np.log10(2 * np.abs(S) + 1e-100), lw=0.9)
axes[1].set(xlabel="Frequency [Hz]", ylabel="Amplitude [dB]", title="FFT of steady-state block")
axes[1].grid(True, which="both", alpha=0.3)
plt.show()

# ── Model ──────────────────────────────────────────────────────────────────────
size = 0.0005
model = lp.Model(geometry)
model.add(
    lp.Solid("cone", "Paper", mesh_size=size),
    lp.Solid("coil", "Copper", mesh_size=size),
    lp.Solid("surround", "Rubber", mesh_size=size),
    lp.Solid("spider", "PhenolicCloth", mesh_size=size),
    lp.Solid("former", "Kapton", mesh_size=size),
    lp.Solid("glue", "SolidGlue", mesh_size=size),
    lp.Solid("dustcap", "Polypropylene", mesh_size=size),
    lp.Clamped("rims", on=lp.Group("surround_rim", "spider_rim")), lp.Clamped("axis", dofs="r"),
    lp.Coil("coil_drive"),
    lp.FluidStructure("wet_front"),
)
model.mesh(mesh_path)

# ── Solve ──────────────────────────────────────────────────────────────────────
study = lp.Transient(model, drive=lp.Force(signal, fs=FS), rho_inf=0.98, nr_tol=1e-6, nr_max_iter=10)
STORE_EVERY = 40               # fields every 40 steps (1.2 kHz): enough for the snapshots and the animation
output = lp.Output(fields=["u"], every=STORE_EVERY, probes={"tip": (0.0, 0.0)})     # tip: full rate (48 kHz)
res = study.solve(output=output)
print(res)
study.save(out_path, case="multitone", f_min=F_MIN, f_max=F_MAX, freqs_tones=freqs,
           t_total=T_RAMP + T_BLOCK, t_ramp=T_RAMP, seed=SEED, force_amplitude=FORCE_AMPLITUDE)
print(f"Saved → {out_path}")
