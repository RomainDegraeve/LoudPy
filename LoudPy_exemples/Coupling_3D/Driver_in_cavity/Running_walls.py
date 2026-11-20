"""
Running_walls.py — the sealed enclosure of Running_script.py with elastic walls: MDF shells (MITC3) on the walls
of the cavity mesh, their panel edges clamped, coupled to the cavity air through their in-vacuo modes (phase I).
The front of the driver and the outside of the walls are in vacuo (cavity configuration).

  scene    : enclosure.sealed_scene() + scene.walls("MDF", thickness=WALL)
  walls    : lp.Modes(scene, physics="walls") — in-vacuo wall modes (panel modes)
  response : lp.FrequencyResponse — 1 N on the coil, the same band as Running_script.py; compared with the rigid
             box computed here (same cavity modes)

What this produces (Results/Files/)
------------------------------------
  - walls_modes.csv          : in-vacuo wall modes (frequency, damping ratio, volume displacement)
  - coupled_walls.h5         : forced response with elastic walls (driver and wall modal amplitudes)
  - walls_response.png       : coil excursion (rigid / elastic walls) and the walls' volume velocity relative to
                               the driver's rear volume velocity
"""
from pathlib import Path

import numpy as np

import loudpy as lp

lp.set_verbosity("info")
from enclosure import FILES, sealed_scene

files = Path(FILES)
WALL = 0.012                                       # MDF wall thickness [m] (thin, to show the panel modes)
FREQS = np.geomspace(20.0, 3000.0, 700)            # forced response band [Hz]
if "--smoke" in __import__("sys").argv:            # quick check: a few frequencies
    FREQS = np.array([50.0, 200.0, 800.0, 1500.0])

rigid = sealed_scene()
elastic = sealed_scene()
elastic.walls("MDF", thickness=WALL)               # lp.Rayleigh() damping of the MDF fit, panel edges clamped
print(elastic)

walls = lp.Modes(elastic, physics="walls").solve(f_max=FREQS[-1] * 2)
f_w, z_w, dv = walls.frequencies, walls.extra("zeta"), walls.extra("volume_disp")
np.savetxt(files / "walls_modes.csv", np.column_stack([f_w, z_w, dv]), delimiter=",",
           header="f_Hz,zeta,volume_displacement_m3_per_unit_modal_amplitude", comments="")
print(f"{len(f_w)} wall modes ≤ {FREQS[-1] * 2:.0f} Hz, first {np.round(f_w[:6], 1)} Hz → walls_modes.csv")

res = lp.FrequencyResponse(elastic, drive=lp.Force(1.0)).solve(FREQS, store=files / "coupled_walls.h5")
res_rigid = lp.FrequencyResponse(rigid, drive=lp.Force(1.0)).solve(FREQS)     # same code, same cavity modes

x_e = res.driver("main").coil_excursion()
x_r = res_rigid.driver("main").coil_excursion()
Q_w, Q_d = res.wall_volume_velocity(), res.driver_volume_velocity()
ratio = np.abs(Q_w.values) / np.abs(Q_d.values)
k = int(np.argmax(ratio))
print(f"walls / driver rear volume velocity: max {20 * np.log10(ratio[k]):.1f} dB at {FREQS[k]:.0f} Hz; "
      f"coil excursion change vs rigid: max {np.abs(x_e.values / x_r.values - 1).max() * 100:.2f} %")

import matplotlib.pyplot as plt
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
a1.loglog(FREQS, np.abs(x_r.values) * 1e3, label="rigid walls")
a1.loglog(FREQS, np.abs(x_e.values) * 1e3, "--", label=f"MDF walls, {WALL * 1e3:.0f} mm")
a1.set_ylabel("|coil excursion| [mm/N]")
a1.legend()
a2.semilogx(FREQS, 20 * np.log10(ratio))
for f in f_w[f_w <= FREQS[-1]]:
    a2.axvline(f, color="0.85", lw=0.6, zorder=0)
a2.set_xlabel("f [Hz]")
a2.set_ylabel("|Q walls| / |Q driver| [dB]")
fig.tight_layout()
fig.savefig(files / "walls_response.png", dpi=150)
print("→", files / "walls_response.png")
