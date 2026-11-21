"""
run.py — three identical HPNEW drivers in a column with a folded transmission line,
on a rigid ground, driven in phase (DRIVE, default 1 N on each voice coil).

  - drivers : the azimuthal-order modes of ../02_azimuthal_modes (m = 0 … 3), the same
              lp.Driver placed three times; their modal coordinates are stacked (top, middle,
              bottom), and they interact through the air
  - air     : ONE domain (lp.Scene.merged) — inside of the line + mouth + exterior half-space,
              P2 FE + exact half-space DtN on a hemisphere; every driver wetted on both faces
  - force   : 1 N on each coil (axisymmetric: drives m = 0); m ≥ 1 of the top and bottom drivers
              are driven by the non-symmetric air load

Run ../02_azimuthal_modes/run.py first (driver modes). The meshes are built on
the first run (one per band) and reused afterwards.

Quick check of a few frequencies (separate column_test.h5):  python3 run.py 40,100,300

What this produces (Results/Files/)
------------------------------------
  - column.h5        : modal amplitudes, DtN coefficients and the pressure at every node, per
                       frequency (loudpy result format 3; the field is written as the sweep goes)
  - air_<f>Hz.msh    : the air mesh of each band
Then run plot.py and plot_3d.py.
"""
import sys
from pathlib import Path

import numpy as np

import loudpy as lp

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

lp.set_verbosity("info")                 # progress messages (loudpy is silent by default)
from column import CENTERS, NAMES, cabinet, driver, line_length, quarter_wave

files = (HERE / "Results/Files")
F_MAX = 1000.0                                 # top of the sweep [Hz] (2 kHz: 169k nodes, > 8 GB RAM)
FREQS = np.unique(np.concatenate([np.arange(20.0, 300.0, 2.0), np.arange(300.0, F_MAX + 1e-9, 10.0)]))
DRIVE = {"top": lp.Force(1.0), "middle": lp.Force(1.0), "bottom": lp.Force(1.0)}
BANDS = [(20.0, 1000.0), (1000.0, 2000.0)]     # one mesh each [Hz]
TAG = "column"
if len(sys.argv) > 1:                          # quick test: python3 run.py 40,100,300
    FREQS = np.array([float(x) for x in sys.argv[1].split(",")])
    TAG = "column_test"

hp = driver()
scene = lp.Scene(cabinet, [hp.place(name, at=c) for name, c in zip(NAMES, CENTERS)])
scene.merged(ground="y", bands=[b for b in BANDS if b[0] < FREQS.max()],
             points_per_wavelength=4.0, max_size=0.07, driver_size=0.004, margin=0.05)
scene.mesh(files)
print(f"{hp} ×{len(NAMES)} | line {line_length():.3f} m, quarter-wave estimate {quarter_wave():.1f} Hz | "
      f"{len(FREQS)} frequencies ≤ {FREQS.max():.0f} Hz")

study = lp.FrequencyResponse(scene, drive=DRIVE)
res = study.solve(FREQS, store=files / f"{TAG}.h5")
print(res, f"→ {files / (TAG + '.h5')}")
