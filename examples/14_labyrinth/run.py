"""
run.py — HPNEW in the labyrinth cabinet on a rigid ground, 1 N on the voice coil.

  - driver : the azimuthal-order modes of ../02_azimuthal_modes (m = 0 … 3, both families)
  - air    : ONE domain (lp.Scene.merged) — inside of the folded line + mouth + exterior
             half-space, P2 FE + exact half-space DtN on a hemisphere; the driver is wetted on
             both faces by the same pressure field. No stuffing.

Run ../02_azimuthal_modes/run.py first (driver modes). The meshes (one per band)
are built on the first run and reused afterwards.

Quick check of a few frequencies (separate labyrinth_test.h5):  python3 run.py 40,70,100

Settings: F_MAX (default 2 kHz: the 3 kHz mesh needs more than 8 GB of RAM without swapping —
set F_MAX = 3000 on a larger machine), frequency steps below.

What this produces (the results <study>.h5 below: results/14_labyrinth/<study>_<stamp>.h5,
stamped — plot.py reads the newest; meshes and caches: Results/Files/)
----------------------------------------------------------------------
  - labyrinth.h5            : driver modal amplitudes, DtN coefficients and the pressure at every
                              node, per frequency (loudpy result format 3, streamed during the sweep)
  - air_<f>Hz.msh           : the air mesh of each band
  - labyrinth_cabinet.step  : the cabinet (walls, panels, driver hole, mouth) for CAD
Then run plot.py and plot_3d.py.

Electrical side: the driver's motor (labyrinth.driver: Re 8 Ω, Bl 7 T·m, Le 4 mH, n 0.8) and the circuit `amp`
(an amplifier of 2.83 V on the coil). The study is driven by 1 N; with one driver its result keeps the coil's unit
response, so it can be driven by `amp` (or any circuit drawn in the interface) after the study, without solving again:
res.redrive(amp). Or solve it driven by the amplifier: lp.FrequencyResponse(scene, drive=amp).

The scene is built at the top (not meshed); meshing and solving run only when this file is run (python3 run.py),
so the interface opens it without computing (python -m loudpy.ui on this folder or file).
"""
import sys
from pathlib import Path

import numpy as np

import loudpy as lp

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

lp.set_verbosity("info")                 # progress messages (loudpy is silent by default)
from labyrinth import X0, Y0, cabinet, driver, export_cabinet, line_length, quarter_wave

files = (HERE / "Results/Files")
files.mkdir(parents=True, exist_ok=True)
F_MAX = 2000.0                                 # top of the sweep [Hz]
FREQS = np.unique(np.concatenate([np.arange(20.0, 300.0, 2.0), np.arange(300.0, F_MAX + 1e-9, 10.0)]))
BANDS = [(20.0, 1000.0), (1000.0, 2000.0), (2000.0, 3000.0)]     # one mesh each [Hz]
TAG = "labyrinth"
hp = driver()
scene = lp.Scene(cabinet, [hp.place("main", at=(X0, Y0))])
scene.merged(ground="y", bands=[b for b in BANDS if b[0] < FREQS.max()],
             points_per_wavelength=4.0, max_size=0.06, driver_size=0.003, margin=0.05)
amp = lp.Circuit(lp.circuit.Source("amp", "in", U=2.83),           # the amplifier on the coil (its motor: the driver's)
                 lp.circuit.Coil("coil", "in", "0", driver="main"))

if __name__ == "__main__":
    if len(sys.argv) > 1:                          # quick test: python3 run.py 40,70,100
        FREQS = np.array([float(x) for x in sys.argv[1].split(",")])
        TAG = "labyrinth_test"
        scene.merged(ground="y", bands=[b for b in BANDS if b[0] < FREQS.max()],
                     points_per_wavelength=4.0, max_size=0.06, driver_size=0.003, margin=0.05)
    export_cabinet(files, hp.r_front)
    scene.mesh(files)
    print(f"{hp} | line {line_length():.3f} m, quarter-wave estimate {quarter_wave():.1f} Hz | "
          f"{len(FREQS)} frequencies ≤ {FREQS.max():.0f} Hz")
    study = lp.FrequencyResponse(scene, drive=lp.Force(1.0))
    res = study.solve(FREQS, store=lp.result_path(HERE, TAG, study))
    print(res)
