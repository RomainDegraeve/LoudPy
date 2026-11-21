"""
Running_vented.py — HPNEW in the vented (bass-reflex) enclosure on a rigid ground, 1 N on the voice coil.

  scene    : enclosure.vented_scene() — ONE air domain (inside of the box + port + outside, P2 FE with the exact
             half-space DtN on a hemisphere); the driver wetted on both faces; the walls around the port refined
  response : lp.FrequencyResponse, 20 Hz – 3 kHz (2 Hz steps below 300 Hz, then 10 Hz), the pressure field streamed
             to the result file as the sweep goes

    python3 LoudPy_exemples/Coupling_3D/Driver_in_cavity/Running_vented.py              full sweep (~40 min)
    python3 LoudPy_exemples/Coupling_3D/Driver_in_cavity/Running_vented.py 50,500       quick test (a few frequencies)

What this produces (Results/Files/)
------------------------------------
  vented_merged.h5           (loudpy result format 3: driver modal amplitudes, DtN coefficients, the air field)
  vented_merged_<f>Hz.msh    one air mesh per band (built on the first run, reused afterwards)
Then run Loading_vented.py and Plot_3D_vented.py. (A faster hybrid model of the same box — modal inside, 1D port —
written with the low-level toolkit is kept in LoudPy_exemples/legacy/vented_hybrid.)
"""
import sys
from pathlib import Path

import numpy as np

import loudpy as lp

lp.set_verbosity("info")                 # progress messages (loudpy is silent by default)
from enclosure import FILES, vented_scene

files = Path(FILES)
FREQS = np.unique(np.concatenate([np.arange(20.0, 300.0, 2.0), np.arange(300.0, 3000.0 + 1e-9, 10.0)]))
tag = "vented_merged"
if len(sys.argv) > 1:                                      # quick test: a list of frequencies
    FREQS = np.array([float(x) for x in sys.argv[-1].split(",")])
    tag = "vented_merged_test"

scene = vented_scene()                                     # enclosure.py: cabinet, driver, merged air, meshed
print(scene)
res = lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve(FREQS, store=files / f"{tag}.h5")
print(res, "→", files / f"{tag}.h5")
