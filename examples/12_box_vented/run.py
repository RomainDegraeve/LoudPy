"""
run.py — HPNEW in the vented (bass-reflex) enclosure on a rigid ground, 1 N on the voice coil.

  scene    : enclosure.vented_scene() — ONE air domain (inside of the box + port + outside, P2 FE with the exact
             half-space DtN on a hemisphere); the driver wetted on both faces; the walls around the port refined
  response : lp.FrequencyResponse, 20 Hz – 3 kHz (2 Hz steps below 300 Hz, then 10 Hz), the pressure field streamed
             to the result file as the sweep goes

    python3 examples/Coupling_3D/Driver_in_cavity/run.py              full sweep (~40 min)
    python3 examples/Coupling_3D/Driver_in_cavity/run.py 50,500       quick test (a few frequencies)

What this produces (the results <study>.h5 below: results/12_box_vented/<study>_<stamp>.h5,
stamped — plot.py reads the newest; meshes and caches: Results/Files/)
----------------------------------------------------------------------
  vented_merged.h5           (loudpy result format 3: driver modal amplitudes, DtN coefficients, the air field)
  vented_merged_<f>Hz.msh    one air mesh per band (built on the first run, reused afterwards)
Then run plot.py and plot_3d.py. (A faster hybrid model of the same box — modal inside, 1D port —
written with the low-level toolkit is kept in examples/legacy/vented_hybrid.)

The scene is built at the top (not meshed); meshing and solving run only when this file is run (python3 run.py),
so the interface opens it without computing (python -m loudpy.ui on this folder or file).
"""
import sys
from pathlib import Path

import numpy as np

import loudpy as lp

lp.set_verbosity("info")                 # progress messages (loudpy is silent by default)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))   # the shared box
from enclosure import MESHES, vented_scene  # noqa: E402

HERE = Path(__file__).resolve().parent
files = Path(__file__).resolve().parent / "Results/Files"     # this example's results
files.mkdir(parents=True, exist_ok=True)
FREQS = np.unique(np.concatenate([np.arange(20.0, 300.0, 2.0), np.arange(300.0, 3000.0 + 1e-9, 10.0)]))
tag = "vented_merged"
scene = vented_scene(mesh=False)                           # enclosure.py: cabinet, driver, merged air

if __name__ == "__main__":
    if len(sys.argv) > 1:                                  # quick test: a list of frequencies
        FREQS = np.array([float(x) for x in sys.argv[-1].split(",")])
        tag = "vented_merged_test"
    scene.mesh(MESHES)
    print(scene)
    study = lp.FrequencyResponse(scene, drive=lp.Force(1.0))
    res = study.solve(FREQS, store=lp.result_path(HERE, tag, study))
    print(res)
