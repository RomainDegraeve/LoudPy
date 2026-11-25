"""
run.py — HPNEW driver (azimuthal-order modes m = 0 … 3) coupled to the modes of a
non-axisymmetric 3D enclosure, forced by its voice coil (../02_azimuthal_modes/run.py
first). The front of the driver is in vacuo (only the enclosure loads it).

  scene    : enclosure.sealed_scene() — lp.Scene(sealed_cabinet, [driver at (X0, Y0)]).cavity(...)
             the cavity mesh (driver cut out with its revolved wet rear profile) is built on the
             first run, its rigid-wall modes are cached next to it (cavity_modes.h5)
  response : lp.FrequencyResponse — 1 N on the coil (axisymmetric: it drives m = 0 only; m ≥ 1
             are driven through the enclosure); coupling S_kn = ∫ (φ_k·n) ψ_n dS on the wet rear
             surface + static correction of the truncated cavity modes
  modes    : lp.Modes — coupled driver–enclosure modes

What this produces (the results <study>.h5 below: results/10_box_cavity/<study>_<stamp>.h5,
stamped — plot.py reads the newest; meshes and caches: Results/Files/)
----------------------------------------------------------------------
  - cavity.msh, cavity_modes.h5 : enclosure mesh and its modes (reused on reruns)
  - coupled.h5                  : forced response (driver modal amplitudes, modal pressures)
  - coupled_modes.h5            : coupled modes (complex s, driver and cavity components)
Then run plot.py and plot_3d.py.

The scene is built at the top (not meshed); meshing and solving run only when this file is run (python3 run.py),
so the interface opens it without computing (python -m loudpy.ui on this folder or file).
"""
import sys
from pathlib import Path

import numpy as np

import loudpy as lp

lp.set_verbosity("info")                 # progress messages (loudpy is silent by default)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))   # the shared box
from enclosure import MESHES, sealed_scene  # noqa: E402

HERE = Path(__file__).resolve().parent
files = Path(__file__).resolve().parent / "Results/Files"     # this example's results
files.mkdir(parents=True, exist_ok=True)
FREQS = np.geomspace(20.0, 3000.0, 700)        # forced response band [Hz]

scene = sealed_scene(mesh=False)               # enclosure.py: cabinet, driver, cavity settings

if __name__ == "__main__":
    scene.mesh(MESHES)
    print(scene)
    study = lp.FrequencyResponse(scene, drive=lp.Force(1.0))
    res = study.solve(FREQS, store=lp.result_path(HERE, 'coupled', study))
    print(res)
    modes = lp.Modes(scene).solve(f_max=FREQS[-1] * 1.5)
    modes.save(HERE, name='coupled_modes')
    print(f"{len(modes)} coupled modes ≤ {FREQS[-1] * 1.5:.0f} Hz")
