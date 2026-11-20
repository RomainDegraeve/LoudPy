"""
Running_script.py — HPNEW driver (azimuthal-order modes m = 0 … 3) coupled to the modes of a
non-axisymmetric 3D enclosure, forced by its voice coil (../../Eigen_study/Fourier/Running_script.py
first). The front of the driver is in vacuo (only the enclosure loads it).

  scene    : enclosure.sealed_scene() — lp.Scene(sealed_cabinet, [driver at (X0, Y0)]).cavity(...)
             the cavity mesh (driver cut out with its revolved wet rear profile) is built on the
             first run, its rigid-wall modes are cached next to it (cavity_modes.h5)
  response : lp.FrequencyResponse — 1 N on the coil (axisymmetric: it drives m = 0 only; m ≥ 1
             are driven through the enclosure); coupling S_kn = ∫ (φ_k·n) ψ_n dS on the wet rear
             surface + static correction of the truncated cavity modes
  modes    : lp.Modes — coupled driver–enclosure modes

What this produces (Results/Files/)
------------------------------------
  - cavity.msh, cavity_modes.h5 : enclosure mesh and its modes (reused on reruns)
  - coupled.h5                  : forced response (driver modal amplitudes, modal pressures)
  - coupled_modes.h5            : coupled modes (complex s, driver and cavity components)
Then run Loading_script.py and Plot_3D_coupled.py.
"""
from pathlib import Path

import numpy as np

import loudpy as lp

lp.set_verbosity("info")                 # progress messages (loudpy is silent by default)
from enclosure import FILES, sealed_scene

files = Path(FILES)
FREQS = np.geomspace(20.0, 3000.0, 700)        # forced response band [Hz]

scene = sealed_scene()                         # enclosure.py: cabinet, driver, cavity settings
print(scene)

res = lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve(FREQS, store=files / "coupled.h5")
print(res, "→", files / "coupled.h5")

modes = lp.Modes(scene).solve(f_max=FREQS[-1] * 1.5)
modes.save(files / "coupled_modes.h5")
print(f"{len(modes)} coupled modes ≤ {FREQS[-1] * 1.5:.0f} Hz → {files / 'coupled_modes.h5'}")
