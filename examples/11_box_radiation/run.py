"""
run.py — HPNEW in its enclosure, standing on a rigid ground, radiating into the
half-space: driver (Fourier modes m = 0 … 3) + enclosure (cavity modes, rear) + exterior air
(3D FE with the exact half-space DtN on a hemisphere, front), forced by 1 N on the voice coil.

  scene : enclosure.exterior_scene() — lp.Scene(sealed_cabinet, …).cavity(…).exterior(…);
          the exterior meshes (one per band, hemisphere minus the box and the driver front) are
          built on the first run and reused afterwards

What this produces (Results/Files/)
------------------------------------
  - exterior_<f>Hz.msh : the exterior mesh of each band
  - exterior.h5        : driver modal amplitudes, cavity modal pressures, DtN coefficients and the
                         exterior pressure at every node (streamed during the sweep)
Then run plot.py, plot_3d.py and plot_3d_views.py.

The scene is built at the top (not meshed); meshing and solving run only when this file is run (python3 run.py),
so the interface opens it without computing (python -m loudpy.ui on this folder or file).
"""
import sys
from pathlib import Path

import numpy as np

import loudpy as lp

lp.set_verbosity("info")                 # progress messages (loudpy is silent by default)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))   # the shared box
from enclosure import MESHES, X0, Y0, exterior_scene  # noqa: E402

files = Path(__file__).resolve().parent / "Results/Files"     # this example's results
files.mkdir(parents=True, exist_ok=True)
FREQS = np.arange(20.0, 3000.0 + 1e-9, 10.0)   # [Hz]
TAG = "exterior"
scene = exterior_scene(mesh=False)

if __name__ == "__main__":
    if len(sys.argv) > 1:                          # quick test: python3 run.py 50,500
        FREQS, TAG = np.array([float(x) for x in sys.argv[1].split(",")]), "exterior_test"
    scene.mesh(MESHES)
    print(scene, f"| hemisphere {np.round(scene.hemisphere['center'], 3)} m, radius {scene.hemisphere['radius']:.3f} m")
    res = lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve(FREQS, store=files / f"{TAG}.h5")

    # check at low frequency: compact source on a rigid ground, far field p·r → ρω²|V_front| / 2π
    i0 = int(np.argmin(np.abs(FREQS - 50.0)))
    far = np.array([[X0 + 20 * np.sin(a), Y0, 20 * np.cos(a)] for a in np.deg2rad([0, 90, 180])])
    pr = np.abs(res.far_field(far)[i0]) * 20.0
    print(f"Far field at {FREQS[i0]:.0f} Hz, 20 m, 0/90/180°: p·r = {np.round(pr, 5)} Pa·m (omnidirectional: "
          f"spread {100 * (pr.max() / pr.min() - 1):.1f} %)")
    print(res, "→", files / f"{TAG}.h5")
