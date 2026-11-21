"""
run.py — the sealed box of 10_box_cavity with ELASTIC walls: MDF shells (MITC3) on the walls of the cavity mesh,
panel edges clamped, coupled to the cavity air through their in-vacuo modes. The driver front and the outside of
the walls are in vacuo (cavity configuration). Needs 02_azimuthal_modes (driver modes).

  scene    : enclosure.sealed_scene() + scene.walls("MDF", thickness=WALL)
  walls    : lp.Modes(scene, physics="walls") — the in-vacuo wall (panel) modes
  response : lp.FrequencyResponse, 1 N on the coil, 20 Hz – 3 kHz (the band of 10_box_cavity)

    python3 examples/13_box_elastic_walls/run.py            full band (~8 min)
    python3 examples/13_box_elastic_walls/run.py --smoke    four frequencies

What this produces (Results/Files/)
------------------------------------
  walls_modes.csv     the wall modes (frequency, damping ratio, volume displacement per unit modal amplitude)
  coupled_walls.h5    the response with elastic walls (driver and wall modal amplitudes)
Then run plot.py (it compares with the rigid box of 10_box_cavity).

The scene is built at the top (not meshed); meshing and solving run only when this file is run (python3 run.py),
so the interface opens it without computing (python -m loudpy.ui on this folder or file).
"""
import sys
from pathlib import Path

import numpy as np

import loudpy as lp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))      # the shared box
from enclosure import MESHES, sealed_scene  # noqa: E402

lp.set_verbosity("info")
files = HERE / "Results/Files"
files.mkdir(parents=True, exist_ok=True)
WALL = 0.012                                         # MDF wall thickness [m] (thin, to show the panel modes)
FREQS = np.geomspace(20.0, 3000.0, 700)              # the band of 10_box_cavity [Hz]
scene = sealed_scene(mesh=False)
scene.walls("MDF", thickness=WALL)                   # lp.Rayleigh() damping of the MDF fit, panel edges clamped

if __name__ == "__main__":
    if "--smoke" in sys.argv:
        FREQS = np.array([50.0, 200.0, 800.0, 1500.0])
    scene.mesh(MESHES)
    print(scene)
    walls = lp.Modes(scene, physics="walls").solve(f_max=FREQS[-1] * 2)
    np.savetxt(files / "walls_modes.csv", np.column_stack([walls.frequencies, walls.extra("zeta"),
                                                           walls.extra("volume_disp")]),
               delimiter=",", header="f_Hz,zeta,volume_displacement_m3", comments="")
    print(f"{len(walls.frequencies)} wall modes, first {np.round(walls.frequencies[:6], 1)} Hz → walls_modes.csv")
    res = lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve(FREQS, store=files / "coupled_walls.h5")
    print(res, "→", files / "coupled_walls.h5")
