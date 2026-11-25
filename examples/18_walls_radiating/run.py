"""
run.py — elastic walls in a labyrinth that radiates: the HPNEW driver in a cabinet drawn as solids (a STEP file —
cabinet.py): six panels, two labyrinth boards folding the path from the driver down to a slot at the bottom of the
front, and the slot's air. Every panel and board is a shell (MITC3) of its own measured thickness, the back in
plywood; the air is ONE domain — inside the labyrinth, through the slot, outside on a rigid ground (3D FE + exact
half-space DtN) — and every wet face of the walls moves with its shell (rigid through the thickness): the boards on
both faces, the panels inside and out. Needs 02_azimuthal_modes (driver modes).

  cabinet  : lp.cabinets.Step("cabinet.step", openings=[a point in the slot]) — solids 0 bottom, 1 left, 2 right,
             3 top, 4 back, 5 front, 6 upper board, 7 lower board (8: the slot, air)
  walls    : scene.walls("MDF", thickness=None, panels={4: Panel("Plywood")})
             thickness=None: each panel's measured through its solid (bottom and front 18, sides and top 15, back
             22, boards 12 mm); the front has the driver's cut-out behind the driver's flange plane (cabinet.py)
  air      : lp.Scene.vented_box — one merged air domain (inside, the slot, outside), 1 kHz bands
  studies  : the in-vacuo wall modes; the response to 1 N on the coil, 20 Hz – 1 kHz, with these walls and with rigid
             walls (the same scene without walls) for comparison

    python3 examples/18_walls_radiating/run.py            the full band (not timed yet: about an hour)
    python3 examples/18_walls_radiating/run.py --smoke    three frequencies

What this produces (the results <study>.h5 below: results/18_walls_radiating/<study>_<stamp>.h5,
stamped — plot.py reads the newest; meshes and caches: Results/Files/)
----------------------------------------------------------------------
  walls_modes.csv   the wall modes (frequency, damping ratio, volume displacement per unit modal amplitude)
  elastic.h5        driver and wall modal amplitudes, DtN coefficients, the pressure in the air everywhere
  rigid.h5          the same with rigid walls
Then run plot.py. 18_walls_radiating.loudpy.json is the same simulation for the interface
(python -m loudpy.ui examples/18_walls_radiating).

The scene is built at the top (not meshed); meshing and solving run only when this file is run (python3 run.py),
so the interface opens it without computing.
"""
import sys
from pathlib import Path

import numpy as np

import loudpy as lp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))      # the driver of examples 10–15
sys.path.insert(0, str(HERE))
from enclosure import X0, driver  # noqa: E402
from cabinet import DRIVER_Y, OPENING, STEP, write_step  # noqa: E402

lp.set_verbosity("info")
files = HERE / "Results/Files"
MESHES = HERE / "meshes"
F_MAX = 1000.0                                       # [Hz]: one air band; wall modes to 2 kHz
FREQS = np.geomspace(20.0, F_MAX, 150)               # [Hz]
PANELS = {4: lp.cabinets.Panel("Plywood")}           # the back in plywood (its measured 22 mm); the rest MDF
if not STEP.exists():
    write_step(STEP)


def make_scene(elastic=True):
    """The driver on the front panel above the upper board, the labyrinth's air merged with the outside through the
    slot; elastic walls with their panels, or rigid ones."""
    cabinet = lp.cabinets.Step(STEP, openings=[OPENING])
    scene = lp.Scene.vented_box(cabinet, driver(), at=(X0, DRIVER_Y), f_max=F_MAX)
    if elastic:
        scene.walls("MDF", thickness=None, panels=PANELS)   # lp.Rayleigh() damping of each material's fit
    return scene


scene = make_scene()

if __name__ == "__main__":
    if "--smoke" in sys.argv:
        FREQS = np.array([60.0, 300.0, 800.0])
    files.mkdir(parents=True, exist_ok=True)
    scene.mesh(MESHES)
    print(scene)
    walls = lp.Modes(scene, physics="walls").solve(f_max=2 * F_MAX)
    np.savetxt(files / "walls_modes.csv", np.column_stack([walls.frequencies, walls.extra("zeta"),
                                                           walls.extra("volume_disp")]),
               delimiter=",", header="f_Hz,zeta,volume_displacement_m3", comments="")
    print(f"{len(walls.frequencies)} wall modes, first {np.round(walls.frequencies[:6], 1)} Hz → walls_modes.csv")
    study = lp.FrequencyResponse(scene, drive=lp.Force(1.0))
    res = study.solve(FREQS, store=lp.result_path(HERE, 'elastic', study))
    print(res)
    rigid = make_scene(elastic=False)
    rigid.mesh(MESHES)                               # the same meshes (reused)
    study = lp.FrequencyResponse(rigid, drive=lp.Force(1.0))
    res = study.solve(FREQS, store=lp.result_path(HERE, 'rigid', study))
    print(res)
