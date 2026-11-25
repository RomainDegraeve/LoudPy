"""
plot.py — the elastic walls of the labyrinth against rigid ones (run.py first).

What this produces
------------------
Results/Figures/
  - walls_radiating.pdf : SPL 1 m in front of the driver (exact DtN expansion), the coil excursion, the walls'
                          volume velocity relative to the driver's (the wall resonances marked), and the RMS
                          normal displacement of each panel
Results/Files/
  - scene_<f>Hz.html    : the SPL in the air and the deformed walls at the strongest wall response (interactive 3D)
"""
import sys
from pathlib import Path

import numpy as np

import loudpy as lp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
sys.path.insert(0, str(HERE))
from enclosure import X0, Z_F  # noqa: E402
from cabinet import DRIVER_Y, PANEL_NAMES  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from names import stored  # noqa: E402  (results where run.py writes them, or stored earlier)
lp.plot.style.use()                                  # the loudpy figure style for this script

files = HERE / "Results/Files"
elastic = stored(HERE, "elastic")
rigid = stored(HERE, "rigid")
ahead = (X0, DRIVER_Y, Z_F + 1.0)                    # 1 m in front of the driver, on its axis [m]
spl_e = elastic.far_field(ahead).spl().rename(label="elastic walls (MDF, plywood back)")
spl_r = rigid.far_field(ahead).spl().rename(label="rigid walls")
x_e = elastic.driver("main").coil_excursion().abs().rename(label="elastic walls")
x_r = rigid.driver("main").coil_excursion().abs().rename(label="rigid walls")
ratio = lp.Data(20 * np.log10(np.abs(elastic.wall_volume_velocity().values)
                              / np.abs(elastic.driver_volume_velocity().values)),
                {"f": elastic.frequencies}, "level", label="walls / driver")
moving = [elastic.wall_displacement(panel=p).rename(label=PANEL_NAMES.get(int(p), p) if p.isdigit() else p)
          for p in elastic.wall_panels]                # RMS normal displacement of each panel [m]
print(f"SPL at 1 m: elastic − rigid within {np.abs(spl_e.values - spl_r.values).max():.2f} dB; walls / driver "
      f"volume velocity: max {ratio.values.max():.1f} dB")

fig, ax = lp.plot.figure("tall", 4, 1, sharex=True)
lp.plot.line([spl_r, spl_e], ax=ax[0])
lp.plot.line([x_r, x_e], ax=ax[1])
ratio.plot(ax=ax[2])
for f in elastic.wall_frequencies[elastic.wall_frequencies <= elastic.frequencies[-1]]:
    ax[2].axvline(f, color="0.8", lw=0.5, zorder=0)
ax[2].set_ylabel(r"$|Q_\mathrm{walls}| / |Q_\mathrm{driver}|$ [dB]")
lp.plot.line(moving, ax=ax[3])
ax[3].set_yscale("log")
for a in ax[:3]:
    a.set_xlabel("")
lp.plot.save(fig, HERE / "Results/Figures/walls_radiating.pdf")
print("✓ figure →", HERE / "Results/Figures/walls_radiating.pdf")

f = float(elastic.frequencies[int(np.argmax(ratio.values))])
view = elastic.view(f)
view.save(files / f"scene_{f:.0f}Hz.html", title=f"elastic walls radiating, {f:.0f} Hz")
print("✓ 3D view →", files / f"scene_{f:.0f}Hz.html")
