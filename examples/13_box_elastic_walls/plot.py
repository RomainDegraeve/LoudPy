"""
plot.py — elastic MDF walls against the rigid box (run.py and 10_box_cavity/run.py first).

What this produces (Results/Figures/)
-------------------------------------
  walls_response.pdf   coil excursion with rigid / elastic walls, and the walls' volume velocity into the box
                       relative to the driver's rear volume velocity (the panel modes marked)
"""
import sys
from pathlib import Path

import numpy as np

import loudpy as lp
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from names import stored  # noqa: E402  (results where run.py writes them, or stored earlier)
HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
lp.plot.style.use()                                  # the loudpy figure style for this script

elastic = stored(HERE, "coupled_walls")
rigid = stored(EXAMPLES / "10_box_cavity", "coupled")     # the same box with rigid walls
x_e = elastic.driver("main").coil_excursion().abs().rename(label="MDF walls, 12 mm")
x_r = rigid.driver("main").coil_excursion().abs().rename(label="rigid walls")
ratio = lp.Data(20 * np.log10(np.abs(elastic.wall_volume_velocity().values)
                              / np.abs(elastic.driver_volume_velocity().values)),
                {"f": elastic.frequencies}, "level", label="walls / driver")
change = np.abs(x_e.values / x_r.values - 1).max() * 100
print(f"walls / driver volume velocity: max {ratio.values.max():.1f} dB; coil excursion change: max {change:.1f} %")

fig, ax = lp.plot.figure("tall", 2, 1, sharex=True)
lp.plot.line([x_r, x_e], ax=ax[0])
ratio.plot(ax=ax[1])
for f in elastic.wall_frequencies[elastic.wall_frequencies <= elastic.frequencies[-1]]:
    ax[1].axvline(f, color="0.8", lw=0.5, zorder=0)
ax[1].set_ylabel(r"$|Q_\mathrm{walls}| / |Q_\mathrm{driver}|$ [dB]")
ax[0].set_xlabel("")
lp.plot.save(fig, HERE / "Results/Figures/walls_response.pdf")
print("✓ figure →", HERE / "Results/Figures/walls_response.pdf")
