"""
cabinet.py — the cabinet of this example as built: six panels, two labyrinth boards and the opening's air, each its
own solid in a STEP file (cabinet.step), the way a CAD model of a real box comes — loudpy takes each solid as a wall
panel and measures its thickness; the opening's solid is air (named by a point in it: lp.cabinets.Step openings=).

    python3 cabinet.py          writes cabinet.step again (run.py does it when the file is missing)

Solids, in the file's order (their keys in scene.walls(panels=…)):
  0 bottom (the box stands on it), 1 left, 2 right, 3 top, 4 back, 5 front (18 mm MDF, its outer face the driver's
  front plane; the driver's cut-out through it behind the driver's flange plane, the opening's slot through it),
  6 upper board, 7 lower board, 8 the opening (air)

The front and the driver: the driver's model (02) meets the cabinet on two planes 1 mm apart, its flange (Z_B, the
rear baffle plane, and Z_F, the front face). The front panel is FRONT thick from its inner face up to Z_F; behind Z_B
it has the driver's cut-out (a hole of radius HOLE, the basket's room), and the last millimetre (Z_B … Z_F) is where
loudpy cuts the driver itself. lp.cabinets.Step checks this: the point just behind the baffle plane must be in the
inside air (here: in the hole), the driver's front on the outer face.

The labyrinth: the driver radiates into the top chamber; the air goes down past the right end of the upper board
(fixed to the left side), back past the left end of the lower board (fixed to the right side), and out through the
slot at the bottom of the front.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
from enclosure import X0, Z_B, Z_F  # noqa: E402

STEP = HERE / "cabinet.step"
W, H, D = 0.20, 0.30, 0.18                    # inside: width (x), height (y), depth (z) [m]
BOTTOM, SIDE, TOP, BACK = 0.018, 0.015, 0.015, 0.022    # panel thicknesses [m]
FRONT = 0.018                                 # the front panel (MDF), its outer face on the driver's front plane [m]
Z_IN = Z_F - FRONT                            # the front panel's inner face (z) [m]
HOLE = 0.058                                  # the driver's cut-out behind its flange (radius) [m]: > its 57.35 mm
BOARD, GAP = 0.012, 0.05                      # labyrinth boards: thickness, the passage at their free end [m]
UPPER, LOWER = BOTTOM + 0.150, BOTTOM + 0.080 # the boards' lower faces (y) [m]
SLOT = (0.10, BOTTOM + 0.015, 0.08, 0.04)     # the opening in the front: x, y, width, height [m]
DRIVER_Y = BOTTOM + 0.232                     # the driver's axis (y) [m]: above the upper board
OPENING = (SLOT[0] + SLOT[2] / 2, SLOT[1] + SLOT[3] / 2, (Z_IN + Z_F) / 2)  # a point of the opening's air
PANEL_NAMES = {0: "bottom", 1: "left", 2: "right", 3: "top", 4: "back", 5: "front", 6: "upper board",
               7: "lower board"}                # the STEP solids (labels for the figures only)


def write_step(path=STEP):
    """Write the cabinet's solids (in metres, the file order above); the inside air is x 0…W, y BOTTOM…BOTTOM+H,
    z Z_IN−D…Z_IN, and the driver's cut-out in the front."""
    import gmsh
    owner = not gmsh.isInitialized()
    if owner:
        gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")
    gmsh.model.add("cabinet")
    occ = gmsh.model.occ
    z0, length = Z_IN - D - BACK, D + BACK + FRONT             # from the back face to the front face
    occ.addBox(-SIDE, 0.0, z0, W + 2 * SIDE, BOTTOM, length, tag=1)            # 0 bottom
    occ.addBox(-SIDE, BOTTOM, z0, SIDE, H + TOP, length, tag=2)                # 1 left
    occ.addBox(W, BOTTOM, z0, SIDE, H + TOP, length, tag=3)                    # 2 right
    occ.addBox(0.0, BOTTOM + H, z0, W, TOP, length, tag=4)                     # 3 top
    occ.addBox(0.0, BOTTOM, z0, W, H, BACK, tag=5)                             # 4 back
    panel = occ.addBox(0.0, BOTTOM, Z_IN, W, H, FRONT, tag=20)                # 5 front …
    hole = occ.addCylinder(X0, DRIVER_Y, Z_IN, 0.0, 0.0, Z_B - Z_IN, HOLE, tag=21)   # the driver's cut-out
    slot = occ.addBox(SLOT[0], SLOT[1], Z_IN, SLOT[2], SLOT[3], FRONT, tag=9)  # 8 the opening's air
    occ.cut([(3, panel)], [(3, hole), (3, slot)], tag=6, removeTool=False)     # … with them through it
    occ.remove([(3, hole)])
    occ.addBox(0.0, UPPER, Z_IN - D, W - GAP, BOARD, D, tag=7)                 # 6 upper board (from the left)
    occ.addBox(GAP, LOWER, Z_IN - D, W - GAP, BOARD, D, tag=8)                 # 7 lower board (from the right)
    occ.synchronize()
    gmsh.write(str(path))
    gmsh.model.remove()
    if owner:
        gmsh.finalize()
    return Path(path)


if __name__ == "__main__":
    print("✓", write_step())
