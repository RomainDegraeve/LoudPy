"""
rect_cavity.py — 3D P2 acoustic FE of CavityModes vs the analytical modes of a
rigid rectangular box,  f = c/2 · sqrt((nx/Lx)² + (ny/Ly)² + (nz/Lz)²).
"""
from pathlib import Path
import itertools
import numpy as np
import gmsh

from loudpy.physics.modal import CavityModes
import loudpy as lp

HERE = Path(__file__).resolve().parent               # this example's folder
EXAMPLES = HERE.parent                               # examples/

LX, LY, LZ, H, F_MAX = 0.30, 0.20, 0.14, 0.012, 3000.0
msh = str(HERE / "Results/Files/rect_box.msh")

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("rect")
v = gmsh.model.occ.addBox(0, 0, 0, LX, LY, LZ)
gmsh.model.occ.synchronize()
faces = [t for _, t in gmsh.model.getBoundary([(3, v)], oriented=False)]
gmsh.model.addPhysicalGroup(3, [v], 1, name="cavity")
gmsh.model.addPhysicalGroup(2, faces[:1], 2, name="interface_driver")
for k, val in (("Mesh.MeshSizeMax", H), ("Mesh.MeshSizeMin", H), ("Mesh.ElementOrder", 2),
               ("Mesh.SecondOrderLinear", 1)):
    gmsh.option.setNumber(k, val)
gmsh.model.mesh.generate(3)
import os; os.makedirs(os.path.dirname(msh), exist_ok=True)
gmsh.write(msh)
gmsh.finalize()

cav = CavityModes.from_msh(msh)
f_fe = cav.solve(F_MAX)
f_an = np.sort([lp.materials["Air"].c / 2 * np.sqrt((a / LX) ** 2 + (b / LY) ** 2 + (c / LZ) ** 2)
                for a, b, c in itertools.product(range(12), repeat=3)])
f_an = f_an[f_an <= F_MAX * 1.02][:len(f_fe)]
n = min(len(f_fe), len(f_an))
err = (f_fe[1:n] - f_an[1:n]) / f_an[1:n] * 100
print(f"{len(cav.nodes)} nodes, V = {cav.volume * 1e3:.3f} L (exact {LX * LY * LZ * 1e3:.3f} L), "
      f"{len(f_fe)} modes ≤ {F_MAX:.0f} Hz (analytical {np.sum(f_an <= F_MAX)})")
for i in range(1, min(n, 16)):
    print(f"  {i:3d}  FE {f_fe[i]:8.2f} Hz   exact {f_an[i]:8.2f} Hz   {err[i - 1]:+.3f} %")
print(f"max |error| over {n - 1} modes: {np.abs(err).max():.3f} %")
