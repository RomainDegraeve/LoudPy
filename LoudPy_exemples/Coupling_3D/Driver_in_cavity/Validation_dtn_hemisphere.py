"""
Validation_dtn_hemisphere.py — exterior FE + half-space DtN (Exterior_dtn) vs analytical
radiation of a hemisphere (radius a) on a rigid ground = sphere in free field:
  - pulsating   (u_r = U):            p = A h_0(kr),                 A k h_0'(ka) = ρω²U
  - oscillating along the ground (x): p = A h_1(kr) cos ψ,  cos ψ = x/r,  A k h_1'(ka) = ρω²U
FE domain a ≤ r ≤ R with the DtN on r = R; the pressure is checked on Γ (r = R) and, through
the DtN expansion, at r = 1 m.
"""
import numpy as np

from loudpy.Studies.Cavity_coupling import _gmsh, surface_quadrature, RHO_AIR, C_AIR
from loudpy.Studies.Exterior_dtn import HalfSpaceDtN, ExteriorFE, _h2

A_IN, R_OUT, H_MESH = 0.10, 0.30, 0.022
FREQS = [100.0, 1000.0, 3000.0]
msh = "LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Files/dtn_shell.msh"

gmsh = _gmsh()
gmsh.clear()
occ = gmsh.model.occ
outer = occ.addSphere(0, 0, 0, R_OUT)
half = occ.addBox(-1, 0, -1, 2, 1, 2)
hemi, _ = occ.intersect([(3, outer)], [(3, half)])
shell, _ = occ.cut(hemi, [(3, occ.addSphere(0, 0, 0, A_IN))])
occ.synchronize()
vol = shell[0][1]
groups = {"dtn": [], "inner": [], "ground": []}
for _, t in gmsh.model.getBoundary([(3, vol)], oriented=False):
    b = gmsh.model.getBoundingBox(2, t)
    if gmsh.model.getType(2, t) == "Sphere":
        groups["dtn" if max(abs(v) for v in b) > 0.2 else "inner"].append(t)
    else:
        groups["ground"].append(t)
gmsh.model.addPhysicalGroup(3, [vol], 1, name="air")
for k, (n, ts) in enumerate(groups.items(), start=2):
    gmsh.model.addPhysicalGroup(2, ts, k, name=n)
for k, v in (("Mesh.MeshSizeMax", H_MESH), ("Mesh.MeshSizeMin", H_MESH * 0.5), ("Mesh.ElementOrder", 2),
             ("Mesh.SecondOrderLinear", 1)):
    gmsh.option.setNumber(k, v)
gmsh.model.mesh.generate(3)
gmsh.write(msh)
gmsh.clear()

dtn = HalfSpaceDtN((0, 0, 0), R_OUT, up=1)
ext = ExteriorFE(msh, dtn, max(FREQS), front="inner")
xq, wq, inodes, W = surface_quadrature(ext.nodes, ext.faces["inner"])
rq = np.linalg.norm(xq, axis=1)
far = np.array([[np.cos(a), np.sin(a) * 0.6, np.sin(a) * 0.8] for a in np.linspace(0, np.pi, 7)])  # r = 1 m
gn = ext.gnodes
print(f"{len(ext.nodes)} nodes, {len(gn)} on Γ")
for f in FREQS:
    w, k = 2 * np.pi * f, 2 * np.pi * f / C_AIR
    for name, n, un in (("pulsating", 0, np.ones(len(xq))), ("oscillating", 1, xq[:, 0] / rq)):
        rhs = np.zeros(len(ext.nodes), complex)
        rhs[inodes] = -RHO_AIR * w ** 2 * (W.T @ (wq * un))
        p, it, info = ext.solve(k, rhs, rtol=1e-10)
        A = RHO_AIR * w ** 2 / (k * _h2(n, k * A_IN, True))
        ang = lambda x: np.ones(len(x)) if n == 0 else x[:, 0] / np.linalg.norm(x, axis=1)
        x_g = ext.nodes[gn]
        exact_g = A * _h2(n, k * np.linalg.norm(x_g, axis=1)) * ang(x_g)
        exact_f = A * _h2(n, k * 1.0) * ang(far)
        p_far = dtn.pressure(far, ext.coefficients(p, k), k)
        eg = np.linalg.norm(p[gn] - exact_g) / np.linalg.norm(exact_g)
        ef = np.linalg.norm(p_far - exact_f) / np.linalg.norm(exact_f)
        print(f"  {f:6.0f} Hz {name:11s}: GMRES {it:3d} it, error on Γ {100 * eg:.3f} %, at 1 m {100 * ef:.3f} %")
