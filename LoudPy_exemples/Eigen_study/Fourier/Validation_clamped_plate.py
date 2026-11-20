"""
Validation_clamped_plate.py — azimuthal-order (Fourier) modes against the analytical
clamped circular plate.

Steel plate, radius a = 0.1 m, thickness h = 2 mm, clamped rim, meshed as an axisymmetric
3D-elasticity solid (Geometries/Plate/clamped_plate.geo). Kirchhoff thin-plate theory gives

    f_mn = λ²_mn / (2π a²) · sqrt(D / (ρ h)),   D = E h³ / (12 (1 − ν²))

with λ²_mn the clamped-plate eigenvalues (m nodal diameters, n nodal circles incl. rim).
The FE uses no plate assumption, so it is slightly lower (shear, rotary inertia): the
difference must stay well under 1 % for the low modes at h/a = 0.02.
"""
import numpy as np

import loudpy as lp

lp.set_verbosity("info")                 # progress messages (loudpy is silent by default)

geometry = "LoudPy_exemples/Geometries/Plate/clamped_plate.geo"
mesh_path = "LoudPy_exemples/Eigen_study/Fourier/Results/Files/plate_mesh.msh"

E, nu, rho, a, h = 2.1e11, 0.3, 7850.0, 0.1, 0.002
# λ²_mn of the clamped circular plate (Leissa, Vibration of Plates), n = 1, 2
LAMBDA2 = {(0, 1): 10.2158, (0, 2): 39.771, (1, 1): 21.260, (1, 2): 60.829,
           (2, 1): 34.877, (2, 2): 84.583, (3, 1): 51.030, (4, 1): 69.666}

steel = lp.Material("Steel", rho=rho, E=E, nu=nu)
model = lp.Model(geometry).add(
    lp.Solid("plate", steel, damping=lp.Undamped(), mesh_size=h / 3),
    lp.Clamped("interface_constrained"),
)
model.mesh(mesh_path)
modes = lp.Modes(model).solve(f_min=10.0, f_max=4000.0, orders=range(5), n_per_shift=10)

D = E * h ** 3 / (12 * (1 - nu ** 2))
scale = np.sqrt(D / (rho * h)) / (2 * np.pi * a ** 2)
print(f"\n{'m':>2} {'n':>2} {'analytical [Hz]':>16} {'FE [Hz]':>10} {'diff [%]':>9}")
worst = 0.0
for m in modes.orders:
    om = modes.order(m)
    U = om.blocks[0].fields["u"]                                        # (n_modes, n_nodes, dpn)
    # bending modes only: drop the in-plane / torsional ones (much higher here, or tiny U_z)
    bend = np.linalg.norm(U[..., 1], axis=1) > 0.5 * np.linalg.norm(U.reshape(len(U), -1), axis=1)
    fb = np.sort(om.frequencies if m == 0 else om.frequencies[bend])
    for n in (1, 2):
        if (m, n) in LAMBDA2 and len(fb) >= n:
            fa = LAMBDA2[(m, n)] * scale
            d = 100 * (fb[n - 1] - fa) / fa
            worst = max(worst, abs(d))
            print(f"{m:2d} {n:2d} {fa:16.2f} {fb[n - 1]:10.2f} {d:9.3f}")
print(f"\nworst difference: {worst:.3f} %")
