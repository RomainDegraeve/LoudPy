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

from loudpy.Studies import Problem, EigenStudy
from loudpy import DomainSpecMecaRayleigh, InterfaceSpecClamped

geo_path = "LoudPy_exemples/Geometries/Plate/clamped_plate.geo"
msh_path = "LoudPy_exemples/Eigen_study/Fourier/Results/Files/plate_mesh.msh"

E, nu, rho, a, h = 2.1e11, 0.3, 7850.0, 0.1, 0.002
# λ²_mn of the clamped circular plate (Leissa, Vibration of Plates), n = 1, 2
LAMBDA2 = {(0, 1): 10.2158, (0, 2): 39.771, (1, 1): 21.260, (1, 2): 60.829,
           (2, 1): 34.877, (2, 2): 84.583, (3, 1): 51.030, (4, 1): 69.666}

problem = Problem(geo_path=geo_path, msh_path=msh_path)
problem.add_sub_domain(DomainSpecMecaRayleigh("plate", material="Steel", E=E, nu=nu, rho=rho,
                                              alpha_ray=0.0, beta_ray=0.0, size=h / 3))
problem.add_interface(InterfaceSpecClamped("interface_constrained"))
problem.mesh(show_mesh_gui=False, reuse_mesh=True)

D = E * h ** 3 / (12 * (1 - nu ** 2))
scale = np.sqrt(D / (rho * h)) / (2 * np.pi * a ** 2)
study = EigenStudy(problem)
print(f"\n{'m':>2} {'n':>2} {'analytical [Hz]':>16} {'FE [Hz]':>10} {'diff [%]':>9}")
worst = 0.0
for m in range(5):
    f, _, modes = study.solve_meca_eigen_fourier(m, f_min=10.0, f_max=4000.0, n_modes_per_shift=10)
    # bending modes only: drop the in-plane / torsional ones (much higher here, or tiny U_z)
    if m == 0:
        bend = [k for k in range(len(f))]
    else:
        bend = [k for k in range(len(f))
                if np.linalg.norm(modes[k][1::3]) > 0.5 * np.linalg.norm(modes[k])]
    fb = np.sort(np.asarray(f)[bend])
    for n in (1, 2):
        if (m, n) in LAMBDA2 and len(fb) >= n:
            fa = LAMBDA2[(m, n)] * scale
            d = 100 * (fb[n - 1] - fa) / fa
            worst = max(worst, abs(d))
            print(f"{m:2d} {n:2d} {fa:16.2f} {fb[n - 1]:10.2f} {d:9.3f}")
print(f"\nworst difference: {worst:.3f} %")
