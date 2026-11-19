# Eigen_study/Fourier — non-axisymmetric modes of an axisymmetric structure

The modes of an axisymmetric structure separate into azimuthal orders m:

    u_r = U_r(r,z) cos mθ,   u_z = U_z(r,z) cos mθ,   u_θ = U_θ(r,z) sin mθ

| m | modes | excited by |
|---|---|---|
| 0 | pistonic, breakup (axisymmetric) | the voice coil |
| 1 | rocking | any asymmetric load: box, neighbouring drivers, off-axis mass |
| ≥ 2 | ovalisation, circumferential modes | asymmetric acoustic loading |

`EigenStudy.solve_meca_eigen_fourier(m, f_min, f_max, n_modes_per_shift)` solves order m on
the usual meridian mesh: 3 DOF per node (U_r, U_z, U_θ) for m ≥ 1, same multishift solver,
Rayleigh / hysteretic damping and normalisation as the axisymmetric study (m = 0 calls it).
The axis conditions are set automatically (m = 1: U_z = 0, U_θ = −U_r; m ≥ 2: all zero);
`_rz_` clamps also block U_θ. Results are saved as `modes/meca` (m = 0) and
`modes/meca_m<m>`; read them with `EigenReader.load_fourier(m)` / `fourier_orders()`.

| file | does |
|---|---|
| `Running_script.py` | HPNEW loudspeaker in vacuo, m = 0 … 4, 20 Hz – 16 kHz → `Results/Files/eigen_fourier.h5` |
| `Loading_script.py` | `mode_chart.pdf` (f vs m), `modes_m<m>.pdf` (meridian shapes) |
| `Plot_3D_modes.py` | `modes_3d_m<m>.pdf`, m = 0 … 3: every mode on the whole revolved structure, 3D cut-away view + top view, max deformation 10 % of the diameter |
| `Validation_clamped_plate.py` | clamped circular steel plate vs Kirchhoff analytical modes, m = 0 … 4 |

Validation (clamped plate, a = 0.1 m, h = 2 mm): all (m, n) modes within 0.6 % of thin-plate
theory, the FE being slightly lower as expected (shear, rotary inertia).

HPNEW: m = 0 identical to `../Meca`; first rocking mode (m = 1) at 171 Hz (pistonic 115 Hz).
