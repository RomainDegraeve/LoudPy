# Coupling_3D/Driver_in_cavity — axisymmetric driver in a non-axisymmetric 3D enclosure

The driver's azimuthal-order (Fourier) modes, for m = 0 … 3, are coupled to the rigid-wall acoustic modes of a 3D enclosure of any shape, by modal superposition (`loudpy.physics.modal`, through `lp.Scene(…).cavity()`).
- The driver stays a cheap 2D meridian model.
- Only the cavity needs 3D FEM (P2 tetrahedra, computed once).
- The front of the driver is in vacuo.

| file | does |
|---|---|
| `Validation_rect_cavity.py` | 3D cavity FEM vs the analytical modes of a rectangular box. Max error 0.02 % over 43 modes ≤ 3 kHz. |
| `enclosure.py` | Demo enclosure: 8.4 L, slanted back, internal brace, off-centre HPNEW. Cabinet builders for `lp.Scene` (sealed, vented), `driver()`, and the configured scenes `sealed_scene()`, `exterior_scene()`, `vented_scene()`. |
| `Running_script.py` | Cavity modes (≤ 6 kHz), Fourier driver modes (≤ 10 kHz, cos and sin families), coupling matrix plus static correction. Then the forced response to 1 N on the coil and the coupled modes (`lp.FrequencyResponse`, `lp.Modes` on the scene). Builds `cavity.msh` on the first run; writes `coupled.h5` and `coupled_modes.h5`. |
| `Loading_script.py` | `response.pdf`, `coupled_modes.pdf`, `cavity_modes.pdf`, `deformation_3d.pdf`. |
| `Plot_3D_coupled.py` | `coupled_3d/coupled_3d_<f0>-<f1>Hz.pdf`: every 10 Hz, 20–3000 Hz. Left: enclosure cut through the driver axis, coloured by pressure, with the deformed driver in place. Right: the driver from the front. Both at the same instant. |

Run `../../Eigen_study/Fourier/Running_script.py` first; it produces the driver modes.

## Equations
q are the modal amplitudes of the driver and P the modal pressures of the cavity:

    (ω_k² + 2iζ_kω_kω − ω²) q_k − ρω² (R q)_k + Σ_n S_kn P_n = F_k
    (ω_n² + iηω_nω − ω²) P_n + ρc²ω² Σ_k S_kn q_k = 0

- S_kn = ∫_S (φ_k·n) ψ_n dS is integrated on the wet rear surface.
- R is the static correction for the truncated cavity modes (the added mass of the near-field air).
- The coil force is axisymmetric, so it only drives m = 0. The m ≥ 1 modes are driven through the asymmetry of the enclosure.

## Checks (demo enclosure)
- **Pistonic mode:** 115.1 Hz in vacuo becomes 120.6 Hz coupled. The air-spring estimate √(f_s² + ρc²Q²/V / 4π²) gives 121.6 Hz; the difference is the added air mass.
- **Below the first enclosure mode (466 Hz):** m ≥ 1 is not excited. Uniform pressure exerts no net force on m ≥ 1.
- **At the first enclosure mode:** the non-axisymmetric orders reach 13 % of the driver's kinetic energy (rocking, m = 1).

## Simplifications of the demo
- The spider, former and coil are acoustically transparent.
- The pocket under the dust cap is closed by a rigid lid at the cone foot.
- Rigid walls, with a modal loss factor `ETA_AIR`.

## Exterior radiation: enclosure standing on a rigid ground
The front radiates into the half-space above a perfectly reflective ground:
- **Exterior air:** 3D P2 FE inside a hemisphere centred on the ground (radius 0.378 m).
- **Radiation condition:** the exact half-space DtN on that hemisphere (`loudpy.physics.exterior`, through `scene.exterior()`). Only the spherical harmonics with n + |m| even are kept, which is the image-source condition.
- **Coupling:** the driver is loaded by the enclosure modes (rear) and the exterior field (front) at once.
- **Solver:** per frequency, preconditioned GMRES, with one sparse LU per 50 Hz. The frequency range uses three meshes, sized for 1, 2 and 3 kHz.

| file | does |
|---|---|
| `enclosure.py` | Shared geometry of the box, driver and ground. |
| `Validation_dtn_hemisphere.py` | Exterior FE + DtN vs the analytical pulsating and oscillating hemisphere on a rigid ground. About 1 % at 100 Hz–3 kHz, on Γ and at 1 m. |
| `Running_exterior.py` | Coupled sweep 20–3000 Hz every 10 Hz, about 18 min. Builds the exterior meshes (hemisphere minus the box and the driver front, one per band) on the first run; writes `exterior.h5` (DtN coefficients, cavity pressures, exterior field). |
| `Loading_exterior.py` | `spl_1m.pdf`, `polars.pdf`, `response_change.pdf`. |
| `Plot_3D_exterior.py` | `exterior_3d/`: every 10 Hz, a vertical cut, a horizontal cut and the driver. |
| `Plot_3D_views.py` | `views_3d.pdf`: full 3D scenes from 6 angles at 100, 470, 1000, 1680 and 2400 Hz. Shows pressure on the ground, on the box walls and on the cuts, inside and outside, plus the deformed driver. |

Check: at 50 Hz the far field is omnidirectional, and p·r is within 1.4 % of a monopole on a rigid ground, ρω²V/2π.


## Sealed box with elastic walls (shells)
`Running_walls.py`: the sealed box of `Running_script.py` with MDF walls 12 mm thick (MITC3 shells on the walls of the cavity mesh, panel edges and the driver cut-out rim clamped, lp.Rayleigh damping of the MDF fit), coupled to the cavity air through their in-vacuo modes; the driver front and the outside of the walls are in vacuo. Writes `walls_modes.csv` (wall modes), `coupled_walls.h5` and `walls_response.png` (coil excursion with rigid / elastic walls; walls' volume velocity relative to the driver's rear volume velocity). `python3 Running_walls.py --smoke` runs 4 frequencies (about 7 min, mostly the cavity modes and the static correction).

## Vented box (bass reflex) on a rigid ground
Walls 15 mm thick, a port of radius 20 mm and length 62 mm through the front face (its tube protrudes into the box). Two models:

| command | model |
|---|---|
| `python3 Running_vented.py merged` | reference: `enclosure.vented_scene()`, ONE air domain (inside + port + outside, FE + half-space DtN), the driver wetted on both faces; the walls around the port are refined (`lp.Cabinet(opening_size=…)`). Writes `vented_merged.h5`. |
| `python3 Running_vented.py hybrid` | fast, low-level toolkit: modal inside + 1D port (plane waves) + exterior FE/DtN. Needs `Build_vented.py` (hybrid meshes) and `Running_script.py` (drive). Writes `vented_hybrid.npz`. |
| `Loading_vented.py` | `vented_response.pdf`: vented vs sealed SPL at 1 m, excursion (port tuning), directivity maps and polars. |
| `Plot_3D_vented.py` | `vented_views_3d.pdf`: the box cut through the driver and port axes, SPL inside and outside, deformed driver. |
