"""
Running_script.py — HPNEW driver (azimuthal-order modes m = 0 … 3) coupled to the modes of a
non-axisymmetric 3D enclosure, forced by its voice coil (Build_cavity.py and
../../Eigen_study/Fourier/Running_script.py first).

  1. cavity modes: 3D P2 acoustic FE of the enclosure (rigid walls), up to CAV_F_MAX
  2. driver modes: Fourier modes of orders ORDERS up to DRV_F_MAX, real and mass-normalised,
     cos and sin families
  3. coupling S_kn = ∫ (φ_k·n) ψ_n dS on the wet rear surface, + static correction of the
     truncated cavity modes
  4. forced response to 1 N on the voice coil (axisymmetric: it only drives m = 0 directly;
     m ≥ 1 are driven through the enclosure), with and without the enclosure
  5. coupled driver–enclosure modes

The front of the driver is in vacuo (only the enclosure loads it).

What this produces (Results/Files/)
------------------------------------
  - cavity_modes.h5   : cavity mesh, κ_n², ψ_n (reused on reruns)
  - coupled.npz       : frequencies, modal amplitudes, pressures, coupled modes, coupling data
Then run Loading_script.py.
"""
from pathlib import Path

import numpy as np

from loudpy.Studies.Cavity_coupling import (CavityModes, DriverModes, DriverCavityCoupling,
                                            sample_profile, RHO_AIR, C_AIR, _gmsh)

here = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity")
geo_path = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
eig_path = "LoudPy_exemples/Eigen_study/Fourier/Results/Files/eigen_fourier.h5"
msh_path = here / "Results/Files/cavity.msh"
cav_path = here / "Results/Files/cavity_modes.h5"
out_path = here / "Results/Files/coupled.npz"

# ── Settings (geometry: see Build_cavity.py) ───────────────────────────────────
X0, Y0, Z_B = 0.125, 0.190, 0.031             # driver axis and rear baffle plane [m]
PROFILE_CURVES, PROFILE_START = [14, 7, 8, 33, 32], (0.05435, Z_B)
COIL_CURVE = 37                                # forced edge of the coil
ORDERS = [0, 1, 2, 3]
DRV_F_MAX = 10000.0                            # driver modes kept [Hz]
CAV_F_MAX = 6000.0                             # cavity modes kept [Hz]
ETA_AIR = 0.02                                 # loss factor of the cavity modes (walls)
FREQS = np.geomspace(20.0, 3000.0, 700)        # forced response band [Hz]
MIC = (0.03, 0.04, Z_B - 0.08)                 # pressure probe in the enclosure [m]

# ── 1. Cavity modes ────────────────────────────────────────────────────────────
if cav_path.exists() and cav_path.stat().st_mtime > msh_path.stat().st_mtime:
    cav = CavityModes.load(cav_path)
    print(f"Cavity modes reused: {cav_path}")
else:
    cav = CavityModes.from_msh(msh_path)
    cav.solve(CAV_F_MAX, c=C_AIR)
    cav.save(cav_path)
print(f"Cavity: {len(cav.nodes)} nodes, V = {cav.volume * 1e3:.2f} L, {len(cav.kappa2)} modes ≤ "
      f"{CAV_F_MAX:.0f} Hz, first: {np.round(cav.freqs[1:7], 1)} Hz")

# ── 2. Driver modes ────────────────────────────────────────────────────────────
drv = DriverModes.from_eigen_h5(eig_path, ORDERS, f_max=DRV_F_MAX)
print(f"Driver: {len(drv)} modes (both families for m ≥ 1): "
      + ", ".join(f"m={m}: {np.sum(drv.m == m)}" for m in ORDERS))

# ── 3. Coupling ────────────────────────────────────────────────────────────────
profile = sample_profile(geo_path, PROFILE_CURVES, PROFILE_START)
cpl = DriverCavityCoupling(cav, drv, profile, center=(X0, Y0), rho=RHO_AIR, c=C_AIR, eta_air=ETA_AIR)
print(f"Coupling: {len(cpl.edges)} wet edges, wet area {cpl.wet_area * 1e4:.1f} cm², "
      f"|S| per order: " + ", ".join(f"m={m}: {np.abs(cpl.S[drv.m == m]).max():.2e}" for m in ORDERS))

# check: pistonic mode + box air spring,  ω_c² ≈ ω_k² + ρc²Q_k²/V
k0 = int(np.argmax(np.abs(cpl.volume_disp) * (drv.m == 0)))
fc = np.sqrt(drv.freq[k0] ** 2 + RHO_AIR * C_AIR ** 2 * cpl.volume_disp[k0] ** 2 / cav.volume
             / (2 * np.pi) ** 2)
print(f"Pistonic mode {drv.freq[k0]:.1f} Hz, Q = {cpl.volume_disp[k0]:.3e} m³ per unit modal "
      f"amplitude → air-spring estimate {fc:.1f} Hz")

# ── 4. Forced response, 1 N on the voice coil ─────────────────────────────────
g = _gmsh()                                     # coil nodes: on the forced curve
g.clear()
g.open(geo_path)
a, b = g.model.getParametrizationBounds(1, COIL_CURVE)
ends = np.array(g.model.getValue(1, COIL_CURVE, [a[0], b[0]])).reshape(2, 3)[:, :2]
g.clear()
t = np.clip(((drv.xy - ends[0]) @ (ends[1] - ends[0])) / np.sum((ends[1] - ends[0]) ** 2), 0, 1)
dist = np.linalg.norm(drv.xy - (ends[0] + t[:, None] * (ends[1] - ends[0])), axis=1)
coil_nodes = np.where(dist < 1e-6)[0]
F = cpl.coil_force(coil_nodes, total=1.0)
print(f"Coil: {len(coil_nodes)} nodes on curve {COIL_CURVE}")

q, P = cpl.forced(FREQS, F, loaded=True)
q0, _ = cpl.forced(FREQS, F, loaded=False)
mic = cav.nearest_node(MIC)
p_mic = P @ cav.psi[mic]

# ── 5. Coupled modes ───────────────────────────────────────────────────────────
s, Xq, XP = cpl.coupled_modes(f_max=FREQS[-1] * 1.5)
print(f"Coupled modes ≤ {FREQS[-1] * 1.5:.0f} Hz: {len(s)}")

np.savez_compressed(
    out_path, freqs=FREQS, q=q, q_vacuo=q0, P=P, p_mic=p_mic, mic=cav.nodes[mic], F=F,
    s=s, Xq=Xq, XP=XP, S=cpl.S, R=cpl.R, volume_disp=cpl.volume_disp,
    m=drv.m, family=drv.family, freq=drv.freq, zeta=drv.zeta, cav_freqs=cav.freqs,
    center=np.array([X0, Y0, Z_B]), orders=np.array(ORDERS), eta_air=ETA_AIR, f_c_estimate=fc)
print(f"Saved → {out_path}")
