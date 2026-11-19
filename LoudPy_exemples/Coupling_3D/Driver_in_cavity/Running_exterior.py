"""
Running_exterior.py — HPNEW in its enclosure, standing on a rigid ground, radiating into the
half-space: driver (Fourier modes m = 0 … 3) + enclosure (cavity modes, rear) + exterior air
(3D FE with the exact half-space DtN on a hemisphere, front), forced by 1 N on the voice coil.

Run first: Build_cavity.py, Running_script.py (enclosure coupling → coupled.npz),
Build_exterior.py (exterior meshes).

What this produces (Results/Files/)
------------------------------------
  - exterior.npz      : frequencies, driver modal amplitudes with / without the exterior,
                        DtN coefficients (pressure anywhere outside the hemisphere), GMRES iterations
  - exterior_field.h5 : per band, the exterior mesh and the pressure at every node and frequency
Then run Loading_exterior.py and Plot_3D_exterior.py.
"""
import json
import time
from pathlib import Path

import h5py
import numpy as np

from loudpy.Studies.Cavity_coupling import (CavityModes, DriverModes, sample_profile, solve_forced,
                                            RHO_AIR, C_AIR)
from loudpy.Studies.Exterior_dtn import HalfSpaceDtN, ExteriorFE, coupled_sweep
from enclosure import geo_path, eig_path, FRONT_CURVES, FRONT_START, X0, Y0

files = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Files")
FREQS = np.arange(20.0, 3000.0 + 1e-9, 10.0)   # [Hz]
CHUNK_HZ = 50.0                                # one LU (preconditioner) per chunk
RTOL = 1e-7                                    # GMRES tolerance

d = np.load(files / "coupled.npz", allow_pickle=True)
cfg = json.loads((files / "exterior.json").read_text())
cav = CavityModes.load(files / "cavity_modes.h5")
drv = DriverModes.from_eigen_h5(eig_path, list(d["orders"]), f_max=float(d["freq"].max()) + 1.0)
assert len(drv) == len(d["F"]), "driver modes changed since Running_script.py"
dtn = HalfSpaceDtN(cfg["center"], cfg["radius"], up=cfg["up"])
front = sample_profile(geo_path, FRONT_CURVES, FRONT_START)
w_s, F = 2 * np.pi * d["freq"], d["F"]

bands = []
for b in cfg["bands"]:
    t0 = time.time()
    ext = ExteriorFE(b["msh"], dtn, b["f_hi"]).couple_driver(drv, front, (X0, Y0))
    bands.append((b["f_lo"], b["f_hi"], ext))
    print(f"Exterior {b['f_lo']:.0f}–{b['f_hi']:.0f} Hz: {len(ext.nodes)} nodes, wet front "
          f"{ext.wet_area * 1e4:.1f} cm² ({time.time() - t0:.0f} s)")

# enclosure only (front in vacuo), same frequencies
q_encl, _ = solve_forced(FREQS, F, w_s, d["zeta"], d["S"], d["R"], cav.kappa2, RHO_AIR, C_AIR,
                         float(d["eta_air"]))

# exterior pressure of every node, per band, written as the sweep goes
h5 = h5py.File(files / "exterior_field.h5", "w")
band_of = np.full(len(FREQS), -1)
row_of = np.full(len(FREQS), -1)
for bi, (f_lo, f_hi, ext) in enumerate(bands):
    g = h5.create_group(f"band{bi}")
    g["nodes"], g["tets"] = ext.nodes, ext.tets
    for k, v in ext.faces.items():
        g[f"faces/{k}"] = v
    sel = [i for i, f in enumerate(FREQS) if f_lo <= f <= f_hi and band_of[i] < 0]
    band_of[sel], row_of[sel] = bi, np.arange(len(sel))
    g["freqs"] = FREQS[sel]
    g.create_dataset("p", (len(sel), len(ext.nodes)), np.complex64)
t_start = time.time()


def store(i, b, p, c_h, q):
    h5[f"band{band_of[i]}/p"][row_of[i]] = p.astype(np.complex64)
    if i % 10 == 0:
        print(f"    {FREQS[i]:.0f} Hz done ({(time.time() - t_start) / 60:.1f} min)", flush=True)


# bands share the boundary frequencies: each frequency goes to the first band containing it
bands_unique = [(lo if bi == 0 else lo + 1e-6, hi, ext) for bi, (lo, hi, ext) in enumerate(bands)]
q, coefs, iters = coupled_sweep(bands_unique, FREQS, F, w_s, d["zeta"], d["S"], d["R"], cav.kappa2,
                                float(d["eta_air"]), RHO_AIR, C_AIR, CHUNK_HZ, RTOL, store)
h5.close()

n_h = max(len(c) for c in coefs)
C = np.zeros((len(FREQS), n_h), complex)
for i, c in enumerate(coefs):
    C[i, :len(c)] = c

# check at low frequency: compact source on a rigid ground, far field p·r → ρω²|V_front| / 2π
i0 = 3
k0 = 2 * np.pi * FREQS[i0] / C_AIR
V = bands[0][2].Bf.sum(axis=0) @ q[i0]                  # front volume displacement
far = np.array([[X0 + 20 * np.sin(a), Y0, 20 * np.cos(a)] for a in np.deg2rad([0, 90, 180])])
pr = np.abs(dtn.pressure(far, C[i0], k0)) * 20.0
est = RHO_AIR * (2 * np.pi * FREQS[i0]) ** 2 * abs(V) / (2 * np.pi)
print(f"Check {FREQS[i0]:.0f} Hz, 20 m, at 0/90/180°: p·r = {np.round(pr, 5)} Pa·m, half-space monopole "
      f"{est:.5f} Pa·m (ratio {pr[0] / est:.3f})")

np.savez_compressed(files / "exterior.npz", freqs=FREQS, q=q, q_encl=q_encl, coefs=C, iters=iters,
                    center=np.array(cfg["center"]), radius=cfg["radius"], up=cfg["up"],
                    band_of=band_of, row_of=row_of, m=drv.m, family=drv.family,
                    orders=d["orders"], drv_center=np.array([X0, Y0]))
print(f"Saved → {files / 'exterior.npz'}  (GMRES iterations: mean {iters.mean():.0f}, max {iters.max()}, "
      f"total {(time.time() - t_start) / 60:.1f} min)")
