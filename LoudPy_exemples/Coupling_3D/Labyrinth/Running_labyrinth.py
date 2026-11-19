"""
Running_labyrinth.py — HPNEW in the labyrinth cabinet on a rigid ground, 1 N on the voice coil:
driver (Fourier modes m = 0 … 3, both families) strongly coupled to ONE air domain (inside of the
folded line + mouth + exterior half-space, P2 FE + exact half-space DtN on a hemisphere); the
driver is wetted on both faces by the same pressure field. No stuffing.

Run first: ../../Eigen_study/Fourier/Running_script.py (driver modes) and Build_labyrinth.py.

Quick check of a few frequencies (separate labyrinth_test.* files):  python3 Running_labyrinth.py 40,70,100

Settings: F_MAX (default 2 kHz: the 3 kHz mesh needs more than 8 GB of RAM without swapping —
set F_MAX = 3000 on a larger machine), frequency steps below.

What this produces (Results/Files/)
------------------------------------
  - labyrinth.npz        : frequencies, driver modal amplitudes, DtN coefficients (pressure
                           anywhere outside the hemisphere), saved after every band
  - labyrinth_field.h5   : per band, the mesh and the pressure at every node (flushed after every
                           frequency: an interrupted run keeps what it computed)
Then run Loading_labyrinth.py and Plot_3D_labyrinth.py.
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from loudpy.Studies.Cavity_coupling import DriverModes, sample_profile, RHO_AIR, C_AIR, _gmsh
from loudpy.Studies.Exterior_dtn import HalfSpaceDtN, ExteriorFE, coupled_sweep
from labyrinth import (geo_path, eig_path, FRONT_CURVES, FRONT_START, REAR_CURVES, REAR_START, COIL_CURVE,
                       X0, Y0, ORDERS, DRV_F_MAX)

files = Path("LoudPy_exemples/Coupling_3D/Labyrinth/Results/Files")
F_MAX = 2000.0                                 # top of the sweep [Hz]
FREQS = np.unique(np.concatenate([np.arange(20.0, 300.0, 2.0), np.arange(300.0, F_MAX + 1e-9, 10.0)]))
CHUNK_HZ, RTOL = 50.0, 1e-7
TAG = "labyrinth"
if len(sys.argv) > 1:                          # quick test: python3 Running_labyrinth.py 40,70,100
    FREQS = np.array([float(x) for x in sys.argv[1].split(",")])
    TAG = "labyrinth_test"

cfg = json.loads((files / "labyrinth.json").read_text())
dtn = HalfSpaceDtN(cfg["center"], cfg["radius"], up=cfg["up"])
drv = DriverModes.from_eigen_h5(eig_path, ORDERS, f_max=DRV_F_MAX)
front = sample_profile(geo_path, FRONT_CURVES, FRONT_START)
rear = sample_profile(geo_path, REAR_CURVES, REAR_START)
w_s, n_s = 2 * np.pi * drv.freq, len(drv)

# voice-coil force: 1 N axial, shared by the nodes of the forced edge (drives m = 0 only)
g = _gmsh()
g.clear()
g.open(geo_path)
a, b = g.model.getParametrizationBounds(1, COIL_CURVE)
ends = np.array(g.model.getValue(1, COIL_CURVE, [a[0], b[0]])).reshape(2, 3)[:, :2]
g.clear()
t = np.clip(((drv.xy - ends[0]) @ (ends[1] - ends[0])) / np.sum((ends[1] - ends[0]) ** 2), 0, 1)
coil = np.where(np.linalg.norm(drv.xy - (ends[0] + t[:, None] * (ends[1] - ends[0])), axis=1) < 1e-6)[0]
F = drv.U[:, coil, 1].mean(axis=1) * (drv.m == 0)
print(f"Driver: {n_s} modes (m = {ORDERS}), coil {len(coil)} nodes | line {cfg['line_length']:.3f} m, "
      f"quarter-wave estimate {cfg['f_quarter_wave']:.1f} Hz | {len(FREQS)} frequencies ≤ {F_MAX:.0f} Hz")

bands = []
for bd in cfg["bands"]:
    if bd["f_lo"] >= F_MAX:
        continue
    ext = ExteriorFE(bd["msh"], dtn, min(bd["f_hi"], F_MAX), front="interface_driver")
    ext.couple_driver(drv, (front, rear), (X0, Y0))               # both faces, same air
    bands.append((bd["f_lo"], bd["f_hi"], ext))
    print(f"  band {bd['f_lo']:.0f}–{bd['f_hi']:.0f} Hz: {len(ext.nodes)} nodes, "
          f"wet surface {ext.wet_area * 1e4:.0f} cm²")

h5 = h5py.File(files / f"{TAG}_field.h5", "w")
band_of, row_of = np.full(len(FREQS), -1), np.full(len(FREQS), -1)
for bi, (lo, hi, ext) in enumerate(bands):
    gr = h5.create_group(f"band{bi}")
    gr["nodes"], gr["tets"] = ext.nodes, ext.tets
    for k, v in ext.faces.items():
        gr[f"faces/{k}"] = v
    sel = [i for i, f in enumerate(FREQS) if (lo if bi == 0 else lo + 1e-6) <= f <= hi and band_of[i] < 0]
    band_of[sel], row_of[sel] = bi, np.arange(len(sel))
    gr["freqs"] = FREQS[sel]
    gr.create_dataset("p", (len(sel), len(ext.nodes)), np.complex64)
t0 = time.time()
done_y, done_c = {}, {}


def save_npz(idx):
    C = np.zeros((len(idx), max(len(done_c[i]) for i in idx)), complex)
    for j, i in enumerate(idx):
        C[j, :len(done_c[i])] = done_c[i]
    np.savez_compressed(files / f"{TAG}.npz", freqs=FREQS[idx], y=np.array([done_y[i] for i in idx]),
                        coefs=C, n_s=n_s, center=np.array(cfg["center"]), radius=cfg["radius"], up=cfg["up"],
                        band_of=band_of[idx], row_of=row_of[idx], m=drv.m, F=F,
                        f_quarter_wave=cfg["f_quarter_wave"], line_length=cfg["line_length"])


def store(i, b, p, c_h, q):
    h5[f"band{band_of[i]}/p"][row_of[i]] = p.astype(np.complex64)
    h5.flush()
    done_y[i], done_c[i] = q, c_h
    if i % 20 == 0:
        print(f"    {FREQS[i]:.0f} Hz done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if row_of[i] == np.count_nonzero(band_of == band_of[i]) - 1:
        save_npz(sorted(done_y))
        print(f"  ✓ band {band_of[i]} finished, results saved ({len(done_y)} frequencies)", flush=True)


bands_u = [(lo if bi == 0 else lo + 1e-6, hi, ext) for bi, (lo, hi, ext) in enumerate(bands)]
coupled_sweep(bands_u, FREQS, F, w_s, drv.zeta, np.zeros((n_s, 0)), np.zeros((n_s, n_s)), np.zeros(0), 0.0,
              RHO_AIR, C_AIR, CHUNK_HZ, RTOL, store)
h5.close()
save_npz(sorted(done_y))
print(f"Saved → {files / (TAG + '.npz')} ({(time.time() - t0) / 60:.1f} min)")
