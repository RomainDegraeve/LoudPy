"""
Running_column.py — three identical HPNEW drivers in a column with a folded transmission line,
on a rigid ground, driven in phase (DRIVE amplitudes, default 1 N on each voice coil).

  - drivers : the same Fourier modes (m = 0 … 3, both families) for the three drivers, placed
              by MultiDriver at their own axes; their modal coordinates are stacked
              (top, middle, bottom), so the dense structural block has 3 × n_modes rows
  - air     : ONE domain — inside of the line + mouth + exterior half-space (P2 FE + exact
              half-space DtN on a hemisphere); each driver is wetted on both faces by it, and the
              drivers interact through it (no extra term: it comes out of the coupling)
  - force   : F = Σ_j a_j F_j, F_j the unit coil force of driver j (axisymmetric, m = 0 only);
              m ≥ 1 of the top and bottom drivers are driven by the non-symmetric air load.

Run first: ../../Eigen_study/Fourier/Running_script.py (driver modes) and Build_column.py.

Quick check of a few frequencies (separate column_test.* files):  python3 Running_column.py 40,100,300

What this produces (Results/Files/)
------------------------------------
  - column.npz       : frequencies, stacked modal amplitudes, DtN coefficients (pressure anywhere
                       outside the hemisphere), saved after every band
  - column_field.h5  : per band, the mesh and the pressure at every node (flushed after every
                       frequency: an interrupted run keeps what it computed)
Then run Loading_column.py and Plot_3D_column.py.
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from loudpy.Studies.Cavity_coupling import DriverModes, MultiDriver, sample_profile, RHO_AIR, C_AIR, _gmsh
from loudpy.Studies.Exterior_dtn import HalfSpaceDtN, ExteriorFE, coupled_sweep
from column import (geo_path, eig_path, FRONT_CURVES, FRONT_START, REAR_CURVES, REAR_START, COIL_CURVE,
                    CENTERS, NAMES, ORDERS, DRV_F_MAX)

files = Path("LoudPy_exemples/Coupling_3D/Column_labyrinth/Results/Files")
F_MAX = 1000.0                                 # top of the sweep [Hz] (2 kHz: 169k nodes, > 8 GB RAM)
FREQS = np.unique(np.concatenate([np.arange(20.0, 300.0, 2.0), np.arange(300.0, F_MAX + 1e-9, 10.0)]))
DRIVE = [1.0, 1.0, 1.0]                        # complex coil force of each driver [N] (top, middle, bottom)
CHUNK_HZ, RTOL = 50.0, 1e-7
TAG = "column"
if len(sys.argv) > 1:                          # quick test: python3 Running_column.py 40,100,300
    FREQS = np.array([float(x) for x in sys.argv[1].split(",")])
    TAG = "column_test"

cfg = json.loads((files / "column.json").read_text())
dtn = HalfSpaceDtN(cfg["center"], cfg["radius"], up=cfg["up"])
drv = DriverModes.from_eigen_h5(eig_path, ORDERS, f_max=DRV_F_MAX)
drivers = MultiDriver(drv, CENTERS, names=NAMES)          # one DriverModes shared by the 3 drivers
front = sample_profile(geo_path, FRONT_CURVES, FRONT_START)
rear = sample_profile(geo_path, REAR_CURVES, REAR_START)
w_s, n_s = 2 * np.pi * drivers.freq, len(drivers)

# coil nodes: on the forced edge of the meridian mesh (the same for the three drivers)
g = _gmsh()
g.clear()
g.open(geo_path)
a, b = g.model.getParametrizationBounds(1, COIL_CURVE)
ends = np.array(g.model.getValue(1, COIL_CURVE, [a[0], b[0]])).reshape(2, 3)[:, :2]
g.clear()
t = np.clip(((drv.xy - ends[0]) @ (ends[1] - ends[0])) / np.sum((ends[1] - ends[0]) ** 2), 0, 1)
coil = np.where(np.linalg.norm(drv.xy - (ends[0] + t[:, None] * (ends[1] - ends[0])), axis=1) < 1e-6)[0]
F = drivers.coil_forces(coil, DRIVE)
print(f"{drivers}\ncoil {len(coil)} nodes, drive {DRIVE} N | line {cfg['line_length']:.3f} m, quarter-wave "
      f"estimate {cfg['f_quarter_wave']:.1f} Hz | {len(FREQS)} frequencies ≤ {F_MAX:.0f} Hz")

bands = []
for bd in cfg["bands"]:
    if bd["f_lo"] >= F_MAX:
        continue
    t0 = time.time()
    ext = ExteriorFE(bd["msh"], dtn, min(bd["f_hi"], F_MAX), front="interface_driver")
    ext.couple_driver(drivers, [(front, rear)] * drivers.n_drivers)     # both faces of each driver
    bands.append((bd["f_lo"], bd["f_hi"], ext))
    print(f"  band {bd['f_lo']:.0f}–{bd['f_hi']:.0f} Hz: {len(ext.nodes)} nodes, wet surface per driver "
          f"{np.round(np.array(ext.wet_areas) * 1e4, 1)} cm² ({time.time() - t0:.0f} s)")

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
                        coefs=C, n_s=n_s, offsets=drivers.offsets, names=np.array(NAMES),
                        centers=np.array(CENTERS), drive=np.array(DRIVE, complex), F=F,
                        center=np.array(cfg["center"]), radius=cfg["radius"], up=cfg["up"],
                        band_of=band_of[idx], row_of=row_of[idx], m=drivers.m,
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


# no cavity modes: the line is part of the FE air (S has no columns, R = 0)
bands_u = [(lo if bi == 0 else lo + 1e-6, hi, ext) for bi, (lo, hi, ext) in enumerate(bands)]
coupled_sweep(bands_u, FREQS, F, w_s, drivers.zeta, np.zeros((n_s, 0)), np.zeros((n_s, n_s)), np.zeros(0),
              0.0, RHO_AIR, C_AIR, CHUNK_HZ, RTOL, store)
h5.close()
save_npz(sorted(done_y))
print(f"Saved → {files / (TAG + '.npz')} ({(time.time() - t0) / 60:.1f} min)")
