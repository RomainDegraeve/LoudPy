"""
Running_vented.py — HPNEW in the vented enclosure on a rigid ground, 1 N on the voice coil
Two models, chosen on the command line:

  python3 Running_vented.py merged   reference (lp.Scene, enclosure.vented_scene()): ONE air
                                     domain (inside + port + outside, FE + half-space DtN); the
                                     driver wetted on both faces. Meshes built on the first run.
  python3 Running_vented.py hybrid   fast: modal inside (rigid walls, inner port mouth as a
                                     piston) + 1D port (plane waves) + exterior FE/DtN (outer mouth
                                     as a piston); generalised coordinates (driver modes, X1, X2).
                                     Low-level toolkit (loudpy.Studies): run Build_vented.py and
                                     Running_script.py first.

What this produces (Results/Files/)
------------------------------------
  merged: vented_merged.h5 (loudpy result format 2, field streamed), vented_merged_<f>Hz.msh
  hybrid: vented_hybrid.npz (frequencies, driver modal amplitudes + port X1, X2, DtN coefficients)
          and vented_hybrid_field.h5 (per band, the mesh and the exterior pressure at every node)
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

import loudpy as lp
from loudpy.Studies.Cavity_coupling import CavityModes, DriverCavityCoupling, RHO_AIR, C_AIR
from loudpy.Studies.Exterior_dtn import HalfSpaceDtN, ExteriorFE, coupled_sweep, port_block
from enclosure import FILES, X0, Y0, PORT_A, PORT_L, driver, vented_scene

MODEL = sys.argv[1] if len(sys.argv) > 1 else "merged"
files = Path(FILES)
FREQS = np.unique(np.concatenate([np.arange(20.0, 300.0, 2.0), np.arange(300.0, 3000.0 + 1e-9, 10.0)]))
if len(sys.argv) > 2:                                      # quick test: list of frequencies
    FREQS = np.array([float(x) for x in sys.argv[2].split(",")])
CAV_F_MAX, CHUNK_HZ, RTOL = 6000.0, 50.0, 1e-7

if MODEL == "merged":
    tag = "vented_merged" if len(sys.argv) <= 2 else "vented_merged_test"
    res = lp.FrequencyResponse(vented_scene(), drive=lp.Force(1.0)).solve(FREQS, store=files / f"{tag}.h5")
    print(res, "→", files / f"{tag}.h5")
    sys.exit()

# ── hybrid (low-level toolkit) ─────────────────────────────────────────────────
cfg = json.loads((files / "vented.json").read_text())
hp = driver()
drv, front, rear = hp.modes, hp.front, hp.rear
F = lp.load(files / "coupled.h5").driver("main").force          # modal force of 1 N on the coil
dtn = HalfSpaceDtN(cfg["center"], cfg["radius"], up=cfg["up"])
w_s, zeta = 2 * np.pi * drv.freq, drv.zeta
n_s, Sp = len(drv), np.pi * PORT_A ** 2
t0 = time.time()

cav_h5 = files / "vented_cavity_modes.h5"
if cav_h5.exists():
    cav = CavityModes.load(cav_h5)
else:
    cav = CavityModes.from_msh(cfg["cavity"]["msh"], surfaces=["port_inner"])
    cav.solve(CAV_F_MAX, c=C_AIR)
    cav.save(cav_h5)
print(f"Cavity: {len(cav.nodes)} nodes, {cav.volume * 1e3:.2f} L, {len(cav.kappa2)} modes ≤ {CAV_F_MAX:.0f} Hz "
      f"({time.time() - t0:.0f} s)")
# X1: air leaving the box into the port (wall displacement into the cavity air = −X1/S)
cpl = DriverCavityCoupling(cav, drv, rear, center=(X0, Y0), rho=RHO_AIR, c=C_AIR, eta_air=0.0,
                           pistons=[(cav.surfaces["port_inner"], -1.0 / Sp), None])
print(f"Cavity coupling ({time.time() - t0:.0f} s): inner mouth area "
      f"{-cpl.volume_disp[n_s] * Sp * 1e4:.2f} cm² (exact {Sp * 1e4:.2f})")
# X2: air leaving the port into the exterior (displacement into the exterior air = +X2/S)
bands = []
for b in cfg["ext"]:
    ext = ExteriorFE(b["msh"], dtn, b["f_hi"], front="interface_driver", surfaces=["port_outer"])
    ext.couple_driver(drv, front, (X0, Y0), pistons=[None, (ext.faces["port_outer"], 1.0 / Sp)])
    bands.append((b["f_lo"], b["f_hi"], ext))
S, R, kappa2, eta = cpl.S, cpl.R, cav.kappa2, 0.0
H_extra, Fy = port_block(PORT_L, PORT_A), np.concatenate([F, [0.0, 0.0]])

for lo, hi, ext in bands:
    print(f"  {MODEL} {lo:.0f}–{hi:.0f} Hz: {len(ext.nodes)} nodes, {ext.Bf.shape[0]} coupled nodes")

h5 = h5py.File(files / f"vented_{MODEL}_field.h5", "w")
band_of, row_of = np.full(len(FREQS), -1), np.full(len(FREQS), -1)
for bi, (lo, hi, ext) in enumerate(bands):
    g = h5.create_group(f"band{bi}")
    g["nodes"], g["tets"] = ext.nodes, ext.tets
    for k, v in ext.faces.items():
        g[f"faces/{k}"] = v
    sel = [i for i, f in enumerate(FREQS) if (lo if bi == 0 else lo + 1e-6) <= f <= hi and band_of[i] < 0]
    band_of[sel], row_of[sel] = bi, np.arange(len(sel))
    g["freqs"] = FREQS[sel]
    g.create_dataset("p", (len(sel), len(ext.nodes)), np.complex64)


done_y, done_c = {}, {}


def save_npz(idx):
    """Results of the frequencies idx (all finished ones while running, all at the end)."""
    C = np.zeros((len(idx), max(len(done_c[i]) for i in idx)), complex)
    for j, i in enumerate(idx):
        C[j, :len(done_c[i])] = done_c[i]
    np.savez_compressed(files / f"vented_{MODEL}.npz", freqs=FREQS[idx], y=np.array([done_y[i] for i in idx]),
                        coefs=C, iters=np.zeros(len(idx), int), n_s=n_s, center=np.array(cfg["center"]),
                        radius=cfg["radius"], up=cfg["up"], band_of=band_of[idx], row_of=row_of[idx],
                        f_b_estimate=cfg["f_b_estimate"], vol_front=bands[0][2].Bf[:, :n_s].sum(axis=0))


def store(i, b, p, c_h, q):
    h5[f"band{band_of[i]}/p"][row_of[i]] = p.astype(np.complex64)
    h5.flush()                                  # an interrupted run keeps every finished frequency
    done_y[i], done_c[i] = q, c_h
    if i % 20 == 0:
        print(f"    {FREQS[i]:.0f} Hz done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if row_of[i] == np.count_nonzero(band_of == band_of[i]) - 1:          # last frequency of its band
        save_npz(sorted(done_y))
        print(f"  ✓ band {band_of[i]} finished: results saved ({len(done_y)} frequencies)", flush=True)


bands_u = [(lo if bi == 0 else lo + 1e-6, hi, ext) for bi, (lo, hi, ext) in enumerate(bands)]
y, coefs, iters = coupled_sweep(bands_u, FREQS, Fy, w_s, zeta, S, R, kappa2, eta, RHO_AIR, C_AIR,
                                CHUNK_HZ, RTOL, store, H_extra=H_extra)
h5.close()
save_npz(list(range(len(FREQS))))
print(f"Saved → {files / f'vented_{MODEL}.npz'}  (GMRES mean {iters.mean():.0f}, max {iters.max()}, "
      f"{(time.time() - t0) / 60:.1f} min)")
