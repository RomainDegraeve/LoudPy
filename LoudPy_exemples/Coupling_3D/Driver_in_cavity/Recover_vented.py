"""
Recover_vented.py — rebuild vented_merged.npz from the pressure fields already written in
vented_merged_field.h5 (e.g. while Running_vented.py merged is still running, or after it was
stopped): for every frequency whose field is stored,
    q = H⁻¹ (F − Bᵀ p)          (driver modal amplitudes, H = in-vacuo modal dynamic matrix)
    c = (2/R²) Gᵀ p|_Γ          (DtN coefficients → pressure anywhere outside the hemisphere)
Then Loading_vented.py / Plot_3D_vented.py work on the frequencies available.
"""
import json
from pathlib import Path

import h5py
import numpy as np

from loudpy.Studies.Cavity_coupling import DriverModes, sample_profile, C_AIR
from loudpy.Studies.Exterior_dtn import HalfSpaceDtN, ExteriorFE
from enclosure import geo_path, eig_path, FRONT_CURVES, FRONT_START, REAR_CURVES, REAR_START, X0, Y0

files = Path("LoudPy_exemples/Coupling_3D/Driver_in_cavity/Results/Files")
cfg = json.loads((files / "vented.json").read_text())
cp = np.load(files / "coupled.npz", allow_pickle=True)
drv = DriverModes.from_eigen_h5(eig_path, list(cp["orders"]), f_max=float(cp["freq"].max()) + 1.0)
dtn = HalfSpaceDtN(cfg["center"], cfg["radius"], up=cfg["up"])
front = sample_profile(geo_path, FRONT_CURVES, FRONT_START)
rear = sample_profile(geo_path, REAR_CURVES, REAR_START)
w_s, zeta, F, n_s = 2 * np.pi * drv.freq, drv.zeta, cp["F"], len(drv)

freqs, ys, coefs, band_of, row_of, vol_front = [], [], [], [], [], None
with h5py.File(files / "vented_merged_field.h5", "r", locking=False) as h5:
    for bi, b in enumerate(cfg["merged"]):
        g = h5[f"band{bi}"]
        fb = g["freqs"][()]
        P = g["p"][()]
        done = np.where(np.abs(P).max(axis=1) > 0)[0]
        print(f"band {b['f_lo']:.0f}–{b['f_hi']:.0f} Hz: {len(done)} / {len(fb)} frequencies stored")
        if not len(done):
            continue
        cache = files / f"vented_merged_recovered_band{bi}.npz"
        if cache.exists() and int(np.load(cache)["n_done"]) == len(done):
            c_ = np.load(cache, allow_pickle=True)
            freqs += list(c_["f"]); ys += list(c_["y"]); coefs += list(c_["c"])
            band_of += [bi] * len(done); row_of += list(done)
            vol_front = c_["vol_front"] if vol_front is None else vol_front
            continue
        n0 = len(freqs)
        ext = ExteriorFE(b["msh"], dtn, b["f_hi"], front="interface_driver").couple_driver(drv, (front, rear), (X0, Y0))
        vol_front = ext.Bf.sum(axis=0) if vol_front is None else vol_front
        for r in done:
            f, p = fb[r], P[r].astype(complex)
            w, k = 2 * np.pi * f, 2 * np.pi * f / C_AIR
            H = np.diag(w_s ** 2 + 2j * zeta * w_s * w - w ** 2)
            ys.append(np.linalg.solve(H, F - ext.Bf.T @ p[ext.pnodes]))
            coefs.append(ext.coefficients(p, k))
            freqs.append(f), band_of.append(bi), row_of.append(r)
        cs = np.zeros((len(done), max(len(c) for c in coefs[n0:])), complex)
        for j, c in enumerate(coefs[n0:]):
            cs[j, :len(c)] = c
        np.savez(cache, f=np.array(freqs[n0:]), y=np.array(ys[n0:]), c=cs, n_done=len(done),
                 vol_front=ext.Bf.sum(axis=0))
        coefs[n0:] = list(cs)
        del ext

C = np.zeros((len(freqs), max(len(c) for c in coefs)), complex)
for i, c in enumerate(coefs):
    C[i, :len(c)] = c
np.savez_compressed(files / "vented_merged.npz", freqs=np.array(freqs), y=np.array(ys), coefs=C,
                    iters=np.zeros(len(freqs), int), n_s=n_s, center=np.array(cfg["center"]),
                    radius=cfg["radius"], up=cfg["up"], band_of=np.array(band_of), row_of=np.array(row_of),
                    f_b_estimate=cfg["f_b_estimate"], vol_front=vol_front)
print(f"✓ vented_merged.npz rebuilt: {len(freqs)} frequencies, {min(freqs):.0f}–{max(freqs):.0f} Hz")
