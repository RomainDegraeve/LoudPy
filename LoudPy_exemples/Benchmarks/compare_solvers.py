"""compare_solvers.py — speed-up and accuracy of a run against the baseline (bench_solvers.py).
    python3 LoudPy_exemples/Benchmarks/compare_solvers.py <tag> [baseline_tag]"""
import sys
from pathlib import Path
import numpy as np

tag = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "baseline"
d = Path("LoudPy_exemples/Benchmarks/Results")
print(f"{'case':10s} {'baseline':>10s} {tag:>10s} {'speed-up':>9s}   accuracy (max relative difference)")
for case in ("fsi_elec", "fsi", "meca", "eigen", "fsi_eigen", "time"):
    a, b = d / f"{case}_{base}.npz", d / f"{case}_{tag}.npz"
    if not (a.exists() and b.exists()):
        continue
    A, B = np.load(a), np.load(b)
    acc = []
    for k in ("U", "P"):
        if k in A and k in B:
            x, y = A[k], B[k]
            if x.ndim > 1:
                e = np.linalg.norm(x - y, axis=-1) / np.maximum(np.linalg.norm(x, axis=-1), 1e-300)
                acc.append(f"{k}: {e.max():.1e}")
    if "freqs" in A and case in ("eigen", "fsi_eigen"):
        fa, fb = A["freqs"], B["freqs"]
        n = min(len(fa), len(fb))
        acc.append(f"{len(fa)} vs {len(fb)} modes, max |Δf|/f {np.max(np.abs(fa[:n] - fb[:n]) / fa[:n]):.1e}")
        acc.append(f"max |Δζ| {np.max(np.abs(A['zeta'][:n] - B['zeta'][:n])):.1e}")
    ta, tb = float(A["t_solve"]), float(B["t_solve"])
    print(f"{case:10s} {ta:9.2f}s {tb:9.2f}s {ta / tb:8.1f}×   " + ", ".join(acc))
