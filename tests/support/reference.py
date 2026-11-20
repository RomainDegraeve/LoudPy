"""
Reference store and comparisons that survive the refactor.

What is compared is chosen so that a pure reorganisation of the code passes, while any change of
the numbers fails:
  - nodal fields are stored with the coordinates of their nodes and matched by coordinates,
    so a new DOF or node ordering does not matter;
  - mode shapes are compared through the MAC (sign and normalisation do not matter) after
    pairing by frequency;
  - everything else (frequencies, probes, scalars) directly, with tight tolerances.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Tolerances. Floating-point results agree across machines and compilers only to a few units in the
# last place per operation, and ill-conditioned quantities amplify that. Measured spread (macOS arm64,
# Linux x86-64, Numba fastmath on / off): ~1e-11 on operators with cancellation (C = αM + βK applied to
# a smooth vector), ~3e-8 on responses near a resonance. The tolerances sit about a decade above the
# spread and decades below any change of the model or the method (≥ 1e-5).
RTOL = 1e-8              # default (smooth quantities)
RTOL_OPERATOR = 1e-10    # assembled operators (fingerprints)
RTOL_RESPONSE = 1e-6     # forced responses (amplified near resonances)
RTOL_COUPLED_MODES = 1e-4   # damped coupled modes in near-degenerate pairs (cos / sin twins of a
#                             non-symmetric problem): a perturbation ε moves them by ~√ε (spread ~3e-6)
MAC_MIN = 1 - 1e-8       # mode-shape agreement


class ReferenceStore:
    def __init__(self, folder: Path, name: str, update: bool):
        self.path = Path(folder) / f"{_safe(name)}.npz"
        self.update = update
        self._new: dict[str, np.ndarray] = {}
        self._ref = None
        if not update:
            if not self.path.exists():
                raise FileNotFoundError(f"no reference {self.path}: run pytest --update-reference first")
            with np.load(self.path, allow_pickle=False) as f:
                self._ref = {k: f[k] for k in f.files}

    # ── plain arrays ───────────────────────────────────────────────────────
    def check(self, rtol=RTOL, atol_rel=1e-12, **arrays):
        """Compare each array with the reference (relative to the max of the reference)."""
        for k, v in arrays.items():
            v = np.asarray(v)
            if self.update:
                self._new[k] = v
                continue
            r = self._get(k)
            assert v.shape == r.shape, f"{k}: shape {v.shape} != reference {r.shape}"
            scale = max(np.abs(r).max(initial=0.0), 1e-300)
            err = np.abs(v - r).max(initial=0.0)
            assert err <= rtol * scale + atol_rel * scale, \
                f"{k}: max |Δ| = {err:.3e} > {rtol:.1e} × max|ref| = {rtol * scale:.3e}"

    # ── nodal fields, matched by coordinates ───────────────────────────────
    def check_nodal(self, key, coords, values, rtol=RTOL, digits=10):
        """values (n_nodes, ...) at coords (n_nodes, dim); compared node by node by coordinates."""
        coords = np.round(np.asarray(coords, float), digits)
        values = np.asarray(values)
        order = np.lexsort(coords.T[::-1])
        coords, values = coords[order], values[order]
        if self.update:
            self._new[f"{key}__coords"] = coords
            self._new[f"{key}__values"] = values
            return
        rc, rv = self._get(f"{key}__coords"), self._get(f"{key}__values")
        assert coords.shape == rc.shape, f"{key}: {len(coords)} nodes != reference {len(rc)}"
        assert np.array_equal(coords, rc), f"{key}: node coordinates differ from the reference (mesh changed?)"
        scale = max(np.abs(rv).max(initial=0.0), 1e-300)
        err = np.abs(values - rv).max(initial=0.0)
        assert err <= rtol * scale, f"{key}: max |Δ| = {err:.3e} > {rtol:.1e} × max|ref| = {rtol * scale:.3e}"

    # ── modes: frequencies + MAC ───────────────────────────────────────────
    def check_modes(self, key, freqs, shapes=None, coords=None, rtol=1e-7, mac_min=MAC_MIN, digits=10,
                    n_shapes=8):
        """
        freqs (n,), shapes (n, n_nodes, dpn) nodal (optional, with coords (n_nodes, dim)).
        Frequencies compared directly (same count, sorted); the shapes of the n_shapes lowest
        modes paired by index and compared with the MAC, after matching the nodes by coordinates.
        """
        freqs = np.asarray(freqs, float)
        o = np.argsort(freqs)
        freqs = freqs[o]
        if shapes is not None:
            coords = np.round(np.asarray(coords, float), digits)
            no = np.lexsort(coords.T[::-1])
            shapes = np.asarray(shapes)[o][:n_shapes, no]
            coords = coords[no]
        if self.update:
            self._new[f"{key}__freqs"] = freqs
            if shapes is not None:
                self._new[f"{key}__coords"] = coords
                self._new[f"{key}__shapes"] = shapes
            return
        rf = self._get(f"{key}__freqs")
        assert len(freqs) == len(rf), f"{key}: {len(freqs)} modes != reference {len(rf)}\n{freqs}\n{rf}"
        err = np.abs(freqs - rf) / np.abs(rf)
        assert err.max() <= rtol, f"{key}: max relative frequency error {err.max():.3e} > {rtol:.1e}"
        if shapes is not None:
            rc, rs = self._get(f"{key}__coords"), self._get(f"{key}__shapes")
            assert np.array_equal(coords, rc), f"{key}: node coordinates differ from the reference"
            macs = np.array([mac(a.ravel(), b.ravel()) for a, b in zip(shapes, rs)])
            bad = np.where(macs < mac_min)[0]
            assert not len(bad), f"{key}: MAC below {mac_min} for modes {bad.tolist()} ({macs[bad]})"

    # ── internals ──────────────────────────────────────────────────────────
    def _get(self, k):
        assert k in self._ref, f"reference {self.path.name} has no entry {k!r} (update the reference?)"
        return self._ref[k]

    def finish(self):
        if self.update and self._new:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.path, **self._new)


def mac(a, b):
    """Modal assurance criterion |aᴴb|² / (aᴴa · bᴴb) (complex allowed)."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real))


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
