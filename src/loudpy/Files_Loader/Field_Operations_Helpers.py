from __future__ import annotations

import numpy as np


def infer_dpn(field_name: str) -> int:
    _DPN_TABLE: dict[str, int] = {
    "u_meca": 2, "v_meca": 2, "a_meca": 2,
    "p_acou": 1, "phi_acou": 1,
}
    return _DPN_TABLE.get(field_name, 1)


def scatter_dofs(raw: np.ndarray, tags: np.ndarray | None,
                 mesh, dpn: int) -> np.ndarray:
    """Place raw DOFs onto mesh nodes; returns (N,) or (N, dpn).

    When tags is None the array is treated as already node-indexed:
    - shape (N,) or (N, dpn)  → returned as-is (node-indexed)
    - shape (N*dpn,)           → reshaped to (N, dpn)  (legacy flat DOF vector)
    """
    N   = mesh.n_nodes
    raw = np.asarray(raw)

    if tags is None:
        if dpn == 1:
            out = np.zeros(N, dtype=raw.dtype)
            n   = min(len(raw), N)
            out[:n] = raw[:n]
            return out
        # 2-D node-indexed: (n_nodes, dpn) — already the right shape
        if raw.ndim == 2:
            out = np.zeros((N, dpn), dtype=raw.dtype)
            n   = min(raw.shape[0], N)
            out[:n] = raw[:n]
            return out
        # legacy flat DOF vector: (n_dofs,) = (n_nodes * dpn,)
        out = np.zeros((N, dpn), dtype=raw.dtype)
        n   = min(len(raw) // dpn, N)
        out[:n] = raw[:dpn * n].reshape(n, dpn)
        return out

    t2r  = mesh.tag_to_row
    rows = np.fromiter((t2r[int(t)] for t in tags), dtype=int, count=len(tags))
    out  = np.zeros((N,) if dpn == 1 else (N, dpn), dtype=raw.dtype)
    if dpn == 1:
        out[rows] = raw[:len(rows)]
    else:
        out[rows] = raw[:dpn * len(rows)].reshape(len(rows), dpn)
    return out


def reduce_field(arr: np.ndarray, component: str) -> np.ndarray:
    """Reduce (N,) or (N,dpn) array to (N,) scalar per component."""
    if component == "complex":
        return arr
    if arr.ndim == 1:
        return {
            "abs":   np.abs(arr),
            "mag":   np.abs(arr),
            "real":  arr.real,
            "imag":  arr.imag,
            "phase": np.angle(arr),
        }.get(component, np.abs(arr))

    # (N, dpn) — vector field
    if component in ("mag", "abs"):
        return np.linalg.norm(np.abs(arr), axis=1)
    if component in ("ux", "x"):  return _real_of(arr[:, 0])
    if component in ("uy", "y"):  return _real_of(arr[:, 1])
    if component == "real":       return arr[:, 0].real
    if component == "imag":       return arr[:, 0].imag
    if component == "phase_dominant":
        dom = np.argmax(np.abs(arr), axis=1)
        return np.angle(arr[np.arange(arr.shape[0]), dom])
    return np.linalg.norm(np.abs(arr), axis=1)


def _real_of(x: np.ndarray) -> np.ndarray:
    return x.real if np.iscomplexobj(x) else x


def SPL(p_mag: np.ndarray, p_ref: float = 20e-6) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(p_mag), 1e-30) / p_ref)
