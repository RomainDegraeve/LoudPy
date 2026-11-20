"""Per-entry scalars of a solve (ω, coil current, …) → Block.extra arrays."""
from __future__ import annotations

import numpy as np


def scalars(dicts) -> dict[str, np.ndarray]:
    """{key: (n,) array} of the scalar values of a list of dicts (NaN where a key is missing)."""
    keys = sorted({k for d in dicts for k in d})
    out = {}
    for k in keys:
        vals = [d.get(k, np.nan) for d in dicts]
        if all(np.isscalar(v) and not isinstance(v, (str, bytes)) for v in vals):
            out[k] = np.array(vals)
    return out
