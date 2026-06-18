import numpy as np
import hashlib
import h5py
import numpy as np
from pathlib import Path
from typing import Any, Mapping

def _snap_key(kind: str, value: float, idx: int | None = None) -> str:
    """Stable, sortable snapshot id."""
    if kind == "freq":
        return f"f_{value:012.4f}Hz"
    if kind == "time":
        return f"t_{value:014.8f}s"
    if kind == "static":
        return f"s_{idx:06d}" if idx is not None else "s_000000"
    return f"{kind}_{value}"


def _mesh_hash(coords: np.ndarray, tris: np.ndarray) -> str:
    """Deterministic short hash to detect identical meshes and avoid duplication."""
    h = hashlib.blake2b(digest_size=8)
    h.update(np.ascontiguousarray(coords).tobytes())
    h.update(np.ascontiguousarray(tris).tobytes())
    return h.hexdigest()


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / Path / set / dataclass to JSON-friendly types."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, Path):           return str(obj)
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return _to_jsonable(vars(obj))
    return repr(obj)


def _write_complex(group: h5py.Group, name: str, arr: np.ndarray, **kw):
    """h5py supports complex natively; just enforce contiguous + compression."""
    arr = np.ascontiguousarray(arr)
    return group.create_dataset(
        name, data=arr,
        compression="gzip", compression_opts=4, shuffle=True, **kw
    )

