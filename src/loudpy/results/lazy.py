"""
LazyArray: a dataset of a result file read on demand (large 3D fields stay on disk).

It behaves like a read-only numpy array for indexing: arr[i], arr[:, k], arr[i, rows] read only
what is asked (the file is opened per access). np.asarray(arr) reads everything.
"""
from __future__ import annotations

import h5py
import numpy as np


class LazyArray:
    def __init__(self, path, name):
        self.path, self.name = str(path), name
        with h5py.File(self.path, "r") as f:
            ds = f[name]
            self.shape, self.dtype = ds.shape, ds.dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        with h5py.File(self.path, "r") as f:
            ds = f[self.name]
            if isinstance(key, tuple) and any(isinstance(k, (list, np.ndarray)) for k in key):
                # h5py needs increasing indices: read sorted unique rows, then reorder
                k0 = key[0]
                rows = np.asarray(key[1]) if len(key) > 1 else None
                if rows is not None and rows.ndim == 1:
                    order = np.argsort(rows)
                    uniq, inv = np.unique(rows[order], return_inverse=True)
                    block = ds[k0, uniq] if not isinstance(k0, (list, np.ndarray)) else ds[:, uniq][k0]
                    out = np.empty(block.shape, block.dtype)
                    picked = block[..., inv]
                    out[..., order] = picked
                    return out
                return np.asarray(ds)[key]
            return ds[key]

    def __array__(self, dtype=None, copy=None):
        with h5py.File(self.path, "r") as f:
            a = f[self.name][()]
        return a.astype(dtype) if dtype is not None else a

    def __repr__(self):
        return f"LazyArray({self.name} in {self.path}, shape={self.shape}, dtype={self.dtype})"
