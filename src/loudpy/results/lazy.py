"""
LazyArray: a dataset of a result file read on demand (large 3D fields stay on disk).

It behaves like a read-only numpy array for indexing: arr[i], arr[:, k], arr[i, rows] read only
what is asked (the file is opened per access). np.asarray(arr) reads everything.
"""
from __future__ import annotations

import h5py
import numpy as np


class LazyArray:
    """A dataset of an HDF5 result file read on demand (rows, node subsets), shaped like a NumPy array."""
    def __init__(self, path, name):
        self.path, self.name = str(path), name
        with h5py.File(self.path, "r") as f:
            ds = f[name]
            self.shape, self.dtype = ds.shape, ds.dtype

    @property
    def ndim(self):
        """Number of dimensions."""
        return len(self.shape)

    @property
    def size(self):
        """Number of elements."""
        return int(np.prod(self.shape))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        with h5py.File(self.path, "r") as f:
            ds = f[self.name]
            if isinstance(key, tuple) and any(isinstance(k, (list, np.ndarray)) for k in key):
                # h5py needs increasing, unique indices along one axis: read the sorted unique
                # nodes (axis 1), then put them back in the order asked for
                k0 = key[0]
                rows = np.asarray(key[1]) if len(key) > 1 else None
                if rows is not None and rows.ndim == 1 and not isinstance(k0, (list, np.ndarray)):
                    uniq, inv = np.unique(rows, return_inverse=True)
                    block = ds[(k0, uniq) + tuple(key[2:])]
                    return np.take(block, inv, axis=1 if np.ndim(block) > 1 and not np.isscalar(k0) else 0)
                return np.asarray(ds)[key]
            return ds[key]

    def __array__(self, dtype=None, copy=None):
        with h5py.File(self.path, "r") as f:
            a = f[self.name][()]
        return a.astype(dtype) if dtype is not None else a

    def __repr__(self):
        return f"LazyArray({self.name} in {self.path}, shape={self.shape}, dtype={self.dtype})"
