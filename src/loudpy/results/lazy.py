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


class Combination:
    """
    Σ_k W[:, k] · A_k, read on demand: the fields of a result driven by its coil forces W (n, K) from the unit
    responses A_k (one per coil, arrays or LazyArray of the same shape (n, …)). Indexed like an array (rows first),
    np.asarray(c) gives the whole sum.
    """
    def __init__(self, arrays, W):
        self.arrays = list(arrays)
        self.W = np.asarray(W)
        self.shape = tuple(self.arrays[0].shape)
        self.dtype = np.result_type(self.arrays[0].dtype, self.W.dtype)

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
        k0 = key[0] if isinstance(key, tuple) else key
        w = self.W[k0] if k0 is not Ellipsis else self.W
        out = None
        for k, a in enumerate(self.arrays):
            part = np.asarray(a[key])
            wk = w[..., k]
            if np.ndim(wk):
                wk = np.reshape(wk, np.shape(wk) + (1,) * (part.ndim - np.ndim(wk)))
            out = part * wk if out is None else out + part * wk
        return out

    def __array__(self, dtype=None, copy=None):
        a = self[:]
        return a.astype(dtype) if dtype is not None else a

    def __repr__(self):
        return f"Combination({len(self.arrays)} unit responses, shape={self.shape})"


def combine(arrays, W):
    """Σ_k W[:, k] · A_k: an array when every A_k is in memory, else a Combination read on demand."""
    W = np.asarray(W)
    if all(isinstance(a, np.ndarray) for a in arrays):
        out = None
        for k, a in enumerate(arrays):
            wk = W[:, k].reshape((-1,) + (1,) * (a.ndim - 1))
            out = a * wk if out is None else out + a * wk
        return out
    return Combination(arrays, W)
