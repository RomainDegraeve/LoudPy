"""
Homogeneous Dirichlet conditions by elimination (shared by the structural assemblers).

The blocked DOFs come from the boundaries returned by `_clamped(objects)` (each with node_tags
and `direction`: the blocked components); the physics layer defines that selection. After
apply_blocked_boundaries(), expand() scatters a reduced vector back to the full DOF numbering
(blocked DOFs = 0).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import issparse


class DirichletElimination:
    """Mixin: needs matrix_size, dofs_per_node, tag_to_local and _clamped(objects); sets _free."""

    def _clamped(self, objects):
        return objects

    def _blocked_dofs(self, interfaces):
        """Blocked global DOF indices from the clamped boundaries."""
        blocked = set()
        for iface in self._clamped(interfaces):
            comps = iface.direction
            for tag in iface.node_tags:
                local = self.tag_to_local[tag]
                for c in comps:
                    blocked.add(self.dofs_per_node * local + c)
        return blocked

    def _free_dofs(self, interfaces):
        blocked = self._blocked_dofs(interfaces)
        all_dofs = np.arange(self.matrix_size)
        return np.setdiff1d(all_dofs, np.fromiter(blocked, dtype=np.int64), assume_unique=True)

    def apply_blocked_boundaries(self, interfaces, *operators):
        """Reduce any number of operators consistently with the blocked DOFs.

        Dispatch rule (by shape):
          - (n_dof, n_dof) square operator        -> A[free, :][:, free]
          - (n_dof,)       vector                 -> v[free]
          - (m, n_dof)     coupling rows x dofs   -> A[:, free]
          - (n_dof, m)     dofs x coupling cols   -> A[free, :]
          - anything else                         -> returned untouched

        Stores `self._free` so `expand()` works afterwards.
        Returns operators in the same order they were passed.
        """
        n = self.matrix_size
        free = self._free_dofs(interfaces)
        self._free = free

        def reduce_one(A):
            if A is None:
                return None
            if issparse(A):
                r, c = A.shape
                if r == n and c == n:
                    return A[free, :][:, free].tocsr()
                elif r == n:
                    return A[free, :].tocsr()
                elif c == n:
                    return A[:, free].tocsr()
                return A
            arr = np.asarray(A)
            if arr.ndim == 1 and arr.shape[0] == n:
                return arr[free]
            if arr.ndim == 2:
                r, c = arr.shape
                if r == n and c == n:
                    return arr[np.ix_(free, free)]
                elif r == n:
                    return arr[free, :]
                elif c == n:
                    return arr[:, free]
            return A

        out = tuple(reduce_one(A) for A in operators)
        return out[0] if len(out) == 1 else out

    def expand(self, u_red):
        """Scatter a reduced vector back to the full DOF size (blocked DOFs = 0)."""
        if self._free is None:
            raise RuntimeError("expand() called before apply_blocked_boundaries(); no free-DOF map.")
        u = np.zeros(self.matrix_size, dtype=u_red.dtype)
        u[self._free] = u_red
        return u
