"""
Structural operators of the axisymmetric solid regions (role "solid").

  MecaAssembler         K (complex with hysteretic damping), M, C = α M + β K   — 2 DOF per node
  MecaAssemblerFourier  the same for azimuthal order m                         — 3 DOF per node
  MecaAssemblerNL       total-Lagrangian tangent K_T, M, C and internal forces  — 2 DOF per node

Blocked DOFs: the boundaries with role "clamped" (DirichletElimination, constraint matrix T
for the Fourier orders).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import issparse

from loudpy.fem.assembly import ElementAssembler
from loudpy.fem.dofs import DirichletElimination
from loudpy.fem.kernels import mass_kernel, coefficient, expand_components
from loudpy.fem.solid import (axisymmetric_kernel, fourier_kernel, material_matrix_D, material_matrix_D6,
                              total_lagrangian_kernel)
from loudpy.physics.registry import select
from loudpy.errors import ModelError, StateError


class _Structural(DirichletElimination, ElementAssembler):
    PTS_PER_TRI = 6
    dofs_per_node = 2

    def __init__(self, objects):
        solids = select(objects, "solid")
        if not solids:
            raise ModelError(f"{type(self).__name__}: no solid region (role 'solid') found.")
        self._setup(solids)
        self._free = None

    def _clamped(self, objects):
        return select(objects, "clamped")


def _mass_blocks(asm, ec, coef, components):
    """Consistent mass blocks ∫ coef N_a N_b 2πr dA on every component, (n_e, (c·n)²)."""
    N, dN, w = asm.tabulation
    n = N.shape[1]
    Ms = np.empty((len(ec), n * n))
    mass_kernel(ec, N, dN, w, coefficient(coef, ec, N), True, Ms)
    return expand_components(Ms, components)


class MecaAssembler(_Structural):
    def assemble(self):
        Kv = np.zeros(self.V_size, dtype=np.complex128)
        Mv = np.zeros(self.V_size, dtype=np.float64)
        Cv = np.zeros(self.V_size, dtype=np.float64)

        for sd, ec, gdofs, s0, s1, n_e in self.prep:
            N, dN, w = self.tabulation
            Ke = np.empty((n_e, self.block))
            axisymmetric_kernel(ec, N, dN, w, material_matrix_D(sd.E, sd.nu), Ke)
            Me = _mass_blocks(self, ec, float(sd.rho), 2)

            eta = getattr(sd, "eta", 0.0)
            alpha = getattr(sd, "alpha_ray", 0.0)
            beta = getattr(sd, "beta_ray", 0.0)

            Ke_flat, Me_flat = Ke.ravel(), Me.ravel()
            Kv[s0:s1] = Ke_flat * (1 + 1j * eta)
            Mv[s0:s1] = Me_flat
            Cv[s0:s1] = alpha * Me_flat + beta * Ke_flat

        return self._to_csr(Kv), self._to_csr(Mv), self._to_csr(Cv)


class MecaAssemblerNL(_Structural):
    """
    Nonlinear (total-Lagrangian) structural assembler, axisymmetric T6.

    nl = MecaAssemblerNL(model.parts)
    K_T, M, C, F_int = nl.assemble_tangent(u_full)     # full (unreduced) operators
    """

    def assemble_tangent(self, u_full: np.ndarray):
        """
        K_T, M, C (Rayleigh) and F_int at the displacement `u_full` (full size, (n_dof,)).

        Returns
        -------
        K_T   : csr_matrix  tangent stiffness  (full size)
        M     : csr_matrix  consistent mass    (full size, state-independent)
        C     : csr_matrix  Rayleigh damping   (full size)
        F_int : ndarray     internal force     (full size)
        """
        Kv = np.zeros(self.V_size, dtype=np.float64)
        Mv = np.zeros(self.V_size, dtype=np.float64)
        Cv = np.zeros(self.V_size, dtype=np.float64)
        F_full = np.zeros(self.matrix_size, dtype=np.float64)

        for sd, ec, gdofs, s0, s1, n_e in self.prep:
            D = material_matrix_D(sd.E, sd.nu)
            rho = float(sd.rho)
            alpha = float(getattr(sd, "alpha_ray", 0.0))
            beta = float(getattr(sd, "beta_ray", 0.0))

            u_e_all = u_full[gdofs]  # (n_e, 12)

            Ke_out = np.empty((n_e, self.block), dtype=np.float64)
            Me_out = np.empty((n_e, self.block), dtype=np.float64)
            Ce_out = np.empty((n_e, self.block), dtype=np.float64)
            Fe_out = np.empty((n_e, self.ndof_e), dtype=np.float64)

            N, dN, w = self.tabulation
            total_lagrangian_kernel(ec, np.ascontiguousarray(u_e_all, dtype=np.float64), N, dN, w, D,
                                    rho, alpha, beta, Ke_out, Me_out, Ce_out, Fe_out)

            Kv[s0:s1] = Ke_out.ravel()
            Mv[s0:s1] = Me_out.ravel()
            Cv[s0:s1] = Ce_out.ravel()

            F_full += np.bincount(np.asarray(gdofs).ravel(), weights=Fe_out.ravel(),
                                  minlength=self.matrix_size)

        return self._to_csr(Kv), self._to_csr(Mv), self._to_csr(Cv), F_full


class MecaAssemblerFourier(ElementAssembler):
    """
    Structural matrices of azimuthal order m (see fem.solid.fourier_kernel), 3 DOF per node
    ordered (U_r, U_z, U_θ), with the constraints of order m eliminated by a
    transformation u_full = T u_red:

      - axis nodes (r = 0), exact regularity of the 3D field, set automatically:
            m = 0 : U_r = U_θ = 0
            m = 1 : U_z = 0 and U_θ = −U_r   (lateral motion of the axis)
            m ≥ 2 : U_r = U_z = U_θ = 0
        clamped boundaries are ignored on the axis (their "_r_" axis condition only
        holds for m = 0);
      - clamped boundaries elsewhere: the components of their direction are
        blocked; a boundary blocking both r and z ("_rz_", a clamp) also blocks U_θ.
    """
    PTS_PER_TRI = 6
    dofs_per_node = 3

    def __init__(self, objects, m: int, axis_tol: float = 1e-9):
        solids = select(objects, "solid")
        if not solids:
            raise ModelError("MecaAssemblerFourier: no solid region (role 'solid') found.")
        if int(m) != m or m < 0:
            raise ValueError(f"azimuthal order m must be a non-negative integer, got {m}")
        self.m = int(m)
        self.subdomains = solids
        self.PTS_PER_TRI = self._cell_width(solids)
        self.ndof_e = self.PTS_PER_TRI * self.dofs_per_node
        self.block = self.ndof_e ** 2
        self.unique_tags = np.unique(np.concatenate([sd.node_tags for sd in solids]))
        self.tag_to_local = {int(t): i for i, t in enumerate(self.unique_tags)}
        self.matrix_size = len(self.unique_tags) * self.dofs_per_node
        r = np.zeros(len(self.unique_tags))
        for sd in solids:
            idx = np.searchsorted(self.unique_tags, sd.node_tags)
            r[idx] = np.asarray(sd.node_coords)[:, 0]
        self.node_r = r
        self.axis_tol = axis_tol * max(1.0, float(np.abs(r).max()))
        self._build_topology()
        self._free = None
        self.T = None

    def assemble(self):
        Kv = np.zeros(self.V_size, dtype=np.complex128)
        Mv = np.zeros(self.V_size, dtype=np.float64)
        Cv = np.zeros(self.V_size, dtype=np.float64)
        for sd, ec, gdofs, s0, s1, n_e in self.prep:
            N, dN, w = self.tabulation
            Ke = np.empty((n_e, self.block))
            fourier_kernel(ec, N, dN, w, material_matrix_D6(sd.E, sd.nu), self.m, Ke)
            # ∫ r dθ = 2π r for m = 0, π r for m ≥ 1: the axisymmetric mass, halved
            Me = _mass_blocks(self, ec, float(sd.rho) * (1.0 if self.m == 0 else 0.5), 3)
            eta = getattr(sd, "eta", 0.0)
            alpha = getattr(sd, "alpha_ray", 0.0)
            beta = getattr(sd, "beta_ray", 0.0)
            Ke_flat, Me_flat = Ke.ravel(), Me.ravel()
            Kv[s0:s1] = Ke_flat * (1 + 1j * eta)
            Mv[s0:s1] = Me_flat
            Cv[s0:s1] = alpha * Me_flat + beta * Ke_flat
        return self._to_csr(Kv), self._to_csr(Mv), self._to_csr(Cv)

    # ------------------------------------------------------------------
    def constraint_matrix(self, interfaces):
        """Sparse T (n_full x n_red) with u_full = T u_red (see the class docstring)."""
        from scipy.sparse import csr_matrix
        on_axis = np.abs(self.node_r) < self.axis_tol
        blocked, slaves = set(), {}                 # slave dof -> (master dof, factor)
        for i in np.flatnonzero(on_axis):
            dr, dz, dt = 3 * i, 3 * i + 1, 3 * i + 2
            if self.m == 0:
                blocked |= {dr, dt}
            elif self.m == 1:
                blocked.add(dz)
                slaves[dt] = (dr, -1.0)
            else:
                blocked |= {dr, dz, dt}
        for iface in select(interfaces, "clamped"):
            comps = set(iface.direction)
            if {0, 1} <= comps:
                comps.add(2)                         # a clamp also holds the azimuthal motion
            for tag in iface.node_tags:
                i = self.tag_to_local.get(int(tag))
                if i is None or on_axis[i]:
                    continue
                blocked |= {3 * i + c for c in comps}
        masters = [d for d in range(self.matrix_size) if d not in blocked and d not in slaves]
        col = {d: k for k, d in enumerate(masters)}
        rows, cols, vals = list(masters), list(range(len(masters))), [1.0] * len(masters)
        for s, (mdof, fac) in slaves.items():
            if mdof in col:
                rows.append(s)
                cols.append(col[mdof])
                vals.append(fac)
        self._free = np.array(masters, dtype=np.int64)
        return csr_matrix((vals, (rows, cols)), shape=(self.matrix_size, len(masters)))

    def apply_blocked_boundaries(self, interfaces, *operators):
        """Reduce square operators (Tᵀ A T) and vectors (Tᵀ v); stores T for expand()."""
        T = self.constraint_matrix(interfaces)
        self.T = T
        n = self.matrix_size

        def reduce_one(A):
            if A is None:
                return None
            if issparse(A) and A.shape == (n, n):
                return (T.T @ A @ T).tocsr()
            arr = np.asarray(A)
            if arr.ndim == 1 and arr.shape[0] == n:
                return T.T @ arr
            return A

        out = tuple(reduce_one(A) for A in operators)
        return out[0] if len(out) == 1 else out

    def expand(self, u_red):
        if self.T is None:
            raise StateError("expand() called before apply_blocked_boundaries().")
        return self.T @ u_red
