import numpy as np
import re
from loudpy.assembly.core.helpers import _local_coords
from loudpy.assembly.kernels.indexing import _fill_ij_square
from loudpy.assembly.core.base import _SparseAssemblerBase
from loudpy.assembly.kernels.elements import (
    _batch_H,
    _batch_K,
    _batch_Q,
    _batch_M,
    _batch_TL,
    _batch_Q_pml, 
    _batch_H_pml
)
from loudpy.Finite_Elements_formulations.FE_kernels import material_matrix_D
import numpy as np
from loudpy.assembly.core.helpers import _local_coords
from loudpy.assembly.kernels.indexing import _fill_ij_square
from loudpy.Finite_Elements_formulations.FE_kernels import material_matrix_D
import numpy as np
from scipy.sparse import issparse

from loudpy.SubDomains.SubDomainMeca import SubDomainMeca
from loudpy.SubDomains.SubDomainAcou import SubDomainAcou



class Assembler(_SparseAssemblerBase):
    """Assembles square FE matrices over T6 triangular subdomains."""

    def _build_topology(self):
        n_elem_total = sum(len(sd.tri) for sd in self.subdomains)
        V_size = n_elem_total * self.block
        I = np.empty(V_size, dtype=np.int32)
        J = np.empty(V_size, dtype=np.int32)

        self.prep = []
        offset = 0
        for sd in self.subdomains:
            n_e = len(sd.tri)
            tri_flat = np.asarray(sd.tri).ravel()

            ec = _local_coords(
                sd.node_tags, sd.node_coords, tri_flat, n_e, self.PTS_PER_TRI
            )

            gidx = np.searchsorted(self.unique_tags, tri_flat).reshape(
                n_e, self.PTS_PER_TRI
            )
            gdofs = np.ascontiguousarray(
                (gidx[..., None] * self.dofs_per_node + np.arange(self.dofs_per_node))
                .reshape(n_e, self.ndof_e)
                .astype(np.int32)
            )

            s0 = offset * self.block
            s1 = s0 + n_e * self.block
            _fill_ij_square(gdofs, I, J, s0)

            self.prep.append((sd, ec, s0, s1, n_e))
            offset += n_e

        self._finalize_pattern(I, J, V_size, (self.matrix_size, self.matrix_size))


class MecaAssembler(Assembler):
    PTS_PER_TRI = 6
    dofs_per_node = 2

    def __init__(self, subdomains):
        subdomains = [s for s in subdomains if isinstance(s, SubDomainMeca)]
        if not subdomains:
            raise ValueError("MecaAssembler: no SubDomainMeca found.")
        self.subdomains = subdomains
        self.ndof_e = self.PTS_PER_TRI * self.dofs_per_node
        self.block = self.ndof_e**2
        self.unique_tags = np.unique(
            np.concatenate([sd.node_tags for sd in subdomains])
        )
        self.tag_to_local = {int(t): i for i, t in enumerate(self.unique_tags)}
        self.matrix_size = len(self.unique_tags) * self.dofs_per_node
        self._build_topology()
        self._free = None

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------
    def assemble(self):
        Kv = np.zeros(self.V_size, dtype=np.complex128)
        Mv = np.zeros(self.V_size, dtype=np.float64)
        Cv = np.zeros(self.V_size, dtype=np.float64)

        for sd, ec, s0, s1, n_e in self.prep:
            Ke = np.empty((n_e, self.block))
            Me = np.empty((n_e, self.block))
            _batch_K(ec, material_matrix_D(sd.E, sd.nu), Ke)
            _batch_M(ec, sd.rho, Me)

            eta = getattr(sd, "eta", 0.0)
            alpha = getattr(sd, "alpha_ray", 0.0)
            beta = getattr(sd, "beta_ray", 0.0)

            Ke_flat, Me_flat = Ke.ravel(), Me.ravel()
            Kv[s0:s1] = Ke_flat * (1 + 1j * eta)
            Mv[s0:s1] = Me_flat
            Cv[s0:s1] = alpha * Me_flat + beta * Ke_flat

        return self._to_csr(Kv), self._to_csr(Mv), self._to_csr(Cv)

    # ------------------------------------------------------------------
    # Dirichlet BCs (elimination)
    # ------------------------------------------------------------------
    def _blocked_dofs(self, interfaces):
        """Collect blocked global DOF indices from constrained interfaces."""
        from loudpy.Interfaces.InterfaceConstrained import (
            InterfaceConstrained,
        )  # local import to avoid cycles

        blocked = set()
        for iface in interfaces:
            if not isinstance(iface, InterfaceConstrained):
                continue
            comps = iface.direction
            for tag in iface.node_tags:
                local = self.tag_to_local[tag]  # provided by base Assembler
                for c in comps:
                    blocked.add(self.dofs_per_node * local + c)
        return blocked

    def _free_dofs(self, interfaces):
        blocked = self._blocked_dofs(interfaces)
        all_dofs = np.arange(self.matrix_size)
        return np.setdiff1d(
            all_dofs, np.fromiter(blocked, dtype=np.int64), assume_unique=True
        )

    def apply_blocked_boundaries(self, interfaces, *operators):
        """Reduce any number of operators consistently with mech BCs.

        Dispatch rule (by shape):
          - (n_dof, n_dof) square mech operator   -> A[free, :][:, free]
          - (n_dof,)       mech vector            -> v[free]
          - (m, n_dof)     coupling rows x mech   -> A[:, free]
          - (n_dof, m)     mech x coupling cols   -> A[free, :]
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

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------
    def expand(self, u_red):
        """Scatter a reduced mech vector back to full mech DOF size.

        Blocked DOFs are filled with 0 (homogeneous Dirichlet).
        """
        if self._free is None:
            raise RuntimeError(
                "expand() called before apply_bcs(); no free-DOF map available."
            )
        u = np.zeros(self.matrix_size, dtype=u_red.dtype)
        u[self._free] = u_red
        return u

_FLOAT = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"

def _parse_pml(name: str):
    if "PML" not in name:
        return None

    info = {
        "use_r": False, "r0": 0.0, "tr": 0.0, "sgn_r": 0.0,
        "use_z": False, "z0": 0.0, "tz": 0.0, "sgn_z": 0.0
    }

    def _get(key, s):
        m = re.search(r"__" + key + r"=" + _FLOAT + r"(?:__|$)", s)
        return float(m.group(1)) if m else None

    if "__r0=" in name:
        info["use_r"] = True
        info["r0"]    = _get("r0", name)
        info["sgn_r"] = -1.0 if "r-" in name else 1.0
        info["tr"]    = _get("tr", name) or _get("t", name) or 0.0

    if "__z0=" in name:
        info["use_z"] = True
        info["z0"]    = _get("z0", name)
        info["sgn_z"] = -1.0 if "z-" in name else 1.0
        info["tz"]    = _get("tz", name) or _get("t", name) or 0.0

    return info


class AcouAssembler(Assembler):
    """Acoustic (1 DOF/node): H and Q matrices, with optional PML layers."""

    PTS_PER_TRI = 6
    dofs_per_node = 1

    def __init__(self, subdomains):
        subdomains = [s for s in subdomains if isinstance(s, SubDomainAcou)]
        self.subdomains = subdomains
        self.ndof_e = self.PTS_PER_TRI * self.dofs_per_node
        self.block  = self.ndof_e ** 2
       
        # Cache PML metadata on each subdomain (parse once)
        for sd in self.subdomains:
            sd._pml_info = _parse_pml(sd.name)
            sd.is_pml    = sd._pml_info is not None

        self.unique_tags = np.unique(
            np.concatenate([sd.node_tags for sd in subdomains])
        )
        self.tag_to_local = {int(t): i for i, t in enumerate(self.unique_tags)}
        self.matrix_size  = len(self.unique_tags) * self.dofs_per_node

        self._build_topology()

    # ---------------------------------------------------------------
    def assemble(self):
        """
        Parameters
        ----------
        omega : float or None
            Angular frequency. Required if any PML subdomain is present.
        """
        has_pml = any(sd.is_pml for sd in self.subdomains)
    

        dtype = np.complex128 if has_pml else np.float64
        Hv = np.zeros(self.V_size, dtype=dtype)
        Qv = np.zeros(self.V_size, dtype=dtype)

        for sd, ec, s0, s1, n_e in self.prep:
            He = np.empty((n_e, self.block), dtype=dtype)
            Qe = np.empty((n_e, self.block), dtype=dtype)
            
            sign_val = sd.orientation

            if sd.is_pml:
                info = sd._pml_info
                sigma_max = float(getattr(sd, "alpha", 1.0))
                p_order   = int(  getattr(sd, "n",     2  ))
                
                

                _batch_H_pml(
                    ec,
                    info["use_r"], info["r0"], info["tr"], info["sgn_r"],
                    info["use_z"], info["z0"], info["tz"], info["sgn_z"],
                    sigma_max, p_order, sd.omega_pml,sd.c, sign_val,
                    He, 
                )
                _batch_Q_pml(
                    ec, sd.c,
                    info["use_r"], info["r0"], info["tr"], info["sgn_r"],
                    info["use_z"], info["z0"], info["tz"], info["sgn_z"],
                    sigma_max, p_order, sd.omega_pml, sign_val,
                    Qe,
                )
                
            else:
                _batch_H(ec, sign_val, He)
                _batch_Q(ec, sd.c, sign_val, Qe)
                rho = sd.rho

            Hv[s0:s1] = He.ravel()
            Qv[s0:s1] = Qe.ravel()

        return self._to_csr(Hv), self._to_csr(Qv), rho

class MecaAssemblerNL(_SparseAssemblerBase):
    """
    Nonlinear (Total-Lagrangian) mechanical assembler for axisymmetric T6.

    Usage
    -----
    >>> nl = MecaAssemblerNL(fem_objects)
    >>> K_red, M_red, C_red, F_int_red = nl.assemble_nl(u_full, interfaces)
    >>> # or, for time-domain use, just call nl.assemble_tangent(u_full)
    >>> # to get full (unreduced) sparse matrices + internal force vector.
    """

    PTS_PER_TRI = 6
    dofs_per_node = 2

    # ------------------------------------------------------------------
    def __init__(self, subdomains):
        subdomains = [s for s in subdomains if isinstance(s, SubDomainMeca)]
        if not subdomains:
            raise ValueError("MecaAssemblerNL: no SubDomainMeca found.")

        self.subdomains = subdomains
        self.ndof_e = self.PTS_PER_TRI * self.dofs_per_node  # 12
        self.block = self.ndof_e**2  # 144
        self.unique_tags = np.unique(
            np.concatenate([sd.node_tags for sd in subdomains])
        )
        self.tag_to_local = {int(t): i for i, t in enumerate(self.unique_tags)}
        self.matrix_size = len(self.unique_tags) * self.dofs_per_node
        self._build_topology()
        self._free = None

    # ------------------------------------------------------------------
    # Topology  (identical pattern to linear assembler)
    # ------------------------------------------------------------------
    def _build_topology(self):
        n_elem_total = sum(len(sd.tri) for sd in self.subdomains)
        V_size = n_elem_total * self.block
        I = np.empty(V_size, dtype=np.int32)
        J = np.empty(V_size, dtype=np.int32)

        self.prep = []
        offset = 0
        for sd in self.subdomains:
            n_e = len(sd.tri)
            tri_flat = np.asarray(sd.tri).ravel()

            ec = _local_coords(
                sd.node_tags, sd.node_coords, tri_flat, n_e, self.PTS_PER_TRI
            )

            gidx = np.searchsorted(self.unique_tags, tri_flat).reshape(
                n_e, self.PTS_PER_TRI
            )
            gdofs = np.ascontiguousarray(
                (gidx[..., None] * self.dofs_per_node + np.arange(self.dofs_per_node))
                .reshape(n_e, self.ndof_e)
                .astype(np.int32)
            )

            s0 = offset * self.block
            s1 = s0 + n_e * self.block
            _fill_ij_square(gdofs, I, J, s0)

            self.prep.append((sd, ec, gdofs, s0, s1, n_e))
            offset += n_e

        self._finalize_pattern(I, J, V_size, (self.matrix_size, self.matrix_size))

    # ------------------------------------------------------------------
    # Core nonlinear assembly  (state-dependent, called every NR iteration)
    # ------------------------------------------------------------------
    def assemble_tangent(self, u_full: np.ndarray):
        """
        Assemble K_T, M, C (Rayleigh), and F_int at the current
        displacement state `u_full`  (full-size, shape (n_dof,)).

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

            # Gather element displacements  (n_e, 12)
            u_e_all = u_full[gdofs]  # fancy index: (n_e, 12)

            Ke_out = np.empty((n_e, self.block), dtype=np.float64)
            Me_out = np.empty((n_e, self.block), dtype=np.float64)
            Ce_out = np.empty((n_e, self.block), dtype=np.float64)
            Fe_out = np.empty((n_e, self.ndof_e), dtype=np.float64)

            _batch_TL(ec, u_e_all, D, rho, alpha, beta, Ke_out, Me_out, Ce_out, Fe_out)

            Kv[s0:s1] = Ke_out.ravel()
            Mv[s0:s1] = Me_out.ravel()
            Cv[s0:s1] = Ce_out.ravel()

            # Scatter internal forces (no atomics needed – serial loop)
            for e in range(n_e):
                for i in range(self.ndof_e):
                    F_full[gdofs[e, i]] += Fe_out[e, i]

        K_T = self._to_csr(Kv)
        M = self._to_csr(Mv)
        C = self._to_csr(Cv)
        return K_T, M, C, F_full

    # ------------------------------------------------------------------
    # Dirichlet BCs  (same logic as linear MecaAssembler)
    # ------------------------------------------------------------------
    def _blocked_dofs(self, interfaces):
        from loudpy.Interfaces.InterfaceConstrained import InterfaceConstrained

        blocked = set()
        for iface in interfaces:
            if not isinstance(iface, InterfaceConstrained):
                continue
            comps = iface.direction
            for tag in iface.node_tags:
                local = self.tag_to_local[tag]
                for c in comps:
                    blocked.add(self.dofs_per_node * local + c)
        return blocked

    def _free_dofs(self, interfaces):
        blocked = self._blocked_dofs(interfaces)
        all_dofs = np.arange(self.matrix_size)
        return np.setdiff1d(
            all_dofs, np.fromiter(blocked, dtype=np.int64), assume_unique=True
        )

    def apply_blocked_boundaries(self, interfaces, *operators):
        """Identical dispatch logic to linear MecaAssembler."""
        from scipy.sparse import issparse

        n = self.matrix_size
        free = self._free_dofs(interfaces)
        self._free = free

        def _reduce(A):
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

        out = tuple(_reduce(A) for A in operators)
        return out[0] if len(out) == 1 else out

    def expand(self, u_red: np.ndarray) -> np.ndarray:
        """Scatter reduced vector back to full DOF size (blocked = 0)."""
        if self._free is None:
            raise RuntimeError("expand() called before apply_blocked_boundaries().")
        u = np.zeros(self.matrix_size, dtype=u_red.dtype)
        u[self._free] = u_red
        return u



