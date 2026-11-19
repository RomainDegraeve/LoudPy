import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from loudpy.solvers.linear import factorize  # noqa: F401
from loudpy.solvers.time import free_dof_reducer, generalized_alpha_newton

from loudpy.physics.structural import MecaAssembler, MecaAssemblerNL
from loudpy.physics.registry import select
from loudpy.fem.elements.tri6_axi import compute_unit_force_T6 as TR6_unit
from loudpy.Studies.Study_class import Problem      
from loudpy.Studies.DofMapMixin import DofMapMixin  # ← direct

class TimeStudy(Problem, DofMapMixin):                  # 
    def __init__(self, problem: Problem):
        super().__init__(geo_path=problem.geo_path,
                        msh_path=problem.msh_path,
                        mat_path=problem.mat_path,
                        subdomains_key=problem.subdomains_key)
        self.fem_objects   = problem.fem_objects
        self.nodes         = problem.nodes
        self.specs_dom     = problem.specs_dom
        self.specs_interf  = problem.specs_interf
        self._problem      = problem
        self._assemblers = {}  # dicts 
        self._results = [] # list of dicts 

        # ── collect interface node tags (same as FreqStudy) ───────────────
        self._interface_meta = {}
        for spec in self.specs_interf:
            tags = self._get_interface_node_tags(spec)
            self._interface_meta[spec.name] = {
                "kind":      spec.__class__.__name__,
                "node_tags": tags,
                "edges":     self._get_interface_edges(spec),

            }


    def assemble_init(self):
        self.assembler_nl = MecaAssemblerNL(self.fem_objects)
        self.forced_iface = select(self.fem_objects, "coil")[0]

        self.forced_elements = [
            [self.assembler_nl.tag_to_local[tag] for tag in edge]
            for edge in self.forced_iface.edges
        ]

        tag_to_pos = {int(tag): i for i, tag in
                      enumerate(self.forced_iface.node_tags)}
        self.node_coords_list = [
            np.array([[self.forced_iface.node_coords[tag_to_pos[tag]][0],
                       self.forced_iface.node_coords[tag_to_pos[tag]][1]]
                      for tag in edge])
            for edge in self.forced_iface.edges
        ]

        #  register
        self._assemblers["u_meca"] = self.assembler_nl

        
    def solve_time_domain_rayleigh_nl(self,
        force_amplitude: float,
        force_direction: str,
        force_signal: np.ndarray,
        dt: float,
        n_steps: int,
        nr_max_iter: int   = 10,
        nr_tol:      float = 1e-5,
        rho_inf:     float = 0.8,
        reuse_tangent: bool = True,
        reuse_max_iter: int = 3,
    ):
        """
        reuse_tangent: modified Newton — the LU of the effective matrix is kept across iterations
        and time steps (with dt ≪ 1/f_max it is dominated by the mass term and barely changes)
        and refactorised only when an iteration count exceeds reuse_max_iter; the convergence
        test (nr_tol) is unchanged, so the solution has the same accuracy.
        """
        # ── Dimensions & free DOFs ────────────────────────────────────────────────
        n_dof = self.assembler_nl.matrix_size
        free  = self.assembler_nl._free_dofs(self.fem_objects)
        self.assembler_nl._free = free
        n_free = len(free)

        dof_map       = np.full(n_dof, -1, dtype=np.int32)
        dof_map[free] = np.arange(n_free)

        # ── Mass matrix (state-independent, assembled once at u=0) ───────────────
        u_zero                        = np.zeros(n_dof, dtype=np.float64)
        K_T_zero, M_full, C_zero, F_int_zero = self.assembler_nl.assemble_tangent(u_zero)
        # Use chained slicing (same as K_red / C_red inside the loop) — NOT np.ix_
        M_red              = M_full[free, :][:, free].tocsc()

        reduce = free_dof_reducer(K_T_zero, free)   # one CSR pattern for every tangent


        # ── Diagnostic: compare NL tangent at u=0 with linear assembler ──────────
        _K_lin, _, _C_lin  = MecaAssembler(self.fem_objects).assemble()
        _K_lin_real        = _K_lin.real
        _dK                = (K_T_zero - _K_lin_real)
        _dC                = (C_zero   - _C_lin.real)
        print("── NL vs Linear assembler at u=0 ──────────────────────────────")
        print(f"  F_int(u=0): max={np.max(np.abs(F_int_zero)):.3e}  norm={np.linalg.norm(F_int_zero):.3e}")
        print(f"  ΔK:  max={np.max(np.abs(_dK.data)):.3e}  rel={np.max(np.abs(_dK.data))/np.max(np.abs(K_T_zero.data)):.3e}")
        print(f"  ΔC:  max={np.max(np.abs(_dC.data)):.3e}  rel={np.max(np.abs(_dC.data))/np.max(np.abs(C_zero.data)):.3e}")
        print("────────────────────────────────────────────────────────────────")

        # ── Spatial force vector ──────────────────────────────────────────────────
        dof_offset = {'_r_': 0, '_z_': 1}
        C_full = np.zeros(n_dof, dtype=np.float64)

        for element, node_coords in zip(self.forced_elements, self.node_coords_list):
            nodal_forces = TR6_unit(node_coords)
            for i, node_idx in enumerate(element):
                dof_idx          = 2 * node_idx + dof_offset[force_direction]
                C_full[dof_idx] += nodal_forces[i]

        C_red_vec = C_full[free]
        total     = C_red_vec.sum()

        
        C_red_vec /= total
        F_red_base = force_amplitude * C_red_vec    # (n_free,)

        print(f"F_red_base  max={np.max(np.abs(F_red_base)):.3e}  sum={F_red_base.sum():.3e}")

        # ── Generalised-α time integration (loudpy.solvers.time) ───────────────────
        def tangent(u_full):
            K_T, _, C, F_int = self.assembler_nl.assemble_tangent(u_full)
            return K_T, C, F_int

        time_array, U, V, A, n_factor = generalized_alpha_newton(
            tangent, M_red, reduce, free, n_dof, F_red_base, force_signal, dt, n_steps,
            rho_inf=rho_inf, nr_max_iter=nr_max_iter, nr_tol=nr_tol,
            reuse_tangent=reuse_tangent, reuse_max_iter=reuse_max_iter)
        results_u, results_v, results_a = list(U), list(V), list(A)

        self._results.append({
            "kind":   "time",
            "time":   time_array,
            "U":      U,
            "V":      V,
            "A":      A,
            "input": {
                "force_signal":    np.asarray(force_signal),
                "force_amplitude": float(force_amplitude),
                "force_direction": str(force_direction),
            },
            "solver_params": {
                "dt":           float(dt),
                "n_steps":      int(n_steps),
                "nr_max_iter":  int(nr_max_iter),
                "nr_tol":       float(nr_tol),
                "rho_inf":      float(rho_inf),
                "t_total":      float(time_array[-1]),
                "force_amplitude": float(force_amplitude),
                "force_direction": str(force_direction),
            },
            "attrs":      {"dt": float(dt), "n_steps": int(n_steps)},
            "interfaces": self._interface_meta,
        })

        return time_array, results_u, results_v, results_a


    def save(self, path, **metadata):
        """
        One-shot persistence: mesh + specs + interfaces + full time history.

        U/V/A are scattered to node-indexed arrays chunk-by-chunk (via
        StudySaver) so peak extra RAM ≈ chunk_size × n_nodes × dpn.
        """
        from pathlib import Path
        from loudpy.Files_Saver import ResultStore, StudySaver

        if not self._results:
            raise RuntimeError("Nothing to save — call solve_time_domain_*() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with ResultStore(path, mode="w") as store:
            auto = {"geo_path": self.geo_path,
                    "msh_path": self.msh_path,
                    "mat_path": self.mat_path}
            for i, r in enumerate(self._results):
                for k, v in r["solver_params"].items():
                    auto[f"run{i}_{k}"] = v
            store.set_metadata(**auto, **metadata)
            if hasattr(self._problem, "specs_dict"):
                store.save_specs(self._problem.specs_dict())
            mesh_id = store.save_mesh_from_problem(self._problem)
            StudySaver(self._problem, store).save_time_runs(
                self._results, self.dof_maps(), mesh_id)

        print(f"✓ saved → {path}")

