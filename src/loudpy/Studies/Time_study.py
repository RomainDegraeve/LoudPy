import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

from loudpy.assembly import MecaAssembler, AcouAssembler, ForceAssembler, FsiAssembler, MecaAssemblerNL
from loudpy.Interfaces.InterfaceForced import InterfaceForced
from loudpy.Finite_Elements_formulations import compute_unit_force_T6 as TR6_unit
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
            }


    def assemble_init(self):
        self.assembler_nl = MecaAssemblerNL(self.fem_objects)
        self.forced_iface = next(s for s in self.fem_objects
                                 if isinstance(s, InterfaceForced))

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
    ):
        # ── Dimensions & free DOFs ────────────────────────────────────────────────
        n_dof = self.assembler_nl.matrix_size
        free  = self.assembler_nl._free_dofs(self.fem_objects)
        self.assembler_nl._free = free
        n_free = len(free)

        dof_map       = np.full(n_dof, -1, dtype=np.int32)
        dof_map[free] = np.arange(n_free)

        # ── Mass matrix (state-independent, assembled once at u=0) ───────────────
        u_zero             = np.zeros(n_dof, dtype=np.float64)
        _, M_full, _, _    = self.assembler_nl.assemble_tangent(u_zero)
        # Use chained slicing (same as K_red / C_red inside the loop) — NOT np.ix_
        M_red              = M_full[free, :][:, free].tocsc()

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

        # ── Generalized-Alpha parameters ──────────────────────────────────────────
        alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
        alpha_f =        rho_inf         / (rho_inf + 1.0)
        gam     = 0.5 - alpha_m + alpha_f
        beta    = 0.25 * (1.0 - alpha_m + alpha_f) ** 2

        c1 = 1.0 / (beta * dt ** 2)
        c2 = gam  / (beta * dt)

        # ── Initial conditions ────────────────────────────────────────────────────
        u_n = np.zeros(n_free, dtype=np.float64)
        v_n = np.zeros(n_free, dtype=np.float64)
        a_n = np.zeros(n_free, dtype=np.float64)

        results_u  = [np.zeros(n_dof, dtype=np.float64)]
        results_v  = [np.zeros(n_dof, dtype=np.float64)]
        results_a  = [np.zeros(n_dof, dtype=np.float64)]
        time_array = [0.0]


        # ── Time loop ─────────────────────────────────────────────────────────────
        for step in range(n_steps):
            t_n1 = (step + 1) * dt

            F_ext_n   = F_red_base * float(np.real(force_signal[step]))
            F_ext_n1  = F_red_base * float(np.real(force_signal[step + 1]))
            F_ext_mid = (1.0 - alpha_f) * F_ext_n1 + alpha_f * F_ext_n

            # ── Predictor ────────────────────────────────────────────────────────
            u_iter = u_n + dt * v_n + dt ** 2 * (0.5 - beta) * a_n
            v_iter = v_n + dt * (1.0 - gam) * a_n
            a_iter = np.zeros_like(a_n)

            # ── Newton-Raphson ────────────────────────────────────────────────────
            res_norm = np.inf
            for k in range(nr_max_iter):
                u_mid = (1.0 - alpha_f) * u_iter + alpha_f * u_n
                v_mid = (1.0 - alpha_f) * v_iter + alpha_f * v_n
                a_mid = (1.0 - alpha_m) * a_iter + alpha_m * a_n

                u_full_mid       = np.zeros(n_dof, dtype=np.float64)
                u_full_mid[free] = u_mid

                K_T_full, _, C_full_it, F_int_full = \
                    self.assembler_nl.assemble_tangent(u_full_mid)

                K_red     = K_T_full[free, :][:, free].tocsc()
                C_red     = C_full_it[free, :][:, free].tocsc()
                F_int_red = F_int_full[free]

                Residual = (F_ext_mid
                            - F_int_red
                            - M_red @ a_mid
                            - C_red @ v_mid)

                res_norm = np.linalg.norm(Residual)
                f_norm   = np.linalg.norm(F_ext_mid)

                # ── Step-0 / iter-0 deep diagnostics ─────────────────────────────
                if step == 0 and k == 0:
            
                    K_eff_diag0 = ((1.0 - alpha_f) * K_red
                                + (1.0 - alpha_f) * c2 * C_red
                                + (1.0 - alpha_m) * c1 * M_red).diagonal()
                    print(f"  K_eff diag mean={K_eff_diag0.mean():.6e}  min={K_eff_diag0.min():.6e}")

                if res_norm < nr_tol * (f_norm + 1e-12):
                    break

                K_eff = (  (1.0 - alpha_f) * K_red
                        + (1.0 - alpha_f) * c2 * C_red
                        + (1.0 - alpha_m) * c1 * M_red)

                delta_u = np.real(spsolve(K_eff, Residual))

                # ── Step-0 / iter-0 correction diagnostics ───────────────────────
                if step == 0 and k == 0:
                    print(f"  delta_u    max={np.max(np.abs(delta_u)):.6e}  norm={np.linalg.norm(delta_u):.6e}")
                    print(f"===================================\n")

                u_iter += delta_u
                a_iter += c1 * delta_u
                v_iter += c2 * delta_u

            else:
                print(f"  ⚠  NR did not converge at step {step}  "
                    f"(res_norm = {res_norm:.3e})")

            # ── Advance history ───────────────────────────────────────────────────
            u_n, v_n, a_n = u_iter, v_iter, a_iter

            u_full_step       = np.zeros(n_dof, dtype=np.float64)
            v_full_step       = np.zeros(n_dof, dtype=np.float64)
            a_full_step       = np.zeros(n_dof, dtype=np.float64)
            u_full_step[free] = u_n
            v_full_step[free] = v_n
            a_full_step[free] = a_n

            results_u.append(u_full_step)
            results_v.append(v_full_step)
            results_a.append(a_full_step)
            time_array.append(t_n1)

            if step % 10 == 0:
                print(f"  step {step:>5}/{n_steps}  t={t_n1:.4f}s  "
                    f"|u|_max={np.max(np.abs(u_n)):.3e}  "
                    f"(NR iters: {k + 1})")

        print("Nonlinear Generalized-Alpha sweep complete.")

        time_array = np.asarray(time_array)
        U = np.asarray(results_u)
        V = np.asarray(results_v)
        A = np.asarray(results_a)

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

