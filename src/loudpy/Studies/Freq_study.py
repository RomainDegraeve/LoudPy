# loudpy/Studies/FreqStudy.py
import numpy as np
from pathlib import Path
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from loudpy.assembly import MecaAssembler, AcouAssembler, ForceAssembler, FsiAssembler
from loudpy.Studies.Study_class import Problem
from loudpy.Studies.DofMapMixin import DofMapMixin


class FreqStudy(Problem, DofMapMixin):
    def __init__(self, problem: Problem):
        super().__init__(geo_path=problem.geo_path,
                         msh_path=problem.msh_path,
                         mat_path=problem.mat_path)
        self.fem_objects   = problem.fem_objects
        self.nodes         = problem.nodes
        self.specs_dom     = problem.specs_dom
        self._problem      = problem
        self._assemblers: dict = {}
        self._results: list[dict] = []
        self.specs_interf  = problem.specs_interf

         # ── collect interface node tags ────────────────────────────────────
        self._interface_meta = {}
        for spec in self.specs_interf:
            tags = self._get_interface_node_tags(spec)
            self._interface_meta[spec.name] = {
                "kind":      spec.__class__.__name__,
                "node_tags": tags,
            }


    def _build_meca(self):
        """Return (meca, K, M, R, F) without applying BCs yet."""
        meca  = MecaAssembler (self.fem_objects)
        force = ForceAssembler(self.fem_objects, meca)
        K, M, R = meca.assemble()
        F       = force.assemble()
        return meca, K, M, R, F

    def _finalize_meca(self, meca, K, M, R, F, Cam=None):
        self.K, self.M, self.R, self.F, self.Cam = meca.apply_blocked_boundaries(
            self.fem_objects, K, M, R, F, Cam)
        self.meca_asm              = meca
        self.n_dof_meca            = self.M.shape[0]
        self._assemblers["u_meca"] = meca

    def assemble_meca(self):
        meca, K, M, R, F = self._build_meca()
        self._finalize_meca(meca, K, M, R, F, Cam=None)

    def assemble_domains(self):
        meca, K, M, R, F = self._build_meca()

        acou = AcouAssembler(self.fem_objects)
        self.H, self.Q, self.rho = acou.assemble()

        Cam = FsiAssembler(self.fem_objects, meca, acou).assemble()

        self._finalize_meca(meca, K, M, R, F, Cam=Cam)

        self.acou_asm              = acou
        self.n_dof_acou            = self.H.shape[0]
        self._assemblers["p_acou"] = acou




    def solve_meca(self, freq: float, force: bool, *, record: bool = True):
        omega  = 2 * np.pi * freq
        omega2 = omega ** 2
        self.omega = omega

        A_meca = self.K + 1j * omega * self.R - omega2 * self.M

        rhs = np.zeros(self.n_dof_meca, dtype=complex)
        rhs[:self.n_dof_meca] = self.F * force

        sol      = spsolve(A_meca.tocsc(), rhs)
        sol_full = self.meca_asm.expand(sol)

        if record:
            self._results.append({
                "kind":   "freq",
                "value":  float(freq),
                "fields": {"u_meca": sol_full},   # ← expanded, no p_acou
                "attrs":  {"omega": float(omega)},
                "solver_params": {"f": float(freq)},
                "interfaces": self._interface_meta,
            })

        return sol_full


    # ───────────────────────── solve + record ─────────────────────────
    def solve_fsi(self, freq: float, force : bool, *, record: bool = True):
        omega  = 2 * np.pi * freq
        omega2 = omega ** 2
        self.omega = omega

        A_meca       = self.K + 1j * omega * self.R - omega2 * self.M
        A_acou       = self.H - omega2 * self.Q
        fsi_coupling = self.rho * omega2 * self.Cam

        system_matrix = bmat(
            [[A_meca,        -self.Cam.T],
             [-fsi_coupling,  A_acou    ]], format="csc")
        rhs = np.zeros(self.n_dof_meca + self.n_dof_acou, dtype=complex)
        rhs[:self.n_dof_meca] = self.F*force

        sol      = spsolve(system_matrix, rhs)
        sol_meca = sol[:self.n_dof_meca]
        sol_acou = sol[self.n_dof_meca:]

        sol_meca_full = self.meca_asm.expand(sol_meca)
        sol_acou_full = sol_acou

        if record:
            self._results.append({
                "kind":   "freq",
                "value":  float(freq),
                "fields": {"u_meca": sol_meca_full,
                           "p_acou": sol_acou_full},
                "attrs":  {"omega": float(omega)},
                "solver_params": {"f": float(freq)},
                 "interfaces": self._interface_meta,   
            })

        return sol_meca_full, sol_acou_full


    # ───────────────────────── one-shot saver ─────────────────────────────
    def save(self, path, **metadata):
        from loudpy.Files_Saver import ResultStore, StudySaver

        if not self._results:
            raise RuntimeError("Nothing to save — call solve_fsi() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with ResultStore(path, mode="w") as store:
            auto = {"geo_path": self.geo_path,
                    "msh_path": self.msh_path,
                    "mat_path": self.mat_path}
            for i, r in enumerate(self._results):
                for k, v in r["solver_params"].items():
                    auto[f"snap{i}_{k}"] = v
            store.set_metadata(**auto, **metadata)
            if hasattr(self._problem, "specs_dict"):
                store.save_specs(self._problem.specs_dict())
            mesh_id = store.save_mesh_from_problem(self._problem)
            StudySaver(self._problem, store).save_freq_snapshots(
                self._results, self.dof_maps(), mesh_id)

        print(f"✓ saved → {path}")
