"""
Modes: eigenmodes of a model.

    modes = lp.Modes(model).solve(f_max=6000)                        structural modes (in vacuo)
    modes = lp.Modes(model).solve(f_max=6000, orders=[0, 1, 2, 3])   azimuthal orders m (3D modes of
                                                                     the axisymmetric structure)
    modes = lp.Modes(model, physics="coupled").solve(f_max=9000, dtn_center=..., ...)
                                                                     structure + air with the exact
                                                                     DtN radiation (lp.Radiation)
    modes = lp.Modes(model, physics="acoustic").solve(f_max=3000)    air only (rigid structure)

Structural modes use several shift-invert solves spread over the band (method="multishift",
default) or one solve with n_modes around f_target (method="arpack").
"""
from __future__ import annotations

from loudpy.results.results import ModeResults
from loudpy.study import records
from loudpy.study.base import Study

PHYSICS = ("structure", "coupled", "acoustic")


class Modes(Study):
    accepts_scene = True

    def __init__(self, model, physics: str = "structure"):
        super().__init__(model)
        if physics not in PHYSICS:
            raise ValueError(f"physics must be one of {PHYSICS}, got {physics!r}")
        self.physics = physics

    def solve(self, f_max: float, f_min: float = 20.0, *, orders=None, method: str = "multishift",
              n_per_shift: int = 20, n_modes: int = 30, f_target: float = 100.0, **options) -> ModeResults:
        """
        f_min, f_max : band of the modes [Hz]
        orders       : azimuthal orders m (structure only; None: the axisymmetric modes, m = 0)
        method       : "multishift" (shifts over the band, n_per_shift modes each) or "arpack"
                       (n_modes around f_target)
        options      : passed to the coupled solver (dtn_center, dtn_half, dtn_order, mac_tol,
                       cover_zeta_max, max_shifts, cover_goal, checkpoint_path, …)
        """
        if self.is_scene:                       # drivers + cavity modes (scene configuration "cavity")
            from loudpy.study import scene_response
            self.results = scene_response.coupled_modes(self, f_max=f_max)
            return self.results
        from loudpy.Studies import EigenStudy
        engine = EigenStudy(self._problem())
        if self.physics == "structure":
            if orders is not None:
                for m in orders:
                    engine.solve_meca_eigen_fourier(m, f_min=f_min, f_max=f_max, n_modes_per_shift=n_per_shift,
                                                    **options)
            elif method == "multishift":
                engine.solve_meca_eigen_multishift(f_min=f_min, f_max=f_max, n_modes_per_shift=n_per_shift,
                                                   **options)
            elif method == "arpack":
                engine.solve_meca_eigen_ARPAC(n_modes=n_modes, f_min=f_min, f_max=f_max, f_target=f_target)
            else:
                raise ValueError(f"method must be 'multishift' or 'arpack', got {method!r}")
        elif self.physics == "coupled":
            engine.solve_fsi_eigen(f_min=f_min, f_max=f_max, n_modes_per_shift=n_per_shift, **options)
        else:
            engine.solve_acoustic_eigen_ARPAC(n_modes=n_modes, f_min=f_min, f_max=f_max, f_target=f_target)
        blocks = records.mode_blocks(engine._results, engine.dof_maps(), engine.fem_objects)
        meta = {"study": "Modes", "physics": self.physics, "geometry": str(self.model.geometry)}
        self.results = records.mode_results(blocks, meta, records.boundaries_of(engine._interface_meta))
        return self.results
