"""
FrequencyResponse: harmonic response of a model to a drive.

    res = lp.FrequencyResponse(model, drive=lp.Force(1.0)).solve(freqs)
    res = lp.FrequencyResponse(model, drive=lp.Voltage(2.83, Re=8, Le=4e-3, Bl=7)).solve(freqs)
    res = lp.FrequencyResponse(model).solve(freqs, remesh=lp.PerWavelength(bands=[20, 160, 1280, 5120]))

The structure alone is solved when the model has no air region, the coupled structure + air
otherwise (coupled= forces the choice). With remesh=, the air (and PML) mesh follows the
wavelength: a new mesh per frequency band, one result block per band.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from loudpy.checks import check
from loudpy.model.drives import Force, Voltage
from loudpy.model.regions import PML, Air
from loudpy.results.results import Block, FrequencyResults
from loudpy.study import records
from loudpy.study.base import Study
from loudpy.errors import ModelError


@dataclass
class PerWavelength:
    """
    Air and PML meshes that follow the wavelength λ = c / f0, f0 = first frequency of each band.

    bands          : frequencies where a new band (and mesh) starts: a frequency f is in the band
                     of the largest edge ≤ f.   Or every: a new mesh every `every` frequencies.
    air_elements   : air element size λ / air_elements, at most air_max_size
                     (default: the mesh_size given to the Air region)
    pml_thickness  : PML thickness in wavelengths;  pml_elements: elements across it
    c              : speed of sound for λ (default: the Air material)
    folder         : where the band meshes are written (default: next to the geometry)
    """
    bands: list = field(default_factory=list)
    every: int | None = None
    air_elements: float = 6.0
    air_max_size: float | None = None
    pml_thickness: float = 0.25
    pml_elements: float = 4.0
    c: float | None = None
    folder: str | Path | None = None

    def __post_init__(self):
        from loudpy.checks import check
        check(self.bands, "frequency", "PerWavelength.bands")
        check(self.every, "positive", "PerWavelength.every")
        check(self.air_elements, "elements_per_wavelength", "PerWavelength.air_elements")
        check(self.air_max_size, "mesh_size", "PerWavelength.air_max_size")
        check(self.pml_thickness, "wavelengths", "PerWavelength.pml_thickness")
        check(self.pml_elements, "positive", "PerWavelength.pml_elements")
        check(self.c, "speed_of_sound", "PerWavelength.c")

    def split(self, freqs) -> list[np.ndarray]:
        """The frequencies grouped by band (in the given order)."""
        freqs = np.asarray(freqs, float)
        if self.every:
            key = np.arange(len(freqs)) // int(self.every)
        elif len(self.bands):
            key = np.searchsorted(np.asarray(self.bands, float), freqs, side="right")
        else:
            raise ModelError("PerWavelength needs bands= or every=")
        out, start = [], 0
        for i in range(1, len(freqs) + 1):
            if i == len(freqs) or key[i] != key[start]:
                out.append(freqs[start:i])
                start = i
        return out


class FrequencyResponse(Study):
    """Harmonic response of a Model (lp.Force or lp.Voltage on the coil) or of an lp.Scene
    (lp.Force, or {driver name: lp.Force}).

        res = lp.FrequencyResponse(model, drive=lp.Force(1.0)).solve(np.geomspace(20, 20000, 200))"""
    accepts_scene = True

    def __init__(self, model, drive=None, coupled: bool | None = None, reuse_factorization: bool = True):
        super().__init__(model)
        self.drive = drive if drive is not None else Force(1.0)
        if self.is_scene:                                    # 3D scene: Force or {driver name: Force}
            drives = self.drive.values() if isinstance(self.drive, dict) else [self.drive]
            if not all(isinstance(d, Force) and not d.is_signal for d in drives):
                raise ModelError("a scene is driven by lp.Force (harmonic) or {driver name: lp.Force}")
            return
        if not isinstance(self.drive, (Force, Voltage)):
            raise ModelError(f"drive must be lp.Force or lp.Voltage, got {self.drive!r}")
        if isinstance(self.drive, Force) and self.drive.is_signal:
            raise ModelError("FrequencyResponse needs a harmonic Force (a complex amplitude), not a signal")
        has_air = any(isinstance(r, Air) for r in model.regions)
        self.coupled = has_air if coupled is None else coupled
        if self.coupled and not has_air:
            raise ModelError("coupled=True needs an air region (lp.Air) in the model")
        if isinstance(self.drive, Voltage) and not self.coupled:
            raise NotImplementedError("a Voltage drive is available for the coupled structure + air model")
        self.reuse_factorization = reuse_factorization

    # ── one mesh ───────────────────────────────────────────────────────────
    def _solve_block(self, freqs, label=""):
        """Responses at `freqs` on the current mesh → one Block (the system comes from the builder)."""
        from loudpy.physics.builder import coupled_builder
        from loudpy.physics.systems import assemble, meshes, nodal
        from loudpy.solvers.linear import SweepSolver
        parts = self._parts()
        s = assemble(parts, air=self.coupled)
        builder = coupled_builder(s, parts, self.drive, self.coupled)
        solver = SweepSolver()
        solver.reuse = self.reuse_factorization and solver.reuse
        d = self.drive
        U, P, attrs = [], [], []
        for f in np.asarray(freqs, float):
            w = 2 * np.pi * f
            z = 1j * w
            x = builder.split(solver.solve(builder.at(z), builder.rhs(z)))
            U.append(s.meca.expand(x["u"]))
            a = {"omega": float(w)}
            if "p" in x:
                P.append(x["p"])
            if "I" in x:
                c = builder.fields["I"].info
                a.update(I_coil=complex(x["I"][0]), Ze=complex(c["Re"] + z ** c["n"] * c["Le"]), Re=c["Re"],
                         Le=c["Le"], n=c["n"], Bl=c["Bl"], U=complex(d.amplitude))
            attrs.append(a)
        ms = meshes(s, parts)
        fields = {"u": nodal(np.array(U), len(s.meca.unique_tags), 2)}
        if P:
            fields["p"] = nodal(np.array(P), len(s.acou.unique_tags), 1)
        meta = {"mesh": str(self.model._mesh_path)}
        self.solver_stats = dict(factorizations=solver.n_factor, gmres_solves=solver.n_gmres,
                                 gmres_iterations=solver.iters)
        return Block(ms, np.asarray(freqs, float), fields, records.scalars(attrs), meta, label), \
            self.model.boundary_nodes()

    # ── public ─────────────────────────────────────────────────────────────
    def solve(self, freqs, remesh: PerWavelength | None = None, store=None, **options) -> FrequencyResults:
        """
        Response at every frequency of `freqs` [Hz] (one block per mesh).
        Scene: store= the result file the 3D pressure field is streamed to (format 2); without it
        only the driver amplitudes, DtN coefficients and cavity pressures are kept.
        """
        if self.is_scene:
            from loudpy.study import scene_response
            self.results = scene_response.solve(self, freqs, store=store, **options)
            return self.results
        freqs = np.atleast_1d(np.asarray(freqs, float))
        check(freqs, "frequency", "FrequencyResponse.solve(freqs)")
        meta = {"study": "FrequencyResponse", "drive": repr(self.drive), "coupled": self.coupled,
                "geometry": str(self.model.geometry)}
        if remesh is None:
            block, bounds = self._solve_block(freqs)
            blocks = [block]
        else:
            blocks, bounds = self._solve_remeshed(freqs, remesh)
        self.results = FrequencyResults(blocks, meta, bounds)
        return self.results

    def _solve_remeshed(self, freqs, rule: PerWavelength):
        if not self.coupled:
            raise ModelError("remesh= adapts the air mesh: it needs the coupled model")
        air = [r for r in self.model.regions if isinstance(r, Air) and not isinstance(r, PML)]
        pml = [r for r in self.model.regions if isinstance(r, PML)]
        saved = [(r, r.mesh_size, getattr(r, "thickness", None), getattr(r, "tuned_to", None)) for r in air + pml]
        c = rule.c or self.model.materials.resolve(air[0].material).c
        cap = rule.air_max_size
        folder = Path(rule.folder) if rule.folder else self.model.geometry.parent
        blocks, bounds = [], {}
        try:
            for band in rule.split(freqs):
                f0 = float(band[0])
                lam = c / f0
                for r, size0, _, _ in saved[:len(air)]:
                    r.mesh_size = min(lam / rule.air_elements, cap if cap is not None else size0)
                for r in pml:
                    r.thickness = rule.pml_thickness * lam
                    r.mesh_size = r.thickness / rule.pml_elements
                    r.tuned_to = f0
                self.model.mesh(folder / f"mesh_{f0:.0f}Hz.msh")
                block, b = self._solve_block(band, label=f"{f0:.0f}Hz")
                block.meta.update(wavelength=lam, band_start=f0)
                blocks.append(block)
                bounds.update(b)
        finally:
            for r, size0, t0, tuned0 in saved:
                r.mesh_size = size0
                if isinstance(r, PML):
                    r.thickness, r.tuned_to = t0, tuned0
        return blocks, bounds
