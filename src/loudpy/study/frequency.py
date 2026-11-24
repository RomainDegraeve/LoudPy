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
from loudpy.errors import ModelError, StateError


@dataclass
class PerWavelength:
    """
    Air and PML meshes that follow the wavelength: one mesh per frequency band.

    bands          : frequencies where a new band (and mesh) starts: a frequency f is in the band
                     of the largest edge ≤ f.   Or every: a new mesh every `every` frequencies.
    air_elements   : air elements per wavelength at the band's top frequency (the next band's start; the last
                     frequency of the last band, or of a group of every=): element size λ_top / air_elements,
                     at most air_max_size (default: the mesh_size given to the Air region)
    pml_thickness  : PML thickness in wavelengths at the band's start (its longest wavelength);
                     pml_elements: elements across it
    c              : speed of sound for λ (default: the Air material)
    folder         : where the band meshes are written (default: next to the geometry)
    size_at        : "top" (above); "start": the air sized at the band's first frequency, as before 0.2.2
                     (meshes built then are reused; the band's top has fewer elements per wavelength)

        rule.plan(model, freqs)        the meshes of a sweep: one per band, with its frequencies and sizes
        rule.mesh_files(model, freqs)  those meshes with their state (built, out of date, missing)
    """
    bands: list = field(default_factory=list)
    every: int | None = None
    air_elements: float = 6.0
    air_max_size: float | None = None
    pml_thickness: float = 0.25
    pml_elements: float = 4.0
    c: float | None = None
    folder: str | Path | None = None
    size_at: str = "top"

    def __post_init__(self):
        from loudpy.checks import check
        check(self.bands, "frequency", "PerWavelength.bands")
        check(self.every, "positive", "PerWavelength.every")
        check(self.air_elements, "elements_per_wavelength", "PerWavelength.air_elements")
        check(self.air_max_size, "mesh_size", "PerWavelength.air_max_size")
        check(self.pml_thickness, "wavelengths", "PerWavelength.pml_thickness")
        check(self.pml_elements, "positive", "PerWavelength.pml_elements")
        check(self.c, "speed_of_sound", "PerWavelength.c")
        if self.size_at not in ("top", "start"):
            raise ModelError(f"PerWavelength.size_at: 'top' or 'start', got {self.size_at!r}")

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

    def _top(self, band) -> float:
        """The highest frequency the band's air mesh is made for."""
        if self.size_at == "start":
            return float(band[0])
        if not self.every and len(self.bands):
            edges = np.asarray(self.bands, float)
            k = int(np.searchsorted(edges, float(band[0]), side="right"))
            if k < len(edges):
                return float(edges[k])                          # the next band's start
        return float(np.max(band))

    def plan(self, model, freqs) -> list[dict]:
        """
        The meshes of a sweep of `model` at `freqs`: one per band, [{"freqs", "start", "top", "wavelength" (at the
        top: the air's elements), "pml_wavelength" (at the start: the PML's thickness), "path"}].
        """
        air = [r for r in model.regions if isinstance(r, Air) and not isinstance(r, PML)]
        if not air:
            raise ModelError("PerWavelength sizes the air mesh: the model has no lp.Air region")
        c = float(self.c or model.materials.resolve(air[0].material).c)  # type: ignore[arg-type]
        folder = Path(self.folder) if self.folder else Path(model.geometry).parent
        out = []
        for band in self.split(np.atleast_1d(np.asarray(freqs, float))):
            f0, top = float(band[0]), self._top(band)
            out.append(dict(freqs=band, start=f0, top=top, wavelength=c / top, pml_wavelength=c / f0,
                            path=folder / f"mesh_{f0:.0f}Hz.msh"))
        return out

    def band_model(self, model, band: dict):
        """A copy of `model` (parameters resolved) with the air and PML sized for one band of plan()."""
        cap = self.air_max_size
        twin = model.resolved()
        for r in twin.regions:
            if isinstance(r, PML):
                r.thickness = self.pml_thickness * band["pml_wavelength"]
                r.mesh_size = r.thickness / self.pml_elements
                r.tuned_to = band["start"]
            elif isinstance(r, Air):
                r.mesh_size = min(band["wavelength"] / self.air_elements, cap if cap is not None else r.mesh_size)
        return twin

    def mesh_files(self, model, freqs) -> list[dict]:
        """
        The band meshes a sweep at `freqs` uses, as Scene.mesh_files gives them: [{"what", "title", "path", "state",
        "band"}], state "built", "out of date" (made with other settings: built again) or "missing". Nothing is
        meshed.
        """
        out = []
        for b in self.plan(model, freqs):
            path = Path(b["path"])
            if not path.exists():
                state = "missing"
            else:
                state = "built" if self.band_model(model, b).mesh_reusable(path) else "out of date"
            out.append(dict(what=f"air_{b['start']:.0f}", title=f"air {b['start']:.0f}–{b['top']:.0f} Hz", path=path,
                            state=state, band=[b["start"], b["top"]]))
        return out


class FrequencyResponse(Study):
    """Harmonic response of a Model (lp.Force or lp.Voltage on the coil) or of an lp.Scene
    (lp.Force or lp.Voltage on every driver's coil, or {driver name: lp.Force} / {driver name: lp.Voltage}: a
    Voltage solves one current per driver with the modes, its back-EMF included; or an lp.Circuit: amplifier,
    filters, passive parts and the coils, solved the same way — loudpy.circuit).

        res = lp.FrequencyResponse(model, drive=lp.Force(1.0)).solve(np.geomspace(20, 20000, 200))"""
    accepts_scene = True

    def __init__(self, model, drive=None, coupled: bool | None = None, reuse_factorization: bool = True):
        super().__init__(model)
        self.drive = drive if drive is not None else Force(1.0)
        if self.is_scene:                                    # 3D scene: Force / Voltage, or one per driver
            from loudpy.circuit import Circuit
            if isinstance(self.drive, Circuit):                     # amplifier, filters, coils: loudpy.circuit
                return
            drives = self.drive.values() if isinstance(self.drive, dict) else [self.drive]
            if not all((isinstance(d, Force) and not d.is_signal) or isinstance(d, Voltage) for d in drives):
                raise ModelError("a scene is driven by lp.Force (harmonic), lp.Voltage or an lp.Circuit, or {driver "
                                 "name: a Force / Voltage}")
            return
        from loudpy.circuit import Circuit
        if not isinstance(self.drive, (Force, Voltage, Circuit)):
            raise ModelError(f"drive must be lp.Force, lp.Voltage or lp.Circuit, got {self.drive!r}")
        if isinstance(self.drive, Force) and self.drive.is_signal:
            raise ModelError("FrequencyResponse needs a harmonic Force (a complex amplitude), not a signal")
        has_air = any(isinstance(r, Air) for r in model.regions)
        self.coupled = has_air if coupled is None else coupled
        if self.coupled and not has_air:
            raise ModelError("coupled=True needs an air region (lp.Air) in the model")
        self.reuse_factorization = reuse_factorization

    # ── one mesh ───────────────────────────────────────────────────────────
    def _solve_block(self, freqs, label="", model=None):
        """Responses at `freqs` on the mesh of `model` (default: the study's) → one Block: the structure and the air
        solved for a unit force on the coil (its unit response), the drive weighting it — a Force its amplitude, a
        Voltage or a Circuit the coil force Bl·I of the circuit solved with the coil's mobility
        (loudpy.physics.electric: exact, and Results.redrive changes the drive afterwards)."""
        from loudpy.physics import electric
        from loudpy.physics.builder import coupled_builder
        from loudpy.physics.systems import assemble, meshes, nodal
        from loudpy.solvers.linear import SweepSolver
        model = self.model if model is None else model
        if model.meshed is None:
            raise StateError("mesh the model first: model.mesh(path)")
        parts = model.parts
        s = assemble(parts, air=self.coupled)
        builder = coupled_builder(s, parts, Force(1.0), self.coupled)          # the coil's unit response
        f_coil = np.asarray(s.F, float).reshape(-1)
        solver = SweepSolver()
        solver.reuse = self.reuse_factorization and solver.reuse
        freqs = np.asarray(freqs, float)
        U, P, Y, attrs = [], [], [], []
        from loudpy import progress
        for k_f, f in enumerate(freqs):
            progress.step("frequency", k_f + 1, len(freqs))
            w = 2 * np.pi * f
            z = 1j * w
            x = builder.split(solver.solve(builder.at(z), builder.rhs(z)))
            U.append(s.meca.expand(x["u"]))
            Y.append(z * (f_coil @ x["u"]))                                  # coil velocity per newton
            if "p" in x:
                P.append(x["p"])
            attrs.append({"omega": float(w)})
        ms = meshes(s, parts)
        basis = {"fields": {"u": [nodal(np.array(U), len(s.meca.unique_tags), 2)]}, "extra": {}}
        if P:
            basis["fields"]["p"] = [nodal(np.array(P), len(s.acou.unique_tags), 1)]
        from loudpy.geometry.mesh_files import fingerprint
        meta = {"mesh": str(model._mesh_path), "mesh_fingerprint": fingerprint(model._mesh_path) or "",
                "basis": {"coils": [None]}}
        self.solver_stats = dict(factorizations=solver.n_factor, gmres_solves=solver.n_gmres,
                                 gmres_iterations=solver.iters)
        bounds = model.boundary_nodes()
        extra = records.scalars(attrs)
        extra["mobility"] = np.array(Y, complex).reshape(-1, 1, 1)
        circuit = electric.as_circuit(self.drive, [None], electric.motors_of(model), model.parameters)
        if circuit is None:
            W, more = electric.force_weights(self.drive, [None], len(freqs)), {}
        else:
            W, E = electric.solve(circuit, freqs, extra["mobility"], [None])
            more = {"e": E, **electric.coil_scalars(circuit, E, freqs)}
        block = Block(ms, freqs, {}, extra, meta, label, bounds, basis).driven(W)
        block.extra.update(more)
        return block, bounds

    # ── public ─────────────────────────────────────────────────────────────
    def solve(self, freqs, remesh: PerWavelength | None = None, store=None, output=None, **options) -> FrequencyResults:
        """
        Response at every frequency of `freqs` [Hz] (one block per mesh).
        Scene: store= the result file the 3D pressure field is streamed to (format 3); without it
        only the driver amplitudes, DtN coefficients and cavity pressures are kept.
        output: an lp.Output (fields kept, every n-th frequency, full-rate probes); 2D models.
        """
        if self.is_scene:
            if output is not None:
                raise ModelError("output= is for 2D models (a scene streams its 3D field with store=)")
            from loudpy.study import scene_response
            self.results = scene_response.solve(self, freqs, store=store, **options)
            return self.results
        freqs = np.atleast_1d(np.asarray(freqs, float))
        check(freqs, "frequency", "FrequencyResponse.solve(freqs)")
        meta = {"study": "FrequencyResponse", "drive": repr(self.drive), "coupled": self.coupled,
                "geometry": str(self.model.geometry)}
        from loudpy.physics.electric import as_circuit, motors_of, motors_to_meta
        motors = motors_of(self.model)                            # the coil's motor (its lp.Coil item)
        circuit = as_circuit(self.drive, [None], motors, self.model.parameters)   # a Voltage: its amplifier, coil
        meta["motors"] = motors_to_meta(motors)
        if circuit is not None:
            circuit.check(freqs)
            meta["circuit"] = circuit.to_dict()
        if remesh is None:
            block, bounds = self._solve_block(freqs)
            blocks = [block]
        else:
            blocks, bounds = self._solve_remeshed(freqs, remesh)
        from loudpy.results.output import apply
        self.results = self._finish(apply(FrequencyResults(blocks, meta, bounds), output))
        return self.results

    def _solve_remeshed(self, freqs, rule: PerWavelength):
        """One mesh per band on a copy of the model (the user's model and items are never changed)."""
        if not self.coupled:
            raise ModelError("remesh= adapts the air mesh: it needs the coupled model")
        from loudpy.geometry.mesh_files import annotate
        blocks, bounds = [], {}
        for plan in rule.plan(self.model, freqs):
            band, f0 = plan["freqs"], plan["start"]
            band_model = rule.band_model(self.model, plan)
            band_model.mesh(plan["path"])
            annotate(plan["path"], role=f"air {f0:.0f}–{plan['top']:.0f} Hz", band=[f0, plan["top"]],
                     air_elements=rule.air_elements, size_at=rule.size_at)
            block, b = self._solve_block(band, label=f"{f0:.0f}Hz", model=band_model)
            block.meta.update(wavelength=plan["wavelength"], band_start=f0, band_top=plan["top"],
                              pml_wavelength=plan["pml_wavelength"])
            blocks.append(block)
            bounds.update(b)
            self._meshed_copy = band_model                 # physical data of the groups for the description
        return blocks, bounds
