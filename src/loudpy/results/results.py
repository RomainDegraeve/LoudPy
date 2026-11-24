"""
Results: the outcome of any study, with one interface.

    res = lp.load("sweep.h5")                    FrequencyResults, ModeResults or TimeResults
    res.frequencies                              (res.times) the coordinate of the entries: NumPy array
    res.names                                    fields: "u" (displacement), "p" (pressure), "v", "a"

    res.field("p", f=1000)                       Field: the values on the mesh at the entry nearest 1 kHz
    res.spl(at=(0.0, 0.3))                       Data: SPL [dB re 20 µPa] over frequency (RMS)
    res.spl(at=lp.Arc(0.3, range(0, 91, 5)))     Data over (f, angle): directivity
    res.displacement(at=(0, 0.02), component="z")   Data: u_z over frequency (complex amplitude)
    res.velocity(...), res.acceleration(...)     jω u, −ω² u (harmonic) or the stored v, a (transient)
    res.probe("p", at=lp.Line((0, .05), (0, .5)))   any field along any selection
    res.scalar("I_coil"), res.impedance()        per-entry quantities as Data

    res.far_field(lp.Arc(1, range(0, 91, 10)), radius=0.2, baffle=z_b)   pressure outside the mesh (Kirchhoff–Helmholtz)
    res.arrays()                                 plain NumPy arrays (values, fields, meshes, per-entry quantities)
    res.save("sweep.h5")                         loudpy result format 3 (HDF5, with the model description)
    res.model, res.study                         what produced them: model tree, groups, geometry files; settings
    res.export("sweep.mat")                      MATLAB / NumPy (.npz) export of every field and mesh

Every Data / Field plots itself (.plot()) and exports (.export("x.csv" | ".mat" | ".npz" | ".vtu"),
.to_pandas()); np.asarray(d) gives the plain array.

A Results is a list of blocks: one block per mesh (a frequency sweep remeshed per band gives one
block per band) or per mode set (one per azimuthal order). The block structure is hidden:
evaluations run over all the blocks in order; `field` finds the block. Fields live on the mesh of
their physics: "u", "v", "a" on the "solid" mesh, "p" on the "air" mesh.
"""
from __future__ import annotations

from typing import Self

from dataclasses import dataclass, field as dc_field

import numpy as np

from loudpy.geometry.mesh import Mesh
from loudpy.results.data import Data, Field, spl  # noqa: F401  (spl re-exported)
from loudpy.results.quantities import COMPONENTS, quantity
from loudpy.results.selection import as_selection
from loudpy.errors import ResultsError

PHYSICS_OF = {"u": "solid", "v": "solid", "a": "solid", "p": "air"}
ELECTRIC_EXTRA = ("weights", "e", "I_coil", "U", "Ze", "Bl", "Re", "Le", "n", "I")    # set by the drive (redrive)
DERIVED = ("strain", "stress", "von_mises", "v", "a", "particle_velocity", "intensity")   # results.derived
ALIASES = {"u_meca": "u", "v_meca": "v", "a_meca": "a", "p_acou": "p"}     # former field names


def canonical(name: str) -> str:
    """The current name of a field (former names such as "u_meca" mapped to "u")."""
    return ALIASES.get(name, name)


@dataclass
class Block:
    """Entries on one set of meshes: values (n,) of the coordinate, fields (n, n_nodes[, dpn])."""
    meshes: dict[str, Mesh]
    values: np.ndarray
    fields: dict[str, np.ndarray]
    extra: dict[str, np.ndarray] = dc_field(default_factory=dict)      # per-entry scalars (zeta, I_coil, …)
    meta: dict = dc_field(default_factory=dict)
    label: str = ""
    boundaries: dict = dc_field(default_factory=dict)  # name → {"node_tags", "edges"} on this block's meshes
    basis: dict = dc_field(default_factory=dict)      # unit responses: {"fields" | "extra": {name: [one per coil]}}

    def __len__(self):
        return len(self.values)

    @property
    def coils(self) -> list | None:
        """The coils of the unit responses (a scene's driver names, [None] for a 2D model's coil), None when the
        block keeps none (one force on several drivers at once)."""
        return self.meta.get("basis", {}).get("coils") if self.basis else None

    def driven(self, W) -> "Block":
        """This block driven by the coil forces W (n, K): its fields and per-entry arrays the unit responses
        weighted and summed (fields read on demand when they are), "weights" W."""
        from loudpy.results.lazy import combine
        W = np.asarray(W, complex).reshape(len(self.values), -1)
        fields, extra = dict(self.fields), dict(self.extra)
        for name, arrays in self.basis.get("fields", {}).items():
            fields[name] = combine(arrays, W)
        for name, arrays in self.basis.get("extra", {}).items():
            extra[name] = combine([np.asarray(a) for a in arrays], W)
        extra["weights"] = W
        return Block(self.meshes, self.values, fields, extra, self.meta, self.label, self.boundaries, self.basis)

    def mesh_of(self, name: str) -> Mesh:
        """The mesh of a field (by its physics, or the block's only mesh)."""
        phys = PHYSICS_OF.get(name)
        if phys in self.meshes:
            return self.meshes[phys]
        if len(self.meshes) == 1:
            return next(iter(self.meshes.values()))
        raise ResultsError(f"no mesh for field {name!r} (meshes: {list(self.meshes)})")


def _at_rows(arr, W):
    """W @ arr[i] for every entry i of a (n, n_nodes[, dpn]) field (arrays or lazy datasets)."""
    W = W.tocsr()
    cols = np.unique(W.indices)                                       # the nodes W uses
    sub = np.asarray(arr[:, cols])                                     # reads only those (lazy datasets)
    Wc = W[:, cols].toarray()
    if sub.ndim == 2:
        return sub @ Wc.T                                              # (n, n_pts)
    return np.einsum("pk,nkd->npd", Wc, sub)                           # (n, n_pts, dpn)


class Results:
    """Results of a study: blocks (one per mesh / mode set) of fields over entries (frequencies,
    modes, times), per-entry quantities (`extra`), metadata and attachments. Read with lp.load."""
    kind = "results"
    coordinate = "value"
    dim = "index"                                  # name of the entry coordinate of evaluated Data
    real = False                                   # values are real (transients), not complex amplitudes

    def __init__(self, blocks: list[Block], meta: dict | None = None, boundaries: dict | None = None,
                 source=None):
        self.blocks = list(blocks)
        self.meta = dict(meta or {})
        self.boundaries = dict(boundaries or {})     # name → {"node_tags", "edges"}
        self.source = source
        self.attachments: dict = {}                  # name → {array name: array, "meta": {...}}
        self.model = None                            # the model description (results.description), format 3
        self.study: dict = {}                        # the study settings, format 3

    def _like(self, blocks) -> Self:
        """Results of the same class with other blocks, the same metadata and description."""
        out = type(self)(blocks, self.meta, self.boundaries, self.source)
        out.attachments, out.model, out.study = self.attachments, self.model, self.study
        return out

    # ── what is there ──────────────────────────────────────────────────────
    @property
    def values(self) -> np.ndarray:
        """The coordinate of every entry, all blocks in order."""
        return np.concatenate([b.values for b in self.blocks]) if self.blocks else np.zeros(0)

    @property
    def names(self) -> list[str]:
        """Names of the fields over all blocks."""
        out = []
        for b in self.blocks:
            out += [n for n in b.fields if n not in out]
        return out

    def __len__(self):
        return sum(len(b) for b in self.blocks)

    def extra(self, name: str) -> np.ndarray:
        """A per-entry quantity (e.g. "zeta", "I_coil") over all blocks, as an array (NaN where missing)."""
        return np.concatenate([np.asarray(b.extra[name]) if name in b.extra else np.full(len(b), np.nan)
                               for b in self.blocks])

    @property
    def scalars(self) -> list[str]:
        """Names of the per-entry quantities (scalar() / extra())."""
        out = []
        for b in self.blocks:
            out += [n for n, v in b.extra.items() if np.ndim(v) == 1 and n not in out]
        return out

    def scalar(self, name: str) -> Data:
        """A per-entry quantity (e.g. "I_coil", "zeta", "force") as Data over the entries."""
        return Data(self.extra(name), {self.dim: self._coord()}, quantity(name), meta=self._meta())

    def mesh(self, physics: str = "solid", block: int = 0) -> Mesh:
        """The mesh of a physics ("solid", "air") in a block."""
        return self.blocks[block].meshes[physics]

    def boundaries_of(self, block: Block) -> dict:
        """{name: {"node_tags", "edges"}} of the boundaries on the meshes of `block` (see results.boundaries)."""
        from loudpy.results.boundaries import boundaries_of
        return boundaries_of(self, block)

    def boundary(self, name: str) -> np.ndarray:
        """Node tags of the boundaries selected by `name` (e.g. "wet_front")."""
        from loudpy.geometry.naming import matches
        sel = [b for n, b in self.boundaries.items() if n == name or matches(name, n)]
        if not sel:
            raise ResultsError(f"no boundary {name!r} (boundaries: {list(self.boundaries)})")
        return np.unique(np.concatenate([np.asarray(b["node_tags"]) for b in sel]))

    def __repr__(self):
        v = self.values
        rng = f"{v.min():.4g}–{v.max():.4g}" if len(v) else "empty"
        return (f"{type(self).__name__}({len(self)} entries, {self.coordinate} {rng}, fields {self.names}, "
                f"{len(self.blocks)} block(s))")

    def _coord(self) -> np.ndarray:
        return self.values

    def _meta(self, **kw) -> dict:
        src = self.source if isinstance(self.source, str) or self.source is None else ", ".join(map(str, self.source))
        out = {"source": src} if src else {}
        return {**out, **kw}

    # ── one field at one entry ─────────────────────────────────────────────
    def _locate(self, index=None, value=None):
        """(block, row, global index) of the entry with global index `index`, or nearest to `value`."""
        if index is None:
            index = 0 if value is None else int(np.argmin(np.abs(self.values - value)))
        if index < 0:
            index += len(self)
        k = index
        for b in self.blocks:
            if k < len(b):
                return b, k, index
            k -= len(b)
        raise IndexError("entry index out of range")

    def field(self, name: str, index: int | None = None, value: float | None = None) -> Field:
        """
        One field on its mesh at one entry (by global index, or nearest to a coordinate value). Besides the
        stored fields: the derived fields "strain", "stress", "von_mises" (from u), "v", "a" (jω u, −ω² u of a
        harmonic result), "particle_velocity", "intensity" (from p) — loudpy.results.derived; stress and the
        air quantities use the physical data of the mesh groups stored with the results (format 3).
        """
        name = canonical(name)
        b, i, g = self._locate(index, value)
        if name not in b.fields and name in DERIVED:
            return self._derived(name, g)
        if name not in b.fields:
            raise ResultsError(f"no field {name!r} (fields: {list(b.fields)})")
        values = np.asarray(b.fields[name][i])
        return Field(name, b.mesh_of(name), np.real(values) if self.real else values, float(b.values[i]),
                     "t" if self.dim == "t" else "f", {k: v[i] for k, v in b.extra.items() if np.ndim(v) == 1},
                     meta=self._meta(index=g, block=b.label, **({"order": int(b.meta["order"])}
                                                                 if "order" in b.meta else {})))

    def _groups(self, what) -> dict:
        if self.model is None or not self.model.groups:
            raise ResultsError(f"{what} needs the physical data of the mesh groups: results saved with loudpy "
                               f"format 3 (or loudpy.results.derived with groups={{…}})")
        return self.model.groups

    def _derived(self, name, index) -> Field:
        from loudpy.results import derived
        if name in ("v", "a"):
            u = self.field("u", index=index)
            if self.real or u.coord != "f":
                raise ResultsError(f"no stored field {name!r} in these results")
            w = 2 * np.pi * u.coordinate
            k = 1j * w if name == "v" else -w ** 2
            return Field(name, u.mesh, k * u.values, u.coordinate, u.coord, u.extra, name,
                         {**u.meta, "derived": "jω u" if name == "v" else "−ω² u"})
        if name in ("strain", "stress", "von_mises"):
            u = self.field("u", index=index)
            if name == "strain":
                return derived.strain(u)
            s = derived.stress(u, self._groups("the stress"))
            return s if name == "stress" else derived.von_mises(s)
        p = self.field("p", index=index)
        fn = derived.particle_velocity if name == "particle_velocity" else derived.intensity
        return fn(p, self._groups(name.replace("_", " ")))

    # ── values over all entries ────────────────────────────────────────────
    def probe(self, name: str, at, component: str | None = None) -> Data:
        """
        A field at `at` (a point, points, lp.Arc, lp.Line …) for every entry: linear interpolation
        on 2D meshes, the nearest node on 3D meshes. Vector fields: one component ("r", "z",
        "theta") or all of them (a "component" dimension). Complex amplitudes for harmonic results.
        """
        if isinstance(name, (tuple, list, np.ndarray)) and isinstance(at, str):   # former order probe(point, name)
            name, at = at, name
        name = canonical(name)
        sel = as_selection(at)
        out = []
        for b in self.blocks:
            if name not in b.fields:
                raise ResultsError(f"no field {name!r} in block {b.label!r} (fields: {list(b.fields)})")
            mesh = b.mesh_of(name)
            out.append(_at_rows(b.fields[name], mesh.interpolation(sel.points(mesh.dim))))
        v = np.concatenate(out, axis=0)                                   # (n, n_pts[, dpn])
        if self.real:
            v = np.real(v)
        q = quantity(name)
        if v.ndim == 3 and component is not None:
            v, q = v[..., COMPONENTS[component]], quantity(f"{name}_{component}")
        coords: dict = {self.dim: self._coord()}
        if sel.dim is None:
            v = v[:, 0]
        else:
            coords[sel.dim] = sel.coord
        if v.ndim == len(coords) + 1:
            coords["component"] = np.array(["r", "z", "theta"][:v.shape[-1]])
        return Data(v, coords, q, meta=self._meta(at=sel.describe()))

    def pressure(self, at) -> Data:
        """Pressure p at `at` over the entries (complex amplitude [Pa])."""
        return self.probe("p", at)

    def spl(self, at, rms: bool = True) -> Data:
        """Sound pressure level [dB re 20 µPa] at `at` (RMS by default)."""
        return self.pressure(at).spl(rms)

    def displacement(self, at, component: str | None = None) -> Data:
        """Displacement u at `at` (one component "r", "z", "theta", or all)."""
        return self.probe("u", at, component)

    def velocity(self, at, component: str | None = None) -> Data:
        """Velocity at `at`: the stored field "v" if there is one, else jω u (harmonic)."""
        if "v" in self.names:
            return self.probe("v", at, component)
        return self.displacement(at, component).velocity()

    def acceleration(self, at, component: str | None = None) -> Data:
        """Acceleration at `at`: the stored field "a" if there is one, else −ω² u (harmonic)."""
        if "a" in self.names:
            return self.probe("a", at, component)
        return self.displacement(at, component).acceleration()

    # ── full-rate probes (lp.Output(probes=…)) ─────────────────────────────
    @property
    def probe_names(self) -> list[str]:
        """Names of the probes recorded with the results (lp.Output(probes=…))."""
        from loudpy.results.output import probe_names
        return probe_names(self)

    def probe_signal(self, name: str, field: str = "u", component: str | None = None) -> Data:
        """A field at a recorded probe for every computed entry (full rate, whatever Output(every=…) kept)."""
        from loudpy.results.output import probe_signal
        return probe_signal(self, name, field, component)

    # ── plain arrays ───────────────────────────────────────────────────────
    def arrays(self, block: int | None = None, fields=None):
        """
        The results as plain NumPy arrays, per block: {coordinate: (n,), field: (n, n_nodes[, dpn]),
        f"{field}_mesh": {"coords", "triangles" (2D, linear sub-triangles), "node_tags"}, per-entry quantities}.
        block: one block's dict (default: a list over the blocks; one block → its dict). This is what
        res.export writes (.mat / .npz).
        """
        from loudpy.results.export import block_arrays
        if block is not None:
            return block_arrays(self, self.blocks[block], fields)
        out = [block_arrays(self, b, fields) for b in self.blocks]
        return out[0] if len(out) == 1 else out

    # ── files ──────────────────────────────────────────────────────────────
    def save(self, path):
        """Write these results in the loudpy result format 3 (loudpy.results.schema): fields, meshes, the
        model description (res.model) and the study settings (res.study)."""
        from loudpy.results.schema import write
        write(self, path)
        return path

    def export(self, path, fields=None):
        """
        Every block (meshes, fields, per-entry quantities) to a file read outside loudpy:
        .mat (MATLAB / Octave: one struct per block), .npz (NumPy: keys block<i>_<name>) or .pvd (ParaView
        series: one .vtu per entry and mesh, the frequency / time / mode as the time step).
        fields: the fields to write (default: all).
        """
        from loudpy.results.export import export_results
        return export_results(self, path, fields)


class FrequencyResults(Results):
    """Harmonic responses: one entry per frequency (blocks: one per mesh of a remeshed sweep)."""
    kind = "frequency"
    coordinate = "frequency"
    dim = "f"

    @property
    def frequencies(self) -> np.ndarray:
        """Frequencies of the entries [Hz]."""
        return self.values

    def field(self, name: str, f: float | None = None, index: int | None = None) -> Field:  # type: ignore[override]
        """One field on its mesh at the frequency nearest f (or at a global entry index)."""
        return super().field(name, index, f)

    def far_field(self, at, radius: float, center=(0.0, 0.0), baffle: float | None = None,
                  c: float | None = None) -> Data:
        """
        Pressure [Pa] at `at` (points (r, z), lp.Arc …) outside a sphere enclosing the loudspeaker, radiated
        into free space: the Kirchhoff–Helmholtz integral of p and ∂p/∂n over the sphere (radius [m], centre
        on the axis; it must lie in the air, not in a PML) — see loudpy.results.farfield. baffle: z of a
        rigid infinite baffle (then the half sphere above it and the image Green's function). c: speed of
        sound (default: the air of the result's model description).
        """
        from loudpy.results.farfield import kirchhoff_helmholtz
        c = c or self._sound_speed()
        sel = as_selection(at)
        out = []
        for b in self.blocks:
            if "p" not in b.fields:
                raise ResultsError(f"no pressure field in block {b.label!r}: far_field needs the air")
            out.append(kirchhoff_helmholtz(b.mesh_of("p"), np.asarray(b.fields["p"]), 2 * np.pi * b.values / c,
                                           sel.points(2), radius, center, baffle))
        v = np.concatenate(out, axis=0)
        coords: dict = {"f": self.frequencies}
        if sel.dim is None:
            v = v[:, 0]
        else:
            coords[sel.dim] = sel.coord
        return Data(v, coords, quantity("p"), meta=self._meta(
            at=sel.describe(), far_field=f"Kirchhoff–Helmholtz over a sphere of radius {radius:g} m"
            + (f", baffle z = {baffle:g} m" if baffle is not None else "")))

    def baffle(self) -> float | None:
        """z [m] of the baffle's front face, read from the mesh (the plane the front air stands on); None in free
        field (loudpy.results.directivity)."""
        from loudpy.results.directivity import baffle
        return baffle(self)

    def directivity(self, radius: float | None = None, angles=None, center=None, mirror: bool = True) -> Data:
        """
        SPL [dB] over (f, angle) on an arc of `radius` [m] around `center` on the axis, angles [°] from the z axis;
        defaults read from the mesh: the centre on the baffle's front face (else the origin), the radius 0.8 × the
        air in front, 0 … 180° when there is air behind (else 0 … 90°); mirror: −θ … θ, as both sides of the axis
        (loudpy.results.directivity).
        """
        from loudpy.results.directivity import directivity
        return directivity(self, radius, angles, center, mirror)

    def volume_velocity(self, on: str) -> Data:
        """Volume velocity Q = ∫ jω u·n dA [m³/s] over the boundary `on` (n out of the structure)."""
        from loudpy.results.values import volume_velocity
        return volume_velocity(self, on)

    def radiated_power(self, on: str | None = None, radius: float | None = None, center=(0.0, 0.0),
                       baffle: float | None = None) -> Data:
        """Acoustic power [W] through the boundary `on` (from u and p) or through a sphere in the air (radius,
        centre on the axis, optional baffle; from p and ∇p) — loudpy.results.values."""
        from loudpy.results.values import radiated_power
        return radiated_power(self, on, radius, center, baffle)

    def thiele_small(self, on: str | None = None, coil: str | None = None) -> dict:
        """Thiele–Small parameters of a voltage-driven result (loudpy.results.values.thiele_small); `on`: the
        radiating boundary (for Sd and Vas)."""
        from loudpy.results.values import thiele_small
        return thiele_small(self, on, coil)

    def _sound_speed(self) -> float:
        groups: dict = self.model.groups if self.model is not None else {}
        cs = {g["c"] for g in groups.values() if g.get("role") in ("air", "pml") and "c" in g}
        if len(cs) == 1:
            return cs.pop()
        raise ResultsError("give the speed of sound c= (these results do not describe one air)")

    @property
    def circuit(self):
        """The circuit of a circuit-driven response over frequency (loudpy.circuit.CircuitResponse: voltage(node),
        current(part), impedance(part), input_impedance(), transfer(node))."""
        from loudpy.circuit import Circuit, CircuitResponse
        data = self.meta.get("circuit") or (getattr(self, "attachments", None) or {}).get("circuit", {}).get("meta")
        if not data:
            raise ResultsError("these results were not driven by a circuit (lp.Circuit)")
        return CircuitResponse(Circuit.from_dict(data), self.frequencies, self.extra("e"))

    @property
    def motors(self) -> dict:
        """The drivers' motors the study had ({driver name, or None for a 2D model's coil: a coil model or None}):
        a voltage or a circuit given later (redrive) drives them."""
        from loudpy.physics.electric import motors_from_meta
        return motors_from_meta(self.meta.get("motors"))

    @property
    def redrivable(self) -> bool:
        """Whether the response keeps one unit response per coil (redrive works): a study driven by a voltage or a
        circuit, or by a force on one coil."""
        return bool(self.blocks) and all(b.coils is not None and "mobility" in b.extra for b in self.blocks)

    def redrive(self, drive) -> Self:
        """
        The same response driven otherwise — another lp.Circuit (filters, crossover, EQ, coil values), an lp.Voltage
        or an lp.Force — without solving the structure and the air again: they respond linearly to the coil forces,
        kept per coil (its unit response) with the coils' mobility (loudpy.physics.electric). Exact.

            res2 = res.redrive(circuit2)               # the crossover changed after the study
            res2.spl(at=pt), res2.impedance(), res2.voltage("c")
        """
        from loudpy.physics import electric
        if not self.redrivable:
            raise ResultsError("these results keep no unit response per coil (computed with one force on several "
                               "drivers together, or by an older loudpy): drive the study by an lp.Voltage or an "
                               "lp.Circuit to change its drive afterwards")
        blocks, circuit = [], None
        for b in self.blocks:
            coils = list(b.coils or [])
            circuit = electric.as_circuit(drive, coils, self.motors)
            freqs = np.asarray(b.values, float)
            extra_keep = {k: v for k, v in b.extra.items() if k not in ELECTRIC_EXTRA}
            if circuit is None:
                W, scalars = electric.force_weights(drive, coils, len(freqs)), {}
            else:
                W, E = electric.solve(circuit, freqs, b.extra["mobility"], coils)
                scalars = {"e": E, **(electric.coil_scalars(circuit, E, freqs) if coils == [None] else {})}
            base = Block(b.meshes, b.values, b.fields, extra_keep, b.meta, b.label, b.boundaries, b.basis)
            nb = base.driven(W)
            nb.extra.update(scalars)
            blocks.append(nb)
        out = self._like(blocks)
        out.meta = dict(self.meta, drive=repr(drive))
        out.attachments = dict(self.attachments)
        out.meta.pop("circuit", None)
        out.attachments.pop("circuit", None)
        if circuit is not None:
            if self.meta.get("scene"):
                out.attachments["circuit"] = dict(meta=circuit.to_dict())
            else:
                out.meta["circuit"] = circuit.to_dict()
        from loudpy.results.description import drive_description
        out.study = {**(self.study or {}), "drive": drive_description(drive), "redriven": True}
        return out

    def input_impedance(self, source: str | None = None) -> Data:
        """The impedance the amplifier of a circuit-driven response sees [Ω]."""
        return self.circuit.input_impedance(source)

    def voltage(self, node: str, ref: str = "0") -> Data:
        """A circuit node's voltage [V]."""
        return self.circuit.voltage(node, ref)

    def current(self, part: str) -> Data:
        """The current through a circuit part [A]."""
        return self.circuit.current(part)

    def transfer(self, node: str, source: str | None = None) -> Data:
        """A circuit node's voltage over its source's U."""
        return self.circuit.transfer(node, source)

    def impedance(self) -> Data:
        """Electrical input impedance Z = U / I_coil over frequency (voltage drive; a circuit: the coil's) [Ω]."""
        if "I_coil" not in self.scalars:
            raise ResultsError("no coil current in these results (lp.Voltage drive needed)")
        return Data(self.extra("U") / self.extra("I_coil"), {"f": self.frequencies}, "Z", meta=self._meta())


class TimeResults(Results):
    """Transient responses: one entry per time step."""
    kind = "time"
    coordinate = "time"
    dim = "t"
    real = True

    @property
    def times(self) -> np.ndarray:
        """Times of the entries [s]."""
        return self.values

    def field(self, name: str, t: float | None = None, index: int | None = None) -> Field:  # type: ignore[override]
        """One field on its mesh at the time nearest t (or at a step index)."""
        return super().field(name, index, t)


class ModeResults(Results):
    """
    Modes: one entry per mode (frequency, damping ratio "zeta"); one block per mode set
    (structural modes, each azimuthal order m, coupled modes …), labelled e.g. "m0", "m1", "fsi".
    Evaluations (probe …) run over the modes: Data over the "mode" index (frequencies in .meta).
    """
    kind = "modes"
    coordinate = "frequency"
    dim = "mode"

    def _coord(self):
        return np.arange(len(self))

    @property
    def frequencies(self) -> np.ndarray:
        """Frequencies of the modes [Hz]."""
        return self.values

    @property
    def zeta(self) -> np.ndarray:
        """Damping ratios of the modes."""
        return self.extra("zeta")

    @property
    def sets(self) -> list[str]:
        """Labels of the mode sets (one per block: orders, coupled set…)."""
        return [b.label for b in self.blocks]

    @property
    def orders(self) -> list[int]:
        """Azimuthal orders m of the mode sets that have one."""
        return sorted(int(b.meta["order"]) for b in self.blocks if "order" in b.meta)

    def order(self, m: int) -> "ModeResults":
        """The modes of azimuthal order m."""
        sel = [b for b in self.blocks if int(b.meta.get("order", -1)) == int(m)]
        if not sel:
            raise ResultsError(f"no modes of order m = {m} (orders: {self.orders})")
        return self._like(sel)

    def set(self, label: str) -> "ModeResults":
        """The modes of one set (by label)."""
        sel = [b for b in self.blocks if b.label == label]
        if not sel:
            raise ResultsError(f"no mode set {label!r} (sets: {self.sets})")
        return self._like(sel)

    def mode(self, k: int, name: str | None = None) -> Field:
        """Shape of mode k (global index; frequency order within each set): "u" (default) or "p"."""
        name = name or ("u" if "u" in self.names else self.names[0])
        f = self.field(name, index=k)
        f.coord = "f"
        return f

    def field(self, name: str, index: int | None = None, f: float | None = None) -> Field:
        return super().field(name, index, f)

    def table(self):
        """pandas DataFrame, one row per mode: set, order m, frequency f [Hz], damping ratio ζ."""
        import pandas as pd
        rows = []
        for b in self.blocks:
            z = b.extra.get("zeta", np.full(len(b), np.nan))
            for k in range(len(b)):
                rows.append({"set": b.label, "order": b.meta.get("order", np.nan), "f [Hz]": float(b.values[k]),
                             "zeta": float(np.real(z[k]))})
        return pd.DataFrame(rows, index=pd.RangeIndex(len(rows), name="mode"))

    def to_pandas(self):
        """The mode table (see table())."""
        return self.table()

    def to_csv(self, path):
        """The mode table as a CSV file."""
        from loudpy.results.data import write_atomic
        return write_atomic(path, lambda tmp: self.table().to_csv(tmp))


KINDS = {"frequency": FrequencyResults, "modes": ModeResults, "time": TimeResults}


def result_class(name: str, kind: str):
    """The Results subclass named in a file (e.g. SceneResults), else the one of its kind."""
    if name == "SceneResults":
        from loudpy.results.scene import SceneResults
        return SceneResults
    return KINDS[kind]
