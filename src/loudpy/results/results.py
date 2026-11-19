"""
Results: the outcome of any study, with one interface.

    res = lp.load("sweep.h5")                # FrequencyResults, ModeResults or TimeResults
    res.frequencies                          # (or res.times) the coordinate of the entries
    res.names                                # fields: "u" (displacement), "p" (pressure), "v", "a"
    res.field("p", f=1000)                   # FieldView: mesh + values at the entry nearest 1 kHz
    res.probe((0.0, 0.3), "p")               # values at the node nearest a point, for every entry
    res.spl(at=(0.0, 0.3))                   # SPL [dB re 20 µPa], RMS by default
    res.save("sweep_v2.h5")

A Results is a list of blocks: one block per mesh (a frequency sweep remeshed per band gives
one block per band) or per mode set (one per azimuthal order). The block structure is hidden:
`frequencies`, `probe`, `spl` run over all the blocks in order; `field` finds the block.

Fields live on the mesh of their physics: "u", "v", "a" on the "solid" mesh, "p" on the "air"
mesh (geometry.Mesh, node order = the field rows).
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from loudpy.geometry.mesh import Mesh

P_REF = 20e-6
PHYSICS_OF = {"u": "solid", "v": "solid", "a": "solid", "p": "air"}
ALIASES = {"u_meca": "u", "v_meca": "v", "a_meca": "a", "p_acou": "p"}     # former field names


def canonical(name: str) -> str:
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

    def __len__(self):
        return len(self.values)

    def mesh_of(self, name: str) -> Mesh:
        phys = PHYSICS_OF.get(name)
        if phys in self.meshes:
            return self.meshes[phys]
        if len(self.meshes) == 1:
            return next(iter(self.meshes.values()))
        raise KeyError(f"no mesh for field {name!r} (meshes: {list(self.meshes)})")


@dataclass
class FieldView:
    """One field at one entry: values (n_nodes[, dpn]) on `mesh`, at `coordinate`."""
    name: str
    mesh: Mesh
    values: np.ndarray
    coordinate: float
    extra: dict = dc_field(default_factory=dict)

    def magnitude(self) -> np.ndarray:
        """|value| per node (Euclidean norm of the moduli for a vector field)."""
        v = np.abs(self.values)
        return v if v.ndim == 1 else np.linalg.norm(v, axis=1)

    def spl(self, rms: bool = True) -> np.ndarray:
        return spl(self.values, rms)

    def triangles(self) -> np.ndarray:
        return self.mesh.triangles()

    def restrict(self, region: str) -> "FieldView":
        """The field on the mesh groups selected by `region` (e.g. "surround"; naming rules)."""
        from loudpy.geometry.naming import matches
        names = [n for n in self.mesh.groups if matches(region, n)]
        if not names:
            raise KeyError(f"no mesh group selected by {region!r} (groups: {list(self.mesh.groups)})")
        tags = np.unique(np.concatenate([self.mesh.groups[n].cells.ravel() for n in names]))
        rows = self.mesh.rows_of(tags)
        sub = Mesh(tags, self.mesh.coords[rows], {n: self.mesh.groups[n] for n in names}, self.mesh.path)
        return FieldView(self.name, sub, self.values[rows], self.coordinate, self.extra)

    def at_nodes(self, tags) -> tuple[np.ndarray, np.ndarray]:
        """(coords, values) at the node tags present on this field's mesh (e.g. res.boundary(...))."""
        pos = self.mesh._positions
        rows = np.array([pos[int(t)] for t in np.asarray(tags) if int(t) in pos], dtype=np.int64)
        return self.mesh.coords[rows], self.values[rows]


def spl(p, rms: bool = True) -> np.ndarray:
    """Sound pressure level [dB re 20 µPa] of complex amplitudes (RMS: |p|/√2; rms=False: |p|)."""
    a = np.abs(np.asarray(p)) / (np.sqrt(2.0) if rms else 1.0)
    return 20.0 * np.log10(np.maximum(a, 1e-300) / P_REF)


class Results:
    kind = "results"
    coordinate = "value"

    def __init__(self, blocks: list[Block], meta: dict | None = None, boundaries: dict | None = None,
                 source=None):
        self.blocks = list(blocks)
        self.meta = dict(meta or {})
        self.boundaries = dict(boundaries or {})     # name → {"node_tags", "edges"}
        self.source = source
        self.attachments: dict = {}                  # name → {array name: array, "meta": {...}}

    # ── what is there ──────────────────────────────────────────────────────
    @property
    def values(self) -> np.ndarray:
        """The coordinate of every entry, all blocks in order."""
        return np.concatenate([b.values for b in self.blocks]) if self.blocks else np.zeros(0)

    @property
    def names(self) -> list[str]:
        out = []
        for b in self.blocks:
            out += [n for n in b.fields if n not in out]
        return out

    def __len__(self):
        return sum(len(b) for b in self.blocks)

    def extra(self, name: str) -> np.ndarray:
        """A per-entry quantity (e.g. "zeta", "I_coil") over all blocks (NaN where missing)."""
        return np.concatenate([b.extra.get(name, np.full(len(b), np.nan)) for b in self.blocks])

    def mesh(self, physics: str = "solid", block: int = 0) -> Mesh:
        return self.blocks[block].meshes[physics]

    def boundary(self, name: str) -> np.ndarray:
        """Node tags of the boundaries selected by `name` (e.g. "interface_acou_meca_front")."""
        from loudpy.geometry.naming import matches
        sel = [b for n, b in self.boundaries.items() if n == name or matches(name, n)]
        if not sel:
            raise KeyError(f"no boundary {name!r} (boundaries: {list(self.boundaries)})")
        return np.unique(np.concatenate([np.asarray(b["node_tags"]) for b in sel]))

    def __repr__(self):
        v = self.values
        rng = f"{v.min():.4g}–{v.max():.4g}" if len(v) else "empty"
        return (f"{type(self).__name__}({len(self)} entries, {self.coordinate} {rng}, fields {self.names}, "
                f"{len(self.blocks)} block(s))")

    # ── one field at one entry ─────────────────────────────────────────────
    def _locate(self, index=None, value=None):
        """(block, row) of the entry with global index `index`, or nearest to `value`."""
        if index is None:
            if value is None:
                index = 0
            else:
                index = int(np.argmin(np.abs(self.values - value)))
        if index < 0:
            index += len(self)
        for b in self.blocks:
            if index < len(b):
                return b, index
            index -= len(b)
        raise IndexError("entry index out of range")

    def field(self, name: str, index: int | None = None, value: float | None = None) -> FieldView:
        name = canonical(name)
        b, i = self._locate(index, value)
        if name not in b.fields:
            raise KeyError(f"no field {name!r} (fields: {list(b.fields)})")
        return FieldView(name, b.mesh_of(name), b.fields[name][i], float(b.values[i]),
                         {k: v[i] for k, v in b.extra.items()})

    # ── values at a point, over all entries ────────────────────────────────
    def probe(self, point, name: str = "p") -> np.ndarray:
        """Values at the node nearest `point` (per block mesh), for every entry: (n,) or (n, dpn)."""
        name = canonical(name)
        out = []
        for b in self.blocks:
            if name not in b.fields:
                raise KeyError(f"no field {name!r} in block {b.label!r}")
            out.append(b.fields[name][:, b.mesh_of(name).nearest(point)])
        return np.concatenate(out)

    def spl(self, at, rms: bool = True) -> np.ndarray:
        """SPL [dB re 20 µPa] of the pressure at the node nearest `at`, for every entry."""
        return spl(self.probe(at, "p"), rms)

    # ── files ──────────────────────────────────────────────────────────────
    def save(self, path):
        """Write these results in the loudpy result format 2 (loudpy.results.schema)."""
        from loudpy.results.schema import write
        write(self, path)
        return path


class FrequencyResults(Results):
    """Harmonic responses: one entry per frequency (blocks: one per mesh of a remeshed sweep)."""
    kind = "frequency"
    coordinate = "frequency"

    @property
    def frequencies(self) -> np.ndarray:
        return self.values

    def field(self, name: str, f: float | None = None, index: int | None = None) -> FieldView:
        return super().field(name, index, f)


class TimeResults(Results):
    """Transient responses: one entry per time step."""
    kind = "time"
    coordinate = "time"

    @property
    def times(self) -> np.ndarray:
        return self.values

    def field(self, name: str, t: float | None = None, index: int | None = None) -> FieldView:
        return super().field(name, index, t)


class ModeResults(Results):
    """
    Modes: one entry per mode (frequency, damping ratio "zeta"); one block per mode set
    (structural modes, each azimuthal order m, coupled modes …), labelled e.g. "m0", "m1", "fsi".
    """
    kind = "modes"
    coordinate = "frequency"

    @property
    def frequencies(self) -> np.ndarray:
        return self.values

    @property
    def zeta(self) -> np.ndarray:
        return self.extra("zeta")

    @property
    def sets(self) -> list[str]:
        return [b.label for b in self.blocks]

    @property
    def orders(self) -> list[int]:
        """Azimuthal orders m of the mode sets that have one."""
        return sorted(int(b.meta["order"]) for b in self.blocks if "order" in b.meta)

    def order(self, m: int) -> "ModeResults":
        """The modes of azimuthal order m."""
        sel = [b for b in self.blocks if int(b.meta.get("order", -1)) == int(m)]
        if not sel:
            raise KeyError(f"no modes of order m = {m} (orders: {self.orders})")
        return ModeResults(sel, self.meta, self.boundaries, self.source)

    def set(self, label: str) -> "ModeResults":
        sel = [b for b in self.blocks if b.label == label]
        if not sel:
            raise KeyError(f"no mode set {label!r} (sets: {self.sets})")
        return ModeResults(sel, self.meta, self.boundaries, self.source)

    def mode(self, k: int, name: str = "u") -> FieldView:
        """Shape of mode k (global index, frequency order within each set)."""
        return self.field(name, index=k)

    def field(self, name: str, index: int | None = None, f: float | None = None) -> FieldView:
        return super().field(name, index, f)


KINDS = {"frequency": FrequencyResults, "modes": ModeResults, "time": TimeResults}


def result_class(name: str, kind: str):
    """The Results subclass named in a file (e.g. SceneResults), else the one of its kind."""
    if name == "SceneResults":
        from loudpy.results.scene import SceneResults
        return SceneResults
    return KINDS[kind]
