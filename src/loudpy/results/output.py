"""
Output control: what a study keeps of its fields (lp.Output), passed to any study's solve(output=…).

    out = lp.Output(fields=["u"], every=20, probes={"tip": (0.0, 0.0), "rim": (0.045, 0.002)})
    res = lp.Transient(model, drive=lp.Force(signal, fs=48000)).solve(output=out)

    res.times                         # every 20th step (the stored fields)
    res.probe_signal("tip", "u", "z") # lp.Data of u_z at the tip at every step (full rate)
    res.probe_names                   # ["tip", "rim"]

fields  names of the fields kept (None: all of them); a probe is still evaluated on every field
every   keep every n-th entry of the fields (time steps, frequencies, modes) — per block, the first entry kept
probes  {name: point}: every field interpolated at the point for every entry before the fields are thinned,
        stored with the results (attachment "probes") and read with res.probe_signal(name, field, component)

The fields are computed in memory as before; Output decides what the results keep and what the file holds
(a 48 000-step transient of the example speaker: 8.7 GB with every field, ~4 MB with every=1000 and a probe).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loudpy.errors import ModelError, ResultsError

PROBES = "probes"


@dataclass(frozen=True)
class Output:
    """What a study keeps of its fields: fields (names, None for all), every (n-th entry: of the whole frequency /
    time list — the same entries with or without remeshed bands —, of each mode set for modes), probes
    ({name: point}, recorded at every entry)."""
    fields: tuple | None = None
    every: int = 1
    probes: dict | None = None

    def __post_init__(self):
        if self.fields is not None:
            names = (self.fields,) if isinstance(self.fields, str) else tuple(self.fields)
            if not all(isinstance(n, str) for n in names):
                raise ModelError(f"Output(fields=…): field names, got {self.fields!r}")
            object.__setattr__(self, "fields", names)
        if int(self.every) != self.every or self.every < 1:
            raise ModelError(f"Output(every=…): a positive integer, got {self.every!r}")
        object.__setattr__(self, "every", int(self.every))
        if self.probes is not None:
            probes = {}
            for name, p in dict(self.probes).items():
                pt = np.asarray(p, float).ravel()
                if pt.size not in (2, 3) or not np.all(np.isfinite(pt)):
                    raise ModelError(f"Output(probes=…): {name!r} must be a point (r, z) or (x, y, z), got {p!r}")
                probes[str(name)] = tuple(float(v) for v in pt)
            object.__setattr__(self, "probes", probes)

    def describe(self) -> dict:
        """The settings as plain values (recorded in the results' metadata)."""
        return {"fields": list(self.fields) if self.fields is not None else None, "every": self.every,
                "probes": {k: list(v) for k, v in (self.probes or {}).items()}}


def apply(res, output: Output | None):
    """Apply an Output to finished results (in place) and return them."""
    if output is None:
        return res
    if not isinstance(output, Output):
        raise ModelError(f"output= takes an lp.Output, got {type(output).__name__}")
    names = res.names
    if output.fields is not None:
        unknown = [n for n in output.fields if n not in names]
        if unknown:
            raise ModelError(f"Output(fields=…): no field {unknown} in these results (fields: {names})")
    if output.probes:
        from loudpy.results.selection import Point
        att: dict = {"values": np.asarray(res._coord())}
        for pname, pt in output.probes.items():
            for f in names:
                if all(f in b.fields for b in res.blocks):
                    att[f"{pname}:{f}"] = np.asarray(res.probe(f, Point(pt)).values)
        att["meta"] = {"points": {k: list(v) for k, v in output.probes.items()}, "dim": res.dim}
        res.attachments[PROBES] = att
    if output.every > 1 or output.fields is not None:
        from loudpy.results.results import Block
        blocks, offset = [], 0
        per_set = res.kind == "modes"                     # modes: every n-th mode of each mode set (order m)
        for b in res.blocks:
            if per_set:
                keep = np.arange(0, len(b), output.every)
            else:                                         # frequencies / times: over the whole list, so that the
                start = (-offset) % output.every          # entries kept do not depend on the remeshed bands
                keep = np.arange(start, len(b), output.every)
            offset += len(b)
            if not len(keep):
                continue
            fields = {k: np.asarray(v)[keep] for k, v in b.fields.items()
                      if output.fields is None or k in output.fields}
            extra = {k: (np.asarray(v)[keep] if np.ndim(v) and len(v) == len(b) else v) for k, v in b.extra.items()}
            blocks.append(Block(b.meshes, np.asarray(b.values)[keep], fields, extra, b.meta, b.label, b.boundaries))
        res.blocks = blocks
    res.meta["output"] = output.describe()
    return res


def probe_names(res) -> list[str]:
    """Names of the probes recorded with the results."""
    att = res.attachments.get(PROBES)
    return list(att["meta"]["points"]) if att else []


def probe_signal(res, name: str, field: str = "u", component: str | None = None):
    """The full-rate values of a field at a probe (lp.Output(probes=…)) as an lp.Data over the coordinate."""
    from loudpy.results.data import Data
    from loudpy.results.quantities import quantity
    from loudpy.results.results import COMPONENTS
    att = res.attachments.get(PROBES)
    if not att:
        raise ResultsError("these results have no probes (solve with output=lp.Output(probes={…}))")
    if name not in att["meta"]["points"]:
        raise ResultsError(f"no probe {name!r} (probes: {list(att['meta']['points'])})")
    key = f"{name}:{field}"
    if key not in att:
        fields = sorted({k.split(':', 1)[1] for k in att if k.startswith(name + ':')})
        raise ResultsError(f"no field {field!r} at probe {name!r} (fields: {fields})")
    v = np.asarray(att[key])
    if res.real:
        v = np.real(v)
    q = quantity(field)
    coords: dict = {att["meta"]["dim"]: np.asarray(att["values"])}
    if v.ndim == 2:
        if component is not None:
            v, q = v[:, COMPONENTS[component]], quantity(f"{field}_{component}")
        else:
            coords["component"] = np.array(["r", "z", "theta"][:v.shape[1]])
    return Data(v, coords, q, meta={"probe": name, "at": att["meta"]["points"][name]})
