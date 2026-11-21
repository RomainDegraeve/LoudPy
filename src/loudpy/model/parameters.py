"""
Parameters: named values a model refers to, for parametric studies (COMSOL's Global Parameters).

    p = lp.Parameters(t_cone=0.3e-3, E_cone=3.5e9, h=1e-3)
    model = lp.Model("speaker.geo", parameters=p)
    model.add(lp.Solid("membranne", "Paper", E=lp.P("E_cone"), mesh_size=lp.P("h")),
              lp.Solid("surround", "Rubber", mesh_size=0.5 * lp.P("h")))

    def builder(occ, p):                       # a geometry builder whose 2nd argument is named p (or params)
        return {"plate": occ.addRectangle(0, 0, 0, 0.1, p["t"])}

    model.with_parameters(E_cone=4e9)          # a copy of the model with other values
    lp.Sweep(lp.Modes(model), "E_cone", [3e9, 3.5e9, 4e9]).solve(f_max=3000)

lp.P(name) stands for a parameter in any numeric field of an item (sizes, PML settings, material overrides,
damping coefficients); `a * lp.P(name) + b` is allowed. The values are resolved, and checked, when the model
is meshed or checked. Parameter values given to a builder enter the mesh signature.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np

from loudpy.errors import ModelError

__all__ = ["Parameters", "P", "resolve"]


@dataclasses.dataclass(frozen=True)
class P:
    """A reference to a parameter: value = scale · parameters[name] + offset."""
    name: str
    scale: float = 1.0
    offset: float = 0.0

    def value(self, params: Mapping) -> float:
        """The value of the reference with these parameters: scale · params[name] + offset."""
        if params is None or self.name not in params:
            known = sorted(params) if params else []
            raise ModelError(f"unknown parameter {self.name!r}; parameters: {known}")
        return self.scale * params[self.name] + self.offset

    def __mul__(self, k):
        return P(self.name, self.scale * float(k), self.offset * float(k))

    __rmul__ = __mul__

    def __truediv__(self, k):
        return self * (1.0 / float(k))

    def __add__(self, k):
        return P(self.name, self.scale, self.offset + float(k))

    __radd__ = __add__

    def __sub__(self, k):
        return self + (-float(k))

    def __repr__(self):
        s = f"P({self.name!r})"
        if self.scale != 1.0:
            s = f"{self.scale:g}*{s}"
        if self.offset:
            s = f"{s} + {self.offset:g}"
        return s


class Parameters(Mapping):
    """Named real values (SI units). Immutable: with_(**changes) gives a new set."""

    def __init__(self, **values):
        def ok(v):
            try:
                return np.isfinite(float(v)) and not isinstance(v, (str, bytes, bool))
            except (TypeError, ValueError):
                return False
        bad = {k: v for k, v in values.items() if not ok(v)}
        if bad:
            raise ModelError(f"parameters must be finite numbers, got {bad}")
        self._values = {k: float(v) for k, v in values.items()}

    def __getitem__(self, name):
        return self._values[name]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def with_(self, **changes) -> "Parameters":
        """A copy with some values changed (only existing names)."""
        unknown = set(changes) - set(self._values)
        if unknown:
            raise ModelError(f"unknown parameters {sorted(unknown)}; parameters: {sorted(self._values)}")
        return Parameters(**{**self._values, **changes})

    def __repr__(self):
        return "Parameters(" + ", ".join(f"{k}={v:g}" for k, v in self._values.items()) + ")"

    def __eq__(self, other):
        return isinstance(other, Parameters) and other._values == self._values

    __hash__ = None


def has_references(obj) -> bool:
    """True if a P appears in the fields of an item (nested dataclasses included)."""
    if isinstance(obj, P):
        return True
    if isinstance(obj, (tuple, list)):
        return any(has_references(x) for x in obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return any(has_references(getattr(obj, f.name)) for f in dataclasses.fields(obj))
    return False


def resolve(obj, params):
    """A copy of an item (dataclass) with every P replaced by its value (nested dataclasses too); the item
    itself when it holds no reference. The copy is validated again by its own checks."""
    if isinstance(obj, P):
        return obj.value(params)
    if isinstance(obj, (tuple, list)):
        return type(obj)(resolve(x, params) for x in obj) if has_references(obj) else obj
    if not (dataclasses.is_dataclass(obj) and not isinstance(obj, type)) or not has_references(obj):
        return obj
    changes = {f.name: resolve(getattr(obj, f.name), params) for f in dataclasses.fields(obj)
               if has_references(getattr(obj, f.name))}
    init = {f.name for f in dataclasses.fields(obj) if f.init}
    out = dataclasses.replace(obj, **{k: v for k, v in changes.items() if k in init})
    for k, v in changes.items():
        if k not in init:
            object.__setattr__(out, k, v)
    return out
