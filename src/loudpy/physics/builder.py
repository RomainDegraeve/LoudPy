"""
Generic system builder: fields (unknowns), terms (matrix blocks times a function of s = jω) and loads.

Every physics contributes terms to the fields it couples; no study builds blocks itself:

    b = SystemBuilder()
    b.add_field(Field("u", n_u, expand=meca.expand))
    b.add(Term("u", "u", K))                 # s⁰
    b.add(Term("u", "u", C, power=1))        # s¹
    b.add(Term("u", "u", M, power=2))        # s²
    b.add(Term("I", "I", Le, func=lambda s: s ** n))     # any function of s (not polynomial)
    b.add_load(Load("u", F))
    A = b.at(1j * omega); x = solve(A, b.rhs()); b.split(x)["u"]
    K, C, M = b.polynomial()                 # for eigenproblems (polynomial terms only)

The contributions of the 2D axisymmetric physics (by role, from the assembled operators of
`physics.systems.assemble`) and of the drives are below: `contribute(builder, system, parts, drive)`.
Frequency-domain form (time convention e^{st}, s = jω):

    u row:  K u + s C u + s² M u − Camᵀ p                 (− Bl b I with a Voltage drive) = F·amplitude
    p row:  ρ s² Cam u + H p + s² Q p                                                      = 0
    I row:  s Bl bᵀ u + (Re + sⁿ Le) I                                                     = U
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.sparse as sp

from loudpy.errors import ModelError

__all__ = ["Field", "Term", "Load", "SystemBuilder", "CONTRIBUTIONS", "contribute", "coupled_builder"]


@dataclass
class Field:
    """An unknown of the system: its name, size and (optionally) the map to full nodal DOFs."""
    name: str
    size: int
    expand: Callable | None = None
    info: dict = field(default_factory=dict)


@dataclass
class Term:
    """A block f(s)·matrix in row field `row`, column field `col`; f(s) = s**power, or `func(s)`."""
    row: str
    col: str
    matrix: object
    power: int = 0
    func: Callable | None = None

    def factor(self, s):
        return self.func(s) if self.func is not None else s ** self.power

    @property
    def polynomial(self) -> bool:
        return self.func is None and self.power in (0, 1, 2)


@dataclass
class Load:
    """A right-hand side on `field`: a vector (or a function of s returning one)."""
    field: str
    vector: object

    def value(self, s):
        return self.vector(s) if callable(self.vector) else self.vector


class SystemBuilder:
    """Collects fields, terms and loads; assembles A(s), the right-hand side and the polynomial form."""

    def __init__(self):
        self.fields: dict[str, Field] = {}
        self.terms: list[Term] = []
        self.loads: list[Load] = []

    # ── declaration ─────────────────────────────────────────────────────────
    def add_field(self, f: Field) -> Field:
        if f.name in self.fields:
            raise ModelError(f"field {f.name!r} declared twice")
        self.fields[f.name] = f
        return f

    def add(self, *terms: Term) -> None:
        for t in terms:
            for name in (t.row, t.col):
                if name not in self.fields:
                    raise ModelError(f"term on unknown field {name!r}; fields: {list(self.fields)}")
            shape = (self.fields[t.row].size, self.fields[t.col].size)
            if t.matrix.shape != shape:
                raise ModelError(f"term ({t.row}, {t.col}) has shape {t.matrix.shape}, expected {shape}")
            self.terms.append(t)

    def add_load(self, load: Load) -> None:
        if load.field not in self.fields:
            raise ModelError(f"load on unknown field {load.field!r}")
        self.loads.append(load)

    # ── layout ──────────────────────────────────────────────────────────────
    @property
    def offsets(self) -> dict[str, int]:
        out, k = {}, 0
        for name, f in self.fields.items():
            out[name] = k
            k += f.size
        return out

    @property
    def size(self) -> int:
        return sum(f.size for f in self.fields.values())

    def split(self, x) -> dict[str, np.ndarray]:
        """The parts of a solution vector (or of the last axis of an array) per field."""
        x = np.asarray(x)
        off = self.offsets
        return {n: x[..., off[n]:off[n] + f.size] for n, f in self.fields.items()}

    # ── assembly ────────────────────────────────────────────────────────────
    def _blocks(self, select):
        names = list(self.fields)
        grid = {(r, c): None for r in names for c in names}
        for t in self.terms:
            v = select(t)
            if v is None:
                continue
            key = (t.row, t.col)
            grid[key] = v if grid[key] is None else grid[key] + v
        rows = []
        for r in names:
            row = [grid[(r, c)] for c in names]
            if all(b is None for b in row):                          # keep the block shape
                row[names.index(r)] = sp.csr_matrix((self.fields[r].size, self.fields[r].size))
            rows.append(row)
        return rows

    def at(self, s) -> sp.csc_matrix:
        """A(s) = Σ f_k(s) B_k, as one sparse matrix (fields in declaration order)."""
        def value(t):
            m = t.matrix
            return sp.csr_matrix(m) * t.factor(s) if not sp.issparse(m) else m * t.factor(s)
        rows = self._blocks(value)
        if len(rows) == 1:
            return sp.csc_matrix(rows[0][0])
        return sp.bmat(rows, format="csc")

    def rhs(self, s=None) -> np.ndarray:
        b = np.zeros(self.size, dtype=complex)
        off = self.offsets
        for load in self.loads:
            v = np.asarray(load.value(s))
            b[off[load.field]:off[load.field] + v.size] += v.ravel()
        return b

    def polynomial(self):
        """(K, C, M) with A(s) = K + s C + s² M; refuses non-polynomial terms."""
        bad = [(t.row, t.col) for t in self.terms if not t.polynomial]
        if bad:
            raise ModelError(f"terms {bad} are not polynomial in s (e.g. a semi-inductance): no "
                             f"polynomial eigenproblem")
        out = []
        for p in (0, 1, 2):
            rows = self._blocks(lambda t: sp.csr_matrix(t.matrix) if t.power == p else None)
            out.append(sp.bmat(rows, format="csc") if len(rows) > 1 else sp.csc_matrix(rows[0][0]))
        return tuple(out)


# ════════════════════════════════════════════════════════════════════════════
#  Contributions of the 2D axisymmetric physics, by role
# ════════════════════════════════════════════════════════════════════════════
def _structure(b, s, parts, drive):
    """Solid regions: K (complex with a loss factor), C = αM + βK, M on the field u."""
    b.add(Term("u", "u", s.K), Term("u", "u", s.C, power=1), Term("u", "u", s.M, power=2))


def _air(b, s, parts, drive):
    """Air and PML regions: H and Q on the field p."""
    b.add(Term("p", "p", s.H), Term("p", "p", s.Q, power=2))


def _fluid_structure(b, s, parts, drive):
    """Coupling with the outward normal of the air: −Camᵀ p on the structure, ρ s² Cam u on the air."""
    b.add(Term("u", "p", -s.Cam.T), Term("p", "u", s.rho * s.Cam, power=2))


def _coil(b, s, parts, drive):
    """The drive on the coil: a force (load on u) or a voltage with its circuit (field I)."""
    from loudpy.model.drives import Force, Voltage
    if s.F is None:
        raise ModelError("the model has no coil boundary (lp.Coil): nothing drives it")
    if isinstance(drive, Voltage):
        n = 1.0 if drive.n is None else float(drive.n)
        f = s.F.reshape(-1, 1)                                   # coil force pattern, sums to 1
        b.add_field(Field("I", 1, info=dict(Re=float(drive.Re), Le=float(drive.Le), n=n, Bl=float(drive.Bl))))
        b.add(Term("u", "I", sp.csr_matrix(-drive.Bl * f)),
              Term("I", "u", sp.csr_matrix(drive.Bl * f.T), power=1),
              Term("I", "I", sp.csr_matrix(np.array([[drive.Re]]))),
              Term("I", "I", sp.csr_matrix(np.array([[drive.Le]])), func=lambda z, n=n: z ** n))
        b.add_load(Load("I", np.array([drive.amplitude], dtype=complex)))
    elif isinstance(drive, Force):
        b.add_load(Load("u", s.F * drive.amplitude))
    else:
        raise ModelError(f"drive must be lp.Force or lp.Voltage, got {drive!r}")


# role → contribution (a new physics: one function here and its role in physics.registry)
CONTRIBUTIONS = {
    "solid": _structure,
    "air": _air,
    "fluid_structure": _fluid_structure,
    "coil": _coil,
}


def contribute(b: SystemBuilder, s, parts, drive, roles) -> None:
    """Let every role present in `roles` (once each, in this order) add its terms to the builder."""
    for role in roles:
        CONTRIBUTIONS[role](b, s, parts, drive)


def coupled_builder(s, parts, drive, coupled: bool) -> SystemBuilder:
    """The frequency-domain system of an assembled 2D model: structure (+ air + coupling) + drive."""
    b = SystemBuilder()
    b.add_field(Field("u", s.n_meca, expand=s.meca.expand))
    roles = ["solid"]
    if coupled:
        b.add_field(Field("p", s.n_acou))
        roles += ["air", "fluid_structure"]
    roles.append("coil")
    contribute(b, s, parts, drive, roles)
    return b
