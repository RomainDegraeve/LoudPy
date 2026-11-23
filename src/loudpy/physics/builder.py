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

__all__ = ["Field", "Term", "LowRank", "Load", "SystemBuilder", "BlockSolver", "CONTRIBUTIONS", "FORMS", "contribute", "coupled_builder",
           "symmetric_builder"]


@dataclass
class Field:
    """An unknown of the system: its name, size and (optionally) the map to full nodal DOFs."""
    name: str
    size: int
    expand: Callable | None = None
    info: dict = field(default_factory=dict)


@dataclass
class Term:
    """
    A block f(s)·B in row field `row`, column field `col`; f(s) = s**power, or `func(s)`. B is a matrix
    (sparse or dense), or a function of s returning one (a frequency-dependent block, e.g. a cavity
    eliminated through its modes) or a `LowRank` operator (e.g. the 3D DtN, dense on the boundary).
    """
    row: str
    col: str
    matrix: object
    power: int = 0
    func: Callable | None = None

    def factor(self, s):
        """f(s): s**power, or func(s)."""
        return self.func(s) if self.func is not None else s ** self.power

    @property
    def polynomial(self) -> bool:
        """True when the term is a constant matrix times s⁰, s¹ or s² (it enters a polynomial eigenproblem)."""
        return self.func is None and self.power in (0, 1, 2) and not callable(self.matrix)

    def value(self, s):
        """f(s)·B(s): a sparse or dense matrix, or a LowRank operator."""
        m = self.matrix(s) if callable(self.matrix) else self.matrix
        f = self.factor(s)
        if isinstance(m, LowRank):
            return m.scaled(f)
        return m * f


class LowRank:
    """
    A low-rank operator block  out[rows] = c · U (core ⊙ (Vᵀ x[cols]))  of shape (n_rows, n_cols):
    U (len(rows), r), core (r,) a vector (diagonal) or (r, r), V (len(cols), r) (default U), c a scale.
    Used for blocks that are dense on a boundary only, never assembled.
    """

    def __init__(self, shape, rows, U, core, cols=None, V=None, scale=1.0):
        self.shape = tuple(shape)
        self.rows = np.asarray(rows)
        self.cols = self.rows if cols is None else np.asarray(cols)
        self.U = U
        self.V = U if V is None else V
        self.core = np.asarray(core)
        self.scale = scale

    def scaled(self, f):
        """The same operator multiplied by f."""
        return LowRank(self.shape, self.rows, self.U, self.core, self.cols, self.V, self.scale * f)

    def __matmul__(self, x):
        y = self.V.T @ x[self.cols]
        y = self.core * y if self.core.ndim == 1 else self.core @ y
        out = np.zeros(self.shape[0], dtype=np.result_type(x, y, complex))
        out[self.rows] = self.scale * (self.U @ y)
        return out


@dataclass
class Load:
    """A right-hand side on `field`: a vector (or a function of s returning one)."""
    field: str
    vector: object

    def value(self, s):
        """The load vector at s."""
        return self.vector(s) if callable(self.vector) else self.vector


class SystemBuilder:
    """Collects fields, terms and loads; assembles A(s), the right-hand side and the polynomial form."""

    def __init__(self):
        self.fields: dict[str, Field] = {}
        self.terms: list[Term] = []
        self.loads: list[Load] = []

    # ── declaration ─────────────────────────────────────────────────────────
    def add_field(self, f: Field) -> Field:
        """Declare an unknown (a field name may be declared once)."""
        if f.name in self.fields:
            raise ModelError(f"field {f.name!r} declared twice")
        self.fields[f.name] = f
        return f

    def add(self, *terms: Term) -> None:
        """Add terms (their fields must be declared, their shapes match the fields' sizes)."""
        for t in terms:
            for name in (t.row, t.col):
                if name not in self.fields:
                    raise ModelError(f"term on unknown field {name!r}; fields: {list(self.fields)}")
            shape = (self.fields[t.row].size, self.fields[t.col].size)
            m_shape = getattr(t.matrix, "shape", None)
            if not callable(t.matrix) and m_shape != shape:
                raise ModelError(f"term ({t.row}, {t.col}) has shape {m_shape}, expected {shape}")
            self.terms.append(t)

    def add_load(self, load: Load) -> None:
        """Add a right-hand side on a declared field."""
        if load.field not in self.fields:
            raise ModelError(f"load on unknown field {load.field!r}")
        self.loads.append(load)

    # ── layout ──────────────────────────────────────────────────────────────
    @property
    def offsets(self) -> dict[str, int]:
        """Offset of each field in the global vector (declaration order)."""
        out, k = {}, 0
        for name, f in self.fields.items():
            out[name] = k
            k += f.size
        return out

    @property
    def size(self) -> int:
        """Total number of unknowns."""
        return sum(f.size for f in self.fields.values())

    def split(self, x) -> dict[str, np.ndarray]:
        """The parts of a solution vector (or of the last axis of an array) per field."""
        x = np.asarray(x)
        off = self.offsets
        return {n: x[..., off[n]:off[n] + f.size] for n, f in self.fields.items()}

    # ── assembly ────────────────────────────────────────────────────────────
    def block(self, row, col, s):
        """Block (row, col) at s: (matrix sum or None, [LowRank operators])."""
        mat, ops = None, []
        for t in self.terms:
            if t.row != row or t.col != col:
                continue
            v = t.value(s)
            if isinstance(v, LowRank):
                ops.append(v)
            else:
                mat = v if mat is None else mat + v
        return mat, ops

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
        """A(s) = Σ f_k(s) B_k, as one sparse matrix (fields in declaration order; no LowRank terms:
        use BlockSolver)."""
        def value(t):
            v = t.value(s)
            if isinstance(v, LowRank):
                raise ModelError("a LowRank term cannot be assembled: solve with BlockSolver")
            return v if sp.issparse(v) else sp.csr_matrix(v)
        rows = self._blocks(value)
        if len(rows) == 1:
            return sp.csc_matrix(rows[0][0])
        return sp.bmat(rows, format="csc")

    def rhs(self, s=None) -> np.ndarray:
        """The right-hand side at s (sum of the loads)."""
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


class BlockSolver:
    """
    Solves A(s) x = b(s) by eliminating `small` fields (dense, factorised exactly) and, if `big` is given,
    iterating on the big field with GMRES on its Schur complement

        (A_bb − A_bs A_ss⁻¹ A_sb) x_b = b_b − A_bs A_ss⁻¹ b_s,   x_s = A_ss⁻¹ (b_s − A_sb x_b)

    A_bb may contain LowRank operators (never assembled). `preconditioner(s)` returns a LinearOperator
    approximating A_bb⁻¹ (e.g. an LU at a nearby frequency); the caller decides when to refresh it.
    Without `big`, the small system is solved directly (several right-hand sides allowed); without small
    fields (small=[]), GMRES runs on the big field alone.
    """

    def __init__(self, builder: SystemBuilder, small, big: str | None = None, rtol=1e-8, restart=150,
                 maxiter=20):
        self.b = builder
        self.small = [small] if isinstance(small, str) else list(small)
        self.big = big
        self.rtol, self.restart, self.maxiter = rtol, restart, maxiter

    def _small_matrix(self, s, rows, cols):
        """Dense block over several fields."""
        sizes_r = [self.b.fields[r].size for r in rows]
        sizes_c = [self.b.fields[c].size for c in cols]
        out = np.zeros((sum(sizes_r), sum(sizes_c)), complex)
        r0 = 0
        for r, nr in zip(rows, sizes_r):
            c0 = 0
            for c, nc in zip(cols, sizes_c):
                m, ops = self.b.block(r, c, s)
                if ops:
                    raise ModelError(f"LowRank term in the eliminated block ({r}, {c})")
                if m is not None:
                    out[r0:r0 + nr, c0:c0 + nc] += m.toarray() if sp.issparse(m) else np.asarray(m)
                c0 += nc
            r0 += nr
        return out

    def _coupling(self, s, rows, cols):
        """Sparse (or None) block between the big field and the small fields."""
        parts = []
        for r in rows:
            row = []
            for c in cols:
                m, ops = self.b.block(r, c, s)
                if ops:
                    raise ModelError(f"LowRank term in the coupling block ({r}, {c})")
                shape = (self.b.fields[r].size, self.b.fields[c].size)
                row.append(sp.csr_matrix(m) if m is not None else sp.csr_matrix(shape))
            parts.append(row)
        return sp.bmat(parts, format="csr")

    def solve(self, s, rhs=None, x0=None, preconditioner=None):
        """Fields at s: (dict of field values, GMRES iterations, GMRES info)."""
        import scipy.linalg as sla
        import scipy.sparse.linalg as spla
        b = self.b.rhs(s) if rhs is None else rhs
        off = self.b.offsets
        if self.small:
            idx_s = np.concatenate([np.arange(off[n], off[n] + self.b.fields[n].size) for n in self.small])
            lu = sla.lu_factor(self._small_matrix(s, self.small, self.small))
            b_s = np.asarray(b)[idx_s]
            if self.big is None:
                return self._split_small(sla.lu_solve(lu, b_s)), 0, 0
        elif self.big is None:
            raise ModelError("BlockSolver needs eliminated fields or a big field")
        nb = self.b.fields[self.big].size
        b_b = np.asarray(b)[off[self.big]:off[self.big] + nb]
        A_bb, ops = self.b.block(self.big, self.big, s)
        A_bb = sp.csr_matrix(A_bb)
        if self.small:
            A_sb = self._coupling(s, self.small, [self.big])
            A_bs = self._coupling(s, [self.big], self.small)

        def mv(v):
            out = A_bb @ v
            for op in ops:
                out = out + op @ v
            return out - A_bs @ sla.lu_solve(lu, A_sb @ v) if self.small else out
        S = spla.LinearOperator((nb, nb), matvec=mv, dtype=complex)
        it = [0]
        rhs_b = b_b - A_bs @ sla.lu_solve(lu, b_s) if self.small else b_b
        x_b, info = spla.gmres(S, rhs_b, x0=x0, M=preconditioner, rtol=self.rtol, restart=self.restart,
                               maxiter=self.maxiter, callback=lambda r: it.__setitem__(0, it[0] + 1),
                               callback_type="pr_norm")
        out = self._split_small(sla.lu_solve(lu, b_s - A_sb @ x_b)) if self.small else {}
        out[self.big] = x_b
        return out, it[0], info

    def _split_small(self, x_s):
        out, k = {}, 0
        for n in self.small:
            m = self.b.fields[n].size
            out[n] = x_s[k:k + m]
            k += m
        return out


# ════════════════════════════════════════════════════════════════════════════
#  Contributions of the 2D axisymmetric physics, by role
# ════════════════════════════════════════════════════════════════════════════
# Two forms of the same physics:
#   "frequency": fields u, p (, I): the forced response, unsymmetric (see the module docstring)
#   "symmetric": fields u, psi (p = −sψ), phi (DtN auxiliaries): the symmetric quadratic eigenproblem
#                (s² Ms + s Cs + Ks) x = 0 of the coupled modes
FORMS = ("frequency", "symmetric")


def _structure(b, s, parts, drive, form, extra):
    """Solid regions: K (complex with a loss factor), C = αM + βK, M on the field u."""
    C = s.C if s.C is not None else sp.csr_matrix(s.K.shape)
    b.add(Term("u", "u", s.K), Term("u", "u", s.M, power=2), Term("u", "u", C, power=1))


def _air(b, s, parts, drive, form, extra):
    """Air and PML regions: H and Q on p (frequency), −H/ρ and −Q/ρ on ψ (symmetric)."""
    if form == "frequency":
        b.add(Term("p", "p", s.H), Term("p", "p", s.Q, power=2))
    else:
        b.add(Term("psi", "psi", -s.H / s.rho), Term("psi", "psi", -s.Q / s.rho, power=2))


def _fluid_structure(b, s, parts, drive, form, extra):
    """Coupling with the outward normal of the air: −Camᵀ p on u and ρ s² Cam u on p (frequency);
    Camᵀ ψ on u and Cam u on ψ, both in s¹ (symmetric)."""
    if form == "frequency":
        b.add(Term("u", "p", -s.Cam.T), Term("p", "u", s.rho * s.Cam, power=2))
    else:
        b.add(Term("u", "psi", s.Cam.T, power=1), Term("psi", "u", s.Cam, power=1))


def _radiation(b, s, parts, drive, form, extra):
    """Exact DtN radiation (symmetric form): Dk, Dc on ψ, auxiliaries φ with G, Gᵀ, Zaa, Caa
    (pieces computed by physics.systems.dtn_system, passed as extra["dtn"])."""
    if form != "symmetric":
        raise ModelError("the exact DtN radiation is available for the coupled modes (symmetric form) only")
    d = extra["dtn"]
    n_aux = d["n_aux"]
    b.add(Term("psi", "psi", d["Dk"]), Term("psi", "psi", d["Dc"], power=1))
    b.add(Term("phi", "phi", sp.csr_matrix((n_aux, n_aux)), power=2))
    if n_aux:
        b.add(Term("psi", "phi", d["G"]), Term("phi", "psi", d["G"].T), Term("phi", "phi", d["Zaa"]),
              Term("phi", "phi", d["Caa"], power=1))


def _coil(b, s, parts, drive, form, extra):
    """The drive on the coil: a force (load on u), a voltage with its circuit (field I), or a circuit
    (loudpy.circuit: its MNA unknowns, field e, its one coil on the model's coil). Nothing in the symmetric form
    (free modes)."""
    from loudpy.circuit import Circuit
    from loudpy.model.drives import Force, Voltage
    if form == "symmetric":
        return
    if s.F is None:
        raise ModelError("the model has no coil boundary (lp.Coil): nothing drives it")
    if isinstance(drive, Circuit):
        drive.check()
        if len(drive.coils) != 1:
            raise ModelError(f"a 2D model has one coil: its circuit holds one Coil part ({len(drive.coils)} given)")
        qe, eq = drive.coupling({drive.coils[0].name: np.asarray(s.F, float).reshape(-1)})
        b.add_field(Field("e", drive.size, info=dict(circuit=drive)))
        b.add(Term("u", "e", sp.csr_matrix(qe)), Term("e", "u", sp.csr_matrix(eq), power=1),
              Term("e", "e", lambda z: sp.csr_matrix(drive.matrix(z))))
        b.add_load(Load("e", drive.rhs()))
        return
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
        raise ModelError(f"drive must be lp.Force, lp.Voltage or lp.Circuit, got {drive!r}")


# role → contribution (a new physics: one function here and its role in physics.registry)
CONTRIBUTIONS = {
    "solid": _structure,
    "air": _air,
    "fluid_structure": _fluid_structure,
    "radiation": _radiation,
    "coil": _coil,
}


def contribute(b: SystemBuilder, s, parts, drive, roles, form: str = "frequency", **extra) -> None:
    """Let every role in `roles` (once each, in this order) add its terms to the builder."""
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}, got {form!r}")
    for role in roles:
        CONTRIBUTIONS[role](b, s, parts, drive, form, extra)


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


def symmetric_builder(s, parts, dtn) -> SystemBuilder:
    """The symmetric u–ψ–φ system (p = −sψ) of the coupled modes with the exact DtN radiation."""
    b = SystemBuilder()
    b.add_field(Field("u", s.n_meca, expand=s.meca.expand))
    b.add_field(Field("psi", s.n_acou))
    b.add_field(Field("phi", dtn["n_aux"]))
    contribute(b, s, parts, None, ["solid", "air", "radiation", "fluid_structure"], form="symmetric", dtn=dtn)
    return b
