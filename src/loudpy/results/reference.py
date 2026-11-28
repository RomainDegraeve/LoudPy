"""
Comparison with reference results: read the text exports of commercial FE software (or any table in that
format) and compare them with loudpy Data.

    ref = lp.read_reference("reference_spl.txt")             # a table / point-graph export (text, % headers)
    ref.columns                                            # ['freq (Hz)', 'abs(p) (Pa)', …]
    d_ref = ref.data("freq", "p")                          # lp.Data of the column matching "p" over "freq"
    cmp = lp.compare(res.pressure(at=(0, 0.3)), d_ref, rtol=0.02)   # interpolated on the common range
    print(cmp)                                             # max error, where, within tolerance?
    cmp.passed, cmp.max_error, cmp.at, cmp.data            # relative error over the coordinate (lp.Data)

The text format: header lines start with "%" ("% Model:", "% Version:", "% Date:", … and, last, the column
names, each with its unit in parentheses); values separated by spaces, tabs or commas; complex values written
1.2+3.4i (or with j). A column is found by an exact name, by its name without the unit, or by a unique substring
("p" finds "abs(p) (Pa)" only when no other column contains "p" as a whole word).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from loudpy.errors import ResultsError

_UNIT = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")


def _bare(column: str) -> str:
    """A column name without its unit in parentheses."""
    m = _UNIT.match(column)
    return m.group(1) if m else column


def _number(tok: str):
    t = tok.strip().replace("j", "i")
    if t.endswith("i"):
        return complex(t[:-1] + "j")
    return float(t)


def _split_header(line: str, n: int) -> list[str]:
    """Column names of a header line: separated by two spaces or more (names may hold one space)."""
    parts = [p for p in re.split(r"\s{2,}|\t|,", line.strip()) if p]
    if len(parts) == n:
        return parts
    single = line.split()
    return single if len(single) == n else [f"col{k}" for k in range(n)]


@dataclass
class ReferenceTable:
    """Columns of a reference export: names (with units), units, values (one array per column), header fields."""
    columns: list[str]
    values: dict[str, np.ndarray]
    meta: dict = field(default_factory=dict)

    @property
    def units(self) -> dict[str, str]:
        """Unit of each column (the text in parentheses at the end of its name, or "")."""
        out = {}
        for c in self.columns:
            m = _UNIT.match(c)
            out[c] = m.group(2) if m else ""
        return out

    def column(self, name: str) -> np.ndarray:
        """Values of a column: exact name, name without its unit, or a unique whole-word match."""
        if name in self.values:
            return self.values[name]
        bare = {_bare(c): c for c in self.columns}
        if name in bare:
            return self.values[bare[name]]
        hits = [c for c in self.columns if re.search(rf"(?<![A-Za-z_]){re.escape(name)}(?![A-Za-z_])", c)]
        if len(hits) == 1:
            return self.values[hits[0]]
        raise ResultsError(f"column {name!r}: {'ambiguous ' + str(hits) if hits else 'not found'} "
                           f"(columns: {self.columns})")

    def data(self, x: str, y: str, quantity: str | None = None, coordinate: str | None = None):
        """lp.Data of column y over column x (sorted by x); quantity: loudpy's name ("p", "spl", "u_z"…, default
        y); coordinate: the coordinate's name (default "f" when x looks like a frequency, else x)."""
        from loudpy.results.data import Data
        xv, yv = np.real(self.column(x)).astype(float), self.column(y)
        o = np.argsort(xv)
        if coordinate is None:
            coordinate = "f" if re.match(r"(freq|f\b)", x, re.I) else x
        return Data(yv[o], {coordinate: xv[o]}, quantity or y, meta={"source": self.meta.get("path", "reference")})


def read_reference(path) -> ReferenceTable:
    """Read a text export (table, point graph, or nodal data with coordinates) as a ReferenceTable."""
    path = Path(path)
    header, rows = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("%"):
            header.append(s[1:].strip())
            continue
        rows.append([_number(t) for t in re.split(r"[\s,;]+", s) if t])
    if not rows:
        raise ResultsError(f"{path}: no data rows")
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        raise ResultsError(f"{path}: rows of different lengths")
    arr = np.array(rows, dtype=complex if any(isinstance(v, complex) for r in rows for v in r) else float)
    meta = {"path": str(path)}
    names = None
    for h in header:
        m = re.match(r"^([A-Za-z][\w ]*?):\s*(.*)$", h)
        if m and len(h.split()) <= 12 and not _looks_like_columns(h, n):
            meta[m.group(1).strip().lower()] = m.group(2).strip()
        else:
            names = h
    if names is None and header:
        names = header[-1]
    cols = _split_header(names, n) if names else [f"col{k}" for k in range(n)]
    if len(set(cols)) != len(cols):
        cols = [f"{c} [{k}]" for k, c in enumerate(cols)]
    values = {}
    for k, c in enumerate(cols):                      # a column written without imaginary parts stays real
        v = arr[:, k]
        values[c] = v.real.copy() if np.iscomplexobj(v) and not np.any(v.imag) else v
    return ReferenceTable(cols, values, meta)


def _looks_like_columns(h: str, n: int) -> bool:
    return len(_split_header(h, n)) == n and n > 1 and "(" in h


@dataclass
class Comparison:
    """loudpy result against a reference over their common coordinate range."""
    data: object                       # lp.Data: relative error over the coordinate
    max_error: float
    at: float
    rtol: float | None
    metric: str
    label: str = ""

    @property
    def passed(self) -> bool | None:
        """True / False against rtol, None without a tolerance."""
        return None if self.rtol is None else bool(self.max_error <= self.rtol)

    def __str__(self):
        dim = self.data.dims[0]
        verdict = "" if self.rtol is None else (" — within" if self.passed else " — OUTSIDE") + f" {self.rtol:g}"
        return (f"{self.label or 'comparison'}: max {self.metric} {self.max_error:.3g} at {dim} = {self.at:g}"
                f" over {self.data.coords[dim][0]:g} … {self.data.coords[dim][-1]:g}{verdict}")


def compare(ours, reference, rtol: float | None = None, metric: str = "relative", label: str = ""):
    """
    Compare two 1D lp.Data over the same coordinate: the reference is interpolated (complex: real and imaginary
    parts; a frequency axis: in log f) onto loudpy's points inside the common range. metric: "relative"
    (|a − b| / max |b|, the default: no blow-up at zeros), "pointwise" (|a − b| / |b|) or "db" (|20 log10 |a|/|b||).
    Returns a Comparison (max error, where, passed against rtol).
    """
    from loudpy.results.data import Data
    if len(ours.dims) != 1 or len(reference.dims) != 1:
        raise ResultsError("compare: two 1D Data (one coordinate each)")
    dim = ours.dims[0]
    if reference.dims[0] != dim:
        raise ResultsError(f"compare: coordinates differ ({dim!r} and {reference.dims[0]!r})")
    x, xr = np.asarray(ours.coords[dim], float), np.asarray(reference.coords[dim], float)
    keep = (x >= xr.min() - 1e-12 * abs(xr.min())) & (x <= xr.max() * (1 + 1e-12))
    if keep.sum() < 1:
        raise ResultsError("compare: no common range")
    x = x[keep]
    a, b = np.asarray(ours.values)[keep], np.asarray(reference.values)
    logx = dim == "f" and xr.min() > 0
    xi, xri = (np.log(x), np.log(xr)) if logx else (x, xr)
    bi = np.interp(xi, xri, np.real(b)) + (1j * np.interp(xi, xri, np.imag(b)) if np.iscomplexobj(b) else 0)
    if metric == "relative":
        err = np.abs(a - bi) / max(float(np.abs(bi).max()), 1e-300)
    elif metric == "pointwise":
        err = np.abs(a - bi) / np.maximum(np.abs(bi), 1e-300)
    elif metric == "db":
        err = np.abs(20 * np.log10(np.maximum(np.abs(a), 1e-300) / np.maximum(np.abs(bi), 1e-300)))
    else:
        raise ResultsError(f"compare: metric 'relative', 'pointwise' or 'db', not {metric!r}")
    k = int(np.argmax(err))
    d = Data(err, {dim: x}, "level" if metric == "db" else "error", meta={"metric": metric})
    return Comparison(d, float(err[k]), float(x[k]), rtol, metric, label)
