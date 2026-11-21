"""
Evaluated results: labelled arrays that know their quantity and units, plot themselves in the
loudpy style and export to NumPy, pandas and files read outside Python.

    spl = res.spl(at=lp.Arc(0.3, range(0, 91, 5)))     Data: spl [dB re 20 µPa] over (f, angle)
    np.asarray(spl), spl.values, spl.coords["f"]        plain NumPy arrays (no copy)
    spl.sel(angle=30)                                   one angle → Data over f
    spl.normalize(angle=0)                              dB re on axis
    spl.plot()                                          frequency × angle map (lines for few columns)
    spl.to_pandas()                                     DataFrame (index f, one column per angle)
    spl.export("spl.csv")                               .csv (Excel, MATLAB, Origin…), .mat, .npz

    u = res.field("u", f=1000)                          Field: values on the mesh at one entry
    u.component("z").plot(), u.plot(deformed=True)
    u.export("u_1kHz.vtu")                              ParaView (.vtu), .csv (node table), .mat, .npz

Data are harmonic amplitudes when complex: abs(), phase(), real, imag, db(), spl() give the parts;
plotting a complex Data shows its modulus.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from loudpy.results.quantities import Quantity, quantity as as_quantity
from loudpy.errors import ResultsError, SelectionError

P_REF = 20e-6


def spl(p, rms: bool = True) -> np.ndarray:
    """Sound pressure level [dB re 20 µPa] of complex amplitudes (RMS: |p|/√2; rms=False: |p|)."""
    a = np.abs(np.asarray(p)) / (np.sqrt(2.0) if rms else 1.0)
    return 20.0 * np.log10(np.maximum(a, 1e-300) / P_REF)


def mat_name(s: str) -> str:
    """A valid MATLAB variable name from any text."""
    s = re.sub(r"\W+", "_", s).strip("_") or "x"
    return s if s[0].isalpha() else "x_" + s


def write_atomic(path: Path, write):
    """Write a file through a temporary file renamed at the end (safe on synced folders); `write(tmp)`
    writes the temporary file. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    write(tmp)
    tmp.replace(path)
    return path


# ════════════════════════════════════════════════════════════════════════════
#  Data: values over named coordinates
# ════════════════════════════════════════════════════════════════════════════
class Data:
    """
    A labelled array: `values` with one named coordinate per axis (`coords`, in axis order), the
    physical `quantity` of the values, a `label` (legend) and `meta` (source, point, drive …).
    """
    __array_priority__ = 1000

    def __init__(self, values, coords: dict, quantity, label: str = "", meta: dict | None = None):
        self.values = np.asarray(values)
        self.coords = {k: np.asarray(v) for k, v in coords.items()}
        self.quantity: Quantity = as_quantity(quantity)
        self.label = label
        self.meta = dict(meta or {})
        if self.values.ndim != len(self.coords):
            raise ValueError(f"values have {self.values.ndim} axes, coords {list(self.coords)}")
        for (k, c), n in zip(self.coords.items(), self.values.shape):
            if len(c) != n:
                raise ValueError(f"coordinate {k!r} has {len(c)} values, axis has {n}")

    # ── NumPy ──────────────────────────────────────────────────────────────
    @property
    def dims(self) -> tuple[str, ...]:
        """Names of the coordinates, in the order of the value axes."""
        return tuple(self.coords)

    @property
    def x(self) -> np.ndarray:
        """The first coordinate (e.g. the frequencies)."""
        return next(iter(self.coords.values()))

    @property
    def shape(self):
        """Shape of the values."""
        return self.values.shape

    @property
    def ndim(self):
        """Number of coordinates."""
        return self.values.ndim

    def __len__(self):
        return len(self.values)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)

    def __getitem__(self, idx):
        return self.values[idx]

    def to_numpy(self) -> np.ndarray:
        """The values as a NumPy array."""
        return self.values

    def __repr__(self):
        def rng(k, c):
            if c.dtype.kind in "fiu" and len(c):
                unit = as_quantity(k).unit
                return f"{k}: {len(c)} ({c.min():.4g}–{c.max():.4g}{' ' + unit if unit else ''})"
            return f"{k}: {len(c)}"
        dims = " × ".join(rng(k, c) for k, c in self.coords.items()) or "scalar"
        kind = "complex " if np.iscomplexobj(self.values) else ""
        lab = f" {self.label!r}" if self.label else ""
        return f"Data({kind}{self.quantity.column()}{lab}, {dims})"

    # ── selection ──────────────────────────────────────────────────────────
    def _axis(self, dim):
        if dim not in self.coords:
            raise ResultsError(f"no dimension {dim!r} (dimensions: {list(self.coords)})")
        return list(self.coords).index(dim)

    def isel(self, **index) -> "Data":
        """Select by position along dimensions: d.isel(angle=0), d.isel(f=slice(0, 10))."""
        values, coords = self.values, dict(self.coords)
        for dim, i in index.items():
            ax = list(coords).index(dim) if dim in coords else self._axis(dim)
            values = np.take(values, np.atleast_1d(np.arange(values.shape[ax])[i]), axis=ax)
            c = coords[dim][i]
            if np.ndim(c) == 0:
                values = values.squeeze(axis=ax)
                coords.pop(dim)
            else:
                coords[dim] = c
        return Data(values, coords, self.quantity, self.label, self.meta)

    def sel(self, **value) -> "Data":
        """
        Select by coordinate value: the nearest one (d.sel(f=1000)), several (d.sel(angle=[0, 30, 60]))
        or a range (d.sel(t=slice(0, 0.05)), d.sel(f=slice(100, None))).
        """
        index: dict = {}
        for dim, v in value.items():
            c = self.coords[dim] if dim in self.coords else self.coords[list(self.coords)[self._axis(dim)]]
            if isinstance(v, slice):
                lo = -np.inf if v.start is None else v.start
                hi = np.inf if v.stop is None else v.stop
                index[dim] = list(np.flatnonzero((c >= lo) & (c <= hi)))
                continue
            if c.dtype.kind in "fiu":
                near = [int(np.argmin(np.abs(c - x))) for x in np.atleast_1d(v)]
            else:
                near = [int(np.flatnonzero(c == x)[0]) for x in np.atleast_1d(v)]
            index[dim] = near[0] if np.ndim(v) == 0 else near
        return self.isel(**index)

    def _new(self, values, quantity=None, label=None, **meta):
        return Data(values, self.coords, quantity or self.quantity, self.label if label is None else label,
                    {**self.meta, **meta})

    # ── parts of complex amplitudes ────────────────────────────────────────
    def abs(self) -> "Data":
        """Modulus |values|."""
        return self._new(np.abs(self.values), self.quantity.part("abs") if np.iscomplexobj(self.values)
                         else self.quantity)

    @property
    def real(self) -> "Data":
        """Real part."""
        return self._new(self.values.real, self.quantity.part("real"))

    @property
    def imag(self) -> "Data":
        """Imaginary part."""
        return self._new(self.values.imag, self.quantity.part("imag"))

    def phase(self, deg: bool = False, unwrap: bool = True) -> "Data":
        """Phase angle (unwrapped along the first dimension) [rad], or [°] with deg=True."""
        ph = np.angle(self.values)
        if unwrap and ph.ndim:
            ph = np.unwrap(ph, axis=0)
        return self._new(np.rad2deg(ph) if deg else ph, self.quantity.part("phase_deg" if deg else "phase"))

    def db(self, ref: float = 1.0) -> "Data":
        """20 log10(|values| / ref)."""
        return self._new(20 * np.log10(np.maximum(np.abs(self.values), 1e-300) / ref), self.quantity.part("db"))

    def spl(self, rms: bool = True) -> "Data":
        """Sound pressure level [dB re 20 µPa] of a pressure (RMS by default)."""
        return self._new(spl(self.values, rms), "spl" if rms else
                         Quantity("spl_peak", "Sound pressure level (peak)", "L_p", "dB re 20 µPa", "spl"))

    # ── operations ─────────────────────────────────────────────────────────
    def normalize(self, **at) -> "Data":
        """
        Relative to the values at a coordinate (e.g. normalize(angle=0): re on axis) or, with no
        argument, to the maximum modulus. dB quantities are subtracted, others divided.
        """
        if at:
            (dim, v), = at.items()
            ref = np.expand_dims(self.sel(**{dim: v}).values, self._axis(dim))
        else:
            ref = np.max(np.abs(self.values)) if not self.quantity.unit.startswith("dB") else np.max(self.values)
        out = self.values - ref if self.quantity.unit.startswith("dB") else self.values / ref
        return self._new(out, self.quantity.relative(), reference=str(at or "max"))

    def _jw(self):
        if "f" not in self.coords:
            raise ValueError("velocity() / acceleration() of harmonic data need a frequency dimension 'f'")
        shape = [1] * self.ndim
        shape[self._axis("f")] = -1
        return (2j * np.pi * self.coords["f"]).reshape(shape)

    def _vector_quantity(self, letter):
        n = self.quantity.name
        if n[:1] in "uva" and (len(n) == 1 or n[1] == "_"):
            return as_quantity(letter + n[1:])
        return self.quantity

    def velocity(self) -> "Data":
        """jω × (harmonic displacement): the velocity."""
        return self._new(self._jw() * self.values, self._vector_quantity("v"))

    def acceleration(self) -> "Data":
        """−ω² × (harmonic displacement): the acceleration."""
        return self._new(self._jw() ** 2 * self.values, self._vector_quantity("a"))

    def __neg__(self):
        return self._new(-self.values)

    def _binary(self, other, op, quantity=None):
        o = other.values if isinstance(other, Data) else np.asarray(other)
        return self._new(op(self.values, o), quantity)

    def __mul__(self, o):
        return self._binary(o, np.multiply)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self._binary(o, np.divide)

    def __rtruediv__(self, o):
        return self._new(np.asarray(o) / self.values)

    def __add__(self, o):
        return self._binary(o, np.add)

    __radd__ = __add__

    def __sub__(self, o):
        return self._binary(o, np.subtract)

    def __rsub__(self, o):
        return self._new(np.asarray(o) - self.values)

    def rename(self, quantity=None, label=None) -> "Data":
        """The same values with another quantity (name or Quantity) and / or legend label."""
        return Data(self.values, self.coords, self.quantity if quantity is None else quantity,
                    self.label if label is None else label, self.meta)

    # ── signal processing ──────────────────────────────────────────────────
    def spectrum(self, start: float | None = None, window: str | None = "hann") -> "Data":
        """
        Single-sided amplitude spectrum of time data (dimension "t"), from t ≥ start (default: the
        last three quarters). window="hann" (amplitude-corrected) or None (rectangular: exact
        amplitudes at frequencies on the FFT grid). Complex amplitudes over "f".
        """
        if self.dims[0] != "t":
            raise ValueError("spectrum() needs time data (first dimension 't')")
        t = self.coords["t"]
        i0 = int(np.searchsorted(t, start)) if start is not None else len(t) // 4
        x = np.real(self.values[i0:])
        n = len(x)
        w = np.hanning(n) if window == "hann" else np.ones(n)
        X = np.fft.rfft(x * (w[:, None] if x.ndim > 1 else w), axis=0) * 2.0 / (n * w.mean())
        f = np.fft.rfftfreq(n, d=float(t[1] - t[0]))
        coords = {"f": f, **{k: v for k, v in self.coords.items() if k != "t"}}
        return Data(X, coords, self.quantity, self.label, {**self.meta, "window": window or "rectangular"})

    def impulse_response(self, fs: float | None = None, df: float = 1.0, taper: bool = True) -> "Data":
        """
        Impulse response (dimension "t") of a one-sided frequency response (dimension "f"): modulus and
        unwrapped phase interpolated on a uniform grid 0…fs/2 (resolution df, window 1/df), zero outside
        the computed band, raised-cosine tapers at the band edges, inverse real FFT.
        """
        if self.dims[0] != "f" or self.ndim != 1:
            raise ValueError("impulse_response() needs 1D frequency data")
        f, H = self.coords["f"].astype(float), self.values.astype(complex)
        fs = 2.0 * f[-1] if fs is None else fs
        n_fft = int(2 ** np.ceil(np.log2(max(fs / df, 256))))
        fu = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        mag = np.interp(fu, f, np.abs(H), left=0.0, right=0.0)
        ph = np.interp(fu, f, np.unwrap(np.angle(H)))
        band = (fu >= f[0]) & (fu <= f[-1])
        Hu = np.zeros_like(fu, dtype=complex)
        Hu[band] = (mag * np.exp(1j * ph))[band]
        if taper:
            idx = np.flatnonzero(band)
            e = max(1, idx.size // 20)
            edge = 0.5 * (1.0 - np.cos(np.pi * np.arange(e) / e))
            Hu[idx[:e]] *= edge
            Hu[idx[-e:]] *= edge[::-1]
        h = np.fft.irfft(Hu, n=n_fft)
        q = Quantity(f"h_{self.quantity.name}", f"Impulse response ({self.quantity.label.lower()})", "h",
                     f"{self.quantity.unit}/s" if self.quantity.unit else "")
        return Data(h, {"t": np.arange(n_fft) / fs}, q, self.label, {**self.meta, "fs": fs, "df": df})

    # ── export ─────────────────────────────────────────────────────────────
    def _columns(self):
        """(names, 2D array) of a table: first coordinate, then the values (re/im for complex)."""
        x = self.x
        vals = self.values.reshape(len(x), -1) if self.ndim else self.values.reshape(1, 1)
        if self.ndim > 1:
            other = list(self.coords)[1]
            heads = [f"{self.quantity.column()} {other}={c}" for c in self.coords[other]]
        else:
            heads = [self.quantity.column()]
        if np.iscomplexobj(vals):
            names = [f"{h} {p}" for h in heads for p in ("re", "im")]
            vals = np.stack([vals.real, vals.imag], axis=2).reshape(len(x), -1)
        else:
            names = heads
        return [as_quantity(list(self.coords)[0]).column()] + names, x, vals

    def to_pandas(self):
        """pandas object: Series (1D, indexed by the coordinate) or DataFrame (2D: one column per value of the second coordinate)."""
        import pandas as pd
        dims = list(self.coords)
        idx = pd.Index(self.coords[dims[0]], name=as_quantity(dims[0]).column())
        if self.ndim == 1:
            return pd.Series(self.values, index=idx, name=self.quantity.column())
        if self.ndim == 2:
            cols = pd.Index(self.coords[dims[1]], name=as_quantity(dims[1]).column())
            return pd.DataFrame(self.values, index=idx, columns=cols)
        raise ValueError("to_pandas() takes 1D or 2D data (use .sel() first)")

    def to_xarray(self):
        """xarray.DataArray (needs xarray)."""
        import xarray as xr
        return xr.DataArray(self.values, coords=self.coords, dims=self.dims, name=self.quantity.name,
                            attrs=dict(units=self.quantity.unit, long_name=self.quantity.label, **self._header()))

    def _header(self) -> dict:
        return {"quantity": self.quantity.name, "label": self.quantity.label, "unit": self.quantity.unit,
                **{k: str(v) for k, v in self.meta.items()}}

    def to_csv(self, path):
        """
        Text table: '#' header lines (quantity, unit, source…), a line of column names, then one
        row per value of the first coordinate (2D data: one column per value of the second;
        complex values: re and im columns). Read with numpy.loadtxt, pandas.read_csv(comment="#"),
        MATLAB readmatrix / readtable, Excel.
        """
        names, x, vals = self._columns()
        head = "\n".join(f"{k}: {v}" for k, v in self._header().items())

        def write(tmp):
            np.savetxt(tmp, np.column_stack([x, vals]), delimiter=",", fmt="%.10g",
                       header=head + "\n" + ",".join(names), comments="# ")
        return write_atomic(path, write)

    def to_mat(self, path):
        """MATLAB / Octave .mat file: the values (complex kept), one variable per coordinate, the units."""
        from scipy.io import savemat
        d = {mat_name(self.quantity.name): self.values}
        for k, c in self.coords.items():
            d[mat_name(k)] = c if c.dtype.kind in "fiuc" else c.astype(str)
        d["info"] = self._header() | {"dims": ",".join(self.dims)}
        return write_atomic(path, lambda tmp: savemat(tmp, d))

    def to_npz(self, path):
        """NumPy .npz: values, one array per coordinate, header (JSON)."""
        arrays = {"values": self.values, **{f"coord_{k}": c for k, c in self.coords.items()}}
        arrays["header"] = np.array(json.dumps(self._header() | {"dims": list(self.dims)}))
        return write_atomic(path, lambda tmp: np.savez(tmp, **arrays))

    def export(self, path):
        """Write by extension: .csv / .txt (text table), .mat (MATLAB), .npz (NumPy)."""
        suffix = Path(path).suffix.lower()
        if suffix in (".csv", ".txt", ".dat"):
            return self.to_csv(path)
        if suffix == ".mat":
            return self.to_mat(path)
        if suffix == ".npz":
            return self.to_npz(path)
        raise ValueError(f"unknown export format {suffix!r}: use .csv, .mat or .npz")

    # ── plot ───────────────────────────────────────────────────────────────
    def plot(self, ax=None, kind: str | None = None, **kwargs):
        """
        Draw in the loudpy style (labels and units from the quantity); returns the axes.
        kind: "line" (default for 1D, and 2D with ≤ 12 columns: one line per column), "map"
        (2D: colour map over both coordinates), "polar" (over "angle"). See loudpy.plot.
        """
        from loudpy import plot
        return plot.data(self, ax=ax, kind=kind, **kwargs)


# ════════════════════════════════════════════════════════════════════════════
#  Field: values on a mesh at one entry
# ════════════════════════════════════════════════════════════════════════════
class Field:
    """
    One field at one entry: `values` (n_nodes,) or (n_nodes, n_components) on `mesh`, at
    `coordinate` (a frequency, a time or a mode frequency; `coord` names it).
    Vector fields (u, v, a) have the components r, z (and theta for azimuthal orders m ≥ 1).
    """

    def __init__(self, name: str, mesh, values, coordinate: float, coord: str = "f", extra: dict | None = None,
                 quantity=None, meta: dict | None = None):
        self.name = name
        self.mesh = mesh
        self.values = np.asarray(values)
        self.coordinate = float(coordinate)
        self.coord = coord
        self.extra = dict(extra or {})
        self.quantity: Quantity = as_quantity(quantity or name)
        self.meta = dict(meta or {})

    def __repr__(self):
        comp = f", {self.values.shape[1]} components" if self.values.ndim > 1 else ""
        unit = as_quantity(self.coord).unit
        return (f"Field({self.quantity.column()} at {self.coord} = {self.coordinate:.5g}{' ' + unit if unit else ''}, "
                f"{self.mesh.n_nodes} nodes{comp})")

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)

    @property
    def is_vector(self) -> bool:
        """True for a vector field (values (n, components))."""
        return self.values.ndim > 1

    def with_values(self, values, quantity, **meta) -> "Field":
        """The same mesh, coordinate and metadata with other nodal values and quantity (a derived scalar or
        component; meta updates the metadata)."""
        return Field(self.name, self.mesh, values, self.coordinate, self.coord, self.extra, quantity,
                     {**self.meta, **meta})

    # ── scalar views ───────────────────────────────────────────────────────
    def component(self, c: str) -> "Field":
        """One component: "r", "z" or "theta" of a vector field; "rr", "zz", "rz", "tt" of strain / stress."""
        if not self.is_vector:
            raise ValueError(f"{self.name} is a scalar field")
        names = list(self.meta.get("components") or ("r", "z", "theta"))
        if c not in names or names.index(c) >= self.values.shape[1]:
            raise ValueError(f"{self.name} has no {c!r} component (components: {names[:self.values.shape[1]]})")
        return self.with_values(self.values[:, names.index(c)], f"{self.name}_{c}")

    def magnitude(self) -> "Field":
        """Modulus per node (Euclidean norm of the component moduli for a vector field)."""
        v = np.abs(self.values)
        return self.with_values(v if v.ndim == 1 else np.linalg.norm(v, axis=1), self.quantity.part("abs"))

    def spl(self, rms: bool = True) -> "Field":
        """Sound pressure level [dB re 20 µPa] of a pressure field."""
        return self.with_values(spl(self.values, rms), "spl")

    @property
    def real(self) -> "Field":
        """Real part."""
        return self.with_values(np.real(self.values), self.quantity.part("real"))

    @property
    def imag(self) -> "Field":
        """Imaginary part."""
        return self.with_values(np.imag(self.values), self.quantity.part("imag"))

    def phase(self, deg: bool = False) -> "Field":
        """Phase [rad, or degrees with deg=True]; for a vector field, of its largest component at each node."""
        v = self.values if not self.is_vector else self.values[np.arange(len(self.values)),
                                                                np.argmax(np.abs(self.values), axis=1)]
        ph = np.angle(v)
        return self.with_values(np.rad2deg(ph) if deg else ph, self.quantity.part("phase_deg" if deg else "phase"))

    def at_phase(self, phi: float) -> "Field":
        """Instantaneous real field Re(value · e^{iφ}) of a harmonic amplitude (φ [rad])."""
        return self.with_values(np.real(self.values * np.exp(1j * phi)), self.quantity.part("real"), phase=phi)

    # ── where ──────────────────────────────────────────────────────────────
    def restrict(self, region) -> "Field":
        """The field on the mesh groups selected by `region`: a label ("surround"; naming rules) or a selector
        of domains (lp.Group, lp.Box, lp.Near and their combinations, evaluated on this field's mesh)."""
        from loudpy.geometry.mesh import Mesh
        from loudpy.geometry.naming import matches
        if isinstance(region, str):
            names = [n for n in self.mesh.groups if matches(region, n)]
        else:
            from loudpy.geometry.selectors import Domains, as_selector
            sel = as_selector(region).domains(Domains.from_mesh(self.mesh))
            names = [n for n in self.mesh.groups if n in sel]
            region = repr(as_selector(region))
        if not names:
            raise SelectionError(f"no mesh group selected by {region!r} (groups: {list(self.mesh.groups)})")
        tags = np.unique(np.concatenate([self.mesh.groups[n].cells.ravel() for n in names]))
        rows = self.mesh.rows_of(tags)
        sub = Mesh(tags, self.mesh.coords[rows], {n: self.mesh.groups[n] for n in names}, self.mesh.path)
        return Field(self.name, sub, self.values[rows], self.coordinate, self.coord, self.extra, self.quantity,
                     {**self.meta, "region": region})

    def at_nodes(self, tags) -> tuple[np.ndarray, np.ndarray]:
        """(coords, values) at the node tags present on this field's mesh (e.g. res.boundary(...))."""
        pos = self.mesh._positions
        rows = np.array([pos[int(t)] for t in np.asarray(tags) if int(t) in pos], dtype=np.int64)
        return self.mesh.coords[rows], self.values[rows]

    def at(self, at) -> "Data | complex":
        """Values at a point (a number or a vector) or along a selection (lp.Line, lp.Arc, …): Data."""
        from loudpy.results.selection import as_selection
        sel = as_selection(at)
        W = self.mesh.interpolation(sel.points(self.mesh.dim))
        v = W @ self.values
        if sel.dim is None:
            return v[0]
        coords = {sel.dim: sel.coord}
        if v.ndim > 1:
            coords["component"] = np.array(list(self.meta.get("components") or ("r", "z", "theta"))[:v.shape[1]])
        meta = {k: v_ for k, v_ in self.meta.items() if k != "components"}
        return Data(v, coords, self.quantity, meta={**meta, self.coord: self.coordinate, "at": sel.describe()})

    def revolve(self, order: int | None = None, **kwargs):
        """3D surface (pyvista) of an axisymmetric displacement: see loudpy.results.surfaces.revolve.
        order: azimuthal order m (default: the field's own, 0 for axisymmetric results)."""
        from loudpy.results.surfaces import revolve
        m = self.meta.get("order", 0) if order is None else order
        return revolve(self, m, **kwargs)

    # ── export ─────────────────────────────────────────────────────────────
    def _arrays(self) -> dict:
        names = ["r", "z"] if self.mesh.dim == 2 else ["x", "y", "z"]
        cols = {n: self.mesh.coords[:, i] for i, n in enumerate(names)}
        v = self.values
        names = list(self.meta.get("components") or ("r", "z", "theta"))
        comps = [""] if v.ndim == 1 else [f"_{c}" for c in names[:v.shape[1]]]
        for j, c in enumerate(comps):
            vj = v if v.ndim == 1 else v[:, j]
            if np.iscomplexobj(vj):
                cols[f"{self.name}{c}_re"], cols[f"{self.name}{c}_im"] = vj.real, vj.imag
            else:
                cols[f"{self.quantity.name}{c}"] = vj
        return cols

    def to_pandas(self):
        """DataFrame: one row per node (node tag index), columns r, z (x, y, z) and the values."""
        import pandas as pd
        return pd.DataFrame(self._arrays(), index=pd.Index(self.mesh.node_tags, name="node"))

    def to_pyvista(self):
        """pyvista mesh (linear sub-cells through the mid-side nodes) with the values as point data."""
        import pyvista as pv
        mesh = self.mesh
        if mesh.dim == 2:
            tri = mesh.triangles(refine=True)
            pts = np.column_stack([mesh.coords, np.zeros(mesh.n_nodes)])
            grid = pv.PolyData(pts, np.hstack([np.full((len(tri), 1), 3), tri]).ravel())
        else:
            from loudpy.plot.view3d import tet_grid
            grid = tet_grid(mesh)
        for k, v in self._arrays().items():
            if k not in ("r", "z", "x", "y"):
                grid.point_data[k] = v
        return grid

    def _header(self):
        return {"field": self.name, "quantity": self.quantity.label, "unit": self.quantity.unit,
                self.coord: self.coordinate, **{k: str(v) for k, v in self.meta.items()}}

    def export(self, path):
        """
        Write by extension: .vtu / .vtk (ParaView: mesh + values), .html (interactive viewer, lp.plot.html),
        .csv (one row per node: coordinates,
        values, re/im for complex), .mat (MATLAB: coordinates, triangles, values), .npz (NumPy).
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".html", ".htm"):
            from loudpy.plot.html import html
            return html(self, path)
        if suffix in (".vtu", ".vtk"):
            return write_atomic(path, lambda tmp: self.to_pyvista().cast_to_unstructured_grid().save(tmp))
        cols = self._arrays()
        if suffix in (".csv", ".txt"):
            head = "\n".join(f"{k}: {v}" for k, v in self._header().items())
            return write_atomic(path, lambda tmp: np.savetxt(
                tmp, np.column_stack(list(cols.values())), delimiter=",", fmt="%.10g",
                header=head + "\n" + ",".join(cols), comments="# "))
        tri = self.mesh.triangles(refine=True) if self.mesh.dim == 2 else np.zeros((0, 3), int)
        if suffix == ".mat":
            from scipy.io import savemat
            d = {mat_name(k): v for k, v in cols.items()}
            d.update(values=self.values, triangles=tri + 1, node_tags=self.mesh.node_tags, info=self._header())
            return write_atomic(path, lambda tmp: savemat(tmp, d))
        if suffix == ".npz":
            return write_atomic(path, lambda tmp: np.savez(tmp, values=self.values, coords=self.mesh.coords,
                                                            triangles=tri, node_tags=self.mesh.node_tags,
                                                            header=np.array(json.dumps(self._header()))))
        raise ValueError(f"unknown export format {suffix!r}: use .vtu, .html, .csv, .mat or .npz")

    # ── plot ───────────────────────────────────────────────────────────────
    def plot(self, show: str | None = None, ax=None, **kwargs):
        """
        Colour map on the mesh (2D: r–z section; 3D: rendered view); returns the axes.
        show: what to colour — "spl" (default for a pressure), "abs" (default otherwise), "real",
        "imag", "phase", or a component "r", "z", "theta". Keywords: see loudpy.plot.field.
        """
        from loudpy import plot
        return plot.field(self, show=show, ax=ax, **kwargs)
