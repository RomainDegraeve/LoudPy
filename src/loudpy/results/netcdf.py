"""
netCDF files of labelled data (lp.Data): data.export("x.nc"), lp.load_data("x.nc").

netCDF is the standard exchange format of scientific arrays: xarray (xr.open_dataset), MATLAB (ncread / ncinfo),
Julia, R, ParaView, Panoply and ncdump read it. loudpy writes the classic (64-bit offset) format with SciPy, so no
extra package is needed:

    variables   one per coordinate (its values; text coordinates as an index with a "labels" attribute),
                the values: <quantity>, or <quantity>_real and <quantity>_imag for complex amplitudes
    attributes  units, long_name, symbol on every variable; globally the loudpy version, the quantity and the
                metadata (JSON)

    spl = res.spl(at=lp.Arc(1.0, range(0, 181, 5)))     # (f, angle)
    spl.export("spl.nc")
    xr.open_dataset("spl.nc")["spl"]                    # in another tool: the same labelled array
    lp.load_data("spl.nc")                              # back as an lp.Data
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from loudpy.errors import ResultsFormatError


def _name(text: str) -> str:
    s = re.sub(r"\W+", "_", str(text)).strip("_") or "x"
    return s if s[0].isalpha() else "x_" + s


def _attr(v) -> bytes:
    """An attribute value as UTF-8 bytes (classic netCDF text is bytes; units such as "°" or "µPa" are not ASCII)."""
    text = v if isinstance(v, str) else json.dumps(v, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
                                                    ensure_ascii=False)
    return text.encode("utf-8")


def _text(v) -> str:
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def write_data(data, path) -> Path:
    """Write an lp.Data as a netCDF file (classic 64-bit offset format)."""
    from scipy.io import netcdf_file
    import loudpy
    from loudpy.results.data import write_atomic
    q = data.quantity
    vname = _name(q.name)

    def write(tmp):
        f = netcdf_file(str(tmp), "w", version=2)
        try:
            f.Conventions = _attr("CF-1.8")
            f.title = _attr(q.label or q.name)
            f.source = _attr(f"loudpy {loudpy.__version__}")
            f.loudpy_quantity = _attr(q.name)
            f.loudpy_label = _attr(data.label or "")
            f.loudpy_meta = _attr(data.meta)
            dims = []
            for k, c in data.coords.items():
                d = _name(k)
                dims.append(d)
                f.createDimension(d, len(c))
                if c.dtype.kind in "fiu":
                    var = f.createVariable(d, "d", (d,))
                    var[:] = np.asarray(c, float)
                else:
                    var = f.createVariable(d, "i", (d,))
                    var[:] = np.arange(len(c))
                    var.labels = _attr([str(x) for x in c])
                cq = _coord_quantity(k)
                var.units = _attr(cq.unit or "1")
                var.long_name = _attr(cq.label or k)
                var.loudpy_name = _attr(k)
            values = np.asarray(data.values)
            parts = [("", values)] if not np.iscomplexobj(values) else [("_real", values.real), ("_imag", values.imag)]
            for suffix, arr in parts:
                var = f.createVariable(vname + suffix, "d", tuple(dims))
                var[...] = np.asarray(arr, float)
                var.units = _attr(q.unit or "1")
                var.long_name = _attr((q.label or q.name) + {"": "", "_real": " (real part)", "_imag": " (imaginary part)"}[suffix])
                if q.symbol:
                    var.symbol = _attr(q.symbol)
        finally:
            f.close()
    return write_atomic(Path(path), write)


def _coord_quantity(name):
    from loudpy.results.quantities import quantity
    return quantity(name)


def read_data(path):
    """Read a netCDF file written by data.export("x.nc") (or any file with one data variable, its coordinate
    variables and optionally <name>_real / <name>_imag) as an lp.Data."""
    from scipy.io import netcdf_file
    from loudpy.results.data import Data
    with netcdf_file(str(path), "r", mmap=False) as f:
        variables = dict(f.variables)
        dims_all = set(f.dimensions)
        data_vars = [k for k in variables if k not in dims_all]
        if not data_vars:
            raise ResultsFormatError(f"{path}: no data variable")
        qname = f.loudpy_quantity.decode("utf-8") if hasattr(f, "loudpy_quantity") else None
        base = _name(qname) if qname else re.sub(r"_(real|imag)$", "", data_vars[0])
        if base in variables:
            var = variables[base]
            values = np.array(var[...], dtype=float)
        elif base + "_real" in variables:
            var = variables[base + "_real"]
            values = np.array(var[...]) + 1j * np.array(variables[base + "_imag"][...])
        else:
            raise ResultsFormatError(f"{path}: no variable {base!r} (variables: {data_vars})")
        coords = {}
        for d in var.dimensions:
            cv = variables.get(d)
            name = cv.loudpy_name.decode("utf-8") if cv is not None and hasattr(cv, "loudpy_name") else d
            if cv is None:
                coords[name] = np.arange(f.dimensions[d])
            elif hasattr(cv, "labels"):
                coords[name] = np.array(json.loads(cv.labels.decode("utf-8")))
            else:
                coords[name] = np.array(cv[:], dtype=float)
        meta = json.loads(f.loudpy_meta.decode("utf-8")) if hasattr(f, "loudpy_meta") else {}
        label = f.loudpy_label.decode("utf-8") if hasattr(f, "loudpy_label") else ""
    return Data(values, coords, qname or base, label=label, meta=meta)
