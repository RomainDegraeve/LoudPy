"""
loudpy.plot.style — one figure style for the whole library (lp.plot, loudpy.Plotter, examples).

    import loudpy as lp
    lp.plot.style.use()                         # the loudpy style for every figure that follows
    fig, ax = lp.plot.style.figure("wide")      # standard sizes: "single", "wide", "square", "tall"
    lp.plot.style.frequency_axis(ax)            # log axis 20 Hz … with 20, 50, 100, 200, 500, 1k … ticks
    ax.set_ylabel(lp.plot.style.LABELS["spl"])
    lp.plot.style.save(fig, "Results/Figures/spl.pdf")

Choices (journal-like, colour-blind safe):
  - serif text and maths (STIX), 9–10 pt, thin axes, ticks inside on all four sides, minor ticks
  - Okabe–Ito colour cycle
  - colormaps by quantity: CMAPS["spl"] (magma), ["pressure"] (signed, RdBu_r), ["displacement"]
    (viridis), ["phase"] (twilight, cyclic), ["level"] (directivity maps, viridis)
  - SPL is RMS, in dB re 20 µPa (lp.spl), everywhere
  - PDF with embedded TrueType fonts (editable text), 300 dpi rasters
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import numpy as np

# Okabe & Ito (2008), colour-blind safe; yellow last (weak on white)
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]

CMAPS = {
    "spl": "magma",
    "pressure": "RdBu_r",
    "displacement": "viridis",
    "phase": "twilight",
    "level": "viridis",
}

LABELS = {
    "f": "Frequency $f$ [Hz]",
    "t": "Time $t$ [s]",
    "spl": "SPL [dB re 20 µPa RMS]",
    "spl_n": "SPL [dB re 20 µPa RMS / N]",
    "spl_v": "SPL [dB re 20 µPa RMS / V]",
    "p": "Pressure $p$ [Pa]",
    "u": "Displacement $|u|$ [m]",
    "uz": "Axial displacement $|u_z|$ [m]",
    "v": "Velocity $|v|$ [m/s]",
    "phase": "Phase [rad]",
    "angle": "Angle [deg]",
    "r": "$r$ [m]",
    "z": "$z$ [m]",
    "zeta": r"Damping ratio $\zeta$",
    "impedance": r"$|Z_e|$ [$\Omega$]",
}

SIZES = {                                     # inches (width, height)
    "single": (3.5, 2.6),                     # one journal column
    "wide": (7.2, 3.2),                       # full page width
    "square": (4.2, 4.0),
    "tall": (7.2, 6.0),
    "field": (4.5, 4.0),                      # 2D axisymmetric field (r, z)
}

RC = {
    "figure.dpi": 110,
    "figure.figsize": SIZES["wide"],
    "figure.constrained_layout.use": False,       # lp.plot passes layout="constrained" itself
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
    "axes.linewidth": 0.7,
    "axes.prop_cycle": mpl.cycler(color=COLORS),
    "axes.grid": True,
    "axes.grid.which": "major",
    "axes.axisbelow": True,
    "axes.formatter.use_mathtext": True,
    "axes.titleweight": "normal",
    "grid.color": "0.85",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.8,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "image.cmap": "viridis",
}


def use() -> None:
    """Apply the loudpy style to every figure created afterwards."""
    mpl.rcParams.update(RC)


def reset() -> None:
    """Back to the matplotlib defaults."""
    mpl.rcdefaults()


@contextmanager
def context():
    """with lp.plot.style.context(): … — the loudpy style inside the block only."""
    with mpl.rc_context(RC):
        yield


def figure(size: str | tuple = "wide", nrows: int = 1, ncols: int = 1, **kwargs):
    """plt.subplots with a standard size (SIZES name, or (width, height) in inches)."""
    import matplotlib.pyplot as plt
    figsize = SIZES[size] if isinstance(size, str) else size
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


_HZ_TICKS = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5])


def _hz(x, _pos=None) -> str:
    return f"{x / 1e3:g}k" if x >= 1e3 else f"{x:g}"


def frequency_axis(ax, lo: float | None = None, hi: float | None = None, label: bool = True, axis: str = "x"):
    """Log frequency axis with readable ticks (20, 50, 100, 200, 500, 1k, 2k …)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    if lo is not None or hi is not None:
        (ax.set_xlim if axis == "x" else ax.set_ylim)(lo, hi)
    a.set_major_locator(FixedLocator(_HZ_TICKS))
    a.set_major_formatter(FuncFormatter(_hz))
    a.set_minor_formatter(NullFormatter())
    ax.grid(True, which="minor", axis=axis, color="0.92", linewidth=0.4)
    if label:
        (ax.set_xlabel if axis == "x" else ax.set_ylabel)(LABELS["f"])
    return ax


def label_panels(axes, x: float = -0.12, y: float = 1.04, start: str = "a") -> None:
    """(a), (b), … in the top-left corner of each axes."""
    for k, ax in enumerate(np.ravel(axes)):
        ax.text(x, y, f"({chr(ord(start) + k)})", transform=ax.transAxes, fontsize=10, va="bottom", ha="left")


def save(fig, path, close: bool = True, **kwargs) -> Path:
    """Save a figure (folder created; written next to the target then renamed: safe on synced folders)."""
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(tmp, **kwargs)
    os.replace(tmp, path)
    if close:
        plt.close(fig)
    return path


@contextmanager
def pdf_pages(path):
    """Multi-page PDF (write-then-rename): with pdf_pages(path) as pdf: pdf.savefig(fig)."""
    from matplotlib.backends.backend_pdf import PdfPages
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.pdf")
    with PdfPages(tmp) as pdf:
        yield pdf
    os.replace(tmp, path)


__all__ = ["COLORS", "CMAPS", "LABELS", "SIZES", "RC", "use", "reset", "context", "figure", "frequency_axis",
           "label_panels", "save", "pdf_pages"]
