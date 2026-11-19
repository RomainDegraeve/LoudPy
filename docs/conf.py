"""Sphinx configuration of the loudpy documentation (build: python3 -m sphinx -b html docs docs/_build/html)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import loudpy  # noqa: E402

project = "loudpy"
author = "Romain Degraeve"
copyright = "2026, Romain Degraeve (GPL-3.0)"
release = loudpy.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]
autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": False,
                           "member-order": "bysource"}
autodoc_typehints = "description"
autodoc_class_signature = "mixed"
napoleon_numpy_docstring = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}
intersphinx_disabled_reftypes = ["*"]          # build offline as well

templates_path = ["_templates"]
exclude_patterns = ["_build", "ARCHITECTURE.md"]
html_theme = "furo"
html_title = f"loudpy {release}"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}
pygments_style = "friendly"
pygments_dark_style = "monokai"
nitpicky = False


def _plain_docstrings(app, what, name, obj, options, lines):
    """
    loudpy docstrings are plain text (aligned "name : description" lists, sketches, |x| for a
    modulus): the first paragraph is kept as text, the rest is shown as a literal block when it
    has indented or aligned lines.
    """
    for i, line in enumerate(lines):
        if "|" in line and not line.lstrip().startswith("+"):
            lines[i] = line.replace("|", r"\|")
    try:
        k = lines.index("")
    except ValueError:
        k = len(lines)
    head, rest = lines[:k], lines[k:]
    # a first paragraph with aligned "key : value" lines is itself a list: literal from the start
    if len(head) > 1 and any(l.startswith(" ") or " : " in l for l in head[1:]):
        head, rest = head[:1], [""] + head[1:] + rest
    if any(l.startswith((" ", "\t")) or " : " in l for l in rest if l.strip()):
        body = [l.replace(r"\|", "|") for l in rest]
        lines[:] = head + ["", "::", ""] + ["    " + l if l.strip() else "" for l in body]
    else:
        lines[:] = head + rest


def setup(app):
    app.connect("autodoc-process-docstring", _plain_docstrings)
