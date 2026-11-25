"""
Where the 2D geometry of a model comes from: lp.Model(geometry).

    "speaker.geo"     a gmsh .geo. Without surfaces (the loudpy curve-loop convention), every closed
                      physical curve group is a domain, made a surface by the mesher. With surfaces (Plane
                      Surface, Physical Surface …), its physical surface groups are the domains.
    "section.step"    a CAD file: .step / .stp / .iges / .igs (lengths converted to metres from the unit of
    "section.brep"    the file) or .brep (in metres). Its faces are the domains; a sketch without faces
                      (curves only) gives the closed regions of its curves.
                      Its DefineConstant[name = value, …] entries are parameters: a model parameter of the
                      same name (lp.Parameters) replaces the value (lp.Sweep over a .geo dimension).
    builder           a function builder(occ) drawing with gmsh's OpenCASCADE kernel (occ = gmsh.model.occ)
                      and returning the named parts {label: surface tag(s) or (dim, tag) pair(s)}:
                          def plate(occ):
                              return {"plate": occ.addRectangle(0, 0, 0, 0.1, 0.002)}

Surfaces (faces, parts of a builder, regions of a sketch) are fragmented together so that the mesh is
conforming. A surface in no named group is a domain of its own, "surface_<tag>": select it by position
(lp.Near, lp.Box); its mesh group then takes the name of the region. Named curve groups (physical curves,
curves returned by a builder) are boundary groups.

GeometryIndex(source) reads it once (one gmsh session): the layout ("loops" or "surfaces"), the domains,
the named curve groups, the domains on each side of every curve, and points along every curve (for the
selectors). The mesher loads the same source the same way (GeometrySource.load), so the entity tags agree.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import re
from pathlib import Path

import numpy as np

from loudpy.errors import MeshError, ModelError

CAD_UNITS = {".step", ".stp", ".iges", ".igs"}            # files that declare their length unit
CAD = CAD_UNITS | {".brep", ".brp"}


class GeometrySource:
    """A .geo, a CAD file (.step/.stp/.iges/.igs/.brep) or a builder function (module docstring)."""

    def __init__(self, geometry, parameters=None):
        self.parameters = parameters
        if callable(geometry) and not isinstance(geometry, (str, os.PathLike)):
            self.builder, self.path = geometry, None
            self.name = f"{getattr(geometry, '__name__', 'builder')}()"
            return
        self.builder, self.path = None, Path(geometry)
        if not self.path.exists():
            raise FileNotFoundError(f"geometry {self.path} not found")
        if self.path.suffix.lower() not in CAD | {".geo"}:
            raise ModelError(f"geometry {self.path.name}: expected a .geo, a CAD file ({', '.join(sorted(CAD))}) "
                             f"or a builder function")
        self.name = self.path.name

    # ── what a mesh built from it depends on ─────────────────────────────────
    def files(self) -> dict:
        """{file: sha256} of the geometry and the files a .geo merges; for a builder its code and constants."""
        if self.builder is not None:
            out: dict = {self.name: _builder_digest(self.builder)}
            if self.takes_parameters:
                out["parameters"] = {k: float(v) for k, v in sorted((self.parameters or {}).items())}
            return out

        def digest(path):
            with open(path, "rb") as fh:
                return hashlib.sha256(fh.read()).hexdigest()
        out = {self.path.name: digest(self.path)}
        if self.path.suffix.lower() == ".geo":
            for m in re.findall(r'Merge\s+"([^"]+)"', self.path.read_text()):
                mp = m if os.path.isabs(m) else os.path.join(self.path.parent, m)
                out[m] = digest(mp) if os.path.exists(mp) else "missing"
            values = self.geo_values()
            if values:
                out["parameters"] = values
        return out

    # ── parameters of a .geo (DefineConstant) ────────────────────────────────
    @property
    def geo_constants(self) -> list[str]:
        """Names declared in DefineConstant[…] of a .geo (empty for other sources)."""
        if self.path is None or self.path.suffix.lower() != ".geo":
            return []
        text = re.sub(r"//[^\n]*", "", self.path.read_text())
        names = []
        for block in re.findall(r"DefineConstant\s*\[(.*?)\]\s*;", text, re.S):
            names += re.findall(r"(?:^|,)\s*([A-Za-z_]\w*)\s*=", block)
        return list(dict.fromkeys(names))

    @property
    def geo_defaults(self) -> dict:
        """{name: default value} of the DefineConstant[…] entries of a .geo that are plain numbers."""
        if self.path is None or self.path.suffix.lower() != ".geo":
            return {}
        text = re.sub(r"//[^\n]*", "", self.path.read_text())
        out = {}
        for block in re.findall(r"DefineConstant\s*\[(.*?)\]\s*;", text, re.S):
            for name, value in re.findall(r"(?:^|,)\s*([A-Za-z_]\w*)\s*=\s*([-+0-9.eE]+)", block):
                try:
                    out[name] = float(value)
                except ValueError:
                    continue
        return out

    def geo_values(self) -> dict:
        """{name: value} of the model parameters that set a DefineConstant of the .geo."""
        params = self.parameters or {}
        return {k: float(params[k]) for k in self.geo_constants if k in params}

    def _open_geo(self, gmsh):
        """Open the .geo; with parameter values, through a wrapper that defines them first (DefineConstant keeps
        a value already defined; relative Merge / Include paths still resolve from the .geo's folder)."""
        values = self.geo_values()
        if not values:
            gmsh.open(str(self.path))
            return
        import tempfile
        lines = [f"{k} = {v!r};" for k, v in values.items()]
        lines.append(f'Include "{self.path.resolve()}";')
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = Path(tmp) / f"{self.path.stem}_parameters.geo"
            wrapper.write_text("\n".join(lines) + "\n")
            gmsh.open(str(wrapper))

    # ── loading into gmsh ────────────────────────────────────────────────────
    def load(self, gmsh) -> dict:
        """
        Put the geometry into the current gmsh model and return its layout:
        {"kind": "loops" | "surfaces", "domains": {name: tags}, "groups": {name: curve tags}, "unnamed": set}.
        loops: domains are closed physical curve groups (curve tags), groups every physical curve group;
        surfaces: domains are named surface groups and unnamed surfaces (surface tags), groups the named
        curve groups (kept as physical curve groups in the model; no physical surface group is left).
        """
        model, occ = gmsh.model, gmsh.model.occ
        if self.builder is not None:
            named = self._run_builder(gmsh)
            return _surfaces(gmsh, named, self.name)
        from loudpy.geometry.gmsh_session import in_metres, max_extent, step_correction
        if self.path.suffix.lower() == ".geo":
            self._open_geo(gmsh)
        else:
            with in_metres(gmsh, self.path.suffix.lower() in CAD_UNITS):
                gmsh.open(str(self.path))
            occ.synchronize()
            k = step_correction(self.path, max_extent(gmsh))            # (OpenCASCADE's unit state: see there)
            if k != 1.0:
                occ.dilate(occ.getEntities(), 0.0, 0.0, 0.0, k, k, k)
        occ.synchronize()
        named = _physical_groups(gmsh)
        if not model.getEntities(2):
            curve_groups = {n: t for (d, n), t in named.items() if d == 1}
            loops = {n: t for n, t in curve_groups.items() if _closed(gmsh, t)}
            if loops:                                              # the curve-loop convention: as it is
                return dict(kind="loops", domains=loops, groups=curve_groups, unnamed=set())
            if not model.getEntities(1):
                raise MeshError(f"{self.name}: no surface and no curve")
            named = _faces_of_sketch(gmsh, named, self.name)
        return _surfaces(gmsh, named, self.name)

    @property
    def takes_parameters(self) -> bool:
        """A builder whose second argument is named p, params or parameters receives the model's parameters."""
        if self.builder is None:
            return False
        import inspect
        try:
            args = list(inspect.signature(self.builder).parameters)
        except (TypeError, ValueError):
            return False
        return len(args) >= 2 and args[1] in ("p", "params", "parameters")

    def _run_builder(self, gmsh) -> dict:
        if self.takes_parameters:
            out = self.builder(gmsh.model.occ, self.parameters if self.parameters is not None else {})
        else:
            out = self.builder(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        named: dict = {}
        for label, items in (out or {}).items():
            for dim, tag in _dimtags(label, items):
                named.setdefault((dim, str(label)), []).append(tag)
        return named

    def __repr__(self):
        return f"GeometrySource({self.name})"


def _dimtags(label, items):
    if isinstance(items, (int, np.integer)):
        return [(2, int(items))]
    if isinstance(items, tuple) and len(items) == 2 and all(isinstance(v, (int, np.integer)) for v in items):
        return [(int(items[0]), int(items[1]))]
    try:
        return [d for it in items for d in _dimtags(label, it)]
    except TypeError:
        raise ModelError(f"builder: {label!r} must map to surface tags or (dim, tag) pairs, got {items!r}") from None


def _builder_digest(fn) -> str:
    """The code of a builder, its defaults, closure values and the simple module constants it uses."""
    h = hashlib.sha256()
    try:
        h.update(inspect.getsource(fn).encode())
    except (OSError, TypeError):
        h.update(getattr(fn, "__qualname__", repr(fn)).encode())
    simple = (int, float, str, bool, tuple, list, type(None))
    code = getattr(fn, "__code__", None)
    values = [getattr(fn, "__defaults__", None)]
    values += [c.cell_contents for c in (getattr(fn, "__closure__", None) or [])
               if isinstance(c.cell_contents, simple)]
    if code is not None:
        g = getattr(fn, "__globals__", {})
        values += [(n, g[n]) for n in code.co_names if n in g and isinstance(g[n], simple)]
    h.update(repr(values).encode())
    return h.hexdigest()


def _physical_groups(gmsh) -> dict:
    """{(dim, name): [tags]} of the named physical groups of dimension 1 and 2."""
    out: dict = {}
    for d, t in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(d, t).strip()
        if name and d in (1, 2):
            out.setdefault((d, name), [])
            out[(d, name)] += [int(e) for e in gmsh.model.getEntitiesForPhysicalGroup(d, t)]
    return out


def drop_physical_groups(gmsh, dim: int | None = None, names=()) -> None:
    """Remove the physical groups (of dimension `dim`, or all) and their names, plus the names `names` (gmsh
    keeps the name of a removed group, also of a group it dropped itself in a boolean operation, and a new
    group of that name would then get no name)."""
    old = [(d, t) for d, t in gmsh.model.getPhysicalGroups() if dim is None or d == dim]
    names = {gmsh.model.getPhysicalName(d, t) for d, t in old} | set(names)
    if old:                                       # (an empty list would remove every group)
        gmsh.model.removePhysicalGroups(old)
    for n in names - {""}:
        gmsh.model.removePhysicalName(n)


def _closed(gmsh, curves) -> bool:
    """Do the curves form closed loops (every end point shared by an even number of them)?"""
    if not curves:
        return False
    ends = [abs(p) for _, p in gmsh.model.getBoundary([(1, c) for c in curves], combined=False, oriented=False)]
    _, counts = np.unique(ends, return_counts=True) if ends else (None, np.array([0]))
    return bool(len(ends) == 0 or (counts % 2 == 0).all())


def _fragment(gmsh, inputs):
    """occ.fragment of the (dim, tag) inputs together; {input: [output tags]}."""
    occ = gmsh.model.occ
    if len(inputs) < 2:
        return {dt: [dt[1]] for dt in inputs}
    _, outmap = occ.fragment(inputs[:1], inputs[1:])
    occ.synchronize()
    return {dt: [t for _, t in pieces] for dt, pieces in zip(inputs, outmap)}


def _faces_of_sketch(gmsh, named, name) -> dict:
    """The closed regions of a sketch (curves only): fragment a rectangle around it with every curve and drop
    the face outside of the sketch. Named curve groups follow the fragmentation."""
    model, occ = gmsh.model, gmsh.model.occ
    curves = model.getEntities(1)
    x0, y0, _, x1, y1, _ = model.getBoundingBox(-1, -1)
    m = 0.1 * max(x1 - x0, y1 - y0)
    rect = occ.addRectangle(x0 - m, y0 - m, 0.0, x1 - x0 + 2 * m, y1 - y0 + 2 * m)
    pieces = _fragment(gmsh, [(2, rect)] + list(curves))
    outer = [t for _, t in model.getEntities(2) if model.isInside(2, t, [x0 - 0.5 * m, y0 - 0.5 * m, 0.0])]
    occ.remove([(2, t) for t in outer])
    occ.synchronize()
    if not model.getEntities(2):
        raise MeshError(f"{name}: the curves of the sketch enclose no region")
    return {k: sorted({p for t in tags for p in pieces.get((k[0], t), [t])}) for k, tags in named.items()}


def _surfaces(gmsh, named, name) -> dict:
    """Fragment the surfaces together (conforming mesh), map the named groups, keep the curve groups."""
    model, occ = gmsh.model, gmsh.model.occ
    surfaces = [(2, t) for _, t in occ.getEntities(2)]
    if not model.getEntities(2):
        raise MeshError(f"{name}: no surface")
    if surfaces:                                  # OpenCASCADE surfaces (the built-in kernel is used as it is)
        curves = sorted({(1, t) for (d, _), tags in named.items() if d == 1 for t in tags})
        pieces = _fragment(gmsh, surfaces + curves)
        named = {k: sorted({p for t in tags for p in pieces.get((k[0], t), [t])}) for k, tags in named.items()}
    drop_physical_groups(gmsh, names={n for _, n in named})
    groups = {n: t for (d, n), t in named.items() if d == 1}
    for n, tags in groups.items():
        model.addPhysicalGroup(1, tags, name=n)
    domains = {n: t for (d, n), t in named.items() if d == 2}
    owned = {t for tags in domains.values() for t in tags}
    unnamed = {f"surface_{t}": [t] for _, t in model.getEntities(2) if t not in owned}
    return dict(kind="surfaces", domains={**domains, **unnamed}, groups=groups, unnamed=set(unnamed))


class GeometryIndex:
    """
    A geometry source read once with gmsh (module docstring):

      kind      "loops" or "surfaces"
      domains   {name: tags} the domain groups, in order (loops: curve tags; surfaces: surface tags)
      unnamed   names of the unnamed surfaces ("surface_<tag>")
      groups    {name: curve tags} the named curve groups (loops: every physical curve group)
      outline   {domain: curve tags} its boundary (loops: its curves)
      sides     {curve: [domains]} the domains on its sides (a curve inside a domain: that domain twice)
      curves    {tag: (samples + 1, 2) points along the curve}
    """

    def __init__(self, source, samples: int = 16):
        from loudpy.geometry.gmsh_session import gmsh_session
        source = source if isinstance(source, GeometrySource) else GeometrySource(source)
        self.source, self.name = source, source.name
        with gmsh_session() as gmsh:
            layout = source.load(gmsh)
            model = gmsh.model
            self.kind = layout["kind"]
            self.domains = {n: [int(t) for t in tags] for n, tags in layout["domains"].items()}
            self.groups = {n: [int(t) for t in tags] for n, tags in layout["groups"].items()}
            self.unnamed = set(layout["unnamed"])
            self.outline: dict = {}
            self.sides: dict = {}
            for n, tags in self.domains.items():
                if self.kind == "loops":
                    self.outline[n] = list(tags)
                    for c in tags:
                        self.sides.setdefault(c, []).append(n)
                    continue
                self.outline[n] = sorted({abs(c) for _, c in model.getBoundary(
                    [(2, s) for s in tags], combined=True, oriented=False)})
                for s in tags:
                    for _, c in model.getBoundary([(2, s)], combined=False, oriented=False):
                        self.sides.setdefault(abs(c), []).append(n)
            self.curves = {}
            for _, c in model.getEntities(1):
                lo, hi = model.getParametrizationBounds(1, c)
                t = np.linspace(lo[0], hi[0], samples + 1)
                self.curves[int(c)] = np.asarray(model.getValue(1, c, t)).reshape(-1, 3)[:, :2]

    def segments(self, tags) -> np.ndarray:
        """(k, 2, 2) segments along the curves `tags`."""
        segs = [np.stack([x[:-1], x[1:]], axis=1) for x in (self.curves[int(t)] for t in tags)]
        return np.concatenate(segs) if segs else np.zeros((0, 2, 2))

    def __repr__(self):
        return (f"GeometryIndex({self.name}: {self.kind}, {len(self.domains)} domains, {len(self.groups)} curve "
                f"groups, {len(self.curves)} curves)")
