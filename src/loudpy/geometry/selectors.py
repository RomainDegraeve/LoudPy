"""
Selectors: which domains a region covers, which edges a boundary covers.

    lp.Group("membranne")                       groups selected by label (naming rules, globs allowed)
    lp.Group("membranne", "dustcap")            several labels
    lp.Box(r=(0.0, 0.01), z=(-0.02, 0.0))       what lies entirely inside a box; a number is a line
                                                (lp.Box(r=0): on the axis); 3D: x=, y=, z=
    lp.Near((0.05, -0.003))                     the domain containing the point, or the boundary curve
                                                nearest to it
    lp.Interface("membranne", "subacou_front")  edges shared by two selections of domains
    lp.BoundaryOf("surround")                   edges on the boundary of a selection of domains
    a | b,  a & b,  a - b                       union, intersection, difference

A string stands for lp.Group(string); a region or boundary of the model (lp.Solid(…), lp.Air(…),
lp.Clamped(…)…) stands for what it covers. Every model item takes its selector as `on=` (default:
its name as a label).

Where they are evaluated:

  * a region covers whole domain groups. Its selector is evaluated before meshing on the domain groups
    of the geometry (Domains.from_geometry: the curve loops of a .geo), or on the domain groups of a mesh
    (Domains.from_mesh, e.g. to restrict a result field);
  * a boundary covers edges of the mesh. Its selector is evaluated after meshing on the boundary edges of
    the 2D mesh (Edges): the edges between two different domain groups or on the outside of the meshed
    domains, plus those of the named curve groups kept in the mesh. An interface therefore needs no curve
    drawn in the geometry, and a mesh read from a file is selected the same way as a fresh one;
  * a PML layer grows from curves of the geometry: its selector is evaluated before meshing on the curves
    of the geometry (Curves), with the same rules (sides = the meshed domains on each side).

A "boundary curve" (lp.Near on edges) is a chain of edges between corners (turn > 30°), junctions (a node
with more than two boundary edges) and changes of the adjacent domains.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from loudpy.errors import SelectionError
from loudpy.geometry import naming

_CORNER_COS = np.cos(np.deg2rad(30.0))          # chains of lp.Near stop where the boundary turns by more
_AXES = {2: ("r", "z"), 3: ("x", "y", "z")}


# ════════════════════════════════════════════════════════════════════════════
#  Selectors
# ════════════════════════════════════════════════════════════════════════════
class Selector:
    """Base class: a selection of domain groups (`domains`) or of boundary edges (`edges`), combined with
    `|` (union), `&` (intersection) and `-` (difference)."""

    def domains(self, ctx: "Domains") -> set[str]:
        """Names of the domain groups the selector selects in the context `ctx` (a Domains). Selectors of edges
        only raise a SelectionError."""
        raise SelectionError(f"{self!r} selects boundary edges, not domains (use it for a boundary)")

    def edges(self, ctx: "Edges | Curves") -> set[int]:
        """Codes of the boundary edges the selector selects in the context `ctx` (an Edges or a Curves).
        Selectors of domains only raise a SelectionError."""
        raise SelectionError(f"{self!r} selects domains, not boundary edges (lp.BoundaryOf(…) gives its edges)")

    def labels(self) -> list[str]:
        """The labels that select named boundary groups (the mesher keeps these curve groups)."""
        return []

    @property
    def by_groups(self) -> bool:
        """True when made of labels only (then a boundary is built from the named groups themselves)."""
        return False

    def __or__(self, other):
        return _Union(self, as_selector(other))

    def __ror__(self, other):
        return _Union(as_selector(other), self)

    def __and__(self, other):
        return _Intersection(self, as_selector(other))

    def __rand__(self, other):
        return _Intersection(as_selector(other), self)

    def __sub__(self, other):
        return _Difference(self, as_selector(other))

    def __rsub__(self, other):
        return _Difference(as_selector(other), self)


class Combinable:
    """Mixin of the model items (regions, boundaries): `|`, `&`, `-` combine what they cover into selectors
    (lp.Interface(cone | dustcap, "subacou_front"))."""

    def __or__(self, other):
        return as_selector(self) | other

    def __ror__(self, other):
        return as_selector(other) | as_selector(self)

    def __and__(self, other):
        return as_selector(self) & other

    def __rand__(self, other):
        return as_selector(other) & as_selector(self)

    def __sub__(self, other):
        return as_selector(self) - other

    def __rsub__(self, other):
        return as_selector(other) - as_selector(self)


def as_selector(x) -> Selector:
    """A Selector from a selector, a label (str) or a model item (region or boundary)."""
    if isinstance(x, Selector):
        return x
    if isinstance(x, str):
        return Group(x)
    if hasattr(x, "selector") and hasattr(x, "name"):
        return _Item(x)
    raise TypeError(f"cannot select with {x!r}: give a label (str), an lp.Group, lp.Box, lp.Near, lp.Interface, "
                    f"lp.BoundaryOf or a region / boundary of the model")


class Group(Selector):
    """The named groups selected by each label (rules of loudpy.geometry.naming; "glue" selects
    submeca_glue_1 and submeca_glue_2, "glue_*" is a glob). Every label must select a group. For domains, a
    label that is the name of a region of the model stands for that region (what it covers)."""

    def __init__(self, *labels: str):
        if not labels or not all(isinstance(s, str) and s.strip() for s in labels):
            raise TypeError(f"lp.Group needs one or more labels (non-empty strings), got {labels!r}")
        self.labels_ = tuple(s.strip() for s in labels)

    def domains(self, ctx):
        out = set()
        for label in self.labels_:
            out |= set(ctx.regions[label]) if label in ctx.regions else \
                _match_labels([label], ctx.names, f"domain group of {ctx.where}")
        return out

    def edges(self, ctx):
        for label in self.labels_:
            if not any(naming.matches(label, n) for n in ctx.curve_groups) and \
                    any(naming.matches(label, n) for n in ctx.domain_names):
                raise SelectionError(f"{label!r} selects domain groups, not boundary edges: for the edges around "
                                     f"them use lp.BoundaryOf({label!r}) or lp.Interface({label!r}, …)")
        names = _match_labels(self.labels_, list(ctx.curve_groups), f"boundary group of {ctx.where}")
        return set().union(*(ctx.curve_groups[n] for n in names))

    def labels(self):
        """The labels of the group (names or glob patterns)."""
        return list(self.labels_)

    @property
    def by_groups(self):
        """True when the selection is made of labels only (one part per named group is kept)."""
        return True

    def __repr__(self):
        return f"Group({', '.join(map(repr, self.labels_))})"


def _match_labels(labels, names, what) -> set[str]:
    out = set()
    for label in labels:
        sel = [n for n in names if naming.matches(label, n)]
        if not sel:
            raise SelectionError(f"{label!r} selects no {what}; groups: {list(names)}")
        out |= set(sel)
    return out


class Box(Selector):
    """What lies entirely inside a box. Each coordinate: (min, max), a number (a line: lp.Box(r=0) is the
    axis) or None (unbounded); 2D (axisymmetric): r, z; 3D: x, y, z. tol [m]: tolerance on the bounds
    (default 1e-6 × the size of the model)."""

    def __init__(self, *, r=None, z=None, x=None, y=None, tol: float | None = None):
        self.bounds = {k: _interval(k, v) for k, v in dict(r=r, x=x, y=y, z=z).items() if v is not None}
        if not self.bounds:
            raise TypeError("lp.Box needs at least one coordinate range (r=, z= in 2D; x=, y=, z= in 3D)")
        self.tol = tol

    def _inside(self, pts: np.ndarray, scale: float) -> np.ndarray:
        """(n, k, dim) points → (n,) all k points inside."""
        dim = pts.shape[-1]
        axes = _AXES[dim]
        bad = [k for k in self.bounds if k not in axes]
        if bad:
            raise SelectionError(f"{self!r}: coordinate(s) {bad} do not exist in {dim}D (use {', '.join(axes)})")
        tol = self.tol if self.tol is not None else 1e-6 * max(scale, 1e-12)
        ok = np.ones(pts.shape[:-1], bool)
        for k, (lo, hi) in self.bounds.items():
            v = pts[..., axes.index(k)]
            ok &= (v >= lo - tol) & (v <= hi + tol)
        return np.asarray(ok.all(axis=-1))

    def domains(self, ctx):
        return {n for n in ctx.names if self._inside(ctx.points(n)[None], ctx.scale)[0]}

    def edges(self, ctx):
        keep = self._inside(ctx.points(), ctx.scale)
        return set(ctx.codes[keep].tolist())

    def __repr__(self):
        parts = [f"{k}={lo:g}" if lo == hi else f"{k}=({lo:g}, {hi:g})" for k, (lo, hi) in self.bounds.items()]
        return f"Box({', '.join(parts)})"


def _interval(name, v):
    if np.ndim(v) == 0:
        return float(v), float(v)
    if len(v) != 2:
        raise TypeError(f"lp.Box({name}=…): give (min, max) or a number, got {v!r}")
    lo, hi = (-np.inf if v[0] is None else float(v[0])), (np.inf if v[1] is None else float(v[1]))
    if lo > hi:
        raise ValueError(f"lp.Box({name}=…): min {lo} > max {hi}")
    return lo, hi


class Near(Selector):
    """For a region: the domain group containing `point` (the nearest one if none does). For a boundary:
    the boundary curve nearest to `point` (on the mesh: a chain of edges between corners, junctions and
    changes of the adjacent domains; on the geometry, e.g. the base of a PML: a curve of the .geo)."""

    def __init__(self, point):
        self.point = np.asarray(point, float).ravel()

    def domains(self, ctx):
        p = self.point[:ctx.dim]
        inside = [n for n in ctx.names if ctx.closed(n) and _inside_outline(ctx.outline(n), p)]
        if len(inside) == 1:
            return set(inside)
        cands = inside or ctx.names
        d = [_distance_to_segments(ctx.outline(n), p) for n in cands] if not inside else \
            [np.prod(np.ptp(ctx.points(n), axis=0)) for n in cands]         # nested: the smallest
        return {cands[int(np.argmin(d))]}

    def edges(self, ctx):
        d = ctx.distance(self.point[:ctx.dim])
        chain = ctx.chains()
        return set(ctx.codes[chain == chain[int(np.argmin(d))]].tolist())

    def __repr__(self):
        return f"Near({tuple(float(v) for v in self.point)})"


class Interface(Selector):
    """The edges shared by a domain of `a` and a domain of `b` (each a label, a selector or a region)."""

    def __init__(self, a, b):
        self.a, self.b = as_selector(a), as_selector(b)

    def edges(self, ctx):
        A, B = ctx.domain_mask(self.a), ctx.domain_mask(self.b)
        g1, g2 = ctx.sides[:, 0], ctx.sides[:, 1]
        keep = (A[g1] & B[g2]) | (B[g1] & A[g2])
        return set(ctx.codes[keep].tolist())

    def __repr__(self):
        return f"Interface({self.a!r}, {self.b!r})"


class BoundaryOf(Selector):
    """The edges on the boundary of the domains of `a` (a label, a selector or a region): one side in `a`,
    the other outside of it (another domain or nothing)."""

    def __init__(self, a):
        self.a = as_selector(a)

    def edges(self, ctx):
        A = ctx.domain_mask(self.a)
        return set(ctx.codes[A[ctx.sides[:, 0]] ^ A[ctx.sides[:, 1]]].tolist())

    def __repr__(self):
        return f"BoundaryOf({self.a!r})"


class _Item(Selector):
    """A region or boundary of the model, standing for what it covers."""

    def __init__(self, item):
        self.item = item

    def domains(self, ctx):
        label = self.item.name
        if label in ctx.regions:
            return set(ctx.regions[label])
        return self.item.selector.domains(ctx)

    def edges(self, ctx):
        return self.item.selector.edges(ctx)

    def labels(self):
        return self.item.selector.labels()

    @property
    def by_groups(self):
        return self.item.selector.by_groups

    def __repr__(self):
        return f"{type(self.item).__name__}({self.item.name!r})"


class _Binary(Selector):
    op = ""

    def __init__(self, a: Selector, b: Selector):
        self.a, self.b = a, b

    def domains(self, ctx):
        return self._combine(self.a.domains(ctx), self.b.domains(ctx))

    def edges(self, ctx):
        return self._combine(self.a.edges(ctx), self.b.edges(ctx))

    def labels(self):
        return list(dict.fromkeys(self.a.labels() + self.b.labels()))

    def __repr__(self):
        return f"({self.a!r} {self.op} {self.b!r})"


class _Union(_Binary):
    op = "|"

    @staticmethod
    def _combine(x, y):
        return x | y

    @property
    def by_groups(self):
        return self.a.by_groups and self.b.by_groups


class _Intersection(_Binary):
    op = "&"

    @staticmethod
    def _combine(x, y):
        return x & y


class _Difference(_Binary):
    op = "-"

    @staticmethod
    def _combine(x, y):
        return x - y

    def labels(self):
        return self.a.labels()            # the groups of b are only removed: no need to keep them


# ════════════════════════════════════════════════════════════════════════════
#  What selectors are evaluated on
# ════════════════════════════════════════════════════════════════════════════
class Domains:
    """
    Domain groups with their outlines (segments (k, 2, dim)): what a region selector is evaluated on.

    names   : the domain groups, in order
    regions : {label of a model region: its groups} (a region given as a selector stands for these)
    where   : description for messages ("HPNEW-Sketch.geo", "the mesh")
    The outlines are only computed when a geometric selector (lp.Box, lp.Near) needs them.
    """

    def __init__(self, names, outlines: dict | Callable[[], dict] | None = None, *, dim: int = 2,
                 regions: dict | None = None, where: str = "the model"):
        self.names = list(names)
        self._outlines = outlines
        self.dim = dim
        self.regions = dict(regions or {})
        self.where = where

    @classmethod
    def from_geometry(cls, index, names, regions=None) -> "Domains":
        """The domain groups `names` of a 2D geometry (its sources.GeometryIndex): outlines sampled on their
        boundary curves."""
        return cls(names, lambda: {n: index.segments(index.outline[n]) for n in names}, dim=2, regions=regions,
                   where=index.name)

    @classmethod
    def from_mesh(cls, mesh, regions=None) -> "Domains":
        """The 2D groups of a Mesh (outline: the boundary edges of each group)."""
        names = mesh.names(2)

        def outlines():
            out = {}
            for n in names:
                rows = _tri_edges(mesh.groups[n].cells)[0]
                codes = _codes(rows)
                u, inv, cnt = np.unique(codes, return_inverse=True, return_counts=True)
                b = rows[cnt[inv] == 1]
                X = mesh.coords_of(b)
                out[n] = np.concatenate([X[:, [0, 2]], X[:, [2, 1]]]) if X.shape[1] == 3 else X[:, :2]
            return out
        return cls(names, outlines, dim=mesh.dim, regions=regions, where="the mesh")

    def outline(self, name) -> np.ndarray:
        """Sampled outline of a domain group (points of its closed boundary)."""
        if callable(self._outlines) or self._outlines is None:
            self._outlines = self._outlines() if callable(self._outlines) else {}
        if name not in self._outlines:
            raise SelectionError(f"no outline for the domain group {name!r} of {self.where}")
        return self._outlines[name]

    def points(self, name) -> np.ndarray:
        """Outline points of a domain group as an (n, dim) array."""
        return self.outline(name).reshape(-1, self.dim)

    def closed(self, name) -> bool:
        """Does the outline close (every segment end shared by an even number of segments)? Only a closed
        outline can contain a point (an open one: e.g. the base curve of a PML layer)."""
        ends = self.outline(name).reshape(-1, self.dim)
        if not len(ends):
            return False
        key = np.round(ends / (1e-9 * max(self.scale, 1e-12))).astype(np.int64)
        _, cnt = np.unique(key, axis=0, return_counts=True)
        return bool((cnt % 2 == 0).all())

    @property
    def scale(self) -> float:
        """Diagonal of the bounding box of all outlines (the length scale of the tolerances)."""
        if not hasattr(self, "_scale"):
            pts = np.concatenate([self.points(n) for n in self.names]) if self.names else np.zeros((1, self.dim))
            self._scale = float(np.linalg.norm(np.ptp(pts, axis=0)))
        return self._scale


class Edges:
    """
    The boundary edges of a 2D mesh: what a boundary selector is evaluated on.

    An edge is identified by a code of its two corner nodes; rows (k, 3) are [corner, corner, mid] node
    tags (the order of a gmsh LINE3), sides (k, 2) the indices (into domain_names) of the domain groups on
    its two sides (−1: none, outside of the meshed domains). The edges are those between two different
    domain groups, on the outside of the meshed domains, and those of the named curve groups (curve_groups:
    {name: set of codes}).
    """

    def __init__(self, mesh, regions: dict | None = None):
        self.mesh = mesh
        self.domain_names = mesh.names(2)
        self.where = "the mesh"
        row_list, side_list = [], []
        for gi, n in enumerate(self.domain_names):
            r, _ = _tri_edges(mesh.groups[n].cells)
            row_list.append(r)
            side_list.append(np.full(len(r), gi))
        rows = np.vstack(row_list) if row_list else np.zeros((0, 3), np.int64)
        gidx = np.concatenate(side_list) if side_list else np.zeros(0, np.int64)
        codes = _codes(rows)
        order = np.argsort(codes, kind="stable")
        codes, rows, gidx = codes[order], rows[order], gidx[order]
        u, first, cnt = np.unique(codes, return_index=True, return_counts=True)
        if (cnt > 2).any():
            raise SelectionError(f"{int((cnt > 2).sum())} mesh edges have more than two triangles")
        g1 = gidx[first]
        g2 = np.where(cnt == 2, gidx[np.minimum(first + 1, len(gidx) - 1)], -1)
        boundary = (cnt == 1) | (g1 != g2)

        self.curve_groups = {}
        known = set(u.tolist())
        extra = {}                                      # curve edges on no meshed triangle
        for n in mesh.names(1):
            c = np.asarray(mesh.groups[n].cells, np.int64)
            cc = _codes(c)
            self.curve_groups[n] = set(cc.tolist())
            extra.update({k: row for k, row in zip(cc.tolist(), c) if k not in known})
        on_curves = np.isin(u, np.fromiter(set().union(*self.curve_groups.values()), np.int64)) \
            if self.curve_groups else np.zeros(len(u), bool)
        keep = boundary | on_curves
        self.codes = u[keep]
        self.rows = rows[first[keep]]
        self.sides = np.stack([g1[keep], g2[keep]], axis=1)
        if extra:
            self.codes = np.concatenate([self.codes, np.fromiter(extra, np.int64, len(extra))])
            self.rows = np.vstack([self.rows, np.array(list(extra.values()), np.int64)[:, :self.rows.shape[1]]])
            self.sides = np.vstack([self.sides, np.full((len(extra), 2), -1)])
        self.domains = Domains.from_mesh(mesh, regions)
        self._chains: np.ndarray | None = None

    @property
    def regions(self):
        """The regions of the model (name → domain groups) available to selectors."""
        return self.domains.regions

    @property
    def dim(self) -> int:
        """Spatial dimension of the mesh."""
        return self.mesh.dim

    @property
    def scale(self) -> float:
        """Diagonal of the bounding box of the mesh."""
        return float(np.linalg.norm(np.ptp(self.mesh.coords, axis=0)))

    def coords_of(self, rows) -> np.ndarray:
        """Coordinates of node tags."""
        return self.mesh.coords_of(rows)

    def points(self) -> np.ndarray:
        """(k, nodes per edge, dim) the nodes of every edge."""
        return self.coords_of(self.rows)

    def distance(self, p) -> np.ndarray:
        """(k,) distance from p to every edge (through its mid node)."""
        X = self.points()
        segs = np.concatenate([X[:, [0, 2]], X[:, [2, 1]]]) if X.shape[1] == 3 else X[:, :2]
        return _distance_to_segments(segs, p, reduce=False).reshape(-1, len(X)).min(axis=0)

    def domain_mask(self, selector: Selector) -> np.ndarray:
        """(n_domains + 1,) bool: the domain groups selected by `selector` (on this mesh); the last entry,
        False, is indexed by the side −1 (no domain)."""
        names = selector.domains(self.domains)
        return np.array([n in names for n in self.domain_names] + [False], bool)

    def rows_of(self, codes) -> np.ndarray:
        """Rows [corner, corner, mid] of a set of edge codes, sorted by code."""
        sel = np.isin(self.codes, np.fromiter(codes, np.int64, len(codes)))
        return self.rows[sel]

    def chains(self) -> np.ndarray:
        """(k,) chain index of every edge: the boundary curves of lp.Near (module docstring)."""
        if self._chains is not None:
            return self._chains
        k = len(self.codes)
        a, b = self.rows[:, 0], self.rows[:, 1]
        X = self.coords_of(self.rows[:, :2])
        label = np.sort(self.sides, axis=1)
        parent = np.arange(k)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        ends = np.concatenate([a, b])
        idx = np.concatenate([np.arange(k), np.arange(k)])
        end = np.concatenate([np.zeros(k, int), np.ones(k, int)])
        order = np.argsort(ends, kind="stable")
        ends, idx, end = ends[order], idx[order], end[order]
        u, first, cnt = np.unique(ends, return_index=True, return_counts=True)
        for f in first[cnt == 2]:
            i, j = idx[f], idx[f + 1]
            if (label[i] != label[j]).any():
                continue
            vi = X[i, 1 - end[f]] - X[i, end[f]]                   # from the node along each edge
            vj = X[j, 1 - end[f + 1]] - X[j, end[f + 1]]
            cos = vi @ vj / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-300)
            if cos < -_CORNER_COS:                                 # nearly opposite: the curve goes on
                parent[find(i)] = find(j)
        self._chains = np.array([find(i) for i in range(k)])
        return self._chains


class Curves:
    """
    The curves of a 2D geometry (its sources.GeometryIndex): what the base of a PML layer is selected on
    (before meshing), with the rules of Edges. codes: curve tags; sides (k, 2): indices into domain_names of
    the meshed domains on each side (−1: none); curve_groups: the named curve groups that are not meshed
    loops {name: set of tags}. Every curve is its own "boundary curve" for lp.Near.
    """

    def __init__(self, index, meshed, regions: dict | None = None):
        self.index = index
        self.domain_names = list(meshed)
        self.where = index.name
        self.dim = 2
        self.curve_groups = {n: set(t) for n, t in index.groups.items() if n not in set(self.domain_names)}
        position = {n: gi for gi, n in enumerate(self.domain_names)}
        sides: dict = {}
        for t, owners in index.sides.items():
            for n in owners:
                if n in position:
                    sides.setdefault(t, []).append(position[n])
        named = set().union(*self.curve_groups.values()) if self.curve_groups else set()
        tags = sorted(set(sides) | named)              # curves of the meshed loops and of the named groups
        sides = {t: sides.get(t, []) for t in tags}
        self.codes = np.array(tags, np.int64)
        bad = [t for t, s in sides.items() if len(s) > 2]
        if bad:
            raise SelectionError(f"curves {bad} of {index.name} bound more than two curve loops")
        self.sides = np.array([(s + [-1, -1])[:2] for s in (sides[t] for t in tags)], np.int64).reshape(-1, 2)
        self.domains = Domains.from_geometry(index, self.domain_names, regions)

    @property
    def regions(self):
        """The regions of the model (name → domain groups) available to selectors."""
        return self.domains.regions

    @property
    def scale(self) -> float:
        """Length scale of the tolerances: the domains' scale, or the curves' bounding box."""
        return self.domains.scale if self.domain_names else \
            float(np.linalg.norm(np.ptp(np.concatenate(list(self.index.curves.values())), axis=0)))

    def points(self) -> np.ndarray:
        """Sampled points of every curve (k, n, dim)."""
        return np.stack([self.index.curves[int(t)] for t in self.codes])

    def distance(self, p) -> np.ndarray:
        """Distance of a point to every curve (k,)."""
        return np.array([_distance_to_segments(self.index.segments([int(t)]), p) for t in self.codes])

    def chains(self) -> np.ndarray:
        """Chain index of each curve (each curve is its own chain)."""
        return np.arange(len(self.codes))

    def domain_mask(self, selector: Selector) -> np.ndarray:
        """Mask over the domain groups (plus one entry for 'outside') of those the selector selects."""
        names = selector.domains(self.domains)
        return np.array([n in names for n in self.domain_names] + [False], bool)


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════


def _tri_edges(cells) -> tuple[np.ndarray, np.ndarray]:
    """Edge rows of 2D cells, ordered as line cells of the same order ([a, b] TRI3 / QUAD4, [a, b, mid]
    TRI6 / QUAD9, [a, b, m1, m2] TRI10), and the cell of each edge."""
    from loudpy.fem.reference import element_for_cells
    cells = np.asarray(cells, np.int64)
    loc = np.array(element_for_cells(2, cells.shape[1]).facet_nodes)
    rows = cells[:, loc].reshape(-1, loc.shape[1])
    return rows, np.repeat(np.arange(len(cells)), len(loc))


def _codes(rows) -> np.ndarray:
    """One int64 per edge from its two corner tags (order-independent)."""
    rows = np.asarray(rows, np.int64)
    lo, hi = np.minimum(rows[:, 0], rows[:, 1]), np.maximum(rows[:, 0], rows[:, 1])
    return lo * (1 << 31) + hi


def _distance_to_segments(segs, p, reduce=True):
    a, b = segs[:, 0], segs[:, 1]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-300), 0.0, 1.0)
    d = np.linalg.norm(a + t[:, None] * ab - p, axis=1)
    return d.min() if reduce else d


def _inside_outline(segs, p) -> bool:
    """Point in the region bounded by the segments (even–odd rule; the segments need no order)."""
    if not len(segs):
        return False
    (x1, y1), (x2, y2) = segs[:, 0].T, segs[:, 1].T
    cross = (y1 > p[1]) != (y2 > p[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = x1 + (p[1] - y1) * (x2 - x1) / (y2 - y1)
    return bool((cross & (p[0] < xc)).sum() % 2)
