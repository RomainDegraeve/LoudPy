"""
Driver: an axisymmetric loudspeaker described by its modes, ready to be placed in a 3D scene.

    solids = lp.Group("cone", "dustcap", "surround")
    hp = lp.Driver.from_modes("eigen_fourier.h5", orders=[0, 1, 2, 3], f_max=10000,
                              geometry="HPNEW-Sketch.geo",
                              rear=lp.Profile(lp.BoundaryOf("air_rear"), start=(0.05435, 0.031),
                                              end=(0.00633, 0.0052)),
                              front=lp.Profile(lp.Interface(solids, "air_front")))
    top = hp.place("top", at=(0.07, 0.58))                  # axis +z through (x, y)
    side = hp.place("side", at=(0.0, 0.2, 0.1), axis=(-1, 0, 0))

The modes come from an eigen result file (lp.Modes with orders=…, format 2 or 1). The coil is the
boundary named `coil` in that file (where the force acts). The wet profiles are polylines in the
meridian plane, sampled from the driver's 2D geometry (a path, a builder, or the lp.Model of the
driver: then its region names can be used): they shape the cut-outs of the driver in the 3D air and
select the wet edges of the meridian mesh.
"""
from __future__ import annotations

from typing import Any

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from loudpy.geometry.frame import Frame, as_frame
from loudpy.errors import ModelError


@dataclass(frozen=True, eq=False)
class Profile:
    """
    Wet profile of a driver: a chain of curves of its 2D geometry, from `start` (on the baffle) towards the
    axis, selected like a boundary (loudpy.geometry.selectors, evaluated on the curves of the geometry):

        lp.Profile(lp.Interface(solids, "air_front"))       the selected curves, which form one chain;
                                                               start: its end farthest from the axis
        lp.Profile(lp.BoundaryOf("air_rear"), start=(0.05435, 0.031), end=(0.00633, 0.0052))
                                                               the shortest way along the selected curves
                                                               from start to end
        lp.Profile([14, 7, 8, 33, 32], start=(0.05435, 0.031))  gmsh curve tags, chained from start

    start / end are snapped to the nearest end point of a selected curve (the given start stays the first
    point of the profile).
    """
    on: Any
    start: tuple | None
    end: tuple | None

    def __init__(self, on, start=None, end=None):
        from loudpy.geometry.selectors import Selector, as_selector
        if not isinstance(on, (Selector, str)) and np.ndim(on) == 1 and all(np.ndim(c) == 0 for c in on):
            on = tuple(int(c) for c in on)                       # curve tags
            if start is None:
                raise ModelError("lp.Profile with curve tags needs start=")
        else:
            on = as_selector(on)
        object.__setattr__(self, "on", on)
        object.__setattr__(self, "start", None if start is None else tuple(float(v) for v in start))
        object.__setattr__(self, "end", None if end is None else tuple(float(v) for v in end))

    @property
    def curves(self) -> tuple | None:
        """The gmsh curve tags given (former form), else None."""
        return self.on if isinstance(self.on, tuple) else None

    def resolve(self, geometry) -> tuple[list[int], tuple | None]:
        """(curve tags in chain order, start point) on `geometry` (a path, a builder or an lp.Model)."""
        if self.curves is not None:
            return list(self.curves), self.start
        from loudpy.geometry.selectors import Curves
        from loudpy.geometry.sources import GeometryIndex
        source, regions = _geometry(geometry)
        index = GeometryIndex(source)
        ctx = Curves(index, list(index.domains), regions)
        try:
            tags = self.on.edges(ctx)
        except Exception as e:
            raise ModelError(f"lp.Profile({self.on!r}): {e}") from None
        if not tags:
            raise ModelError(f"lp.Profile({self.on!r}) selects no curve of {index.name}")
        return _chain(index, sorted(tags), self.start, self.end, ctx.scale, repr(self.on))

    def sample(self, geometry, n_per_curve: int = 40) -> list[np.ndarray]:
        """The profile as chained (n, 2) polylines, one per curve (n_per_curve points each)."""
        from loudpy.geometry.meshing3d import sample_profile
        curves, start = self.resolve(geometry)
        if self.curves is not None and isinstance(geometry, (str, Path)):
            where = str(geometry)                   # tags given by hand: the curves of the file as it is
        else:
            where = _geometry(geometry)[0]          # tags from a selection: the geometry loaded for it
        return sample_profile(where, curves, start, n_per_curve)

    def __repr__(self):
        return f"Profile({self.on!r}, start={self.start}, end={self.end})"


def _geometry(geometry):
    """(GeometrySource, {region: domain groups}) of a path, a builder or an lp.Model."""
    from loudpy.geometry.sources import GeometrySource
    if hasattr(geometry, "source") and hasattr(geometry, "selection"):        # an lp.Model
        from loudpy.model.regions import PML
        sel = geometry.selection()
        return geometry.source, {r.name: sel[r.name] for r in geometry.regions if not isinstance(r, PML)}
    return (geometry if isinstance(geometry, GeometrySource) else GeometrySource(geometry)), {}


def _chain(index, tags, start, end, scale, what) -> tuple[list[int], tuple]:
    """Order curves into a chain from `start` (shortest way to `end` when given; else the curves must form one
    chain, started at its end farthest from the axis when start is None)."""
    import heapq
    tol = 1e-9 * max(scale, 1e-12)
    ends = {t: (index.curves[t][0], index.curves[t][-1]) for t in tags}
    points: list[np.ndarray] = []

    def vertex(p):
        for i, q in enumerate(points):
            if np.linalg.norm(q - p) <= tol:
                return i
        points.append(np.asarray(p, float))
        return len(points) - 1

    link = {t: (vertex(a), vertex(b)) for t, (a, b) in ends.items()}
    length = {t: float(np.linalg.norm(np.diff(index.curves[t], axis=0), axis=1).sum()) for t in tags}
    at: dict = {}
    for t, (a, b) in link.items():
        at.setdefault(a, []).append(t)
        at.setdefault(b, []).append(t)
    P = np.array(points)

    def snap(p, name):
        d = np.linalg.norm(P - np.asarray(p, float)[:2], axis=1)
        i = int(np.argmin(d))
        if d[i] > 1e-3 * max(scale, 1e-12):
            raise ModelError(f"lp.Profile({what}): {name} {tuple(p)} is not an end point of a selected curve "
                             f"(nearest {tuple(np.round(P[i], 6))}, {d[i]:.2e} m away)")
        return i

    if start is None:
        free = [v for v, ts in at.items() if len(ts) == 1]
        if len(free) != 2:
            raise ModelError(f"lp.Profile({what}): the selected curves do not form one open chain: give start= "
                             f"(and end=)")
        s = max(free, key=lambda v: abs(P[v][0]))
        start = tuple(float(x) for x in P[s])
    else:
        s = snap(start, "start")
    if end is not None:                                   # shortest way (Dijkstra) from s to e
        e = snap(end, "end")
        dist, prev, heap = {s: 0.0}, {}, [(0.0, s)]
        while heap:
            d, v = heapq.heappop(heap)
            if v == e:
                break
            if d > dist.get(v, np.inf):
                continue
            for t in at.get(v, []):
                w = link[t][1] if link[t][0] == v else link[t][0]
                if d + length[t] < dist.get(w, np.inf):
                    dist[w], prev[w] = d + length[t], (v, t)
                    heapq.heappush(heap, (d + length[t], w))
        if e not in dist:
            raise ModelError(f"lp.Profile({what}): no way from start to end along the selected curves")
        order, v = [], e
        while v != s:
            v, t = prev[v]
            order.append(t)
        return order[::-1], start
    order, used, v = [], set(), s                           # walk the chain
    while True:
        nxt = [t for t in at.get(v, []) if t not in used]
        if not nxt:
            break
        if len(nxt) > 1:
            raise ModelError(f"lp.Profile({what}): the selected curves branch at {tuple(np.round(P[v], 6))}: "
                             f"give end=")
        t = nxt[0]
        order.append(t)
        used.add(t)
        v = link[t][1] if link[t][0] == v else link[t][0]
    if len(used) != len(tags):
        raise ModelError(f"lp.Profile({what}): {len(tags) - len(used)} selected curve(s) are not on the chain "
                         f"from start")
    return order, start


class Driver:
    """An axisymmetric driver described by its modes (lp.Driver.from_modes), placed in 3D scenes with .place()."""
    def __init__(self, modes, rear, front, coil_nodes, name: str = "driver", source=None, motor=None):
        """
        modes      : DriverModes (meridian mesh, real mass-normalised modes, both families)
        rear, front: wet profiles (lists of (n, 2) polylines, baffle → axis), front may be None
        coil_nodes : rows of the meridian mesh where the coil force acts
        motor      : its electrical motor (loudpy.circuit: Semi, Leach, LR2, Wright), or None
        """
        from loudpy.circuit import motor_from_dict
        self.modes, self.rear, self.front = modes, rear, front
        self.coil_nodes = np.asarray(coil_nodes, dtype=np.int64)
        self.name, self.source = name, source
        self.motor = motor_from_dict(motor)

    @classmethod
    def from_modes(cls, path, orders, f_max=None, *, geometry, rear: Profile, front: Profile | None = None,
                   coil: str | None = None, name: str | None = None, motor=None) -> "Driver":
        """A driver from an eigen result file of its axisymmetric model: the modes of the azimuthal orders
        `orders` (up to f_max), the coil nodes (the boundary of the model's lp.Coil item, or the boundary named
        `coil`), its motor (the lp.Coil item's motor=, as the file describes it; `motor` overrides it) and the wet
        profiles `rear` / `front` selected on its 2D geometry (a path, a builder or its lp.Model)."""
        from loudpy.circuit import motor_from_dict
        from loudpy.physics.modal import DriverModes
        from loudpy.results import load
        modes = DriverModes.from_eigen_h5(path, list(orders), f_max=f_max)
        res = load(path)
        items = (res.model.get("boundaries") or []) if res.model is not None else []
        coil_items = [b for b in items if b.get("type") == "Coil"]
        if coil is None:
            coils = [b["name"] for b in coil_items]
            if len(coils) != 1:
                raise ModelError(f"{path}: {len(coils)} lp.Coil item(s) in its model description; name the coil "
                                 f"boundary with coil= (boundaries: {list(res.boundaries)})")
            coil = coils[0]
        tags = res.boundary(coil)
        tagset = {int(t) for t in tags}
        coil_nodes = np.array([i for i, t in enumerate(modes.node_tags) if int(t) in tagset], dtype=np.int64)
        if not len(coil_nodes):
            raise ModelError(f"no node of the boundary {coil!r} on the modal mesh of {path}")
        own = next((b.get("motor") for b in coil_items if b.get("name") == coil), None)
        return cls(modes, rear.sample(geometry), front.sample(geometry) if front else None, coil_nodes,
                   name or "driver", source=dict(path=str(path), orders=list(orders), f_max=f_max,
                                                 geometry=str(geometry), rear=rear, front=front, coil=coil,
                                                 motor=motor_from_dict(motor)),
                   motor=motor if motor is not None else motor_from_dict(own))

    @property
    def rear_cutout(self) -> list[np.ndarray]:
        """Rear profile closed to the axis at the cone foot (rigid lid): the shape cut out of the air."""
        last = self.rear[-1][-1]
        return self.rear + [np.array([last, [0.0, last[1]]])] if abs(last[0]) > 1e-12 else self.rear

    @property
    def r_rear(self) -> float:
        """Radius where the rear profile starts (the baffle) [m]."""
        return float(self.rear[0][0][0])

    @property
    def r_front(self) -> float:
        """Radius where the front profile starts [m]."""
        return float(self.front[0][0][0]) if self.front else self.r_rear

    @property
    def z_baffle(self) -> float:
        """Meridian z of the rear baffle plane (first point of the rear profile)."""
        return float(self.rear[0][0][1])

    @property
    def z_front(self) -> float:
        """Local z of the start of the front profile [m]."""
        return float(self.front[0][0][1]) if self.front else self.z_baffle

    def place(self, name: str, at=(0.0, 0.0), axis=None, ref=None) -> "PlacedDriver":
        """This driver in a scene: axis along `axis` (default +z) through `at` (origin of meridian z)."""
        frame = as_frame(at) if axis is None else Frame.from_axis(np.r_[np.asarray(at, float), [0.0] * (3 - len(at))],
                                                                  axis, ref)
        return PlacedDriver(name, self, frame)

    def __repr__(self):
        return (f"Driver({self.name!r}, {len(self.modes)} modes, coil {len(self.coil_nodes)} nodes"
                f"{', motor ' + type(self.motor).__name__ if self.motor is not None else ''})")


@dataclass
class PlacedDriver:
    """A driver placed in a scene: its name, the Driver and its Frame (Driver.place)."""
    name: str
    driver: Driver
    frame: Frame

    @property
    def motor(self):
        """Its electrical motor (the driver's), or None."""
        return self.driver.motor

    def outline_points(self, n_theta: int = 16) -> np.ndarray:
        """Points of the revolved front (or rear) profile in global coordinates (hemisphere sizing)."""
        prof = np.vstack(self.driver.front or self.driver.rear)
        th = np.linspace(0, 2 * np.pi, n_theta)
        local = np.array([(r * np.cos(a), r * np.sin(a), z) for r, z in prof for a in th])
        return self.frame.to_global(local)
