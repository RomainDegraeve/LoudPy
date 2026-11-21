"""
Where to evaluate a result (like COMSOL's cut points and cut lines): every `at=` argument takes
a point, several points or one of these selections.

    (0.0, 0.3)                          one point (r, z) in 2D, (x, y, z) in 3D
    lp.Points([(0, .3), (.1, .3)])      several points: a "point" dimension
    lp.Polar(0.3, 30)                   at 0.3 m from the origin, 30° from the axis (2D: r = R sin θ, z = R cos θ)
    lp.Arc(0.3, range(0, 91, 2))        the same for several angles: an "angle" dimension [°]
    lp.Line((0, 0.05), (0, 0.5), n=100) points along a segment: an "s" dimension (distance [m])

3D: Polar / Arc take center=(x, y, z), the axis (direction of 0°) and the plane of the angles:
    lp.Arc(1.0, range(-180, 181, 5), center=(0.1, 0.3, 0.03), axis="z", toward="x")
"""
from __future__ import annotations

import numpy as np

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


class Selection:
    """Points with an optional dimension: `points` (n, d), `dim` (None for one point), `coord` (n,)."""
    dim: str | None = None

    def points(self, d: int) -> np.ndarray:
        """The evaluation points in d dimensions (n, d)."""
        raise NotImplementedError

    @property
    def coord(self) -> np.ndarray | None:
        """Values of the coordinate along the selection (angles, distances, labels), or None for one point."""
        return None

    def describe(self) -> str:
        """Short text of the selection (for titles and metadata)."""
        return repr(self)


class Point(Selection):
    """One point: (r, z) in 2D, (x, y, z) in 3D."""

    def __init__(self, *xyz):
        self.xyz = np.asarray(xyz[0] if len(xyz) == 1 else xyz, float).ravel()

    def points(self, d):
        return self.xyz[None, :d] if len(self.xyz) >= d else np.r_[self.xyz, np.zeros(d - len(self.xyz))][None]

    def describe(self):
        return "(" + ", ".join(f"{v:g}" for v in self.xyz) + ") m"

    def __repr__(self):
        return f"Point{tuple(float(v) for v in self.xyz)}"


class Points(Selection):
    """Several points; labels name them along the "point" dimension (default: their coordinates)."""
    dim = "point"

    def __init__(self, points, labels=None):
        self.xyz = np.atleast_2d(np.asarray(points, float))
        self.labels = np.asarray(labels if labels is not None else
                                 ["(" + ", ".join(f"{v:g}" for v in p) + ")" for p in self.xyz])

    def points(self, d):
        return self.xyz[:, :d]

    @property
    def coord(self):
        return self.labels

    def describe(self):
        return f"{len(self.xyz)} points"

    def __repr__(self):
        return f"Points({len(self.xyz)})"


class Arc(Selection):
    """
    Points at `radius` from `center`, at `angles` [°] from the axis.
    2D (axisymmetric, r–z): the axis is z, θ toward +r:  (r, z) = center + R (sin θ, cos θ).
    3D: `axis` (0°) and `toward` (+90°) set the plane: "x", "y", "z" or vectors.
    """
    dim = "angle"

    def __init__(self, radius, angles, center=None, axis="z", toward="x"):
        self.radius = float(radius)
        self.angles = np.atleast_1d(np.asarray(angles, float))
        self.center = None if center is None else np.asarray(center, float)
        self.axis, self.toward = axis, toward

    def points(self, d):
        th = np.deg2rad(self.angles)
        if d == 2:
            c = np.zeros(2) if self.center is None else self.center[:2]
            return c + self.radius * np.column_stack([np.sin(th), np.cos(th)])
        e0 = np.asarray(_AXES.get(self.axis, self.axis), float)
        e1 = np.asarray(_AXES.get(self.toward, self.toward), float)
        e0, e1 = e0 / np.linalg.norm(e0), e1 / np.linalg.norm(e1)
        c = np.zeros(3) if self.center is None else self.center
        return c + self.radius * (np.cos(th)[:, None] * e0 + np.sin(th)[:, None] * e1)

    @property
    def coord(self):
        return self.angles

    def describe(self):
        return f"R = {self.radius:g} m"

    def __repr__(self):
        return f"Arc(R={self.radius:g}, {len(self.angles)} angles {self.angles.min():g}–{self.angles.max():g}°)"


class Polar(Arc):
    """One point (one angle) or an arc (several angles) at `radius` from the centre; see Arc."""

    def __init__(self, radius, angle, center=None, axis="z", toward="x"):
        super().__init__(radius, angle, center, axis, toward)
        self.single = np.ndim(angle) == 0
        if self.single:
            self.dim = None

    @property
    def coord(self):
        return None if self.single else self.angles

    def describe(self):
        return f"R = {self.radius:g} m, θ = {self.angles[0]:g}°" if self.single else super().describe()


class Line(Selection):
    """n points from `start` to `end`; dimension "s", the distance from `start` [m]."""
    dim = "s"

    def __init__(self, start, end, n: int = 100):
        self.start, self.end, self.n = np.asarray(start, float), np.asarray(end, float), int(n)

    def points(self, d):
        a, b = self.start[:d], self.end[:d]
        return a + np.linspace(0.0, 1.0, self.n)[:, None] * (b - a)

    @property
    def coord(self):
        return np.linspace(0.0, np.linalg.norm(self.end - self.start), self.n)

    def describe(self):
        return f"line {tuple(self.start)} → {tuple(self.end)}"

    def __repr__(self):
        return f"Line({tuple(self.start)} → {tuple(self.end)}, n={self.n})"


def as_selection(at) -> Selection:
    """A Selection from a point, an (n, d) array of points, or a Selection."""
    if isinstance(at, Selection):
        return at
    a = np.asarray(at, float)
    if a.ndim == 1:
        return Point(a)
    if a.ndim == 2:
        return Points(a)
    raise ValueError(f"cannot evaluate at {at!r}: give a point, an (n, d) array or an lp.Arc / lp.Line / …")
