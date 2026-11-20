"""
Driver: an axisymmetric loudspeaker described by its modes, ready to be placed in a 3D scene.

    hp = lp.Driver.from_modes("eigen_fourier.h5", orders=[0, 1, 2, 3], f_max=10000,
                              geometry="HPNEW-Sketch.geo",
                              rear=lp.Profile([14, 7, 8, 33, 32], start=(0.05435, 0.031)),
                              front=lp.Profile([5, 4, 3, 2, 1, 113, 22, 21, 20], start=(0.05735, 0.032)))
    top = hp.place("top", at=(0.07, 0.58))                  # axis +z through (x, y)
    side = hp.place("side", at=(0.0, 0.2, 0.1), axis=(-1, 0, 0))

The modes come from an eigen result file (lp.Modes with orders=…, format 2 or 1). The coil is the
boundary named `coil` in that file (where the force acts). The wet profiles are polylines in the
meridian plane, sampled from the driver's 2D geometry: they shape the cut-outs of the driver in
the 3D air and select the wet edges of the meridian mesh.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loudpy.geometry.frame import Frame, as_frame
from loudpy.errors import ModelError


@dataclass(frozen=True)
class Profile:
    """Wet profile: chained curves of the 2D geometry, from `start` (on the baffle) to the axis."""
    curves: tuple
    start: tuple

    def __init__(self, curves, start):
        object.__setattr__(self, "curves", tuple(int(c) for c in curves))
        object.__setattr__(self, "start", tuple(float(v) for v in start))

    def sample(self, geometry, n_per_curve: int = 40) -> list[np.ndarray]:
        from loudpy.geometry.meshing3d import sample_profile
        return sample_profile(str(geometry), list(self.curves), self.start, n_per_curve)


class Driver:
    """An axisymmetric driver described by its modes (lp.Driver.from_modes), placed in 3D scenes with .place()."""
    def __init__(self, modes, rear, front, coil_nodes, name: str = "driver", source=None):
        """
        modes      : DriverModes (meridian mesh, real mass-normalised modes, both families)
        rear, front: wet profiles (lists of (n, 2) polylines, baffle → axis), front may be None
        coil_nodes : rows of the meridian mesh where the coil force acts
        """
        self.modes, self.rear, self.front = modes, rear, front
        self.coil_nodes = np.asarray(coil_nodes, dtype=np.int64)
        self.name, self.source = name, source

    @classmethod
    def from_modes(cls, path, orders, f_max=None, *, geometry, rear: Profile, front: Profile | None = None,
                   coil: str = "interface_forced", name: str | None = None) -> "Driver":
        from loudpy.physics.modal import DriverModes
        from loudpy.results import load
        modes = DriverModes.from_eigen_h5(path, list(orders), f_max=f_max)
        tags = load(path).boundary(coil)
        tagset = {int(t) for t in tags}
        coil_nodes = np.array([i for i, t in enumerate(modes.node_tags) if int(t) in tagset], dtype=np.int64)
        if not len(coil_nodes):
            raise ModelError(f"no node of the boundary {coil!r} on the modal mesh of {path}")
        return cls(modes, rear.sample(geometry), front.sample(geometry) if front else None, coil_nodes,
                   name or "driver", source=dict(path=str(path), orders=list(orders), f_max=f_max,
                                                 geometry=str(geometry)))

    @property
    def rear_cutout(self) -> list[np.ndarray]:
        """Rear profile closed to the axis at the cone foot (rigid lid): the shape cut out of the air."""
        last = self.rear[-1][-1]
        return self.rear + [np.array([last, [0.0, last[1]]])] if abs(last[0]) > 1e-12 else self.rear

    @property
    def r_rear(self) -> float:
        return float(self.rear[0][0][0])

    @property
    def r_front(self) -> float:
        return float(self.front[0][0][0]) if self.front else self.r_rear

    @property
    def z_baffle(self) -> float:
        """Meridian z of the rear baffle plane (first point of the rear profile)."""
        return float(self.rear[0][0][1])

    @property
    def z_front(self) -> float:
        return float(self.front[0][0][1]) if self.front else self.z_baffle

    def place(self, name: str, at=(0.0, 0.0), axis=None, ref=None) -> "PlacedDriver":
        """This driver in a scene: axis along `axis` (default +z) through `at` (origin of meridian z)."""
        frame = as_frame(at) if axis is None else Frame.from_axis(np.r_[np.asarray(at, float), [0.0] * (3 - len(at))],
                                                                  axis, ref)
        return PlacedDriver(name, self, frame)

    def __repr__(self):
        return f"Driver({self.name!r}, {len(self.modes)} modes, coil {len(self.coil_nodes)} nodes)"


@dataclass
class PlacedDriver:
    name: str
    driver: Driver
    frame: Frame

    def outline_points(self, n_theta: int = 16) -> np.ndarray:
        """Points of the revolved front (or rear) profile in global coordinates (hemisphere sizing)."""
        prof = np.vstack(self.driver.front or self.driver.rear)
        th = np.linspace(0, 2 * np.pi, n_theta)
        local = np.array([(r * np.cos(a), r * np.sin(a), z) for r, z in prof for a in th])
        return self.frame.to_global(local)
