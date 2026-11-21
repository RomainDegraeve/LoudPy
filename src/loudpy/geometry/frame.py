"""
Frame: placement and orientation of an axisymmetric part (a driver) in a 3D model.

The meridian (r, z) of the part lives in the local frame x_l = (x − origin) @ rotation, whose z_l
axis is the part axis (the +z of the meridian geometry) and whose x_l axis is θ = 0.
"""
from __future__ import annotations

import numpy as np

from loudpy.geometry.gmsh_session import gmsh_module


class Frame:
    """
    Local frame of a driver: x_l = (x − origin) @ rotation. The columns of `rotation` are the
    local axes in global coordinates. z_l is the driver axis (the +z of the meridian geometry,
    towards the front), x_l is θ = 0, and `origin` is the point of the axis where the meridian
    z = 0. A 2D center (x0, y0) is the frame (x0, y0, 0) with the identity rotation, which is
    the convention of the single-driver code (axis along global z, absolute meridian z).
    """

    def __init__(self, origin=(0.0, 0.0, 0.0), rotation=None):
        o = np.zeros(3)
        o[:len(origin)] = np.asarray(origin, float)
        self.origin = o
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, float)
        if not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-9):
            raise ValueError("rotation must be orthonormal")
        if np.linalg.det(self.rotation) < 0:
            raise ValueError("rotation must be right-handed (det = +1)")

    @classmethod
    def from_axis(cls, origin, axis, ref=None) -> "Frame":
        """Frame with z_l along `axis` and x_l (θ = 0) towards `ref` (projected; default: global x, or y)."""
        z = np.asarray(axis, float)
        z = z / np.linalg.norm(z)
        if ref is None:
            ref = (1.0, 0.0, 0.0) if abs(z[0]) < 0.9 else (0.0, 1.0, 0.0)
        x = np.asarray(ref, float) - np.dot(ref, z) * z
        if np.linalg.norm(x) < 1e-9:
            raise ValueError("ref is parallel to the axis")
        x /= np.linalg.norm(x)
        return cls(origin, np.column_stack([x, np.cross(z, x), z]))

    @property
    def axis(self):
        return self.rotation[:, 2]

    @property
    def is_identity(self):
        return np.array_equal(self.rotation, np.eye(3))

    @property
    def axis_aligned(self):
        a = np.abs(self.rotation)
        return bool(np.all(np.isclose(a, 0.0, atol=1e-12) | np.isclose(a, 1.0, atol=1e-12)))

    def to_local(self, x):
        return (np.asarray(x, float) - self.origin) @ self.rotation

    def to_global(self, xl):
        return np.asarray(xl, float) @ self.rotation.T + self.origin

    def place(self, dim_tags):
        """
        Move gmsh OCC entities built in the local frame (axis z through the origin) to their
        global place, by an exact rotation + translation (affineTransform would approximate
        the shapes by B-splines).
        """
        from scipy.spatial.transform import Rotation
        occ = gmsh_module().model.occ
        rv = Rotation.from_matrix(self.rotation).as_rotvec()
        angle = float(np.linalg.norm(rv))
        if angle > 1e-14:
            occ.rotate(dim_tags, 0.0, 0.0, 0.0, *(rv / angle), angle)
        occ.translate(dim_tags, *self.origin)

    def __repr__(self):
        return f"Frame(origin={self.origin.tolist()}, axis={self.axis.round(6).tolist()})"


def as_frame(center):
    """Frame from a frame, a 2D center (x0, y0) or a 3D origin (axis along global z)."""
    return center if isinstance(center, Frame) else Frame(center)


def is_single(center):
    """True for one center/frame, False for a sequence of them."""
    return isinstance(center, Frame) or np.ndim(center) == 1 and np.isscalar(center[0])
