"""
Directivity of axisymmetric (r, z) results: the baffle the loudspeaker is mounted in, read from the mesh, and the SPL
on an arc centred on it, mirrored to −θ … θ (the field does not depend on the azimuth, so θ and −θ are the same
direction seen from the two sides of the axis).

    z0 = res.baffle()                     z [m] of the baffle's front face; None in free field
    R, angles = arc(res, center)          the default arc: 0.8 × the air in front of the centre (and behind it
                                          when there is as much air there: 0 … 180°, else 0 … 90°), every 2°
    d = res.directivity(radius=None, angles=None, center=None, mirror=True)
                                          SPL [dB] over (f, angle), centred on (0, baffle) by default

The baffle is found on the pressure mesh: the front air (the air domain reaching highest, PML left out) lies on a
plane z = z0 out to its outer radius, the structure reaches that plane or goes below it, and the only air below the
plane is close to the axis (in front of a recessed cone). Air wrapping around the structure is free field.
"""
from __future__ import annotations

import numpy as np

from loudpy.errors import ResultsError


def _nodes(res, physics: str) -> np.ndarray | None:
    try:
        return np.asarray(res.blocks[0].mesh_of(physics).coords, float)[:, :2]
    except Exception:  # noqa: BLE001  (no such physics in these results)
        return None


def _air(res) -> list[np.ndarray]:
    """The (r, z) nodes of each air domain (PML left out) of the pressure mesh."""
    mesh = res.blocks[0].mesh_of("p")
    groups = res.model.groups if getattr(res, "model", None) is not None else {}
    X = np.asarray(mesh.coords, float)[:, :2]
    out = [X[mesh.rows_of(np.unique(np.asarray(g.cells).ravel()))] for n, g in mesh.groups.items()
           if g.dim == 2 and (groups.get(n) or {}).get("role") != "pml" and not n.lower().startswith("pml")]
    return out or [X]


def _check(res):
    if getattr(res, "kind", "") == "scene" or type(res).__name__ == "SceneResults":
        raise ResultsError("3D scenes: res.spl(at=lp.Arc(radius, angles, center, axis, toward)) — no axis of symmetry")
    if _nodes(res, "p") is None:
        raise ResultsError("no pressure field in these results: the directivity needs the air")


def baffle(res) -> float | None:
    """z [m] of the baffle's front face of a 2D (r, z) result, read from its mesh; None in free field."""
    _check(res)
    front = max(_air(res), key=lambda Y: Y[:, 1].max())
    r_out = float(front[:, 0].max())
    tol = 1e-6 * max(r_out, float(np.abs(front).max()), 1e-3)
    z0 = float(front[np.abs(front[:, 0] - r_out) <= tol, 1].min())
    below = front[front[:, 1] < z0 - tol]
    if len(below) and below[:, 0].max() > 0.9 * r_out:               # the air wraps around the structure
        return None
    solid = _nodes(res, "u")
    if solid is not None and solid[:, 1].min() > z0 + tol:          # the structure stands above that plane
        return None
    return z0


def arc(res, center=None) -> tuple[float, np.ndarray]:
    """The default directivity arc around `center` (default (0, baffle)): radius 0.8 × the air in front (and behind,
    when there is as much: angles 0 … 180°, else 0 … 90°), rounded to 2 digits; angles every 2°."""
    _check(res)
    z0 = float(center[1]) if center is not None else (baffle(res) or 0.0)
    X = np.concatenate(_air(res))
    tol = 1e-6 * max(1.0, float(np.abs(X).max()))
    up, down = X[X[:, 1] >= z0 - tol], X[X[:, 1] < z0 - tol]
    front = min(up[:, 0].max(), up[:, 1].max() - z0) if len(up) else 0.0
    rear = min(down[:, 0].max(), z0 - down[:, 1].min()) if len(down) else 0.0
    if front <= 0:
        raise ResultsError(f"no air in front of z = {z0:g} m: give the radius and the centre")
    both = rear >= 0.5 * front
    R = 0.8 * (min(front, rear) if both else front)
    return float(f"{R:.2g}"), np.arange(0.0, (180.0 if both else 90.0) + 1e-9, 2.0)


def directivity(res, radius: float | None = None, angles=None, center=None, mirror: bool = True):
    """SPL [dB] over (f, angle) on an arc of `radius` [m] around `center` (default: (0, baffle), the baffle's front
    face found on the mesh, else the origin), angles [°] from the z axis (default: see arc); mirror: −θ … θ."""
    from loudpy.results.selection import Arc
    _check(res)
    z_b = baffle(res) if center is None else None
    center = (0.0, z_b or 0.0) if center is None else tuple(float(c) for c in center)
    if radius is None or angles is None:
        R, a = arc(res, center)
        radius = R if radius is None else radius
        angles = a if angles is None else angles
    d = res.spl(at=Arc(float(radius), np.asarray(angles, float), center))
    d.meta.update(center=f"(r, z) = (0, {center[1]:g}) m", radius=f"{float(radius):g} m",
                  **({"baffle": f"z = {z_b:g} m"} if z_b is not None else {}))
    return d.mirror("angle") if mirror else d
