"""
Far field of a 2D axisymmetric pressure field (azimuthal order 0) by the Kirchhoff–Helmholtz integral.

The sources (the loudspeaker) are enclosed by a sphere S of radius R about a point of the axis: in the
meridian plane, an arc of the air mesh. Outside S the field radiated into free space is

    p(x) = ∮_S [ p(y) ∂G(x, y)/∂n_y − G(x, y) ∂p/∂n(y) ] dS_y,      G = e^{−jkR} / (4πR)   (e^{jωt})

with n pointing out of the sphere. The surface integral is a meridian integral (Gauss–Legendre in the
angle from the axis) of an azimuthal integral (trapezoidal rule, exponentially accurate for a periodic
integrand). With a rigid baffle (the plane z = z_b, the half-space z ≥ z_b): S is the half sphere above the
plane and G is the Neumann Green's function G(x, y) + G(x', y) (x' the image of x), whose normal derivative
vanishes on the plane, so the baffle contributes nothing.

p and ∂p/∂n on the arc come from the FE field (its own basis, loudpy.geometry.mesh.Mesh.interpolation):
the result needs only the result file.
"""
from __future__ import annotations

import numpy as np

from loudpy.errors import SelectionError


def sphere_arc(radius: float, center, baffle: float | None, n: int):
    """Meridian arc of the sphere: points (n, 2), outward normals (n, 2), arc-length weights (n,)."""
    r0, z0 = float(center[0]), float(center[1])
    if abs(r0) > 1e-12:
        raise SelectionError("the centre of the integration sphere must be on the axis (r = 0)")
    if baffle is None:
        th_max = np.pi
    else:
        cos_max = (baffle - z0) / radius
        if not -1.0 < cos_max < 1.0:
            raise SelectionError(f"the sphere (centre z = {z0:g}, radius {radius:g}) does not cross the baffle "
                                 f"z = {baffle:g}")
        th_max = float(np.arccos(cos_max))
    x, w = np.polynomial.legendre.leggauss(n)
    th = 0.5 * th_max * (x + 1.0)
    nrm = np.column_stack([np.sin(th), np.cos(th)])
    pts = np.array([r0, z0]) + radius * nrm
    return pts, nrm, 0.5 * th_max * w * radius


def kirchhoff_helmholtz(mesh, P, k, at, radius: float, center=(0.0, 0.0), baffle: float | None = None,
                        n_arc: int | None = None, n_phi: int | None = None) -> np.ndarray:
    """
    Pressure (n_f, n_points) at the meridian points `at` (r, z) from the nodal pressures P (n_f, n_nodes) on
    the 2D mesh, wavenumbers k (n_f,), over the sphere (radius, centre on the axis), optional baffle z = z_b.
    """
    at = np.atleast_2d(np.asarray(at, float))[:, :2]
    P = np.atleast_2d(P)
    k = np.atleast_1d(np.asarray(k, float))
    c = np.asarray(center, float)
    if np.any(np.linalg.norm(at - c, axis=1) <= radius * (1 + 1e-9)):
        raise SelectionError(f"far-field points must lie outside the integration sphere (radius {radius:g} m)")
    if baffle is not None and np.any(at[:, 1] < baffle - 1e-12):
        raise SelectionError(f"far-field points must lie on the side of the baffle (z ≥ {baffle:g})")
    th_max = np.pi if baffle is None else float(np.arccos(np.clip((baffle - c[1]) / radius, -1, 1)))
    k_max = float(k.max()) if len(k) else 0.0
    n_arc = n_arc or max(64, int(np.ceil(4 * k_max * radius * th_max)) + 32)
    n_phi = n_phi or max(64, int(np.ceil(2 * k_max * radius)) + 48)
    pts, nrm, ds = sphere_arc(radius, c, baffle, n_arc)
    W, (Gr, Gz) = mesh.interpolation(pts, gradient=True)
    outside = (Gr.getnnz(axis=1) == 0) & (Gz.getnnz(axis=1) == 0)
    if outside.any():
        raise SelectionError(f"the integration sphere leaves the mesh (e.g. at r, z = {np.round(pts[outside][0], 5)}): "
                             f"choose a radius inside the air")
    ps = (W @ P.T).T                                                     # (n_f, n_s)
    dpdn = (Gr @ P.T).T * nrm[:, 0] + (Gz @ P.T).T * nrm[:, 1]
    phi = np.linspace(0.0, np.pi, n_phi)                                 # even in φ: [0, π], doubled
    wphi = np.full(n_phi, np.pi / (n_phi - 1))
    wphi[[0, -1]] *= 0.5
    wphi *= 2.0
    cosphi = np.cos(phi)
    rs, zs = pts[:, 0], pts[:, 1]
    weight = (ds * rs)[None, :, None] * wphi[None, None, :]              # dS = r ds dφ
    targets = [at] if baffle is None else [at, np.column_stack([at[:, 0], 2 * baffle - at[:, 1]])]
    out = np.zeros((len(k), len(at)), complex)
    for x in targets:
        rx, zx = x[:, 0][:, None, None], x[:, 1][:, None, None]
        R = np.sqrt(rx ** 2 + rs[None, :, None] ** 2 - 2 * rx * rs[None, :, None] * cosphi[None, None, :]
                    + (zx - zs[None, :, None]) ** 2)                     # (n_x, n_s, n_φ)
        proj = (rs[None, :, None] - rx * cosphi[None, None, :]) * nrm[None, :, 0, None] \
            + (zs[None, :, None] - zx) * nrm[None, :, 1, None]
        for i, ki in enumerate(k):
            e = np.exp(-1j * ki * R) / (4 * np.pi * R)
            dG = -(1 + 1j * ki * R) * e / R ** 2 * proj
            out[i] += np.einsum("xsf,sf->x", dG * weight, np.broadcast_to(ps[i][:, None], (len(rs), n_phi))) \
                - np.einsum("xsf,sf->x", e * weight, np.broadcast_to(dpdn[i][:, None], (len(rs), n_phi)))
    return out
