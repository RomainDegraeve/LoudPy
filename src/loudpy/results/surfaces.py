"""
3D surfaces of scene results for plotting (pyvista datasets).

    driver_surface(modes, q, frame)    deformed driver: its meridian boundary revolved, displaced by
                                       the modal amplitudes q (all azimuthal orders summed)
    cavity_grid(cavity)                linear-tetrahedron grid of a cavity mesh (P2 corner nodes)
"""
from __future__ import annotations

import numpy as np

from loudpy.geometry.frame import Frame


def driver_surface(drv, q, center=(0.0, 0.0, 0.0), theta=(0.0, 2 * np.pi),
                   n_theta=73, deformation=0.10, phase=None, u_max=None):
    """
    Deformed driver (boundary of the meridian mesh revolved over `theta`, placed at `center`:
    a translation (x0, y0[, z0]), or an lp.Frame for any orientation; θ is measured from its
    x_l axis) for the modal amplitudes q (complex, all orders summed): real part at `phase`
    (default: phase of the largest displacement), max |u| = deformation × driver diameter.
    u_max: displacement [m] mapped to that deformation and to |u| = 1 (default: the max of this
    surface); pass a common value (and phase) to draw several drivers at one scale and instant.
    Returns (pv.PolyData with point data "|u|" (normalised) and "|u| [m]", phase).
    """
    import pyvista as pv
    t = drv.tris
    e = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    edges = uniq[cnt == 1]
    nodes = np.unique(edges)
    loc = np.full(len(drv.xy), -1)
    loc[nodes] = np.arange(len(nodes))
    th = np.linspace(theta[0], theta[1], n_theta)
    ur, ut, uz = drv.displacement(q, nodes, th)
    if phase is None:
        amp = np.abs(ur) ** 2 + np.abs(ut) ** 2 + np.abs(uz) ** 2
        j = np.argmax(amp.ravel())
        big = max((ur.ravel()[j], ut.ravel()[j], uz.ravel()[j]), key=abs)
        phase = -np.angle(big)
    ur, ut, uz = (np.real(x * np.exp(1j * phase)) for x in (ur, ut, uz))
    mag = np.sqrt(ur ** 2 + ut ** 2 + uz ** 2)
    ref = max(mag.max() if u_max is None else u_max, 1e-30)
    sc = deformation * 2 * drv.xy[:, 0].max() / ref
    r, z = drv.xy[nodes, 0][:, None], drv.xy[nodes, 1][:, None]
    rr = r + sc * ur
    pts = np.column_stack([(rr * np.cos(th) - sc * ut * np.sin(th)).ravel(),
                           (rr * np.sin(th) + sc * ut * np.cos(th)).ravel(),
                           (z + sc * uz).ravel()])
    if isinstance(center, Frame):
        pts = center.to_global(pts)
    else:
        pts = pts + np.array([center[0], center[1], center[2] if len(center) > 2 else 0.0])
    ia, ib = loc[edges[:, 0]], loc[edges[:, 1]]
    k = np.arange(n_theta - 1)
    quads = np.stack([np.full((len(edges), n_theta - 1), 4), ia[:, None] * n_theta + k,
                      ib[:, None] * n_theta + k, ib[:, None] * n_theta + k + 1,
                      ia[:, None] * n_theta + k + 1], axis=2)
    surf = pv.PolyData(pts, quads.reshape(-1))
    surf["|u|"] = (mag / ref).ravel()
    surf["|u| [m]"] = mag.ravel()
    return surf, phase


def cavity_grid(cav):
    """Linear-tetrahedron pyvista grid of the cavity (P2 corner nodes), point data "node"."""
    import pyvista as pv
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: cav.tets[:, :4]}, cav.nodes)
    grid["node"] = np.arange(len(cav.nodes))
    return grid
