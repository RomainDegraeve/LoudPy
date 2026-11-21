"""
What a selector selects, drawn on the mesh (lp.plot.selection).

    lp.plot.selection(model, lp.Interface("coil", "former"))      2D model: selected edges over the r–z mesh
    lp.plot.selection(model, "surround")                          selected domain groups filled
    lp.plot.selection(scene, "walls")                             3D scene: named surfaces of its meshes
    lp.plot.selection(mesh, ["walls", "interface_driver"])        a 3D Mesh (TRI3 / TRI6 surface groups)

The model must be meshed (selections are evaluated on the mesh, as the solver sees them). 3D views use
matplotlib's 3D axes (no OpenGL needed), drawing the corner triangles of the surface cells.
"""
from __future__ import annotations

import numpy as np

from loudpy.errors import SelectionError
from loudpy.plot import style

SELECTED = "#D55E00"           # Okabe–Ito vermillion
OTHER = "0.85"


def _model_2d(model, selector, kind, ax, title):
    from matplotlib.collections import LineCollection
    mesh = model.meshed
    if mesh is None:
        from loudpy.errors import StateError
        raise StateError("lp.plot.selection: mesh the model first (model.mesh(…))")
    xy = np.asarray(mesh.coords, float)[:, :2]
    tri = mesh.triangles()
    if ax is None:
        _, ax = style.figure("field")
    ax.triplot(xy[:, 0], xy[:, 1], tri, color=OTHER, lw=0.3)
    found = []
    if kind in (None, "domain"):
        try:
            groups = model.select(selector, kind="domain")
        except SelectionError:
            groups = []
        if groups:
            t = mesh.triangles(groups)
            ax.tripcolor(xy[:, 0], xy[:, 1], t, np.ones(len(xy)), cmap="Oranges", vmin=0, vmax=1.6, shading="flat")
            found.append(f"{len(groups)} domain group(s)")
    if kind in (None, "boundary"):
        try:
            edges = np.asarray(model.select(selector, kind="boundary"))
        except SelectionError:
            edges = np.zeros((0, 3), int)
        if len(edges):
            rows = mesh.rows_of(edges.ravel()).reshape(edges.shape)                 # node tags → rows
            order = [0] + list(range(2, rows.shape[1])) + [1]                       # corner, mid(s), corner
            seg = xy[rows[:, order]]
            ax.add_collection(LineCollection(seg, colors=SELECTED, linewidths=2.0))
            found.append(f"{len(edges)} edge(s)")
    if not found:
        raise SelectionError(f"{selector!r} selects nothing on the mesh")
    ax.set_aspect("equal")
    ax.grid(False, which="both")
    ax.set_xlabel(style.LABELS["r"])
    ax.set_ylabel(style.LABELS["z"])
    ax.set_title(f"{selector!r}: {', '.join(found)}" if title is None else title)
    return ax


def _surfaces_3d(nodes, faces: dict, selected, ax, title, other_alpha=0.12):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=style.SIZES["field"], layout="constrained")
        ax = fig.add_subplot(projection="3d")
    sel = [selected] if isinstance(selected, str) else list(selected)
    missing = [s for s in sel if s not in faces or not len(faces[s])]
    if missing:
        raise SelectionError(f"no surface {missing} (surfaces: {[k for k, v in faces.items() if len(v)]})")
    for name, tris in faces.items():
        if not len(tris):
            continue
        on = name in sel
        poly = nodes[np.asarray(tris)[:, :3]]
        ax.add_collection3d(Poly3DCollection(poly, facecolor=SELECTED if on else OTHER,
                                             edgecolor="0.3" if on else "none", linewidths=0.1,
                                             alpha=1.0 if on else other_alpha))
    lo, hi = nodes.min(axis=0), nodes.max(axis=0)
    c, r = 0.5 * (lo + hi), 0.5 * float((hi - lo).max())
    ax.set(xlim=(c[0] - r, c[0] + r), ylim=(c[1] - r, c[1] + r), zlim=(c[2] - r, c[2] + r),
           xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(", ".join(sel) if title is None else title)
    return ax


def _mesh_faces(mesh):
    nodes = np.asarray(mesh.coords, float)
    faces = {}
    for name, g in mesh.groups.items():
        if g.dim == 2:
            faces[name] = mesh.rows_of(np.asarray(g.cells).ravel()).reshape(np.shape(g.cells))
    return nodes, faces


@style.styled
def selection(model, selector, kind: str | None = None, ax=None, title=None, mesh: str | None = None):
    """
    Draw what `selector` selects. model: a meshed lp.Model (2D: selector = any selector or item; kind
    "domain", "boundary" or None for both), an lp.Scene (selector: surface group name(s) of its meshes —
    "walls", "interface_driver", "dtn", "ground"…; mesh: "cavity" (default when there is one) or the index of a
    band mesh), or a 3D Mesh. Returns the axes.
    """
    from loudpy.geometry.mesh import Mesh
    from loudpy.scene.scene import Scene
    if isinstance(model, Mesh):
        nodes, faces = _mesh_faces(model)
        return _surfaces_3d(nodes, faces, selector, ax, title)
    if isinstance(model, Scene):
        from loudpy.geometry.meshing3d import read_p2_mesh
        if model.folder is None:
            from loudpy.errors import StateError
            raise StateError("lp.plot.selection: mesh the scene first (scene.mesh(folder))")
        if (mesh in (None, "cavity")) and model.cavity_mesh is not None:
            path = model.cavity_mesh
        else:
            path = model.band_meshes[int(mesh or 0)]["path"]
        names = ["walls", "interface_driver", "interface_driver_*", "interface_front", "interface_front_*",
                 "dtn", "ground", "openings"]
        nodes, _, faces = read_p2_mesh(path, names)
        return _surfaces_3d(nodes, faces, selector, ax, title)
    return _model_2d(model, selector, kind, ax, title)
