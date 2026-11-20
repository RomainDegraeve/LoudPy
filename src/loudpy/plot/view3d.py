"""
3D views: pyvista scenes rendered off screen into matplotlib axes, with loudpy colormaps and
matplotlib colour bars (so 3D figures share fonts, sizes and labels with the 2D ones).

    v = lp.plot.View3D(size=(900, 700))
    v.add(lp.plot.view3d.tet_grid(field.mesh, field.spl().values), quantity="spl", clim=(60, 110))
    v.add(res.driver("top").surface(500)[0], scalars="|u|", cmap="viridis", colorbar=False)
    v.camera((1.0, 0.7, 1.0), focus=(0.1, 0.1, 0), up=(0, 1, 0))
    ax = v.show(ax)                                    # image + colour bars in the axes

Datasets:
    tet_grid(mesh, values)        P2 tetrahedra as linear sub-tetrahedra through the mid-edge nodes
    tri_surface(mesh, group, values)   a TRI6 surface group as linear sub-triangles
    clip(grid, normal, origin)    the part of a grid on one side of a plane (a cut-away view)
    isolines(meshes, levels)      iso-value lines of point data on surfaces
    edges(meshes)                 feature edges (wall corners) of surfaces
    dashed(a, b)                  a dashed segment (outlines of removed parts)
"""
from __future__ import annotations

import numpy as np

from loudpy.plot import style
from loudpy.results.quantities import quantity as as_quantity

_SUB_TET = np.array([[0, 4, 6, 7], [4, 1, 5, 9], [6, 5, 2, 8], [7, 9, 8, 3],
                     [4, 8, 5, 6], [4, 8, 6, 7], [4, 8, 7, 9], [4, 8, 9, 5]])        # gmsh TET10 → 8 tetrahedra
_SUB_TRI = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])                    # gmsh TRI6 → 4 triangles


def _rows(mesh, cells):
    return mesh.rows_of(cells) if not np.array_equal(mesh.node_tags, np.arange(mesh.n_nodes)) else np.asarray(cells)


_GRIDS: dict = {}                     # (id(mesh), group) → linear grid (several frequencies share a mesh)


def tet_grid(mesh, values=None, name: str = "values", group: str | None = None):
    """pyvista UnstructuredGrid of the TET10 groups (or one group) of a 3D Mesh, values as point data."""
    import pyvista as pv
    key = (id(mesh), group)
    if key not in _GRIDS:
        names = [group] if group else [n for n, g in mesh.groups.items() if g.cell_type == "tet10"]
        tets = np.vstack([_rows(mesh, mesh.groups[n].cells) for n in names])
        if len(_GRIDS) > 8:
            _GRIDS.clear()
        _GRIDS[key] = (mesh, pv.UnstructuredGrid({pv.CellType.TETRA: tets[:, _SUB_TET].reshape(-1, 4)},
                                                 np.asarray(mesh.coords, float)))
    grid = _GRIDS[key][1].copy(deep=False)
    if values is not None:
        grid.point_data[name] = np.asarray(values)
    return grid


def tri_surface(mesh_or_nodes, group_or_tris, values=None, name: str = "values"):
    """pyvista PolyData of a TRI6 surface: (Mesh, group name) or (nodes (n, 3), tri6 rows (m, 6))."""
    import pyvista as pv
    if hasattr(mesh_or_nodes, "groups"):
        mesh = mesh_or_nodes
        nodes, tris = np.asarray(mesh.coords, float), _rows(mesh, mesh.groups[group_or_tris].cells)
    else:
        nodes, tris = np.asarray(mesh_or_nodes, float), np.asarray(group_or_tris)
    t3 = tris[:, _SUB_TRI].reshape(-1, 3) if tris.shape[1] == 6 else tris
    surf = pv.PolyData(nodes, np.hstack([np.full((len(t3), 1), 3), t3]).ravel())
    if values is not None:
        surf.point_data[name] = np.asarray(values)
    return surf


def clip(grid, normal, origin, invert: bool = True):
    """A cut-away: the part of a grid behind the plane (normal, origin), the side the normal points
    away from (the side toward the normal is removed: put the camera there)."""
    return grid.clip(normal=normal, origin=origin, invert=invert)


def isolines(meshes, levels, scalars: str = "values"):
    """Iso-value lines of point data `scalars` at `levels` on one or several surfaces (None if none)."""
    import pyvista as pv
    meshes = meshes if isinstance(meshes, (list, tuple)) else [meshes]
    lines = [ln for ln in (m.contour(isosurfaces=list(levels), scalars=scalars) for m in meshes) if ln.n_points]
    return pv.merge(lines) if lines else None


def edges(meshes, angle: float = 35.0):
    """Feature edges (folds sharper than `angle` degrees: wall corners) of one or several surfaces."""
    import pyvista as pv
    meshes = meshes if isinstance(meshes, (list, tuple)) else [meshes]
    return pv.merge([m.extract_feature_edges(feature_angle=angle, boundary_edges=False, manifold_edges=False,
                                             non_manifold_edges=False) for m in meshes])


def dashed(a, b, dash: float = 0.008):
    """Dashed segment a → b (pieces of length `dash`)."""
    import pyvista as pv
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.linalg.norm(b - a) / dash))
    pts = np.linspace(a, b, n + 1)
    return pv.merge([pv.Line(pts[i], pts[i + 1]) for i in range(0, n, 2)])


class View3D:
    """An off-screen 3D scene drawn into matplotlib axes (see the module docstring)."""

    def __init__(self, size=(900, 700), background: str = "white", smooth: bool = True):
        self.size, self.background, self.smooth = tuple(size), background, smooth
        self.items, self.bars = [], []
        self.cam = None

    def add(self, dataset, scalars: str | None = None, quantity=None, cmap=None, clim=None, colorbar: bool | str = True,
            label: str | None = None, **kwargs) -> "View3D":
        """
        Add a pyvista dataset. scalars: point-data name (default: its active scalars); quantity: the
        quantity of the values (name or Quantity) for the colormap and the colour-bar label; clim:
        colour range; colorbar: False, True or "shared" (one bar for the datasets with the same
        colormap, range and label); other keywords go to pyvista add_mesh (color, opacity, line_width, show_edges …).
        """
        q = as_quantity(quantity) if quantity is not None else None
        if scalars is None and dataset.point_data.keys() and "color" not in kwargs:
            scalars = dataset.point_data.active_scalars_name or list(dataset.point_data.keys())[0]
        if scalars is not None:
            cmap = cmap or style.CMAPS.get(q.kind if q else "plain", "viridis")
            vals = np.asarray(dataset.point_data[scalars])
            clim = tuple(clim) if clim is not None else (float(np.nanmin(vals)), float(np.nanmax(vals)))
            bar = (cmap, clim, label or (q.axis_label() if q else scalars), colorbar)
            if colorbar and not (colorbar == "shared" and any(b[:3] == bar[:3] for b in self.bars)):
                self.bars.append(bar)
        self.items.append((dataset, dict(scalars=scalars, cmap=cmap, clim=clim, **kwargs)))
        return self

    def camera(self, position, focus=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0), view_angle: float | None = None,
               parallel: float | None = None, zoom: float | None = None) -> "View3D":
        """Camera position, focal point, view-up vector, view angle [°] (perspective) or parallel
        projection with `parallel` = half the visible height [m]; zoom: keep the direction but fit the
        scene to the view, then zoom by this factor."""
        self.cam = (tuple(position), tuple(focus), tuple(up), view_angle, parallel, zoom)
        return self

    def view(self, plane: str = "iso", azimuth: float = 0.0, elevation: float = 0.0, zoom: float = 1.0) -> "View3D":
        """A standard view ("iso", "xy", "xz", "yz", …), then turned by azimuth / elevation [°] and zoomed."""
        self.cam = ("preset", plane, azimuth, elevation, zoom)
        return self

    def image(self) -> np.ndarray:
        """The rendered RGB image."""
        import pyvista as pv
        pl = pv.Plotter(off_screen=True, window_size=list(self.size))
        pl.set_background(self.background)
        if self.smooth:
            pl.enable_anti_aliasing("ssaa")
        for ds, kw in self.items:
            kw = {k: v for k, v in kw.items() if v is not None}
            if kw.get("scalars") is not None:
                kw.setdefault("interpolate_before_map", True)
            pl.add_mesh(ds, show_scalar_bar=False, smooth_shading=kw.pop("smooth_shading", False), **kw)
        if self.cam is not None and self.cam[0] == "preset":
            _, plane, az, el, zoom = self.cam
            if plane == "iso":
                pl.view_isometric()
            else:
                pl.camera_position = plane
            pl.camera.azimuth, pl.camera.elevation = az, el
            pl.reset_camera()
            pl.camera.zoom(zoom)
        elif self.cam is not None:
            pos, foc, up, ang, par, zoom = self.cam
            pl.camera_position = [pos, foc, up]
            if par is not None:
                pl.enable_parallel_projection()
                pl.camera.parallel_scale = par
            elif ang is not None:
                pl.camera.view_angle = ang
            if zoom is not None:
                pl.reset_camera()
                pl.camera.zoom(zoom)
        else:
            pl.view_isometric()
        img = pl.screenshot(return_img=True)
        pl.close()
        return img

    def show(self, ax=None, title: str | None = None, colorbars: bool = True):
        """Draw the image (and its colour bars) into `ax` (a new figure without it); returns the axes.
        colorbars=False: no bars here (draw them once for several panels with colorbars(fig, axes, view.bars))."""
        if ax is None:
            _, ax = style.figure((self.size[0] / 140, self.size[1] / 140))
        ax.imshow(self.image(), interpolation="lanczos")
        ax.set_axis_off()
        if colorbars:
            colorbars_for(ax.figure, ax, self.bars)
        if title:
            ax.set_title(title, fontsize=9)
        return ax


def colorbars_for(fig, axes, bars, **kwargs):
    """Matplotlib colour bars (cmap, clim, label, …) for one axes or several (shared)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    kwargs = {"shrink": 0.75, "pad": 0.01, **kwargs}
    for cmap, clim, label, *_ in bars:
        fig.colorbar(ScalarMappable(Normalize(*clim), cmap), ax=axes, label=label, **kwargs)


def field3d(f, show=None, ax=None, cmap=None, vmin=None, vmax=None, title=None, colorbar=True, camera=None,
            cut=None, size=(900, 700), range_db: float | None = 50.0, **kwargs):
    """
    A 3D Field: its domain surface coloured (SPL for a pressure), optionally cut by a plane
    cut=(normal, origin) to show the inside. camera: (position, focus, up[, view angle]).
    """
    from loudpy.plot.fields import _scalar
    vals, q, cm, symmetric = _scalar(f, show)
    if symmetric and vmin is None and vmax is None:
        vmax = float(np.nanmax(np.abs(vals))) or 1.0
        vmin = -vmax
    if q.name == "spl" and vmin is None and range_db is not None:
        top = float(np.nanpercentile(vals, 99.5)) if vmax is None else vmax
        vmin, vmax = top - range_db, top
    grid = tet_grid(f.mesh, vals, "v")
    shown = clip(grid, *cut) if cut is not None else grid.extract_surface(algorithm="dataset_surface")
    v = View3D(size)
    v.add(shown, scalars="v", quantity=q, cmap=cmap or cm, colorbar=colorbar,
          clim=(vmin if vmin is not None else float(np.nanmin(vals)), vmax if vmax is not None else float(np.nanmax(vals))),
          **kwargs)
    if camera is not None:
        v.camera(*camera)
    from loudpy.plot.fields import _title
    return v.show(ax, title=_title(f) if title is None else title)
