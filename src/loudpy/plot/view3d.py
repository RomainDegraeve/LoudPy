"""
3D views: pyvista scenes rendered off screen into matplotlib axes, with loudpy colormaps and
matplotlib colour bars (so 3D figures share fonts, sizes and labels with the 2D ones).

    v = lp.plot.View3D(size=(900, 700))
    v.add(lp.plot.view3d.tet_grid(field.mesh, field.spl().values), quantity="spl", clim=(60, 110))
    v.add(res.driver("top").surface(500)[0], scalars="|u|", cmap="viridis", colorbar=False)
    v.camera((1.0, 0.7, 1.0), focus=(0.1, 0.1, 0), up=(0, 1, 0))
    ax = v.show(ax)                                    # image + colour bars in the axes

    v.save("view.png")                                 # .png .jpg .pdf .svg: the figure (image + colour bars)
    v.save("view.html")                                # interactive, self-contained (no server, no package)
    v.save("view.gltf")                                # 3D model with its colours (Blender, PowerPoint, web)
    v.save("view.vtm")                                 # the datasets with their values (ParaView)
    v.interactive()                                    # a window to turn the scene (or inline in Jupyter)

    lp.plot.view(res.field("p", f=500), cut=((1, 0, 0), (0, 0, 0)))    a View3D of a Field (3D, or a 2D
                                                                         displacement revolved)

Datasets:
    tet_grid(mesh, values)        P2 tetrahedra as linear sub-tetrahedra through the mid-edge nodes
    tri_surface(mesh, group, values)   a TRI6 surface group as linear sub-triangles
    clip(grid, normal, origin)    the part of a grid on one side of a plane (a cut-away view)
    isolines(meshes, levels)      iso-value lines of point data on surfaces
    edges(meshes)                 feature edges (wall corners) of surfaces
    dashed(a, b)                  a dashed segment (outlines of removed parts)
"""
from __future__ import annotations

from pathlib import Path

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
        self.items: list = []
        self.bars: list = []
        self.cam: tuple | None = None

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
        else:
            bar = None
        self.items.append((dataset, dict(scalars=scalars, cmap=cmap, clim=clim, **kwargs), bar))
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

    def plotter(self, off_screen: bool = True, scalar_bars: bool = False):
        """The pyvista Plotter of this scene (datasets, camera); scalar_bars: pyvista's own bars (labelled as the
        colour bars of the figure) for interactive windows."""
        import pyvista as pv
        pl = pv.Plotter(off_screen=off_screen, window_size=list(self.size))
        pl.set_background(self.background)
        if self.smooth and off_screen:
            pl.enable_anti_aliasing("ssaa")
        shown: set = set()
        for ds, kw, bar in self.items:
            kw = {k: v for k, v in kw.items() if v is not None}
            if kw.get("scalars") is not None:
                kw.setdefault("interpolate_before_map", True)
            sb = bool(scalar_bars and bar is not None and bar[2] not in shown)
            if sb:
                assert bar is not None
                shown.add(bar[2])
                kw["scalar_bar_args"] = dict(title=_short(bar[2]), color="black", fmt="%.3g", title_font_size=13,
                                             label_font_size=11, vertical=True, position_x=0.84,
                                             position_y=0.08 + 0.46 * (len(shown) - 1), height=0.36, width=0.05)
            pl.add_mesh(ds, show_scalar_bar=sb, smooth_shading=kw.pop("smooth_shading", False), **kw)
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
        return pl

    def image(self, scale: int = 1) -> np.ndarray:
        """The rendered RGB image (scale: × the size, for high-resolution images)."""
        pl = self.plotter()
        img = pl.screenshot(return_img=True, scale=scale)
        pl.close()
        return np.asarray(img)

    def interactive(self, title: str | None = None):
        """Open the scene in a window to turn, zoom and pan (in Jupyter: inline, with pyvista's notebook backend);
        returns when the window is closed."""
        pl = self.plotter(off_screen=False, scalar_bars=True)
        if title:
            pl.add_text(title, font_size=10, color="black")
        pl.show(title=title or "loudpy")

    def to_pyvista(self):
        """The datasets of the scene as one pyvista MultiBlock (with their point data)."""
        import pyvista as pv
        return pv.MultiBlock([ds for ds, *_ in self.items])

    def save(self, path, title: str | None = None, scale: int = 2, dpi: int = 300) -> Path:
        """
        Save the scene by extension:
            .png .jpg .tif .pdf .svg   the figure: image (rendered at `scale` × size) + colour bars + title
            .html                      interactive viewer in one self-contained file (lp.plot.html's viewer)
            .gltf .glb                 3D model with the colours (Blender, PowerPoint, Windows 3D viewer, the web)
            .vtm                       the datasets and their values (ParaView: File > Open)
        Folders are created; returns the path.
        """
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower()
        if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf", ".svg", ".eps"):
            ax = self.show(title=title, scale=scale)
            return style.save(ax.figure, path, dpi=dpi)
        if ext == ".html":
            from loudpy.plot.html import html_view
            return html_view(self, path, title=title)
        if ext in (".gltf", ".glb"):
            pl = self.plotter()
            tmp = path.with_name(path.stem + ".tmp.gltf")
            import warnings
            with warnings.catch_warnings():                      # volumes are exported as their surfaces
                warnings.simplefilter("ignore", UserWarning)
                pl.export_gltf(str(tmp))
            pl.close()
            if ext == ".glb":
                _gltf_to_glb(tmp, path)
                tmp.unlink()
            else:
                tmp.replace(path)
            return path
        if ext in (".vtm", ".vtmb"):
            self.to_pyvista().save(str(path))
            return path
        raise ValueError(f"View3D.save: unknown extension {ext!r} (.png .jpg .pdf .svg .html .gltf .glb .vtm)")

    @style.styled
    def show(self, ax=None, title: str | None = None, colorbars: bool = True, scale: int = 1):
        """Draw the image (and its colour bars) into `ax` (a new figure without it); returns the axes.
        colorbars=False: no bars here (draw them once for several panels with colorbars(fig, axes, view.bars))."""
        if ax is None:
            _, ax = style.figure((self.size[0] / 140, self.size[1] / 140))
        ax.imshow(self.image(scale), interpolation="lanczos")
        ax.set_axis_off()
        if colorbars:
            colorbars_for(ax.figure, ax, self.bars)
        if title:
            ax.set_title(title, fontsize=9)
        return ax


def _short(label: str) -> str:
    """A colour-bar title for VTK: the symbol and unit of a loudpy label ("Sound pressure level $L_p$ [dB…]" →
    "L_p [dB…]"), without TeX markup."""
    if "$" in label:
        head = label[:label.index("$")].strip()
        label = label[label.index("$"):]
        if head and len(head) <= 12:                      # keep a short qualifier ("Driver")
            label = head + " " + label
    return style.plain_label(label).strip()


def _gltf_to_glb(src, dst) -> None:
    """A .gltf with embedded (base64) buffers → one binary .glb."""
    import base64
    import json
    import struct
    from pathlib import Path
    doc = json.loads(Path(src).read_text())
    blob = b""
    for b in doc.get("buffers", []):
        uri = b.pop("uri", "")
        if uri.startswith("data:"):
            data = base64.b64decode(uri.split(",", 1)[1])
        else:
            data = (Path(src).parent / uri).read_bytes()
        b["byteOffset_glb"] = len(blob)
        blob += data + b"\0" * (-len(data) % 4)
    offsets = [b.pop("byteOffset_glb") for b in doc.get("buffers", [])]
    for v in doc.get("bufferViews", []):                       # one buffer in a .glb
        v["byteOffset"] = v.get("byteOffset", 0) + offsets[v.get("buffer", 0)]
        v["buffer"] = 0
    doc["buffers"] = [{"byteLength": len(blob)}] if blob else []
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + (8 + len(blob) if blob else 0))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js
    if blob:
        out += struct.pack("<II", len(blob), 0x004E4942) + blob
    Path(dst).write_bytes(out)


def colorbars_for(fig, axes, bars, **kwargs):
    """Matplotlib colour bars (cmap, clim, label, …) for one axes or several (shared)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    kwargs = {"shrink": 0.75, "pad": 0.01, **kwargs}
    for cmap, clim, label, *_ in bars:
        fig.colorbar(ScalarMappable(Normalize(*clim), cmap), ax=axes, label=label, **kwargs)


def view(f, show=None, cut=None, cmap=None, vmin=None, vmax=None, colorbar=True, camera=None, size=(900, 700),
         range_db: float | None = 50.0, sweep: float = 2 * np.pi, deformation: float = 0.10, **kwargs) -> View3D:
    """
    A View3D of a Field, to show, save (.png .pdf .html .gltf .vtm) or turn (interactive()):
    a 3D field: its domain surface coloured (SPL for a pressure), optionally cut by a plane cut=(normal, origin);
    a 2D axisymmetric displacement: the structure revolved over `sweep` [rad] and deformed (Field.revolve),
    coloured by |u|. camera: (position, focus, up[, view angle]).
    """
    from loudpy.errors import ResultsError
    from loudpy.plot.fields import scalar_values
    v = View3D(size)
    if f.mesh.dim == 2:
        if np.ndim(f.values) != 2:
            raise ResultsError("a 3D view of a 2D result revolves a displacement field (\"u\"); a pressure is "
                               "shown on its r–z section (Field.plot) or in lp.plot.html(field, mirror=True)")
        v.add(f.revolve(sweep=sweep, deformation=deformation), scalars="|u|", clim=(0, 1), colorbar=colorbar,
              label=r"$|u|/|u|_{max}$", cmap=cmap or style.CMAPS["displacement"], smooth_shading=True, **kwargs)
        if camera is None:
            v.view("xz", azimuth=35, elevation=25, zoom=1.2)
        else:
            v.camera(*camera)
        return v
    vals, q, cm, symmetric = scalar_values(f, show)
    if symmetric and vmin is None and vmax is None:
        vmax = float(np.nanmax(np.abs(vals))) or 1.0
        vmin = -vmax
    if q.name == "spl" and vmin is None and range_db is not None:
        top = float(np.nanpercentile(vals, 99.5)) if vmax is None else vmax
        vmin, vmax = top - range_db, top
    grid = tet_grid(f.mesh, vals, "v")
    shown = clip(grid, *cut) if cut is not None else grid.extract_surface(algorithm="dataset_surface")
    lo = vmin if vmin is not None else float(np.nanmin(vals))
    hi = vmax if vmax is not None else float(np.nanmax(vals))
    v.add(shown, scalars="v", quantity=q, cmap=cmap or cm, colorbar=colorbar, clim=(lo, hi), **kwargs)
    if camera is not None:
        v.camera(*camera)
    return v


@style.styled
def field3d(f, show=None, ax=None, cmap=None, vmin=None, vmax=None, title=None, colorbar=True, camera=None,
            cut=None, size=(900, 700), range_db: float | None = 50.0, **kwargs):
    """A 3D Field drawn into matplotlib axes (see view)."""
    from loudpy.plot.fields import field_title
    v = view(f, show, cut, cmap, vmin, vmax, colorbar, camera, size, range_db, **kwargs)
    return v.show(ax, title=field_title(f) if title is None else title)
