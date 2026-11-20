"""
3D views of scene results (lp.Scene + lp.FrequencyResponse with an FE air field): the SPL on the
ground, on the walls and on a cut plane, iso-SPL lines, wall edges and the deformed drivers at one
instant and one scale, from several cameras, in one figure.

    cams = {"section": lp.plot.Camera((-1.5, .3, -.1), focus=(.07, .3, -.1), up=(0, 1, 0), parallel=0.4),
            "front":   lp.plot.Camera((.07, .3, 2.0), focus=(.07, .3, 0), up=(0, 1, 0), parallel=0.35, cut=False)}
    fig = lp.plot.scene_view(res, f=100, cameras=cams, cut=((-1, 0, 0), (0.07, 0, 0)))

The colour scale is common to every surface: `range_db` below the maximum SPL farther than
`near_field` from the drivers (the near field saturates). keep(points) → bool can trim the cut
plane (e.g. keep the inside of the cabinet and the front only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loudpy.plot import style, view3d


@dataclass
class Camera:
    """A camera of scene_view: position, focus, view up, perspective view angle [°] or parallel
    projection (half visible height [m]); cut=False shows the whole scene (no cut plane)."""
    position: tuple
    focus: tuple = (0.0, 0.0, 0.0)
    up: tuple = (0.0, 1.0, 0.0)
    view_angle: float | None = 30.0
    parallel: float | None = None
    cut: bool = True


def _driver_centres(res):
    return np.array([d["origin"] for d in res.scene.get("drivers", [])], float).reshape(-1, 3)


def scene_surfaces(res, f, cut=None, keep=None, near_field: float = 0.08, groups=("ground", "walls", "openings"),
                   cavity=None, wall_offset: float = 0.002):
    """
    SPL surfaces of a scene result at the frequency nearest f: dict with "cut" and "full" (lists of
    pyvista surfaces with point data "spl"), "edges_cut" / "edges_full" (wall corners), "top" (the
    maximum SPL beyond near_field from the drivers) and "f". cavity: a CavityModes (or True when the
    results store the cavity cells) adds the pressure inside the enclosure (cavity modes) on the cut
    and on its walls, drawn `wall_offset` inside.
    """
    p = res.field("p", f=f)
    L = p.spl().values
    mesh = p.mesh
    centres = _driver_centres(res)
    far = np.min([np.linalg.norm(mesh.coords - c, axis=1) for c in centres], axis=0) > near_field \
        if len(centres) else np.ones(len(L), bool)
    faces = {k: view3d.tri_surface(mesh, k, L, "spl") for k in groups if k in mesh.groups and len(mesh.groups[k].cells)}
    walls = [s for k, s in faces.items() if k != "ground"]
    out = dict(full=list(faces.values()), edges_full=view3d.edges(walls) if walls else None,
               top=float(L[far].max()), f=p.coordinate)
    if cut is not None:
        normal, origin = np.asarray(cut[0], float), np.asarray(cut[1], float)
        plane = view3d.tet_grid(mesh, L, "spl").slice(normal=normal, origin=origin)
        if keep is not None:
            plane["keep"] = np.asarray(keep(plane.points), float)
            plane = plane.clip_scalar(scalars="keep", value=0.5, invert=False)
        walls_cut = [s.clip(normal=normal, origin=origin) for s in walls]
        out["cut"] = ([faces["ground"]] if "ground" in faces else []) + walls_cut + [plane]
        out["edges_cut"] = view3d.edges(walls_cut) if walls_cut else None
    if cavity is not None:
        inside = res.cavity_field(f, cavity=None if cavity is True else cavity)
        Li = inside.spl().values
        g = view3d.tet_grid(inside.mesh, Li, "spl")
        walls_in = g.extract_surface(algorithm="dataset_surface").compute_normals(
            point_normals=True, cell_normals=False, auto_orient_normals=True)
        walls_in.points = walls_in.points - wall_offset * walls_in["Normals"]
        out["top"] = max(out["top"], float(Li.max()))
        if cut is not None:
            out["cut"] += [walls_in.clip(normal=normal, origin=origin), g.slice(normal=normal, origin=origin)]
    return out


@style.styled
def scene_view(res, f, cameras: dict, cut=None, keep=None, range_db: float = 50.0, contour_db: float | None = 3.0,
               near_field: float = 0.08, deformation: float = 0.10, outline=None, ground=None, size=(900, 760),
               ncols: int | None = None, title: str | None = None, cavity=None):
    """
    One figure, one panel per camera ({name: Camera}), of the SPL surfaces and deformed drivers of a
    scene result at the frequency nearest f. cut=(normal, origin): the cut plane (cameras with
    cut=True show the cut-away scene); outline: a pyvista dataset drawn black on the cut views (e.g.
    the dashed outline of the removed part, view3d.dashed); ground: a dataset drawn grey under it all;
    cavity: also the pressure inside the enclosure (see scene_surfaces), on the same scale.
    """
    sc = scene_surfaces(res, f, cut, keep, near_field, cavity=cavity)
    clim = (sc["top"] - range_db, sc["top"])
    levels = np.arange(np.ceil(clim[0] / contour_db) * contour_db, clim[1] + 1e-9, contour_db) if contour_db else None
    drivers = res.driver_surfaces(sc["f"], deformation)
    n = len(cameras)
    ncols = ncols or (2 if n <= 4 else 3)
    nrows = int(np.ceil(n / ncols))
    w = style.SIZES["wide"][0]
    fig, axs = style.figure((w, w / ncols * size[1] / size[0] * nrows + 0.5), nrows, ncols, squeeze=False)
    for ax in axs.ravel():
        ax.set_axis_off()
    for ax, (name, cam) in zip(axs.ravel(), cameras.items()):
        parts = sc["cut"] if (cam.cut and "cut" in sc) else sc["full"]
        v = view3d.View3D(size)
        if ground is not None:
            v.add(ground, color="#d9d9d9", lighting=False)
        for m in parts:
            v.add(m, scalars="spl", quantity="spl", clim=clim, lighting=False, colorbar="shared")
        if levels is not None:
            lines = view3d.isolines(parts, levels, "spl")
            if lines is not None:
                v.add(lines, color="black", line_width=0.7, opacity=0.45)
        e = sc["edges_cut" if (cam.cut and "cut" in sc) else "edges_full"]
        if e is not None and e.n_points:
            v.add(e, color="#202020", line_width=1.6)
        for s in drivers.values():
            v.add(s, scalars="|u|", clim=(0, 1), smooth_shading=True, specular=0.3, colorbar="shared",
                  cmap=style.CMAPS["displacement"], label=r"Driver $|u|/|u|_{max}$")
        if outline is not None and cam.cut:
            v.add(outline, color="black", line_width=1.0)
        v.camera(cam.position, cam.focus, cam.up, view_angle=cam.view_angle, parallel=cam.parallel)
        v.show(ax, title=name, colorbars=False)
    view3d.colorbars_for(fig, axs, v.bars, shrink=0.5)
    head = f"{sc['f']:.0f} Hz: SPL top {sc['top']:.0f} dB beyond {100 * near_field:.0f} cm of the drivers"
    if contour_db:
        head += f", iso-lines every {contour_db:g} dB"
    fig.suptitle(head if title is None else f"{title} — {head}", fontsize=9)
    return fig
