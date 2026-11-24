"""
3D views of scene results (lp.Scene + lp.FrequencyResponse with an FE air field): the SPL on the
ground, on the walls and on a cut plane, iso-SPL lines, wall edges and the deformed drivers at one
instant and one scale, from several cameras, in one figure.

    fig = res.plot(100)                          six default views (cut through the driver axis, dashed outline)
    fig = res.plot(100, cameras=["from above", "side, facing the cut"], cut=None)
    v = res.view(100)                            one view as a View3D: v.save("x.html" | ".glb"), v.interactive()
    cams = {"section": lp.plot.Camera((-1.5, .3, -.1), focus=(.07, .3, -.1), up=(0, 1, 0), parallel=0.4)}
    fig = res.plot(100, cameras=cams, cut=((-1, 0, 0), (0.07, 0, 0)))      your own cameras and cut

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
    """A camera of a scene view (SceneResults.plot / view, View3D.camera): position, focus, view up, perspective view angle [°] or parallel
    projection (half visible height [m]); cut=False shows the whole scene (no cut plane)."""
    position: tuple
    focus: tuple = (0.0, 0.0, 0.0)
    up: tuple = (0.0, 1.0, 0.0)
    view_angle: float | None = 30.0
    parallel: float | None = None
    cut: bool = True


def default_cameras(bounds, drivers=(), view_angle: float = 25.0) -> dict:
    """
    Six default cameras of a scene of `bounds` ((2, 3) lower / upper corners) with drivers at `drivers` (points),
    in the scene frame (y up, drivers facing +z, a cut plane x = driver x seen from −x):
    front left (cut side), front right, side facing the cut, from above, rear right, close-up on the driver.
    """
    b = np.asarray(bounds, float).reshape(2, 3)
    focus = b.mean(axis=0)
    diag = float(np.linalg.norm(b[1] - b[0]))
    dist = 0.8 * diag / np.tan(np.radians(view_angle) / 2)
    up = (0.0, 1.0, 0.0)

    def cam(direction, f=focus, d=dist, angle=view_angle):
        u = np.asarray(direction, float) / np.linalg.norm(direction)
        return Camera(tuple(np.asarray(f) + d * u), tuple(f), up, view_angle=angle)
    cams = {"front left (cut side)": cam((-0.55, 0.45, 0.70)),
            "front right (outer walls)": cam((0.70, 0.40, 0.62)),
            "side, facing the cut": cam((-1.0, 0.2, 0.0)),
            "from above": cam((-0.22, 1.0, 0.30)),
            "rear right (back and side walls)": cam((0.62, 0.38, -0.70))}
    drv = np.asarray(drivers, float).reshape(-1, 3)
    if len(drv):
        cams["close-up on the driver (cut side)"] = cam((-0.80, 0.32, 0.56), f=drv[0], d=0.38 * dist, angle=30.0)
    return cams


def scene_bounds(res) -> np.ndarray:
    """(2, 3) bounds of a scene result's structure: its walls (or the whole air mesh) and its drivers."""
    mesh = res.field("p", index=0).mesh
    groups = [g for g in ("walls", "openings") if g in mesh.groups and len(mesh.groups[g].cells)]
    pts = mesh.coords[mesh.rows_of(np.unique(np.concatenate([mesh.groups[g].cells.ravel() for g in groups])))] \
        if groups else mesh.coords
    pts = np.vstack([pts, _driver_centres(res)])
    return np.array([pts.min(axis=0), pts.max(axis=0)])


def _dashes(points, dash):
    """Segments (pairs of points) of a dashed polyline: dash on, dash off along its arc length."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] <= 0:
        return []
    marks = np.arange(0.0, s[-1], dash)
    pts = np.column_stack([np.interp(marks, s, points[:, k]) for k in range(3)])
    pts = np.vstack([pts, points[-1]])
    return [(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]


def _outline(res, cut, dash: float = 0.008, angle: float = 35.0):
    """The dashed outline of the part of the cabinet removed by the cut plane (normal, origin): the wall
    corners (folds sharper than `angle`) on the side the normal points to, as one pyvista line set."""
    import pyvista as pv
    mesh = res.field("p", index=0).mesh
    walls = [view3d.tri_surface(mesh, g) for g in ("walls", "openings") if g in mesh.groups and len(mesh.groups[g].cells)]
    if not walls:
        return None
    normal, origin = np.asarray(cut[0], float), np.asarray(cut[1], float)
    e = view3d.edges(walls, angle)
    if e is None or not e.n_points:
        return None
    removed = e.clip(normal=normal, origin=origin, invert=False)
    lines = removed.extract_all_edges().strip() if removed.n_points else None
    if lines is None or not lines.n_points:
        return None
    parts = []
    cells, k = lines.lines, 0
    while k < len(cells):
        n = cells[k]
        ids = cells[k + 1:k + 1 + n]
        k += n + 1
        parts += [pv.Line(a, b) for a, b in _dashes(lines.points[ids], dash)]
    return pv.merge(parts) if parts else None


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


def _panel(sc, drivers, cam, clim, levels, ground, outline, size) -> view3d.View3D:
    """The View3D of one camera of scene_figure."""
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
    return v


CUT_NORMALS = {"x": (-1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
#   the half removed by an axis cut: x < value (seen from −x, the driver's side cut), y > value (seen from above),
#   z > value (seen from the front); "-x", "-y", "-z" remove the other half


def resolve_cut(res, cut):
    """The cut plane (normal, origin) of a scene view: "driver" = the vertical plane x = x of the first driver,
    the half x < that removed (seen from −x); ("x" | "y" | "z", value [m]): the plane through value across that
    axis, the half x < value, y > value or z > value removed ("-x" … the other half); None: no cut; or (normal,
    origin) as given."""
    if isinstance(cut, (tuple, list)) and len(cut) == 2 and isinstance(cut[0], str):
        name = cut[0].strip().lower()
        if name.lstrip("-") not in CUT_NORMALS:
            raise ValueError(f"an axis cut is ('x' | 'y' | 'z' | '-x' | '-y' | '-z', value), got {cut!r}")
        k = "xyz".index(name.lstrip("-"))
        normal = np.asarray(CUT_NORMALS[name.lstrip("-")], float) * (-1.0 if name.startswith("-") else 1.0)
        origin = np.zeros(3)
        origin[k] = float(cut[1])
        return (tuple(normal), tuple(origin))
    if isinstance(cut, str):
        if cut != "driver":
            raise ValueError(f"cut must be 'driver', None or (normal, origin), got {cut!r}")
        c = _driver_centres(res)
        x0 = float(c[0, 0]) if len(c) else float(np.mean(scene_bounds(res)[:, 0]))
        return ((-1.0, 0.0, 0.0), (x0, 0.0, 0.0))
    return cut


def resolve_cameras(res, cameras, cut):
    """{name: Camera}: the default cameras (None), a subset of them by name (a list), or the given dict; every
    camera shows the whole scene when there is no cut."""
    from dataclasses import replace
    defaults = default_cameras(scene_bounds(res), _driver_centres(res))
    if cameras is None:
        cams = defaults
    elif isinstance(cameras, Camera):
        cams = {"view": cameras}
    elif isinstance(cameras, dict):
        cams = dict(cameras)
    else:
        names = [cameras] if isinstance(cameras, str) else list(cameras)
        unknown = [n for n in names if n not in defaults]
        if unknown:
            raise ValueError(f"no default camera {unknown}; default cameras: {list(defaults)}")
        cams = {n: defaults[n] for n in names}
    return cams if cut is not None else {k: replace(c, cut=False) for k, c in cams.items()}


def cut_camera(res, cut, view_angle: float = 25.0) -> Camera:
    """A camera on the removed side of a cut plane (normal, origin), a little from the front, above and left."""
    b = scene_bounds(res)
    focus = b.mean(axis=0)
    dist = 0.8 * float(np.linalg.norm(b[1] - b[0])) / np.tan(np.radians(view_angle) / 2)
    d = np.array([-0.35, 0.45, 0.70]) + 1.2 * np.asarray(cut[0], float)
    up = (0.0, 0.0, -1.0) if abs(d[1]) > 0.9 * np.linalg.norm(d) else (0.0, 1.0, 0.0)
    return Camera(tuple(focus + dist * d / np.linalg.norm(d)), tuple(focus), up, view_angle=view_angle)


def marks(v: view3d.View3D, points=None, arcs=None, size: float | None = None, color: str = "#c0392b"):
    """Observation points (spheres) and arcs (tubes: lp.Arc selections or (n, 3) points) drawn in a 3D view, `size`
    [m] across (default 2 % of what the view shows; the tubes a third of it): solid, they stay visible on a cut
    plane through them."""
    import pyvista as pv
    pts = np.zeros((0, 3)) if points is None else np.atleast_2d(np.asarray(points, float))
    lines = [np.asarray(a.points(3) if hasattr(a, "points") else a, float) for a in (arcs or ())]
    lines = [p[np.all(np.isfinite(p), axis=1)] for p in lines]
    if size is None:
        found = [np.asarray(ds.points) for ds, _, _ in v.items if ds.n_points] + [pts] + lines
        every = np.vstack([e for e in found if len(e)]) if any(len(e) for e in found) else np.zeros((1, 3))
        size = 0.02 * float(np.linalg.norm(every.max(axis=0) - every.min(axis=0)) or 1.0)
    if len(pts):
        v.add(pv.PolyData(pts).glyph(geom=pv.Sphere(radius=size / 2), scale=False, orient=False), color=color,
              smooth_shading=True, colorbar=False)
    for p in lines:
        if len(p) >= 2:
            v.add(pv.lines_from_points(p).tube(radius=size / 6), color=color, smooth_shading=True, colorbar=False)
    return v


def scene_view_one(res, f, camera=None, cut="driver", keep=None, range_db: float = 50.0,
                   contour_db: float | None = 3.0, near_field: float = 0.08, deformation: float = 0.10,
                   outline="auto", ground=None, size=(900, 760), cavity=None, surfaces=None, points=None,
                   arcs=None) -> view3d.View3D:
    """
    One view of a scene result as a View3D (SceneResults.view): camera = a default camera's name (default:
    "front left (cut side)", or the front without a cut; for another cut, a camera on its removed side) or an
    lp.plot.Camera; surfaces: the air mesh's surfaces coloured by the SPL (res.air_surfaces; default the ground,
    the walls and the openings); points, arcs: drawn (marks); other arguments as scene_figure.
    """
    cut = resolve_cut(res, cut)
    axis_x = cut is not None and np.allclose(np.asarray(cut[0], float), CUT_NORMALS["x"])
    if camera is None and cut is not None and not axis_x:
        camera = cut_camera(res, cut)
    if camera is None or isinstance(camera, str):
        name = camera or ("front left (cut side)" if cut is not None else "front right (outer walls)")
        camera = resolve_cameras(res, [name], cut)[name]
    if isinstance(outline, str) and outline == "auto":
        outline = _outline(res, cut) if cut is not None else None
    groups = tuple(surfaces) if surfaces is not None else ("ground", "walls", "openings")
    sc = scene_surfaces(res, f, cut, keep, near_field, groups=groups, cavity=cavity)
    clim = (sc["top"] - range_db, sc["top"])
    levels = np.arange(np.ceil(clim[0] / contour_db) * contour_db, clim[1] + 1e-9, contour_db) if contour_db else None
    v = _panel(sc, res.driver_surfaces(sc["f"], deformation), camera, clim, levels, ground, outline, size)
    n_items = len(v.items)
    marks(v, points, arcs)
    if len(v.items) > n_items and v.cam is not None:                 # back off until the marks are in view
        far = np.vstack([np.asarray(ds.points) for ds, _, _ in v.items[n_items:]])
        b = scene_bounds(res)
        every = np.vstack([b, far])
        pos, focus = np.asarray(v.cam[0], float), np.asarray(v.cam[1], float)
        grow = float(np.linalg.norm(every.max(axis=0) - every.min(axis=0)) / np.linalg.norm(b[1] - b[0]))
        if grow > 1.0:
            centre = 0.5 * (every.max(axis=0) + every.min(axis=0))
            v.cam = (tuple(centre + grow * (pos - focus)), tuple(centre), *v.cam[2:])
    return v


def wall_view(mesh, u, u_n, groups=None, show: str = "real", deformation: float = 0.05, label: str | None = None,
              camera=None, size=(900, 760), labels: bool | dict = True) -> view3d.View3D:
    """
    Shell walls (a TRI3 Mesh) deformed by the translations u (n, 3) and coloured by the normal displacement u_n (n,)
    re its largest |u_n|; complex values (a response) are drawn at the instant where that largest u_n is real: "real"
    (signed, −1 … 1) or "abs" (|u_n|, 0 … 1). The largest displacement is `deformation` × the walls' size; groups:
    the mesh groups drawn (None: "walls"); the outline of the walls at rest in thin grey. labels: each panel drawn
    named at its centre (a STEP solid "solid 4", a Box side "left" …; a dict {panel: name} names them yourself;
    False: none).
    """
    import pyvista as pv
    groups = ["walls"] if groups is None else list(groups)
    missing = [g for g in groups if g not in mesh.groups]
    if missing:
        raise ValueError(f"no group {missing} in the walls' mesh (groups: {list(mesh.groups)})")
    X = np.asarray(mesh.coords, float)
    u, u_n = np.asarray(u), np.asarray(u_n)
    big = int(np.argmax(np.abs(u_n)))
    phase = np.exp(-1j * np.angle(u_n[big])) if np.iscomplexobj(u_n) else 1.0
    ur, unr = np.real(u * phase), np.real(u_n * phase)
    top = float(np.abs(u_n).max()) or 1.0
    size_w = float(np.linalg.norm(X.max(axis=0) - X.min(axis=0)))
    amp = float(np.linalg.norm(ur, axis=1).max()) or 1.0
    moved = X + (deformation * size_w / amp) * ur
    vals, clim, cmap = (unr / top, (-1.0, 1.0), style.CMAPS["pressure"]) if show == "real" else \
        (np.abs(u_n) / top, (0.0, 1.0), style.CMAPS["displacement"])
    label = label or (r"$u_n / |u_n|_{max}$" if show == "real" else r"$|u_n| / |u_n|_{max}$")
    v = view3d.View3D(size)
    out = np.zeros(3)                                             # where the walls move most, seen from outside
    for g in groups:
        tris = view3d._rows(mesh, mesh.groups[g].cells)
        if not len(tris):
            continue
        cn = np.cross(X[tris[:, 1]] - X[tris[:, 0]], X[tris[:, 2]] - X[tris[:, 0]])   # n × 2 area, into the air
        out -= (np.abs(u_n)[tris].mean(axis=1)[:, None] * cn).sum(axis=0)
        faces = np.hstack([np.full((len(tris), 1), 3), tris]).ravel()
        rest = pv.PolyData(X, faces)
        v.add(rest.extract_feature_edges(feature_angle=35.0, boundary_edges=True, manifold_edges=False,
                                         non_manifold_edges=False), color="#9aa5ae", line_width=0.8, colorbar=False)
        surf = pv.PolyData(moved, faces)
        surf.point_data["u_n"] = vals
        v.add(surf, scalars="u_n", cmap=cmap, clim=clim, colorbar="shared", label=label, smooth_shading=True,
              ambient=0.45, diffuse=0.6, specular=0.0)
    if labels is not False:
        names = labels if isinstance(labels, dict) else {}
        pts, texts = [], []
        for g in mesh.groups:
            if not g.startswith("panel ") or not (g in groups or "walls" in groups):
                continue
            rows = np.unique(view3d._rows(mesh, mesh.groups[g].cells))
            key = g[6:]
            pts.append(moved[rows].mean(axis=0))
            texts.append(str(names.get(key, names.get(g, f"solid {key}" if key.isdigit() else key))))
        v.label(np.array(pts).reshape(-1, 3), texts)
    b = np.array([X.min(axis=0), X.max(axis=0)])
    if camera is None:
        cam = default_cameras(b)["front right (outer walls)"]
        if np.linalg.norm(out) > 1e-12:
            focus = np.asarray(cam.focus, float)
            d = out / np.linalg.norm(out) + np.array([0.25, 0.35, 0.25])
            camera = Camera(tuple(focus + np.linalg.norm(np.subtract(cam.position, focus)) * d / np.linalg.norm(d)),
                            tuple(focus), cam.up, view_angle=cam.view_angle)
        else:
            camera = cam
    v.camera(camera.position, camera.focus, camera.up, view_angle=camera.view_angle, parallel=camera.parallel)
    return v


def wall_mode_view(res, k: int, panels=None, show: str = "real", deformation: float = 0.05, **kwargs):
    """Wall mode k of the in-vacuo wall modes of a scene (lp.Modes(scene, physics="walls")): its shape deformed and
    coloured by the normal displacement (wall_view); panels: the panels drawn (None: all)."""
    u, u_n = res.field("u", index=k), res.field("u_n", index=k)
    mesh = u.mesh
    groups = None if panels is None else [f"panel {p}" if f"panel {p}" in mesh.groups else str(p) for p in panels]
    return wall_view(u.mesh, u.values, u_n.values, groups, show, deformation, **kwargs)


def wall_mode_panels(res) -> list[str]:
    """The panels of the wall modes' shell mesh (["walls"] for uniform walls)."""
    mesh = res.field("u_n", index=0).mesh
    found = [g[6:] for g in mesh.groups if g.startswith("panel ")]
    return found or ["walls"]


def balloon_view(res, f, radius: float | None = None, center=None, range_db: float = 40.0, n=(31, 73),
                 size=(900, 760)) -> view3d.View3D:
    """
    The directivity balloon of a scene result at the frequency nearest f: the SPL on the half sphere above the
    ground of `radius` around `center` (default: the DtN hemisphere's centre on the ground, radius max(1 m, 2 × the
    hemisphere's)), drawn as a surface whose distance to the centre is the level above top − range_db (the top
    1), coloured by the SPL; the drivers' axes drawn from the centre (their direction).
    """
    import pyvista as pv
    from loudpy.errors import ResultsError
    h = res.scene.get("hemisphere")
    if not h:
        raise ResultsError("a balloon needs the outside air (exterior / merged): no DtN hemisphere in these results")
    c = np.asarray(h["center"] if center is None else center, float)
    R = float(radius) if radius else max(1.0, 2.0 * float(h["radius"]))
    up = int(h["up"])
    e_up = np.eye(3)[up]
    e1, e2 = (np.eye(3)[k] for k in range(3) if k != up)
    th = np.linspace(0.0, np.pi / 2, n[0])
    ph = np.linspace(0.0, 2 * np.pi, n[1])
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    dirs = (np.cos(TH)[..., None] * e_up + (np.sin(TH) * np.cos(PH))[..., None] * e1
            + (np.sin(TH) * np.sin(PH))[..., None] * e2).reshape(-1, 3)
    i = int(np.argmin(np.abs(res.frequencies - f)))
    p = res._far_at(c + R * dirs, i)
    L = 20 * np.log10(np.maximum(np.abs(p) / np.sqrt(2), 1e-30) / 2e-5)
    top = float(L.max())
    rho = np.clip((L - (top - range_db)) / range_db, 0.0, 1.0)
    grid = pv.StructuredGrid(*(dirs * rho[:, None]).reshape(n[0], n[1], 3).transpose(2, 0, 1))
    grid.point_data["spl"] = L.reshape(n[0], n[1]).ravel(order="F")
    v = view3d.View3D(size)
    v.add(grid, scalars="spl", quantity="spl", clim=(top - range_db, top), smooth_shading=True, specular=0.2,
          label=f"SPL at {R:g} m [dB]")
    v.add(pv.Disc(inner=0.0, outer=1.05, normal=tuple(e_up), c_res=72), color="#9aa5ae", opacity=0.25,
          colorbar=False)                                         # the ground
    for d in res.scene.get("drivers", []):
        ax = np.asarray(d["rotation"], float)[:, 2] if "rotation" in d else np.array([0.0, 0.0, 1.0])
        v.add(pv.Arrow(start=(0.0, 0.0, 0.0), direction=ax / np.linalg.norm(ax), scale=1.15, tip_length=0.06,
                       shaft_radius=0.006, tip_radius=0.02), color="#22303c", colorbar=False)
    d = 0.9 * e1 + 0.7 * e_up + 1.2 * e2
    v.camera(tuple(2.8 * d / np.linalg.norm(d)), tuple(0.25 * e_up), tuple(e_up), view_angle=30.0, zoom=0.9)
    return v


@style.styled
def scene_figure(res, f, cameras=None, cut="driver", keep=None, range_db: float = 50.0,
                 contour_db: float | None = 3.0, near_field: float = 0.08, deformation: float = 0.10, outline="auto",
                 ground=None, size=(900, 760), ncols: int | None = None, title: str | None = None, cavity=None):
    """
    One figure, one panel per camera, of the SPL surfaces and deformed drivers of a scene result at the frequency
    nearest f (SceneResults.plot). cameras: None (the six default cameras), names of default cameras, or
    {name: lp.plot.Camera}; cut: "driver" (the vertical plane through the first driver's axis), None (the whole
    scene) or (normal, origin); outline: "auto" (the dashed corners of the removed part), None, or a pyvista
    dataset; ground: a dataset drawn grey under it all; keep(points) → bool trims the cut plane; cavity: also the
    pressure inside the enclosure (scene_surfaces), on the same scale.
    """
    cut = resolve_cut(res, cut)
    cameras = resolve_cameras(res, cameras, cut)
    if isinstance(outline, str) and outline == "auto":
        outline = _outline(res, cut) if cut is not None else None
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
        v = _panel(sc, drivers, cam, clim, levels, ground, outline, size)
        v.show(ax, title=name, colorbars=False)
    view3d.colorbars_for(fig, axs, v.bars, shrink=0.5)
    head = f"{sc['f']:.0f} Hz: SPL top {sc['top']:.0f} dB beyond {100 * near_field:.0f} cm of the drivers"
    if contour_db:
        head += f", iso-lines every {contour_db:g} dB"
    fig.suptitle(head if title is None else f"{title} — {head}", fontsize=9)
    return fig
