"""
3D geometry builders (gmsh OCC) and P2 mesh I/O for the 3D coupled models.

  - driver cut-outs in an enclosure (add_driver_cutout, carve_drivers) and in the exterior air
    (add_driver_front), placed by a Frame (any orientation);
  - classification of the boundary surfaces per driver (classify_driver_faces);
  - meshing with named physical groups (mesh_cavity, mesh_exterior, mesh_volume);
  - reading a P2 tetrahedral mesh with its named surfaces (read_p2_mesh);
  - sampling the wet profile of a driver from its 2D meridian geometry (sample_profile).
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

import numpy as np

from loudpy.geometry.frame import Frame, as_frame, _is_single
from loudpy.geometry.gmsh_session import gmsh_module as _gmsh


def sample_profile(geo_path, curves, start, n_per_curve=40):
    """
    Chained polylines (r, z) of the curves of a 2D (meridian) geometry, from the point `start`.
    Returns one (n, 2) array per curve, each starting where the previous one ends.
    """
    gmsh = _gmsh()
    gmsh.clear()
    gmsh.open(str(geo_path))
    segs = []
    for t in curves:
        a, b = gmsh.model.getParametrizationBounds(1, t)
        u = np.linspace(a[0], b[0], n_per_curve)
        segs.append(np.array(gmsh.model.getValue(1, t, u)).reshape(-1, 3)[:, :2])
    gmsh.clear()
    out, last = [], np.asarray(start, float)
    for t, p in zip(curves, segs):
        if np.linalg.norm(p[-1] - last) < np.linalg.norm(p[0] - last):
            p = p[::-1]
        if np.linalg.norm(p[0] - last) > 1e-5:
            raise ValueError(f"curve {t} does not start at {last} (gap {np.linalg.norm(p[0] - last):.2e})")
        p[0] = last
        out.append(p)
        last = p[-1]
    return out


def add_driver_cutout(segments, center=(0.0, 0.0), z_top=None):
    """
    In the current gmsh OCC model, the two solids that turn a box into the rear cavity of the
    driver (axis along z through `center`, profile in (r, z); `center` may be a Frame,
    then everything is built in its local frame, and r_c, z_b are local):

      E : the driver and the air in front of its wet profile (region above the profile,
          up to z_top) — to be cut from the cavity;
      C : the cylinder r ≤ R_c between the baffle plane and z_top — to be fused with the box,
          so that the pockets of the profile above the baffle (e.g. under a surround roll)
          belong to the cavity.

    segments : chained (r, z) polylines of the wet profile, from the baffle (first point,
               r = R_c, z = z_b) to the axis (last point, r = 0). Straight polylines become
               lines, the others splines.
    Returns (E, C, R_c, z_b) with E, C volume tags.
    """
    gmsh = _gmsh()
    occ = gmsh.model.occ
    frame = as_frame(center)
    x0, y0, z0 = frame.origin if frame.is_identity else (0.0, 0.0, 0.0)
    pts = np.vstack([segments[0][:1]] + [s[1:] for s in segments])
    r_c, z_b = pts[0]
    if abs(pts[-1, 0]) > 1e-9:
        raise ValueError("the wet profile must end on the axis (r = 0)")
    z_top = float(z_top if z_top is not None else pts[:, 1].max() + 0.01)

    def P(r, z):
        return occ.addPoint(x0 + float(r), y0, z0 + float(z))

    def curve(p, tags):
        ok = np.allclose(np.cross(p[-1] - p[0], p - p[0]), 0.0, atol=1e-12)
        return occ.addLine(tags[0], tags[-1]) if ok else occ.addSpline(tags)

    # E: start after the leading points on the cylinder r = R_c (they bound C, not E)
    lead = 0
    while lead + 1 < len(pts) and abs(pts[lead + 1, 0] - r_c) < 1e-9:
        lead += 1
    lines, first, cur = [], None, None
    count = 0
    for s in segments:
        keep = [j for j in range(len(s)) if count + j >= lead]
        count += len(s) - 1
        if len(keep) < 2:
            continue
        p = s[keep]
        tags = [cur if (cur is not None and k == 0) else P(*q) for k, q in enumerate(p)]
        if first is None:
            first = tags[0]
        lines.append(curve(p, tags))
        cur = tags[-1]
    top_axis = P(0.0, z_top)
    top_out = P(r_c, z_top)
    lines += [occ.addLine(cur, top_axis), occ.addLine(top_axis, top_out), occ.addLine(top_out, first)]
    face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    rev = occ.revolve([(2, face)], x0, y0, z0, 0.0, 0.0, 1.0, 2 * np.pi)
    E = [t for d, t in rev if d == 3][0]
    C = occ.addCylinder(x0, y0, z0 + z_b, 0.0, 0.0, z_top - z_b, r_c)
    if not frame.is_identity:
        frame.place([(3, E), (3, C)])
    return E, C, float(r_c), float(z_b)


def carve_drivers(box, cutouts, extra_cuts=()):
    """
    Cavity of a box with one or several drivers cut out: (box − ∪E_j − extra) ∪ (∪C_j − ∪E_j),
    with cutouts the (E, C, …) tuples of add_driver_cutout and box / extra_cuts volume tags.
    Cutting before fusing is robust when the baffle is a slanted face (fusing C onto it first
    can make the OCC cut fail). Returns the list of volume tags.
    """
    occ = _gmsh().model.occ
    E = [(3, c[0]) for c in cutouts]
    a, _ = occ.cut([(3, box)], E + [(3, t) for t in extra_cuts], removeTool=False)
    b, _ = occ.cut([(3, c[1]) for c in cutouts], E)
    vol, _ = occ.fuse(a, b)
    occ.synchronize()
    return [t for d, t in vol if d == 3]


def _local_extent(tag, frame, n_per_curve=24):
    """
    (r_max, z_min, z_max) of a surface in a driver frame. Axis-aligned frames: from the
    bounding box (r_max is its half-width, as in the single-driver code); otherwise from points
    sampled on the boundary curves.
    """
    gmsh = _gmsh()
    if frame.axis_aligned:
        lo_hi = np.array(gmsh.model.getBoundingBox(2, tag)).reshape(2, 3)
        corners = np.array([[lo_hi[i, 0], lo_hi[j, 1], lo_hi[k, 2]]
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        xl = frame.to_local(corners)
        lo, hi = xl.min(0), xl.max(0)
        return max(hi[0], -lo[0], hi[1], -lo[1]), lo[2], hi[2]
    pts = [np.array(gmsh.model.occ.getCenterOfMass(2, tag))]
    for _, c in gmsh.model.getBoundary([(2, tag)], oriented=False):
        a, b = gmsh.model.getParametrizationBounds(1, abs(c))
        u = np.linspace(a[0], b[0], n_per_curve)
        pts.append(np.array(gmsh.model.getValue(1, abs(c), u)).reshape(-1, 3))
    xl = frame.to_local(np.vstack(pts))
    return np.hypot(xl[:, 0], xl[:, 1]).max(), xl[:, 2].min(), xl[:, 2].max()


def classify_driver_faces(tags, frames, r_max, z_plane, rtol=1e-4, atol=1e-9, extra_planes=()):
    """
    For each surface tag, the index of the first driver whose cylinder r_l ≤ r_max[j] contains
    it, except the flat faces z_l = z_plane[j] (baffle or box face) and z_l = each of
    `extra_planes` (e.g. both faces of the baffle in a merged air domain), else −1.
    """
    frames = [as_frame(f) for f in frames]
    r_max = np.broadcast_to(np.asarray(r_max, float), (len(frames),))
    z_plane = np.broadcast_to(np.asarray(z_plane, float), (len(frames),))
    out = []
    for t in tags:
        owner = -1
        for j, fr in enumerate(frames):
            rm, zmin, zmax = _local_extent(t, fr)
            planes = (z_plane[j],) + tuple(extra_planes)
            flat = abs(zmax - zmin) < atol and min(abs(zmax - z) for z in planes) < atol
            if rm <= r_max[j] * (1 + rtol) + atol and not flat:
                owner = j
                break
        out.append(owner)
    return np.array(out, int)


def mesh_cavity(msh_path, cavity_volume, center, r_c, z_b, h_near, h_far, d_near=0.005,
                d_far=0.06):
    """
    Mesh the cavity volume (P2 tetrahedra, straight edges) and write it with the physical groups
    "cavity" (volume), "interface_driver" (wet surface: boundary faces within r ≤ R_c of the
    driver axis, off the baffle plane) and "walls" (the rest).

    Several drivers: center, r_c and z_b are lists (one center or Frame per driver; r_c
    and z_b may be scalars); the wet surfaces are "interface_driver_0", "interface_driver_1", …
    and the first return value is one list of surface tags per driver.
    """
    gmsh = _gmsh()
    occ = gmsh.model.occ
    occ.synchronize()
    single = _is_single(center)
    frames = [center] if single else list(center)
    bnd = [t for d, t in gmsh.model.getBoundary([(3, cavity_volume)], oriented=False)]
    owner = classify_driver_faces(bnd, frames, r_c, z_b)
    drivers = [[t for t, o in zip(bnd, owner) if o == j] for j in range(len(frames))]
    walls = [t for t, o in zip(bnd, owner) if o < 0]
    for j, d in enumerate(drivers):
        if not d:
            raise ValueError(f"no wet surface found for driver {j} ({as_frame(frames[j])})")
    gmsh.model.addPhysicalGroup(3, [cavity_volume], 1, name="cavity")
    if single:
        gmsh.model.addPhysicalGroup(2, drivers[0], 2, name="interface_driver")
    else:
        for j, d in enumerate(drivers):
            gmsh.model.addPhysicalGroup(2, d, 10 + j, name=f"interface_driver_{j}")
    gmsh.model.addPhysicalGroup(2, walls, 3, name="walls")
    driver = drivers[0] if single else drivers

    f = gmsh.model.mesh.field
    fd = f.add("Distance")
    f.setNumbers(fd, "SurfacesList", [t for d in drivers for t in d])
    f.setNumber(fd, "Sampling", 60)
    ft = f.add("Threshold")
    f.setNumber(ft, "InField", fd)
    for k, v in (("SizeMin", h_near), ("SizeMax", h_far), ("DistMin", d_near), ("DistMax", d_far)):
        f.setNumber(ft, k, v)
    f.setAsBackgroundMesh(ft)
    for k, v in (("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeFromPoints", 0),
                 ("Mesh.MeshSizeFromCurvature", 0), ("Mesh.ElementOrder", 2),
                 ("Mesh.SecondOrderLinear", 1), ("Mesh.Algorithm3D", 10)):
        gmsh.option.setNumber(k, v)
    gmsh.model.mesh.generate(3)
    Path(msh_path).parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(msh_path))
    return driver, walls


def read_p2_mesh(msh_path, surfaces):
    """
    Nodes (used by the tetrahedra), TET10 connectivity and, for each physical surface name in
    `surfaces`, its TRI6 faces — from a gmsh mesh (second order).
    """
    gmsh = _gmsh()
    gmsh.clear()
    gmsh.open(str(msh_path))
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(tags)
    tags, xyz = tags[order], xyz.reshape(-1, 3)[order]
    row = lambda t: np.searchsorted(tags, t)
    _, _, enodes = gmsh.model.mesh.getElements(3)
    tets = row(np.concatenate(enodes).astype(np.int64)).reshape(-1, 10)
    names = {gmsh.model.getPhysicalName(2, t): t for _, t in gmsh.model.getPhysicalGroups(2)}
    wanted = []
    for name in surfaces:                         # glob patterns ("interface_driver_*") expand
        if any(ch in name for ch in "*?["):
            wanted += sorted(fnmatch.filter(names, name))
        else:
            wanted.append(name)
    faces = {}
    for name in dict.fromkeys(wanted):
        if name not in names:
            faces[name] = np.zeros((0, 6), np.int64)
            continue
        tris = []
        for ent in gmsh.model.getEntitiesForPhysicalGroup(2, names[name]):
            types, _, en = gmsh.model.mesh.getElements(2, ent)
            tris += [row(nn.astype(np.int64)).reshape(-1, 6) for ty, nn in zip(types, en) if ty == 9]
        faces[name] = np.vstack(tris)
    gmsh.clear()
    used = np.unique(tets)
    remap = -np.ones(len(tags), np.int64)
    remap[used] = np.arange(len(used))
    return xyz[used], remap[tets], {k: remap[v] for k, v in faces.items()}


def add_driver_front(segments, center, z_low):
    """
    In the current gmsh OCC model, the solid under the wet front profile of the driver
    (revolved about z through `center`) between z_low and the profile, and the cylinder
    r ≤ R_f, z_low ≤ z ≤ z_front that it replaces in the box. segments: chained (r, z)
    polylines from the box front face (first point, r = R_f, z = z_front) to the axis.
    `center` may be a Frame: then everything is built in its local frame (z_low, R_f,
    z_front are local). Returns (rev, cyl, R_f, z_front).
    """
    gmsh = _gmsh()
    occ = gmsh.model.occ
    frame = as_frame(center)
    x0, y0, z0 = frame.origin if frame.is_identity else (0.0, 0.0, 0.0)
    r_f, z_f = segments[0][0]
    P = lambda r, z: occ.addPoint(x0 + float(r), y0, z0 + float(z))
    lines, first, cur = [], None, None
    for s in segments:
        tags = [cur if (cur is not None and k == 0) else P(*q) for k, q in enumerate(s)]
        first = first or tags[0]
        ok = np.allclose(np.cross(s[-1] - s[0], s - s[0]), 0.0, atol=1e-12)
        lines.append(occ.addLine(tags[0], tags[-1]) if ok else occ.addSpline(tags))
        cur = tags[-1]
    a, b = P(0.0, z_low), P(r_f, z_low)
    lines += [occ.addLine(cur, a), occ.addLine(a, b), occ.addLine(b, first)]
    face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    rev = [t for d, t in occ.revolve([(2, face)], x0, y0, z0, 0, 0, 1, 2 * np.pi) if d == 3][0]
    cyl = occ.addCylinder(x0, y0, z0 + z_low, 0, 0, z_f - z_low, r_f)
    if not frame.is_identity:
        frame.place([(3, rev), (3, cyl)])
    return rev, cyl, float(r_f), float(z_f)


def mesh_exterior(msh_path, volume, dtn: HalfSpaceDtN, center, r_f, z_front, h_near, h_far,
                  d_near=0.004, d_far=0.08):
    """
    Mesh the exterior (P2, straight edges) with the physical surfaces "dtn" (hemisphere),
    "ground", "interface_front" (wet front of the driver) and "walls" (box).

    Several drivers: center, r_f and z_front are lists (one center or Frame per driver;
    r_f and z_front may be scalars); the wet fronts are "interface_front_0", "interface_front_1", …
    """
    gmsh = _gmsh()
    gmsh.model.occ.synchronize()
    single = _is_single(center)
    frames = [center] if single else list(center)
    fronts = ["interface_front"] if single else [f"interface_front_{j}" for j in range(len(frames))]
    groups = {"dtn": [], "ground": [], **{nm: [] for nm in fronts}, "walls": []}
    rest = []
    for _, t in gmsh.model.getBoundary([(3, volume)], oriented=False):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, t)
        lo, hi = (xmin, ymin, zmin), (xmax, ymax, zmax)
        if gmsh.model.getType(2, t) == "Sphere":
            groups["dtn"].append(t)
        elif abs(hi[dtn.up] - lo[dtn.up]) < 1e-6 and abs(hi[dtn.up] - dtn.center[dtn.up]) < 1e-6:
            groups["ground"].append(t)
        else:
            rest.append(t)
    owner = classify_driver_faces(rest, frames, r_f, z_front, atol=1e-6)
    for t, o in zip(rest, owner):
        groups[fronts[o] if o >= 0 else "walls"].append(t)
    for nm in fronts:
        if not groups[nm]:
            raise ValueError(f"no wet surface found for {nm}")
    gmsh.model.addPhysicalGroup(3, [volume], 1, name="air")
    for k, (name, ts) in enumerate(groups.items(), start=2):
        gmsh.model.addPhysicalGroup(2, ts, k, name=name)
    f = gmsh.model.mesh.field
    fd = f.add("Distance")
    f.setNumbers(fd, "SurfacesList", [t for nm in fronts for t in groups[nm]])
    f.setNumber(fd, "Sampling", 60)
    ft = f.add("Threshold")
    f.setNumber(ft, "InField", fd)
    for k, v in (("SizeMin", h_near), ("SizeMax", h_far), ("DistMin", d_near), ("DistMax", d_far)):
        f.setNumber(ft, k, v)
    f.setAsBackgroundMesh(ft)
    for k, v in (("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeFromPoints", 0),
                 ("Mesh.MeshSizeFromCurvature", 0), ("Mesh.ElementOrder", 2),
                 ("Mesh.SecondOrderLinear", 1), ("Mesh.Algorithm3D", 10)):
        gmsh.option.setNumber(k, v)
    gmsh.model.mesh.generate(3)
    Path(msh_path).parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(msh_path))
    return {k: len(v) for k, v in groups.items()}


def mesh_volume(msh_path, volume, classify, refine, h_far):
    """
    Mesh one air volume (P2, straight edges). classify(surface tag) → physical surface name;
    refine: {name: (h, d_near, d_far)} size ramps from the surfaces of that name up to h_far.
    Returns {name: number of surfaces}.
    """
    gmsh = _gmsh()
    gmsh.model.occ.synchronize()
    groups = {}
    for _, t in gmsh.model.getBoundary([(3, volume)], oriented=False):
        groups.setdefault(classify(t), []).append(t)
    gmsh.model.addPhysicalGroup(3, [volume], 1, name="air")
    for k, (name, ts) in enumerate(sorted(groups.items()), start=2):
        gmsh.model.addPhysicalGroup(2, ts, k, name=name)
    f = gmsh.model.mesh.field
    fields = []
    for name, (h, d0, d1) in refine.items():
        if name not in groups:
            continue
        fd = f.add("Distance")
        f.setNumbers(fd, "SurfacesList", groups[name])
        f.setNumber(fd, "Sampling", 60)
        ft = f.add("Threshold")
        f.setNumber(ft, "InField", fd)
        for k, v in (("SizeMin", h), ("SizeMax", h_far), ("DistMin", d0), ("DistMax", d1)):
            f.setNumber(ft, k, v)
        fields.append(ft)
    fm = f.add("Min")
    f.setNumbers(fm, "FieldsList", fields)
    f.setAsBackgroundMesh(fm)
    for k, v in (("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeFromPoints", 0),
                 ("Mesh.MeshSizeFromCurvature", 0), ("Mesh.MeshSizeMax", h_far),
                 ("Mesh.ElementOrder", 2), ("Mesh.SecondOrderLinear", 1), ("Mesh.Algorithm3D", 10)):
        gmsh.option.setNumber(k, v)
    gmsh.model.mesh.generate(3)
    Path(msh_path).parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(msh_path))
    return {k: len(v) for k, v in groups.items()}
