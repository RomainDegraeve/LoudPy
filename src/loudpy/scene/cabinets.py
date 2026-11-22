"""
Ready-made cabinets for lp.Scene: a parametric box, or your own CAD file.

    box = lp.cabinets.Box(width=0.20, height=0.30, depth=0.18, depth_top=0.11,          # slanted back
                          wall=0.015, braces=[lp.cabinets.Brace(y=0.08, thickness=0.015, depth=0.08)],
                          ports=[lp.cabinets.Port(at=(0.06, 0.06), radius=0.02, length=0.062)])
    cad = lp.cabinets.Step("my_box.step", openings=[lp.cabinets.Port(at=(0.06, 0.06), radius=0.02,
                                                                     length=0.062, wall=0)])
    scene = lp.Scene.vented_box(box, driver, at=(0.125, 0.19), f_max=3000)      # or lp.Scene(box, [...])

Frame (both): the drivers' axis is +z (the front faces +z), y is up, x across; the box stands on the ground
y = 0. The drivers set the front: the inside air ends on their rear baffle plane and the front face of the
box is their front plane (Driver.z_baffle, Driver.z_front), so a box needs no front coordinate.

Box: inner dimensions width (x, from 0), height (y, from `bottom`), depth (z, back to front at the bottom),
depth_top (at the top: slanted back; default = depth); `wall` the thickness of the sides, top and back (0: the
outer box is the inside up to the front face), `bottom` that of the bottom (default 0: the inside air stands on
the ground). Braces: horizontal shelves across the width, from the back wall; ports: round tubes through the
front, `length` from the front face inwards, tube wall `wall` inside the box (0: a plain hole).

Step: a CAD file (STEP / IGES / BREP, any unit: read in metres) of the cabinet as built — walls, panels,
braces, port tubes, as one or several solids, in the frame above (translate= / scale= otherwise). The inside
air is found by loudpy: the closed void enclosed by the solids (and the ground y = `ground` when the box has no
bottom, as a box standing on the floor of the scene; ground=None: the file must be closed). Driver holes are closed at the drivers'
positions (a plug between their baffle and front planes); port bores must be named, as openings= (an
lp.cabinets.Port, or a point inside a solid of the file that is the bore's air). Checked when meshing: one
closed inside void (else the message says what is open), every driver's baffle on the inside air and its front
on the outside.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from loudpy.errors import MeshError, ModelError

__all__ = ["Box", "Brace", "Port", "Step", "volume"]


@dataclass(frozen=True)
class Brace:
    """A horizontal brace across the inside: at height y (from the inner bottom), thickness, depth from the back."""
    y: float
    thickness: float = 0.015
    depth: float = 0.08

    def __post_init__(self):
        from loudpy.model.parameters import has_references
        if has_references(self):                                  # lp.P: checked once resolved
            return
        for name in ("thickness", "depth"):
            if not getattr(self, name) > 0:
                raise ModelError(f"lp.cabinets.Brace: {name} must be > 0 m, got {getattr(self, name)!r}")


@dataclass(frozen=True)
class Port:
    """A round port through the front face: axis at (x, y), radius, length from the front face inwards, tube
    wall thickness inside the box (0: no tube)."""
    at: tuple
    radius: float
    length: float
    wall: float = 0.003

    def __post_init__(self):
        from loudpy.model.parameters import has_references
        if has_references(self):                                  # lp.P: checked once resolved
            return
        for name in ("radius", "length"):
            if not getattr(self, name) > 0:
                raise ModelError(f"lp.cabinets.Port: {name} must be > 0 m, got {getattr(self, name)!r}")
        if not self.wall >= 0:
            raise ModelError(f"lp.cabinets.Port: wall must be ≥ 0 m, got {self.wall!r}")

    def tuning(self, volume: float, c: float | None = None) -> float:
        """Helmholtz estimate f_b = c/2π √(S / (V L')), L' = L + 0.85a + 0.6a (one flanged, one free end) [Hz];
        c: speed of sound (default: the library's Air)."""
        if c is None:
            from loudpy.model.materials import materials
            c = float(materials["Air"].c)
        S = np.pi * self.radius ** 2
        return float(c / (2 * np.pi) * np.sqrt(S / (volume * (self.length + 1.45 * self.radius))))

    def _bore(self, occ, z_out):
        x, y = self.at
        return occ.addCylinder(x, y, z_out - self.length, 0, 0, self.length, self.radius)

    def _tube(self, occ, z_out, z_in):
        """The tube wall inside the box (z_out − length → z_in), or None."""
        z0 = z_out - self.length
        if self.wall <= 0 or z0 >= z_in:
            return None
        x, y = self.at
        t_out = occ.addCylinder(x, y, z0, 0, 0, z_in - z0, self.radius + self.wall)
        t_in = occ.addCylinder(x, y, z0, 0, 0, z_in - z0, self.radius)
        wall, _ = occ.cut([(3, t_out)], [(3, t_in)])
        return [t for d, t in wall if d == 3]


def _front_planes(drivers):
    """(rear baffle z, front z) shared by the drivers (axis +z), from their profiles."""
    if not drivers:
        raise ModelError("a ready-made cabinet takes its front from the drivers: place at least one")
    zb, zf = set(), set()
    for d in drivers:
        R = np.asarray(d.frame.rotation, float)
        if not np.allclose(R, np.eye(3), atol=1e-9):
            raise ModelError(f"driver {d.name!r}: ready-made cabinets have the drivers on the front face "
                             f"(axis +z); write a cabinet builder for other orientations")
        z0 = float(np.asarray(d.frame.origin, float)[2])
        zb.add(round(z0 + d.driver.z_baffle, 9))
        zf.add(round(z0 + d.driver.z_front, 9))
    if len(zb) > 1 or len(zf) > 1:
        raise ModelError(f"the drivers' baffle / front planes differ ({sorted(zb)} / {sorted(zf)}): one front face")
    return zb.pop(), zf.pop()


@dataclass(frozen=True)
class Box:
    """A parametric box (module docstring). Lengths in m."""
    width: float
    height: float
    depth: float
    depth_top: float | None = None
    wall: float = 0.0
    bottom: float = 0.0
    braces: tuple = ()
    ports: tuple = ()
    port_size: float | None = 0.005          # mesh size on the walls around the ports (merged air)
    front: float | None = None               # inside front plane z (set from the drivers by lp.Scene)
    front_face: float | None = None          # outer front face z (set from the drivers by lp.Scene)

    def __post_init__(self):
        from loudpy.model.parameters import has_references
        object.__setattr__(self, "braces", tuple(self.braces))
        object.__setattr__(self, "ports", tuple(self.ports))
        if has_references(self):                                  # lp.P: checked once resolved
            return
        if self.port_size is not None and not self.port_size > 0:
            raise ModelError(f"lp.cabinets.Box: port_size must be > 0 m (or None), got {self.port_size!r}")
        for name in ("width", "height", "depth"):
            if not getattr(self, name) > 0:
                raise ModelError(f"lp.cabinets.Box: {name} must be > 0 m, got {getattr(self, name)!r}")
        if self.depth_top is not None and not self.depth_top > 0:
            raise ModelError("lp.cabinets.Box: depth_top must be > 0 m")
        if self.wall < 0 or self.bottom < 0:
            raise ModelError("lp.cabinets.Box: wall and bottom thicknesses are ≥ 0")
        for b in self.braces:
            if not (0 <= b.y and b.y + b.thickness <= self.height):
                raise ModelError(f"lp.cabinets.Box: brace at y = {b.y} m is outside the inner height")
        for p in self.ports:
            x, y = p.at
            if not (p.radius <= x <= self.width - p.radius and self.bottom + p.radius <= y + 1e-12
                    and y <= self.bottom + self.height - p.radius):
                raise ModelError(f"lp.cabinets.Box: port at {p.at} (radius {p.radius} m) is not on the front face")

    # the front planes come from the drivers
    def for_drivers(self, drivers) -> "Box":
        """The box with its front planes taken from the placed drivers (called by lp.Scene); the drivers must lie on
        the front face and overlap neither each other nor a port."""
        zb, zf = _front_planes(drivers)
        self._check_drivers(drivers)
        for given, z, what in ((self.front, zb, "front (rear baffle plane)"), (self.front_face, zf, "front_face")):
            if given is not None and abs(given - z) > 1e-9:
                raise ModelError(f"lp.cabinets.Box: {what} = {given} m, but the drivers' plane is z = {z} m")
        return replace(self, front=zb, front_face=zf)

    def _check_drivers(self, drivers):
        from loudpy.model.parameters import P
        if any(isinstance(getattr(self, n), P) for n in ("width", "height", "bottom")):
            return                                                # the face is a parameter: checked once resolved
        ports = [p for p in self.ports if not any(isinstance(v, P) for v in (p.radius, p.wall, *p.at))]
        circles = []
        for d in drivers:
            r = float(max(d.driver.r_front if d.driver.front else 0.0, d.driver.r_rear))
            c = np.asarray(d.frame.origin, float)[:2]
            if c[0] - r < -1e-12 or c[0] + r > self.width + 1e-12 or c[1] - r < self.bottom - 1e-12 \
                    or c[1] + r > self.bottom + self.height + 1e-12:
                raise ModelError(f"lp.cabinets.Box: driver {d.name!r} (radius {r * 1e3:.0f} mm at x = {c[0]:g}, "
                                 f"y = {c[1]:g} m) is not on the front face ({self.width:g} × {self.height:g} m)")
            circles.append((d.name, c, r))
        for i, (a, ca, ra) in enumerate(circles):
            for b, cb, rb in circles[i + 1:]:
                if np.linalg.norm(ca - cb) < ra + rb - 1e-12:
                    raise ModelError(f"lp.cabinets.Box: drivers {a!r} and {b!r} overlap")
            for p in ports:
                if np.linalg.norm(ca - np.asarray(p.at, float)) < ra + p.radius + p.wall - 1e-12:
                    raise ModelError(f"lp.cabinets.Box: driver {a!r} overlaps the port at {p.at}")

    def _planes(self):
        if self.front is None or self.front_face is None:
            raise ModelError("lp.cabinets.Box: no front plane — use it in an lp.Scene (the drivers set it), or "
                             "give front= and front_face=")
        return self.front, self.front_face

    def back_z(self, y):
        """z of the inner back wall at height y (slanted when depth_top is set)."""
        zb, _ = self._planes()
        top = self.depth if self.depth_top is None else self.depth_top
        return zb - self.depth + (self.depth - top) * (np.asarray(y, float) - self.bottom) / self.height

    def _prism(self, occ, x0, x1, y0, y1, z_front, back_offset):
        pts = [(y0, z_front), (y0, self.back_z(y0) - back_offset), (y1, self.back_z(y1) - back_offset), (y1, z_front)]
        p = [occ.addPoint(x0, float(y), float(z)) for y, z in pts]
        loop = occ.addCurveLoop([occ.addLine(p[i], p[(i + 1) % 4]) for i in range(4)])
        return [t for d, t in occ.extrude([(2, occ.addPlaneSurface([loop]))], x1 - x0, 0, 0) if d == 3][0]

    def __call__(self, occ):
        """The lp.Cabinet of the box in the current gmsh OCC model."""
        from loudpy.scene.scene import Cabinet
        zb, zf = self._planes()
        b, t, H = self.bottom, self.wall, self.height
        inner = self._prism(occ, 0.0, self.width, b, b + H, zb, 0.0)
        outer = self._prism(occ, -t, self.width + t, 0.0, b + H + t, zf, t)
        cuts = [occ.addBox(0.0, b + br.y, float(self.back_z(b + br.y + br.thickness)) - 0.01, self.width,
                           br.thickness, br.depth + 0.01) for br in self.braces]
        openings = []
        for p in self.ports:
            tube = p._tube(occ, zf, zb)
            cuts += tube or []
            openings.append(p._bore(occ, zf))
        return Cabinet(inner=inner, cuts=cuts, outer=outer, openings=openings,
                       opening_size=self.port_size if self.ports else None)

    def corners(self, outer: bool = True) -> np.ndarray:
        """The 8 corners of the outer (or inner) box (x, y, z)."""
        zb, zf = self._planes()
        if outer:
            t = self.wall
            xs, ys, zfront, off = (-t, self.width + t), (0.0, self.bottom + self.height + t), zf, t
        else:
            xs, ys, zfront, off = (0.0, self.width), (self.bottom, self.bottom + self.height), zb, 0.0
        return np.array([(x, y, z) for x in xs for y in ys for z in (zfront, float(self.back_z(y)) - off)])

    def signature(self) -> str:
        """What the meshes depend on."""
        return "lp.cabinets.Box/1 " + repr(self)

    def describe(self) -> str:
        """A short description (scene tree)."""
        slant = "" if self.depth_top is None else f"–{self.depth_top:g}"
        extra = [f"walls {self.wall:g} m"] if self.wall else []
        extra += [f"{len(self.braces)} brace(s)"] if self.braces else []
        extra += [f"{len(self.ports)} port(s)"] if self.ports else []
        return (f"Box {self.width:g} × {self.height:g} × {self.depth:g}{slant} m inside"
                + (f" ({', '.join(extra)})" if extra else ""))


@dataclass(frozen=True)
class Step:
    """A cabinet from a CAD file (module docstring)."""
    path: str | Path
    openings: tuple = ()                     # lp.cabinets.Port, or points (x, y, z) inside bore solids of the file
    inner: tuple | None = None               # a point of the inside air, when the solids enclose several voids
    scale: float = 1.0                       # applied after reading in metres (e.g. a file drawn at 1:10)
    translate: tuple = (0.0, 0.0, 0.0)       # applied after the scale
    port_size: float | None = 0.005
    plug_drivers: bool = True                # close the driver holes at the drivers' positions
    ground: float | None = 0.0               # y of a rigid ground closing the bottom (None: the file is closed)
    drivers: tuple = field(default=(), repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "openings", tuple(self.openings))
        if isinstance(self.scale, (int, float)) and not self.scale > 0:
            raise ModelError(f"lp.cabinets.Step: scale must be > 0, got {self.scale!r}")
        if isinstance(self.port_size, (int, float)) and not self.port_size > 0:
            raise ModelError(f"lp.cabinets.Step: port_size must be > 0 m (or None), got {self.port_size!r}")
        if not self.path.exists():
            raise ModelError(f"lp.cabinets.Step: {self.path} not found")
        if self.path.suffix.lower() not in (".step", ".stp", ".iges", ".igs", ".brep", ".brp"):
            raise ModelError(f"lp.cabinets.Step: {self.path.name} is not a STEP / IGES / BREP file")

    def for_drivers(self, drivers) -> "Step":
        """The cabinet with the placed drivers (their holes are plugged; called by lp.Scene)."""
        return replace(self, drivers=tuple(drivers))

    def _front(self):
        return _front_planes(self.drivers) if self.drivers else (None, None)

    def __call__(self, occ):
        """The lp.Cabinet in the current gmsh OCC model: the inside void of the file's solids (module docstring)."""
        from loudpy.geometry.gmsh_session import gmsh_module
        from loudpy.scene.scene import Cabinet
        g = gmsh_module()
        g.option.setString("Geometry.OCCTargetUnit", "M")
        solids = [t for d, t in occ.importShapes(str(self.path), highestDimOnly=True) if d == 3]
        if not solids:
            raise MeshError(f"{self.path.name}: no solid (the cabinet must be solids: walls, panels, tubes)")
        dims = [(3, t) for t in solids]
        if self.scale != 1.0:
            occ.dilate(dims, 0.0, 0.0, 0.0, self.scale, self.scale, self.scale)
        if any(self.translate):
            occ.translate(dims, *map(float, self.translate))
        occ.synchronize()
        zb, zf = self._front()
        bores, picked = [], set()
        for o in self.openings:
            if isinstance(o, Port):
                if zf is None:
                    raise ModelError("lp.cabinets.Step: a Port opening needs the drivers (the front face plane)")
                bores.append(o._bore(occ, zf))
            else:
                hit = [t for t in solids if t not in picked and g.model.isInside(3, t, list(map(float, o)))]
                if len(hit) != 1:
                    raise ModelError(f"lp.cabinets.Step: the opening point {tuple(o)} is inside {len(hit)} solids of "
                                     f"{self.path.name} (one expected: the bore's air)")
                picked.add(hit[0])
                bores.append(hit[0])
        walls = [t for t in solids if t not in picked]
        plugs = []
        if self.plug_drivers:
            for d in self.drivers:
                r = max(d.driver.r_rear, d.driver.r_front)
                a = d.frame.to_global(np.array([[0.0, 0.0, d.driver.z_baffle]]))[0]
                h = d.driver.z_front - d.driver.z_baffle
                if h > 0:
                    plugs.append(occ.addCylinder(*a, 0.0, 0.0, h, r))
        occ.synchronize()
        tools = walls + bores + plugs
        lo_hi = np.array([g.model.getBoundingBox(3, t) for t in tools]).reshape(-1, 2, 3)
        lo, hi = lo_hi[:, 0].min(axis=0), lo_hi[:, 1].max(axis=0)
        m = 0.05 * float(np.linalg.norm(hi - lo))
        box = occ.addBox(*(lo - m), *(hi - lo + 2 * m))
        closers = []                                              # the ground closes a box without a bottom
        if self.ground is not None and self.ground > lo[1] - m:
            closers.append(occ.addBox(lo[0] - m, lo[1] - m, lo[2] - m, hi[0] - lo[0] + 2 * m,
                                      self.ground - lo[1] + m, hi[2] - lo[2] + 2 * m))
        _, mp = occ.fragment([(3, box)], [(3, t) for t in tools + closers])
        occ.synchronize()
        pieces = [[t for d, t in m_ if d == 3] for m_ in mp]
        solid_pieces = {t for ps in pieces[1:] for t in ps}               # tools and the ground slab
        bore_set = {t for ps in pieces[1 + len(walls):1 + len(walls) + len(bores)] for t in ps}   # air, also
        wall_pieces = {t for ps in pieces[1:1 + len(walls)] for t in ps} - bore_set           # where a wall is
        plug_pieces = {t for ps in pieces[1 + len(walls) + len(bores):1 + len(tools)] for t in ps} - bore_set
        ground_pieces = {t for ps in pieces[1 + len(tools):] for t in ps}
        voids = [t for t in pieces[0] if t not in solid_pieces]
        blo, bhi = lo - m, hi + m

        def touches_outside(t):
            b = np.array(g.model.getBoundingBox(3, t)).reshape(2, 3)
            tol = 1e-3 * m                                       # OCC boxes are enlarged by the tolerances
            return bool(np.any(np.abs(b[0] - blo) < tol) or np.any(np.abs(b[1] - bhi) < tol))
        outside = [t for t in voids if touches_outside(t)]
        inside = [t for t in voids if t not in outside]
        if self.inner is not None:
            inside = [t for t in inside if g.model.isInside(3, t, list(map(float, self.inner)))]
        if len(inside) != 1:
            if not inside:
                raise MeshError(f"{self.path.name}: no closed inside air — the cabinet is open to the outside. Name "
                                f"the port bores (openings=[lp.cabinets.Port(…) or a point inside the bore solid]); "
                                f"driver holes are closed at the drivers' positions, so check that the drivers are "
                                f"placed on the front panel (translate= moves the file)")
            vols = ", ".join(f"{occ.getMass(3, t) * 1e3:.3g} L" for t in inside)
            raise MeshError(f"{self.path.name}: {len(inside)} separate inside voids ({vols}); give inner=(x, y, z) "
                            f"a point of the one the drivers radiate into")
        inner = inside[0]
        for d in self.drivers:                                     # the drivers sit on the front panel
            behind = d.frame.to_global(np.array([[0.0, 0.0, d.driver.z_baffle - 2e-4]]))[0]
            ahead = d.frame.to_global(np.array([[0.0, 0.0, d.driver.z_front + 2e-4]]))[0]
            if not g.model.isInside(3, inner, behind.tolist()):
                raise MeshError(f"{self.path.name}: driver {d.name!r} at {np.round(d.frame.origin, 4).tolist()}: the "
                                f"point behind its baffle {np.round(behind, 4).tolist()} is not in the inside air — "
                                f"the baffle plane (z = {behind[2] + 2e-4:.4f} m) must be on the inner face of the "
                                f"front panel or in its hole (translate= moves the file)")
            if any(g.model.isInside(3, t, ahead.tolist()) for t in wall_pieces | plug_pieces):
                raise MeshError(f"{self.path.name}: driver {d.name!r}: its front (z = {ahead[2] - 2e-4:.4f} m) is "
                                f"inside the cabinet — the drivers' front plane must be the outer front face")
        keep_inner = occ.copy([(3, inner)])[0][1]
        keep_bores = [occ.copy([(3, t)])[0][1] for t in sorted(bore_set)]
        shell = sorted(wall_pieces | plug_pieces | bore_set | {inner})
        fused, _ = occ.fuse([(3, shell[0])], [(3, t) for t in shell[1:]])
        outer = [t for d, t in fused if d == 3]
        occ.remove([(3, t) for t in outside + sorted(ground_pieces - wall_pieces - plug_pieces - bore_set)],
                   recursive=True)
        occ.synchronize()
        if len(outer) != 1:
            raise MeshError(f"{self.path.name}: the cabinet is {len(outer)} separate solids (expected one body)")
        return Cabinet(inner=keep_inner, cuts=[], outer=outer[0], openings=keep_bores,
                       opening_size=self.port_size if keep_bores else None)

    def signature(self) -> str:
        """What the meshes depend on: the file's content and the settings."""
        digest = hashlib.sha256(Path(self.path).read_bytes()).hexdigest()[:16]
        return (f"lp.cabinets.Step/1 {digest} openings={self.openings!r} inner={self.inner!r} scale={self.scale!r} "
                f"translate={tuple(self.translate)!r} port_size={self.port_size!r} plug_drivers={self.plug_drivers!r}")

    def describe(self) -> str:
        """A short description (scene tree)."""
        return f"Step {Path(self.path).name}" + (f" ({len(self.openings)} opening(s))" if self.openings else "")


def volume(cabinet, drivers=()) -> float:
    """The inside air volume of a cabinet (lp.cabinets.Box / Step, or a builder occ → lp.Cabinet): the inside
    minus the cuts and the openings [m³]. Drivers: placed drivers, needed for a Box without front planes."""
    from loudpy.geometry import gmsh_session
    if hasattr(cabinet, "for_drivers") and drivers:
        cabinet = cabinet.for_drivers(drivers)
    with gmsh_session() as g:
        g.model.add("volume")
        occ = g.model.occ
        cab = cabinet(occ)
        tools = [(3, t) for t in list(cab.cuts) + list(cab.openings)]
        vol, _ = occ.cut([(3, cab.inner)], tools) if tools else ([(3, cab.inner)], None)
        occ.synchronize()
        return float(sum(occ.getMass(d, t) for d, t in vol if d == 3))
