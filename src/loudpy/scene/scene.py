"""
Scene: placed drivers in a 3D cabinet, with the air around them.

    def cabinet(occ):                                     # gmsh OCC, called once per mesh
        return lp.Cabinet(inner=add_box(occ, Z_B), cuts=[brace(occ)], outer=add_box(occ, Z_F))

    scene = lp.Scene(cabinet, drivers=[hp.place("main", at=(X0, Y0))])
    scene.cavity(mesh_size=(0.003, 0.012), f_max=6000)        # A: cabinet air as cavity modes
    scene.exterior(ground="y", bands=[(20, 1000), (1000, 2000)])   # + B: outside air, FE + exact DtN
    #  or  scene.merged(ground="y", bands=[…])                 # C: one air domain inside + outside
    scene.mesh("Results/Files")

Configurations
  A  cavity()                 cabinet air as rigid-wall cavity modes; the driver fronts in vacuo
  B  cavity() + exterior()    the same, plus the air outside (P2 FE, half-space DtN on a hemisphere
                              centred on the ground), loading the driver fronts
  C  merged()                 one FE air domain: inside of the cabinet + its openings + outside;
                              every driver wetted on both faces

The cabinet builder returns a Cabinet:
  inner     the inside air volume before the drivers are cut out (A, C)
  cuts      solids removed from the inside air (braces, panels, port tube walls)
  outer     the outer solid of the cabinet (B, C)
  openings  air volumes joining inside and outside (port bore, mouth slot) (C)
Scene adds the driver cut-outs (rear profiles, closed by a lid at the cone foot), the driver plugs
on the front (front profiles), the hemisphere, the classification of the wet surfaces per driver,
and the meshes (one per frequency band for the FE air, sized per wavelength).
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path

import logging

import numpy as np

from loudpy.geometry.gmsh_session import gmsh_module
from loudpy.model.materials import C_AIR, RHO_AIR

log = logging.getLogger("loudpy")

AXES = {"x": 0, "y": 1, "z": 2}


@dataclass
class Cabinet:
    """
    What a cabinet builder returns (OCC volume tags):
      inner         the air inside the cabinet (before the drivers are cut out)
      cuts          solids removed from the inside air (braces, panels, port tube walls…)
      outer         the outer envelope of the cabinet (exterior and merged air)
      openings      solids joining the inside air to the outside (ports, slots; merged air)
      opening_size  mesh size on the walls around the openings [m] (None: the default sizes);
                    those faces form the group "openings" (within opening_margin of an opening)
    """
    inner: int | None = None
    cuts: list = field(default_factory=list)
    outer: int | None = None
    openings: list = field(default_factory=list)
    opening_size: float | None = None
    opening_margin: float = 0.005


@dataclass
class _Cavity:
    mesh_size: tuple = (0.003, 0.012)      # (near the drivers, far) [m]
    near: tuple = (0.005, 0.06)            # size ramp distances from the drivers [m]
    f_max: float = 6000.0                  # cavity modes kept [Hz]
    eta: float = 0.0                       # modal loss factor of the cavity
    file: str = "cavity.msh"
    cache_modes: bool = True               # keep the cavity modes next to the mesh (<mesh>_modes.h5)


@dataclass
class _OpenAir:
    kind: str                              # "exterior" or "merged"
    ground: str = "y"
    level: float = 0.0
    margin: float = 0.05                   # hemisphere radius beyond the structure [m]
    bands: list = field(default_factory=lambda: [(20.0, 1000.0)])
    points_per_wavelength: float = 4.0
    max_size: float = 0.06
    driver_size: float = 0.003
    near: tuple = (0.004, 0.08)
    plug_from: float = 0.0                 # local z where the driver plug starts (exterior)
    file: str = "{kind}_{f_hi:.0f}Hz.msh"


class Scene:
    """Drivers placed in a 3D cabinet, with the air as cavity modes, exterior FE/DtN, or one merged domain.

        scene = lp.Scene(cabinet, [hp.place("main", at=(x0, y0))])
        scene.cavity(...) / scene.exterior(...) / scene.merged(...)
        scene.mesh(folder)"""
    def __init__(self, cabinet, drivers, rho: float = RHO_AIR, c: float = C_AIR):
        """cabinet: callable(occ) → Cabinet;  drivers: PlacedDriver list (Driver.place)."""
        self.cabinet = cabinet
        self.drivers = list(drivers)
        names = [d.name for d in self.drivers]
        if len(set(names)) != len(names):
            raise ValueError(f"driver names must be unique: {names}")
        self.rho, self.c = rho, c
        self._cavity: _Cavity | None = None
        self._open: _OpenAir | None = None
        self.folder: Path | None = None
        self.cavity_mesh: Path | None = None
        self.band_meshes: list[dict] = []
        self.hemisphere: dict | None = None

    # ── configuration ──────────────────────────────────────────────────────
    def cavity(self, mesh_size=(0.003, 0.012), f_max=6000.0, eta=0.0, near=(0.005, 0.06), file="cavity.msh",
               cache_modes=True):
        """The cabinet air as rigid-wall cavity modes (configurations A and B)."""
        if self._open is not None and self._open.kind == "merged":
            raise ValueError("a merged air domain already contains the cabinet air")
        self._cavity = _Cavity(tuple(mesh_size), tuple(near), f_max, eta, file, cache_modes)
        return self

    def exterior(self, ground="y", level=0.0, margin=0.05, bands=((20.0, 1000.0),), points_per_wavelength=4.0,
                 max_size=0.06, driver_size=0.003, near=(0.004, 0.08), plug_from=0.0,
                 file="exterior_{f_hi:.0f}Hz.msh"):
        """The air outside the cabinet: P2 FE + exact half-space DtN, one mesh per band (configuration B)."""
        self._open = _OpenAir("exterior", ground, level, margin, [tuple(b) for b in bands], points_per_wavelength,
                              max_size, driver_size, tuple(near), plug_from, file)
        return self

    def merged(self, ground="y", level=0.0, margin=0.05, bands=((20.0, 1000.0),), points_per_wavelength=4.0,
               max_size=0.06, driver_size=0.003, near=(0.004, 0.08), plug_from=0.0,
               file="air_{f_hi:.0f}Hz.msh"):
        """One air domain: inside + openings + outside (configuration C)."""
        if self._cavity is not None:
            raise ValueError("a merged air domain replaces the cavity modes: remove cavity()")
        self._open = _OpenAir("merged", ground, level, margin, [tuple(b) for b in bands], points_per_wavelength,
                              max_size, driver_size, tuple(near), plug_from, file)
        return self

    @property
    def configuration(self) -> str:
        if self._open is not None and self._open.kind == "merged":
            return "merged"
        if self._cavity is not None and self._open is not None:
            return "cavity+exterior"
        if self._cavity is not None:
            return "cavity"
        if self._open is not None:
            return "exterior"
        raise ValueError("configure the scene first: cavity(), exterior() or merged()")

    # ── meshing ────────────────────────────────────────────────────────────
    def mesh(self, folder, reuse=True):
        """
        Build (or reuse) the meshes in `folder`. reuse=True: a mesh is kept when its signature
        (cabinet builder, drivers, settings) is unchanged; "always": an existing file is used as
        it is; False: rebuild.
        """
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        config = self.configuration
        if self._cavity is not None:
            self.cavity_mesh = self._build("cavity", self.folder / self._cavity.file, reuse, self._mesh_cavity)
        if self._open is not None:
            self.hemisphere = self._hemisphere()
            self.band_meshes = []
            o = self._open
            for f_lo, f_hi in o.bands:
                h = min(o.max_size, self.c / f_hi / o.points_per_wavelength)
                path = self.folder / o.file.format(kind=o.kind, f_hi=f_hi)
                built = self._build(f"{o.kind}_{f_hi:.0f}", path, reuse,
                                    lambda p, h=h: self._mesh_open(p, h), extra={"h": h})
                self.band_meshes.append(dict(f_lo=f_lo, f_hi=f_hi, path=built, h=h))
        log.info(f"Scene ({config}): {len(self.drivers)} driver(s)"
              + (f", cavity mesh {self.cavity_mesh.name}" if self.cavity_mesh else "")
              + (f", {len(self.band_meshes)} band mesh(es), hemisphere radius {self.hemisphere['radius']:.3f} m"
                 if self.band_meshes else ""))
        return self

    # signature: what a mesh depends on
    def _signature(self, what, extra=None):
        try:
            src = inspect.getsource(self.cabinet)
        except (OSError, TypeError):
            src = getattr(self.cabinet, "__qualname__", repr(self.cabinet))
        drivers = []
        for d in self.drivers:
            prof = np.concatenate([np.vstack(d.driver.rear), np.vstack(d.driver.front or d.driver.rear)])
            drivers.append([d.name, d.frame.origin.round(12).tolist(), d.frame.rotation.round(12).tolist(),
                            hashlib.sha256(np.ascontiguousarray(prof).tobytes()).hexdigest()[:16]])
        mesh_keys = {"cavity": ("mesh_size", "near"),                 # only what changes the meshes
                     "open": ("kind", "ground", "level", "margin", "points_per_wavelength", "max_size",
                              "driver_size", "near", "plug_from")}
        kind = "cavity" if what == "cavity" else "open"             # a mesh depends on its own settings only
        v = self._cavity if kind == "cavity" else self._open
        settings = {kind: {n: getattr(v, n) for n in mesh_keys[kind]}}
        return json.loads(json.dumps(dict(what=what, cabinet=hashlib.sha256(src.encode()).hexdigest()[:16],
                                          drivers=drivers, settings=settings, extra=extra), default=str))

    def _build(self, what, path, reuse, builder, extra=None):
        sig_path = Path(str(path) + ".json")
        sig = self._signature(what, extra)
        if path.exists() and (reuse == "always" or
                              (reuse and sig_path.exists() and json.loads(sig_path.read_text()) == sig)):
            log.info(f"  mesh reused: {path}")
            return path
        builder(path)
        sig_path.write_text(json.dumps(sig, indent=1))
        return path

    def _start(self, name):
        g = gmsh_module()
        g.clear()
        g.model.add(name)
        return g, g.model.occ, self.cabinet(g.model.occ)

    @staticmethod
    def _keep_only(occ, keep):
        """Remove every volume of the model but `keep` (the builder may create unused solids)."""
        occ.synchronize()
        g = gmsh_module()
        other = [(3, t) for _, t in g.model.getEntities(3) if t not in set(keep)]
        if other:
            occ.remove(other, recursive=True)
            occ.synchronize()

    def _single(self):
        return len(self.drivers) == 1

    def _frames(self):
        return self.drivers[0].frame if self._single() else [d.frame for d in self.drivers]

    def _mesh_cavity(self, path):
        from loudpy.geometry.meshing3d import add_driver_cutout, carve_drivers, mesh_cavity
        g, occ, cab = self._start("cavity")
        cuts = [add_driver_cutout(d.driver.rear_cutout, center=d.frame) for d in self.drivers]
        vols = carve_drivers(cab.inner, cuts, extra_cuts=list(cab.cuts))
        if len(vols) != 1:
            raise ValueError(f"the cavity is {len(vols)} volumes (expected one)")
        self._keep_only(occ, vols)
        r_c = [d.driver.r_rear for d in self.drivers]
        z_b = [d.driver.z_baffle for d in self.drivers]
        cv = self._cavity
        mesh_cavity(path, vols[0], self._frames(), r_c[0] if self._single() else r_c,
                    z_b[0] if self._single() else z_b, cv.mesh_size[0], cv.mesh_size[1], cv.near[0], cv.near[1])
        g.clear()

    def _vertices(self, occ, solids):
        g = gmsh_module()
        occ.synchronize()
        pts = g.model.getBoundary([(3, t) for t in solids], combined=False, oriented=False, recursive=True)
        return np.array([g.model.getValue(0, abs(p), []) for d, p in pts if d == 0])

    def _hemisphere(self):
        """Centre (on the ground, below the middle of the structure) and radius of the DtN hemisphere."""
        g, occ, cab = self._start("sizing")
        corners = self._vertices(occ, [cab.outer])
        g.clear()
        pts = np.vstack([corners] + [d.outline_points() for d in self.drivers])
        up = AXES[self._open.ground]
        center = np.zeros(3)
        for ax in range(3):
            if ax == up:
                center[ax] = self._open.level
            elif ax == 0:
                center[ax] = 0.5 * (corners[:, 0].min() + corners[:, 0].max())
            else:
                center[ax] = 0.5 * (corners[:, ax].min() + max(corners[:, ax].max(), pts[:, ax].max()))
        radius = float(np.linalg.norm(pts - center, axis=1).max() + self._open.margin)
        return dict(center=center.tolist(), radius=radius, up=up)

    def dtn(self):
        from loudpy.physics.exterior import HalfSpaceDtN
        h = self.hemisphere
        return HalfSpaceDtN(h["center"], h["radius"], up=h["up"])

    def _hemi(self, occ):
        h, R = self.hemisphere, self.hemisphere["radius"]
        c, up = np.array(h["center"]), h["up"]
        sph = occ.addSphere(*c, R)
        lo = c - 2 * R
        lo[up] = c[up]
        ext = np.full(3, 4 * R)
        ext[up] = 2 * R
        half = occ.addBox(*lo, *ext)
        hemi, _ = occ.intersect([(3, sph)], [(3, half)])
        return hemi

    def _exterior_air(self, occ, cab):
        from loudpy.geometry.meshing3d import add_driver_front
        hemi = self._hemi(occ)
        plugs = [add_driver_front(d.driver.front, d.frame, self._open.plug_from) for d in self.drivers]
        box, _ = occ.cut([(3, cab.outer)], [(3, p[1]) for p in plugs])
        solid, _ = occ.fuse(box, [(3, p[0]) for p in plugs])
        air, _ = occ.cut(hemi, solid)
        return air

    def _mesh_open(self, path, h_far):
        from loudpy.geometry.meshing3d import (add_driver_cutout, carve_drivers, classify_driver_faces,
                                               mesh_exterior, mesh_volume)
        o = self._open
        g, occ, cab = self._start(o.kind)
        if o.kind == "exterior":
            air = self._exterior_air(occ, cab)
            vols = [t for d, t in air if d == 3]
            self._keep_only(occ, vols)
            if len(vols) != 1:
                raise ValueError(f"the exterior air is {len(vols)} volumes (expected one)")
            r_f = [d.driver.r_front for d in self.drivers]
            z_f = [d.driver.z_front for d in self.drivers]
            mesh_exterior(path, vols[0], self.dtn(), self._frames(), r_f[0] if self._single() else r_f,
                          z_f[0] if self._single() else z_f, o.driver_size, h_far, o.near[0], o.near[1])
        else:
            boxes = [np.array(occ.getBoundingBox(3, t)).reshape(2, 3) for t in cab.openings]
            boxes = [(b[0] - cab.opening_margin, b[1] + cab.opening_margin) for b in boxes]
            cuts = [add_driver_cutout(d.driver.rear_cutout, center=d.frame) for d in self.drivers]
            inside = [(3, t) for t in carve_drivers(cab.inner, cuts, extra_cuts=list(cab.cuts))]
            ext = self._exterior_air(occ, cab)
            air, _ = occ.fuse(inside + ext, [(3, t) for t in cab.openings]) if cab.openings else \
                occ.fuse(inside[:1], inside[1:] + ext)
            occ.synchronize()
            vols = [t for d, t in air if d == 3]
            if len(vols) != 1:
                raise ValueError(f"the air is {len(vols)} volumes: the inside must connect to the outside")
            self._keep_only(occ, vols)
            names = ["interface_driver"] if self._single() else \
                [f"interface_driver_{j}" for j in range(len(self.drivers))]
            frames = [d.frame for d in self.drivers]
            r_f = [d.driver.r_front for d in self.drivers]
            z_planes = [d.driver.z_front for d in self.drivers]
            z_bafs = sorted({d.driver.z_baffle for d in self.drivers})
            up, level = AXES[o.ground], o.level

            def classify(t):
                gm = gmsh_module()
                lo_hi = np.array(gm.model.getBoundingBox(2, t)).reshape(2, 3)
                if gm.model.getType(2, t) == "Sphere":
                    return "dtn"
                if abs(lo_hi[1, up] - level) < 1e-6 and abs(lo_hi[0, up] - level) < 1e-6:
                    return "ground"
                j = classify_driver_faces([t], frames, r_f, z_planes, atol=1e-6, extra_planes=z_bafs)[0]
                if j >= 0:
                    return names[j]
                if cab.opening_size and any(np.all(lo_hi[0] >= lo) and np.all(lo_hi[1] <= hi) for lo, hi in boxes):
                    return "openings"
                return "walls"
            refine = {n: (o.driver_size, o.near[0], o.near[1]) for n in names}
            if cab.opening_size:
                refine["openings"] = (cab.opening_size, o.near[0], o.near[1])
            mesh_volume(path, vols[0], classify, refine, h_far)
        g.clear()

    def __repr__(self):
        try:
            cfg = self.configuration
        except ValueError:
            cfg = "not configured"
        return f"Scene({cfg}, drivers {[d.name for d in self.drivers]})"
