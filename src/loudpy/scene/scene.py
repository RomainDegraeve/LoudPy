"""
Scene: placed drivers in a 3D cabinet, with the air around them.

    def cabinet(occ):                                     # gmsh OCC, called once per mesh
        return lp.Cabinet(inner=add_box(occ, Z_B), cuts=[brace(occ)], outer=add_box(occ, Z_F))

    scene = lp.Scene(cabinet, drivers=[hp.place("main", at=(X0, Y0))])
    scene.cavity(mesh_size=(0.003, 0.012), f_max=6000)        # A: cabinet air as cavity modes
    scene.exterior(ground="y", bands=[(20, 1000), (1000, 2000)])   # + B: outside air, FE + exact DtN
    #  or  scene.merged(ground="y", bands=[…])                 # C: one air domain inside + outside
    scene.mesh("meshes")

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
from loudpy.geometry.mesh_files import same_signature, write_sidecar
from loudpy.model.materials import Material, materials
from loudpy.errors import MeshError, ModelError, StateError

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
class _Walls:
    material: object = "MDF"               # library name or lp.Material (solid)
    thickness: float = 0.018               # [m]
    damping: object = None                 # lp.Rayleigh() (None), lp.Hysteretic(), lp.Undamped()
    fixed: object = "edges"                # "edges" (panel edges and the driver cut-out rims), or f(points) → bool
    f_max: float | None = None             # wall modes kept [Hz] (None: the cavity's f_max)
    angle: float = 30.0                    # a fold sharper than this is a panel edge [deg]
    panels: dict | None = None             # {panel key: lp.cabinets.Panel} (a STEP solid's index, a Box side)
    mesh_size: tuple | None = None         # merged air: (near the drivers, far) [m] of the shells' inside-air mesh


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
    def __init__(self, cabinet, drivers, air: "str | Material" = "Air", parameters=None):
        """cabinet: callable(occ) → Cabinet, or cabinet(occ, p) receiving the parameters;
        drivers: PlacedDriver list (Driver.place);
        air: a fluid of the material library or an lp.Material (lp.air(temperature=…, humidity=…));
        parameters: lp.Parameters referred to by lp.P(name) in the settings and by cabinet(occ, p)."""
        from loudpy.model.parameters import Parameters
        self.parameters = parameters if parameters is not None else Parameters()
        air = air if isinstance(air, Material) else materials[air]
        if air.kind != "fluid":
            raise ModelError(f"Scene air: {air.name!r} is a {air.kind}, not a fluid")
        air.require("rho", "c")
        self.air = air
        self.drivers = list(drivers)
        if hasattr(cabinet, "for_drivers"):                       # lp.cabinets: the drivers set the front
            cabinet = cabinet.for_drivers(self.drivers)
        self.cabinet = cabinet
        names = [d.name for d in self.drivers]
        if len(set(names)) != len(names):
            raise ModelError(f"driver names must be unique: {names}")
        self._cavity: _Cavity | None = None              # settings as resolved for the last meshing
        self._open: _OpenAir | None = None
        self._walls: _Walls | None = None
        self._walls_spec: _Walls | None = None
        self._cavity_spec: _Cavity | None = None         # settings as given (may hold lp.P references)
        self._open_spec: _OpenAir | None = None
        self.folder: Path | None = None
        self.cavity_mesh: Path | None = None
        self.walls_mesh: Path | None = None                               # the inside air for the shells (merged)
        self.band_meshes: list[dict] = []
        self.hemisphere: dict | None = None
        self._keep: tuple = ()                                            # meshes used as they are (keep())

    @property
    def rho(self) -> float:
        """Density of the air [kg/m³]."""
        return float(self.air.rho)  # type: ignore[arg-type]   # required at construction (air.require)

    @property
    def c(self) -> float:
        """Speed of sound in the air [m/s]."""
        return float(self.air.c)  # type: ignore[arg-type]

    # ── ready-made scenes ──────────────────────────────────────────────────
    @staticmethod
    def _placed(driver, at, name):
        from loudpy.scene.driver import PlacedDriver
        if isinstance(driver, PlacedDriver):
            return [driver]
        if isinstance(driver, (list, tuple)) and driver and all(isinstance(d, PlacedDriver) for d in driver):
            return list(driver)
        if at is None:
            raise ModelError("give the driver's position at=(x, y) on the front face (or placed drivers)")
        return [driver.place(name, at=at)]

    @staticmethod
    def bands(f_max: float, width: float = 1000.0, f_min: float = 20.0) -> list:
        """Frequency bands of the FE air meshes up to f_max: [(20, 1000), (1000, 2000), …] [Hz]."""
        edges = [f_min] + list(np.arange(width, f_max, width)) + [f_max]
        return [(float(a), float(b)) for a, b in zip(edges, edges[1:]) if b > a]

    @classmethod
    def sealed_box(cls, cabinet, driver, at=None, *, f_max: float = 3000.0, radiating: bool = False,
                   ground: str = "y", eta: float = 0.02, air="Air", name: str = "main", cavity_file="cavity.msh",
                   air_file="exterior_{f_hi:.0f}Hz.msh") -> "Scene":
        """
        A driver in a closed cabinet: the inside air as cavity modes (to 2 f_max, loss factor eta); radiating=True
        adds the outside air on a rigid ground (FE + exact half-space DtN, 1 kHz bands to f_max). cabinet: an
        lp.cabinets.Box / Step or a builder; driver: an lp.Driver placed at `at` (x, y), or placed drivers. Mesh
        it with scene.mesh(folder).
            scene = lp.Scene.sealed_box(lp.cabinets.Box(0.2, 0.3, 0.18), hp, at=(0.1, 0.19), radiating=True)
        """
        scene = cls(cabinet, cls._placed(driver, at, name), air=air)
        scene.cavity(mesh_size=(0.003, 0.012), f_max=2 * f_max, eta=eta, file=cavity_file)
        if radiating:
            scene.exterior(ground=ground, bands=cls.bands(f_max), points_per_wavelength=4.5, max_size=0.06,
                           driver_size=0.003, margin=0.05, file=air_file)
        return scene

    @classmethod
    def vented_box(cls, cabinet, driver, at=None, *, f_max: float = 3000.0, ground: str = "y", air="Air",
                   name: str = "main", air_file="air_{f_hi:.0f}Hz.msh") -> "Scene":
        """
        A driver in an open cabinet (ports, slots, a transmission line): one FE air domain inside, through the
        openings and outside on a rigid ground (exact half-space DtN), 1 kHz bands to f_max; every driver wetted
        on both faces. cabinet: an lp.cabinets.Box with ports / Step with openings, or a builder.
            scene = lp.Scene.vented_box(lp.cabinets.Box(0.2, 0.3, 0.18, wall=0.015, ports=[port]), hp, at=(0.1, 0.19))
        """
        scene = cls(cabinet, cls._placed(driver, at, name), air=air)
        scene.merged(ground=ground, bands=cls.bands(f_max), points_per_wavelength=4.0, max_size=0.06,
                     driver_size=0.003, margin=0.05, file=air_file)
        return scene

    # ── default views ──────────────────────────────────────────────────────
    def bounds(self) -> np.ndarray:
        """(2, 3) lower and upper corners of the cabinet's outer solid and the drivers [m]."""
        g, occ, cab = self._start("bounds")
        tag = cab.outer if cab.outer is not None else cab.inner
        b = np.array(occ.getBoundingBox(3, tag)).reshape(2, 3)
        g.clear()
        pts = np.vstack([b] + [d.outline_points() for d in self.drivers])
        return np.array([pts.min(axis=0), pts.max(axis=0)])

    def cameras(self) -> dict:
        """Default cameras of the scene's 3D views (lp.plot.default_cameras of its bounds and drivers)."""
        from loudpy.plot.scene3d import default_cameras
        return default_cameras(self.bounds(), [d.frame.origin for d in self.drivers])

    def cavity(self, mesh_size=(0.003, 0.012), f_max=6000.0, eta=0.0, near=(0.005, 0.06), file="cavity.msh",
               cache_modes=True):
        """The cabinet air as rigid-wall cavity modes (configurations A and B)."""
        if self._open is not None and self._open.kind == "merged":
            raise ModelError("a merged air domain already contains the cabinet air")
        self._cavity = self._cavity_spec = _Cavity(tuple(mesh_size), tuple(near), f_max, eta, file, cache_modes)
        return self

    def walls(self, material="MDF", thickness=0.018, damping=None, fixed="edges", f_max=None, angle=30.0,
              panels=None, mesh_size=None):
        """
        Elastic cabinet walls: shells (MITC3) on the walls of the cavity mesh, coupled to the cavity air through
        their in-vacuo modes (configuration cavity). material: library name or lp.Material; thickness [m] (or
        lp.P); damping: lp.Rayleigh() (default), lp.Hysteretic(), lp.Undamped(); fixed: "edges" (the panel edges
        — folds sharper than `angle` — and the rims of the driver cut-outs are clamped) or a function of the
        node coordinates (n, 3) returning a mask of clamped nodes; f_max: wall modes kept (default: the cavity's).
        The shells lie on the inner faces of the walls (no offset), one per panel (a board wet on both faces: one
        side). With the outside air (cavity() + exterior()) the outer face of each wall moves with its shell (rigid
        through the thickness) and radiates; else it is in vacuo. With one merged air domain (merged(), an lp.cabinets
        Box / Step: openings, a labyrinth's boards) every wet face of the walls moves with its shell in that air.

        Panels of their own (an lp.cabinets.Box / Step): thickness=None — each panel's thickness measured on the
        cabinet (a STEP solid through its wet face, a Box wall); panels={key: lp.cabinets.Panel(material, thickness,
        damping)} — a STEP solid by its index in the file (0, 1, …) or a Box side ("left", "right", "bottom", "top",
        "back", "front"); what a Panel leaves None is the walls' own. Faces on no panel (driver plugs, the ground,
        port tubes) stay rigid.

        mesh_size: with one merged air domain, the element sizes (near the drivers, far) [m] of the inside-air mesh
        the shells lie on (None: 3 mm, 12 mm); with cavity modes the shells lie on the cavity mesh: its own sizes.
        """
        from loudpy.scene.cabinets import Panel
        panels = {int(k) if isinstance(k, str) and k.isdigit() else k: v      # a STEP solid's index (JSON: text)
                  for k, v in (panels or {}).items()}
        for key, pnl in panels.items():
            if not isinstance(pnl, Panel):
                raise ModelError(f"walls panels: {key!r} → lp.cabinets.Panel(material, thickness, damping), "
                                 f"got {pnl!r}")
        self._walls = self._walls_spec = _Walls(material, thickness, damping, fixed, f_max, angle,
                                                dict(panels) if panels else None,
                                                None if mesh_size is None else tuple(mesh_size))
        return self

    def exterior(self, ground="y", level=0.0, margin=0.05, bands=((20.0, 1000.0),), points_per_wavelength=4.0,
                 max_size=0.06, driver_size=0.003, near=(0.004, 0.08), plug_from=0.0,
                 file="exterior_{f_hi:.0f}Hz.msh"):
        """The air outside the cabinet: P2 FE + exact half-space DtN, one mesh per band (configuration B)."""
        self._open = self._open_spec = _OpenAir("exterior", ground, level, margin, [tuple(b) for b in bands],
                                                points_per_wavelength, max_size, driver_size, tuple(near),
                                                plug_from, file)
        return self

    def merged(self, ground="y", level=0.0, margin=0.05, bands=((20.0, 1000.0),), points_per_wavelength=4.0,
               max_size=0.06, driver_size=0.003, near=(0.004, 0.08), plug_from=0.0,
               file="air_{f_hi:.0f}Hz.msh"):
        """One air domain: inside + openings + outside (configuration C)."""
        if self._cavity is not None:
            raise ModelError("a merged air domain replaces the cavity modes: remove cavity()")
        self._open = self._open_spec = _OpenAir("merged", ground, level, margin, [tuple(b) for b in bands],
                                                points_per_wavelength, max_size, driver_size, tuple(near),
                                                plug_from, file)
        return self

    def keep(self, meshes=()):
        """
        Meshes used as they are while their file exists, even when their settings changed since (out of date):
        "cavity", "walls" (the inside air of the elastic walls), "air" (every air band), one band by its top
        frequency ("air 2000"), or "all". keep() or keep([]): none — a mesh out of date is built again. A mesh file
        made elsewhere can take the place of one this way (it must carry the groups loudpy writes).
            scene.keep(["cavity"])          # a fine cavity mesh kept while the bands are rebuilt
        """
        names = [meshes] if isinstance(meshes, str) else list(meshes)
        for n in names:
            band = n.startswith("air ") and n[4:].replace(".", "", 1).isdigit()
            ok = n in ("cavity", "walls", "air", "all") or band
            if not ok:
                raise ModelError(f"keep: 'cavity', 'walls', 'air', 'air <top frequency>' or 'all', got {n!r}")
        self._keep = tuple(names)
        return self

    @property
    def settings(self) -> dict:
        """
        The air, the walls and the kept meshes as given (lp.P references kept): {"cavity", "air", "walls", "keep"}.
        "cavity", "air" (its kind: "exterior" or "merged") and "walls" are settings objects or None — read them, or
        copy one changed with dataclasses.replace and give it back with restore(). Made by cavity(), exterior(),
        merged(), walls() and keep().
            before = scene.settings;  …;  scene.restore(before)          # undo
        """
        return {"cavity": self._cavity_spec, "air": self._open_spec, "walls": self._walls_spec, "keep": self._keep}

    def restore(self, settings: dict) -> "Scene":
        """Set the air, the walls and the kept meshes from a dict like scene.settings (a key left out: none)."""
        cav, air, walls = settings.get("cavity"), settings.get("air"), settings.get("walls")
        if cav is not None and air is not None and getattr(air, "kind", None) == "merged":
            raise ModelError("a merged air domain replaces the cavity modes: give cavity=None with it")
        self._cavity = self._cavity_spec = cav
        self._open = self._open_spec = air
        self._walls = self._walls_spec = walls
        self._keep = tuple(settings.get("keep", ()))
        return self

    def air_hemisphere(self) -> dict | None:
        """The hemisphere of the outside air's exact boundary {"center", "radius", "up"}: the meshed scene's, else
        measured on the cabinet and drivers as mesh() will (nothing meshed); None without an outside air."""
        if self._open_spec is None:
            return None
        if self.hemisphere:
            return dict(self.hemisphere)
        from loudpy.model.parameters import resolve
        saved = self._open
        self._open = resolve(self._open_spec, self.parameters)
        try:
            return dict(self._hemisphere())
        finally:
            self._open = saved

    def _kept(self, what: str) -> bool:
        """keep() names this mesh ("cavity", "walls", or a band "<kind>_<f_hi>")."""
        return self._named(what, self._keep)

    @staticmethod
    def _named(what: str, names) -> bool:
        """`names` ("cavity", "walls", "air", "air <f_hi>", "all": keep() / mesh(only=)) include this mesh."""
        k = {names} if isinstance(names, str) else set(names)
        if "all" in k or what in k:
            return True
        _kind, _, f_hi = what.partition("_")
        return bool(f_hi) and ("air" in k or any(n.startswith("air ") and abs(float(n[4:]) - float(f_hi)) < 0.5
                                                 for n in k))

    @property
    def configuration(self) -> str:
        """"cavity", "cavity+exterior", "exterior" or "merged", from the configured air."""
        if self._open is not None and self._open.kind == "merged":
            return "merged"
        if self._cavity is not None and self._open is not None:
            return "cavity+exterior"
        if self._cavity is not None:
            return "cavity"
        if self._open is not None:
            return "exterior"
        raise StateError("configure the scene first: cavity(), exterior() or merged()")

    # ── meshing ────────────────────────────────────────────────────────────
    def mesh(self, folder, reuse=True, only=None):
        """
        Build (or reuse) the meshes in `folder`. reuse=True: a mesh is kept when its signature
        (cabinet builder, drivers, settings) is unchanged; "always": an existing file is used as
        it is; False: rebuild. only: build (or reuse) just these meshes — "cavity", "walls", "air"
        (every band), "air <f_hi>" (one band), as keep() names them — and leave the others alone
        (a mesh-only run: the scene is then not ready to solve).
        """
        from loudpy.model.parameters import resolve
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        config = self.configuration
        self._cavity = resolve(self._cavity_spec, self.parameters)          # lp.P → values
        self._open = resolve(self._open_spec, self.parameters)
        self._walls = resolve(self._walls_spec, self.parameters)
        if self._walls is not None and self._walls.mesh_size is not None and self._cavity is not None:
            raise ModelError("walls(mesh_size=) is for one merged air domain: with cavity modes the shells lie on "
                             "the cavity mesh (set cavity(mesh_size=))")
        def wanted(what):
            return only is None or self._named(what, only)
        if self._cavity is not None and wanted("cavity"):
            self.cavity_mesh = self._build("cavity", self.folder / self._cavity.file, reuse, self._mesh_cavity,
                                           info=dict(role="cavity", mesh_size=list(self._cavity.mesh_size)))
        if self._walls is not None and self._open is not None and self._open.kind == "merged" and wanted("walls"):
            name = "walls_" + Path(self._open.file.format(kind="merged", f_hi=self._open.bands[0][1])).stem + ".msh"
            self.walls_mesh = self._build("walls", self.folder / name, reuse, self._mesh_walls,
                                          info=dict(role="inside air of the walls",
                                                    mesh_size=list(self._walls_sizes().mesh_size)))
        if self._open is not None:
            self.hemisphere = self._hemisphere()
            self.band_meshes = []
            o = self._open
            for f_lo, f_hi in o.bands:
                if not wanted(f"{o.kind}_{f_hi:.0f}"):
                    continue
                h = min(o.max_size, self.c / f_hi / o.points_per_wavelength)
                path = self.folder / o.file.format(kind=o.kind, f_hi=f_hi)
                built = self._build(f"{o.kind}_{f_hi:.0f}", path, reuse,
                                    lambda p, h=h: self._mesh_open(p, h), extra={"h": h},
                                    info=dict(role=f"{o.kind} air {f_lo:g}–{f_hi:g} Hz", band=[f_lo, f_hi], h=h))
                self.band_meshes.append(dict(f_lo=f_lo, f_hi=f_hi, path=built, h=h))
        log.info(f"Scene ({config}): {len(self.drivers)} driver(s)"
              + (f", cavity mesh {self.cavity_mesh.name}" if self.cavity_mesh else "")
              + (f", {len(self.band_meshes)} band mesh(es), hemisphere radius {self.hemisphere['radius']:.3f} m"
                 if self.band_meshes else ""))
        return self

    def mesh_files(self, folder=None) -> list[dict]:
        """
        The meshes mesh(folder) builds, named as it names them — the cavity, the inside air the walls sit on
        (merged air with walls), each air band: [{"what", "title", "path", "state"}], state "built", "out of date"
        (its settings, cabinet or drivers changed since: mesh() builds it again), "kept" (out of date, but keep()
        names it: used as it is) or "missing". Nothing is meshed
        here. folder: the one the scene was meshed in by default.
        """
        from loudpy.model.parameters import resolve
        folder = Path(folder) if folder is not None else self.folder
        if folder is None:
            raise StateError("give the folder of the meshes (or mesh the scene first)")
        saved = (self._cavity, self._open, self._walls)
        cav, opn = resolve(self._cavity_spec, self.parameters), resolve(self._open_spec, self.parameters)
        self._cavity, self._open, self._walls = cav, opn, resolve(self._walls_spec, self.parameters)
        try:
            items: list[tuple] = []
            if cav is not None:
                items.append(("cavity", "cavity", folder / cav.file, None))
            if self._walls is not None and opn is not None and opn.kind == "merged":
                name = "walls_" + Path(opn.file.format(kind="merged", f_hi=opn.bands[0][1])).stem + ".msh"
                items.append(("walls", "inside air of the walls", folder / name, None))
            if opn is not None:
                for f_lo, f_hi in opn.bands:
                    h = min(opn.max_size, self.c / f_hi / opn.points_per_wavelength)
                    items.append((f"{opn.kind}_{f_hi:.0f}", f"{opn.kind} {f_lo:g}–{f_hi:g} Hz",
                                  folder / opn.file.format(kind=opn.kind, f_hi=f_hi), {"h": h}))
            out = []
            for what, title, path, extra in items:
                state = "missing"
                if path.exists():
                    sig_path = Path(str(path) + ".json")
                    try:
                        same = sig_path.exists() and same_signature(json.loads(sig_path.read_text()),
                                                                     self._signature(what, extra))
                    except (OSError, ValueError):
                        same = False
                    state = "built" if same else "kept" if self._kept(what) else "out of date"
                out.append(dict(what=what, title=title, path=path, state=state))
            return out
        finally:
            self._cavity, self._open, self._walls = saved

    # signature: what a mesh depends on
    def _signature(self, what, extra=None):
        if hasattr(self.cabinet, "signature"):                    # lp.cabinets.Box / Step (parameters resolved)
            src = self._cabinet().signature()
        else:
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
        kind = "cavity" if what in ("cavity", "walls") else "open"  # a mesh depends on its own settings only
        if what == "walls":
            v = self._walls_sizes()
        else:
            v = (self._cavity or _Cavity()) if kind == "cavity" else self._open
        settings = {kind: {n: getattr(v, n) for n in mesh_keys[kind]}}
        sig = dict(what=what, cabinet=hashlib.sha256(src.encode()).hexdigest()[:16], drivers=drivers,
                   settings=settings, extra=extra, p2_nodes="curved")           # mid-edge nodes on the surfaces
        if self._cabinet_takes_parameters:
            sig["parameters"] = {k: float(v) for k, v in sorted(self.parameters.items())}
        return json.loads(json.dumps(sig, default=str))

    @property
    def _cabinet_takes_parameters(self) -> bool:
        try:
            args = list(inspect.signature(self.cabinet).parameters)
        except (TypeError, ValueError):
            return False
        return len(args) >= 2 and args[1] in ("p", "params", "parameters")

    def _build(self, what, path, reuse, builder, extra=None, info=None):
        sig_path = Path(str(path) + ".json")
        sig = self._signature(what, extra)
        if path.exists() and (reuse and sig_path.exists() and same_signature(json.loads(sig_path.read_text()), sig)):
            log.info(f"  mesh reused: {path}")              # (a sidecar written before says what it is by "what")
            return path
        if path.exists() and reuse and (reuse == "always" or self._kept(what)):     # reuse=False: rebuilt
            log.info(f"  mesh kept as it is: {path}")
            return path
        builder(path)
        write_sidecar(path, sig, **(info or {}))
        return path

    def _start(self, name):
        g = gmsh_module()
        g.clear()
        g.model.add(name)
        g.option.setNumber("General.NumThreads", 1)                      # deterministic meshes
        occ = g.model.occ
        return g, occ, self.cabinet(occ, self.parameters) if self._cabinet_takes_parameters else self._cabinet()(occ)

    def _cabinet(self):
        """The cabinet with its lp.P references resolved (lp.cabinets dataclasses; a builder as it is)."""
        from loudpy.model.parameters import resolve
        return resolve(self.cabinet, self.parameters)

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
            raise MeshError(f"the cavity is {len(vols)} volumes (expected one)")
        self._keep_only(occ, vols)
        r_c = [d.driver.r_rear for d in self.drivers]
        z_b = [d.driver.z_baffle for d in self.drivers]
        cv = self._cavity
        mesh_cavity(path, vols[0], self._frames(), r_c[0] if self._single() else r_c,
                    z_b[0] if self._single() else z_b, cv.mesh_size[0], cv.mesh_size[1], cv.near[0], cv.near[1])
        g.clear()

    def _walls_sizes(self) -> "_Cavity":
        """The sizes of the inside-air mesh the shells lie on in a merged air domain (walls(mesh_size=), else the
        cavity's defaults)."""
        w = self._walls
        return _Cavity(mesh_size=tuple(w.mesh_size)) if w is not None and w.mesh_size is not None else _Cavity()

    def _mesh_walls(self, path):
        """The inside air meshed for the shells of the elastic walls (a merged air domain has no cavity of its own):
        walls(mesh_size=), else the cavity's default sizes."""
        saved = self._cavity
        self._cavity = self._walls_sizes()
        try:
            self._mesh_cavity(path)
        finally:
            self._cavity = saved

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
        """The half-space DtN on the scene's hemisphere (after meshing)."""
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
                raise MeshError(f"the exterior air is {len(vols)} volumes (expected one)")
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
                raise MeshError(f"the air is {len(vols)} volumes: the inside must connect to the outside")
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

    # ── the model pattern (as lp.Model) ────────────────────────────────────
    def to_script(self, folder=None, title: str | None = None) -> str:
        """The scene as Python code that builds it again (loudpy.model.script): an lp.cabinets.Box / Step cabinet and
        drivers made by lp.Driver.from_modes; folder: where the script will be (paths relative to it)."""
        from loudpy.model.script import scene_script
        return scene_script(self, folder, title)

    def copy(self) -> "Scene":
        """A new, unmeshed scene with the same cabinet, drivers, air, parameters and settings."""
        import copy as _copy
        s = Scene(self.cabinet, self.drivers, self.air, self.parameters)
        s._cavity = s._cavity_spec = _copy.copy(self._cavity_spec)
        s._open = s._open_spec = _copy.copy(self._open_spec)
        s._walls = s._walls_spec = _copy.copy(self._walls_spec)
        return s

    def with_parameters(self, **values) -> "Scene":
        """A copy of the scene (unmeshed) with other parameter values."""
        s = self.copy()
        s.parameters = self.parameters.with_(**values)
        return s

    def check(self) -> list[str]:
        """Check the scene without meshing: configuration, drivers, air, parameters. Raises ModelError
        listing every problem; returns notes."""
        from loudpy.model.parameters import resolve
        problems, notes = [], []
        try:
            self.configuration
        except StateError as e:
            problems.append(str(e))
        if not self.drivers:
            problems.append("no driver (Driver.place)")
        if self._walls_spec is not None and self._cavity_spec is None and \
                (self._open_spec is None or self._open_spec.kind != "merged"):
            problems.append("elastic walls need the air: cavity() (+ exterior()) or merged()")
        for spec in (self._cavity_spec, self._open_spec, self._walls_spec):
            try:
                resolve(spec, self.parameters)
            except ModelError as e:
                problems.append(str(e))
        if self._open_spec is not None and not self._open_spec.bands:
            problems.append("no frequency band for the FE air")
        if problems:
            raise ModelError("the scene has problems:\n  - " + "\n  - ".join(problems))
        if self._cavity_spec is not None and self._open_spec is None:
            notes.append("cavity only: the driver fronts radiate in vacuo")
        return notes

    def tree(self) -> str:
        """The scene as a tree (as print(model) for a 2D model)."""
        try:
            cfg = self.configuration
        except StateError:
            cfg = "not configured"
        name = self._cabinet().describe() if hasattr(self.cabinet, "describe") else \
            f"{getattr(self.cabinet, '__name__', 'cabinet')}()"
        lines = [f"Scene {name}  (3D, configuration {cfg}, P2 tetrahedra, curved)"]
        if len(self.parameters):
            lines.append(f"├─ Parameters: {', '.join(f'{k} = {v:g}' for k, v in self.parameters.items())}")
        lines.append(f"├─ Air: {self.air.name} (rho {self.air.rho:g} kg/m³, c {self.air.c:g} m/s)")
        lines.append(f"├─ Drivers ({len(self.drivers)})")
        for k, d in enumerate(self.drivers):
            lines.append(f"│  {'└' if k == len(self.drivers) - 1 else '├'}─ {d.name!r}: {d.driver.name}, at "
                         f"{np.round(d.frame.origin, 4).tolist()}")
        if self._cavity_spec is not None:
            c = self._cavity_spec
            lines.append(f"├─ Cavity: modes to {c.f_max} Hz, loss factor {c.eta}, mesh {c.mesh_size}")
        if self._walls_spec is not None:
            w = self._walls_spec
            name = getattr(w.material, "name", w.material)
            th = "measured" if w.thickness is None else f"{w.thickness} m"
            lines.append(f"├─ Walls: {name}, {th} shells (MITC3), fixed {w.fixed if isinstance(w.fixed, str) else 'by function'}"
                         f", modes to {w.f_max or 'the default (cavity f_max, or 2 × the air bands)'} Hz"
                         + (f", {len(w.panels)} panel(s) of their own" if w.panels else ""))
        if self._open_spec is not None:
            o = self._open_spec
            lines.append(f"├─ {o.kind.capitalize()} air: bands {o.bands}, ground {o.ground}, "
                         f"{o.points_per_wavelength} points per wavelength")
        meshes = ([self.cavity_mesh.name] if self.cavity_mesh else []) + [b["path"].name for b in self.band_meshes]
        lines.append(f"└─ Meshes: {', '.join(meshes) if meshes else 'not meshed'}")
        return "\n".join(lines)

    def __str__(self):
        return self.tree()

    def __repr__(self):
        try:
            cfg = self.configuration
        except ValueError:
            cfg = "not configured"
        return f"Scene({cfg}, drivers {[d.name for d in self.drivers]})"
