"""
Model: a 2D axisymmetric model — a geometry (.geo of the loudpy convention), its regions and
boundaries, its materials.

    model = lp.Model("HPNEW-Sketch.geo")
    model.add(
        lp.Solid("membranne", "Paper", mesh_size=0.5e-3),
        lp.Solid("surround", "Rubber", damping=lp.Hysteretic()),
        lp.Clamped("interface_constrained"),
        lp.Coil("interface_forced"),
    )
    mesh = model.mesh("Results/mesh.msh")          # reused when nothing changed

Every name must select at least one physical group of the geometry, and no group may be claimed
by two regions; otherwise model.mesh() raises with the list of groups. model.groups() lists them.
"""
from __future__ import annotations

from pathlib import Path

from loudpy.geometry import naming
from loudpy.geometry.gmsh_session import physical_groups
from loudpy.geometry.mesh import Mesh
from loudpy.geometry.meshing2d import DEFAULT_MESH_GRADING
from loudpy.model import _legacy
from loudpy.model.boundaries import Boundary
from loudpy.model.materials import MaterialLibrary, load_materials
from loudpy.model.regions import PML, Region

SUBDOMAINS_KEY = "sub"


def _is_domain_group(name: str) -> bool:
    t = naming.tokens(name)
    return bool(t) and (naming.is_prefix(t[0]) or t[0].upper() == "PML")


class Model:
    """An axisymmetric loudspeaker model: a geometry (.geo), regions, boundaries and materials.

        model = lp.Model("speaker.geo")
        model.add(lp.Solid("membrane", "Paper"), lp.Clamped("interface_constrained"), lp.Coil("interface_forced"))
        model.mesh("mesh.msh")                 # gmsh, P2 triangles; reused while the inputs are unchanged"""
    def __init__(self, geometry, materials=None, *, mesh_order: int = 2,
                 grading: float | None = DEFAULT_MESH_GRADING):
        """
        geometry  : .geo file (loudpy convention: physical curve groups submeca_*, subacou_*,
                    interface_*, PML_*)
        materials : material library — a MaterialLibrary, a JSON path, or None (loudpy's default)
        grading   : growth of the element size away from the structure (None: gmsh default)
        """
        self.geometry = Path(geometry)
        if not self.geometry.exists():
            raise FileNotFoundError(f"geometry {self.geometry} not found")
        self.materials: MaterialLibrary = materials if isinstance(materials, MaterialLibrary) \
            else load_materials(materials)
        self.mesh_order = mesh_order
        self.grading = grading
        self.regions: list[Region] = []
        self.boundaries: list[Boundary] = []
        self.meshed: Mesh | None = None
        self._fem_objects = None
        self._nodes = None
        self._mesh_path: Path | None = None

    # ── description ────────────────────────────────────────────────────────
    def add(self, *items) -> "Model":
        """Add regions and boundaries (or lists of them). Returns the model."""
        for it in items:
            if isinstance(it, (list, tuple)):
                self.add(*it)
            elif isinstance(it, Region):
                self.regions.append(it)
            elif isinstance(it, Boundary):
                self.boundaries.append(it)
            else:
                raise TypeError(f"cannot add {it!r} to a Model (expected a region or a boundary)")
        self.meshed = None
        return self

    def __getitem__(self, name):
        """The region or boundary added with this name."""
        for it in self.regions + self.boundaries:
            if it.name == name:
                return it
        raise KeyError(name)

    def groups(self, kind: str | None = None) -> list[str]:
        """Physical groups of the geometry: kind "domain", "boundary" or None (all)."""
        names = [n for d, n in physical_groups(self.geometry, dim=1)]
        names = list(dict.fromkeys(names))
        if kind == "domain":
            return [n for n in names if _is_domain_group(n)]
        if kind == "boundary":
            return [n for n in names if not _is_domain_group(n)]
        return names

    def selection(self) -> dict[str, list[str]]:
        """{item name: physical groups it selects}, checked (see the module docstring)."""
        domains, bounds = self.groups("domain"), self.groups("boundary")
        out, owner = {}, {}
        for r in self.regions:
            sel = [g for g in domains if naming.matches(r.name, g)]
            if not sel:
                raise ValueError(f"{type(r).__name__}({r.name!r}) selects no domain group of "
                                 f"{self.geometry.name}; domain groups: {domains}")
            for g in sel:
                if g in owner:
                    raise ValueError(f"group {g!r} is selected by both {owner[g]!r} and {r.name!r}")
                owner[g] = r.name
            out[r.name] = sel
        for b in self.boundaries:
            sel = [g for g in bounds if naming.matches(b.name, g)]
            if not sel:
                raise ValueError(f"{type(b).__name__}({b.name!r}) selects no boundary group of "
                                 f"{self.geometry.name}; boundary groups: {bounds}")
            out[b.name] = sel
        return out

    # ── meshing ────────────────────────────────────────────────────────────
    def mesh(self, path=None, *, reuse: bool = True, deterministic: bool = False,
             show: bool = False) -> Mesh:
        """
        Mesh the model (only the selected domains and boundaries) and write it to `path`
        (default: next to the geometry) with its signature file.

        reuse         : load the saved mesh when it was built from the same geometry and settings
        deterministic : one meshing thread, so that the same model always gives the same mesh
        show          : open the gmsh window after meshing
        """
        from loudpy.geometry.meshing2d import MeshManager
        self.selection()                                        # checks the names first
        path = Path(path) if path is not None else self.geometry.with_suffix(".msh")
        specs_dom, specs_if = self._specs()
        mm = MeshManager(specs_dom, SUBDOMAINS_KEY, mesh_order=self.mesh_order, grading=self.grading,
                         interfaces=specs_if, match=naming.matches, deterministic=deterministic)
        ok, why = mm.can_reuse(str(self.geometry), str(path)) if reuse else (False, "")
        if ok:
            domains, boundaries, nodes = mm.load_mesh(str(path))
        else:
            domains, boundaries, nodes = mm.generate_mesh(str(self.geometry), str(path), True,
                                                          show_gui=show, extract_mesh_data=True)
        self._fem_objects = self._physics_objects(domains, boundaries)
        self._nodes = nodes
        self._mesh_path = path
        self.meshed = Mesh.from_axisymmetric(domains, boundaries, nodes, str(path))
        return self.meshed

    # ── bridge to the current studies (until the studies are rebuilt) ──────
    def as_problem(self):
        """A meshed loudpy.Studies.Problem equivalent to this model (EigenStudy, FreqStudy, …)."""
        from loudpy.Studies import Problem
        if self.meshed is None:
            raise RuntimeError("call model.mesh() first")
        specs_dom, specs_if = self._specs()
        p = Problem(geo_path=str(self.geometry), msh_path=str(self._mesh_path),
                    mat_path=str(self.materials.path or ""), subdomains_key=SUBDOMAINS_KEY,
                    mesh_grading=self.grading)
        p.add_sub_domain(*specs_dom)
        p.add_interface(*specs_if)
        p.fem_objects, p.nodes = self._fem_objects, self._nodes
        return p

    def _specs(self):
        return ([_legacy.region_spec(r, self.materials) for r in self.regions],
                [_legacy.boundary_spec(b) for b in self.boundaries])

    def _physics_objects(self, domains, boundaries):
        """Same objects and order as PhysicalEntitiesManager, with exact group selection."""
        out = []
        for g in domains:
            for r in self.regions:
                if naming.matches(r.name, g.name):
                    out.append(_legacy.region_object(r, g, self.materials))
        for g in boundaries:
            for b in self.boundaries:
                if naming.matches(b.name, g.name):
                    out.append(_legacy.boundary_object(b, g))
        return out

    def __repr__(self):
        return (f"Model({self.geometry.name}: {len(self.regions)} regions, {len(self.boundaries)} boundaries, "
                f"{'meshed' if self.meshed else 'not meshed'})")
