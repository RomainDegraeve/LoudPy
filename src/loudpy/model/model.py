"""
Model: a 2D axisymmetric model — a geometry, its regions and boundaries, its materials.

The geometry (loudpy.geometry.sources) is a .geo (curve loops of the loudpy convention, or surfaces), a
CAD file (.step/.stp/.iges/.igs/.brep: faces, or the closed regions of a sketch) or a builder function
drawing with gmsh's OpenCASCADE kernel.

    model = lp.Model("HPNEW-Sketch.geo")
    model.add(
        lp.Solid("cone", "Paper", mesh_size=0.5e-3),
        lp.Solid("surround", "Rubber", damping=lp.Hysteretic()),
        lp.Clamped("rims", on=lp.Group("surround_rim", "spider_rim")), lp.Clamped("axis", dofs="r"),
        lp.Coil("coil_drive"),
    )
    mesh = model.mesh("Results/mesh.msh")          # reused when nothing changed

Each item covers what its selector `on` selects (default: its name as a label; loudpy.geometry.selectors):
a region covers whole domain groups of the geometry, selected before meshing; a boundary covers edges of
the mesh, selected after meshing (lp.Interface, lp.BoundaryOf, lp.Box, lp.Near need no curve drawn in the
geometry). Every selector must select something, every label a group, and no domain group may be claimed by
two regions; otherwise model.mesh() raises with the list of groups. model.groups() lists the groups,
model.selection() what each label selects, model.select(…) evaluates a selector on the meshed model.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from loudpy.geometry import naming
from loudpy.geometry.naming import PMLLayer
from loudpy.geometry.selectors import Curves, Domains, Edges, as_selector
from loudpy.geometry.sources import GeometryIndex, GeometrySource
from loudpy.geometry.mesh import Mesh
from loudpy.geometry.meshing2d import DEFAULT_MESH_GRADING, DomainSizing
from loudpy.model.boundaries import Boundary
from loudpy.model.materials import MaterialLibrary, load_materials
from loudpy.model.mesh_controls import MeshSize
from loudpy.model.parameters import Parameters, resolve
from loudpy.model.regions import PML, Air, Region, Solid
from loudpy.errors import MeshError, ModelError, SelectionError, StateError

log = logging.getLogger("loudpy.model")

class Model:
    """An axisymmetric loudspeaker model: a geometry (.geo), regions, boundaries and materials.
    lp.Model(cabinet, dim=3, drivers=[…]) gives the 3D model of drivers in a cabinet (an lp.Scene), which
    has the same pattern: parameters, print / tree(), check(), copy(), with_parameters(), mesh(), studies.

        model = lp.Model("speaker.geo")
        model.add(lp.Solid("membrane", "Paper"), lp.Clamped("rims", on=lp.Group("surround_rim", "spider_rim")), lp.Clamped("axis", dofs="r"), lp.Coil("coil_drive"))
        model.mesh("mesh.msh")                 # gmsh, P2 triangles; reused while the inputs are unchanged"""
    def __new__(cls, geometry=None, *args, dim: int = 2, **kwargs):
        """lp.Model(cabinet, dim=3, drivers=[…], air=…, parameters=…) builds a 3D model: an lp.Scene."""
        if dim == 3:
            from loudpy.scene.scene import Scene
            return Scene(geometry, *args, **kwargs)
        if dim != 2:
            raise ModelError(f"dim must be 2 (axisymmetric) or 3 (scene), got {dim!r}")
        return super().__new__(cls)

    def __init__(self, geometry, materials=None, *, mesh_order: int = 2,
                 grading: float | None = DEFAULT_MESH_GRADING, parameters: Parameters | None = None, dim: int = 2):
        """
        geometry  : a .geo (curve loops: every closed physical curve group is a domain; or surfaces), a
                    CAD file (.step/.stp/.iges/.igs/.brep) or a builder function builder(occ) → {label:
                    surface tags} (loudpy.geometry.sources)
        materials : material library — a MaterialLibrary, a JSON path, or None (loudpy's default)
        grading   : growth of the element size away from the structure (None: gmsh default)
        parameters: lp.Parameters referred to by lp.P(name) in the items and by a builder(occ, p)
        """
        self.parameters = parameters if parameters is not None else Parameters()
        if not isinstance(self.parameters, Parameters):
            raise ModelError(f"parameters must be lp.Parameters, got {type(parameters).__name__}")
        self.source = GeometrySource(geometry, self.parameters)
        self.geometry = self.source.path if self.source.path is not None else geometry
        self.materials: MaterialLibrary = materials if isinstance(materials, MaterialLibrary) \
            else load_materials(materials)
        self.mesh_order = mesh_order
        self.grading = grading
        self.regions: list[Region] = []
        self.boundaries: list[Boundary] = []
        self.mesh_controls: list[MeshSize] = []
        self.meshed: Mesh | None = None
        self.parts: list = []                  # physics parts of the meshed model (loudpy.physics.parts)
        self._mesh_path: Path | None = None
        self._selection: dict = {}             # of the last meshing: {item: groups of the geometry}
        self._region_groups: dict = {}         # {region: mesh groups}
        self._item_parts: dict = {}            # {boundary: its parts}
        self._edges: Edges | None = None       # the boundary edges of the mesh (selector evaluation)
        self._pml_bases: dict = {}             # {PML: (base curve tags, {tag: layer name})}
        self._pml_layer: dict = {}             # {PML surface of the mesh: PMLLayer}
        self._index: GeometryIndex | None = None   # the geometry, read by the last selection()
        self._plan_names: dict = {}            # {region: its mesh group names}

    # ── description ────────────────────────────────────────────────────────
    def add(self, *items) -> "Model":
        """Add regions, boundaries and mesh controls (lp.MeshSize), or lists of them. Returns the model."""
        for it in items:
            if isinstance(it, (list, tuple)):
                self.add(*it)
                continue
            if isinstance(it, MeshSize):
                self.mesh_controls.append(it)
                self.meshed = None
                continue
            if not isinstance(it, (Region, Boundary)):
                raise ModelError(f"cannot add {it!r} to a Model (expected a region, a boundary or lp.MeshSize)")
            if any(o.name == it.name for o in self.regions + self.boundaries):
                raise ModelError(f"the model already has an item named {it.name!r} (names are the labels of the "
                                 f"results: give each item its own)")
            if isinstance(it, Region):
                self.regions.append(it)
            else:
                self.boundaries.append(it)
        self.meshed = None
        return self

    def copy(self) -> "Model":
        """A new, unmeshed model with the same geometry, materials, settings and copies of the items (the
        studies change copies, never the user's model)."""
        import copy as _copy
        m = Model.__new__(Model)
        Model.__init__(m, self.source.path if self.source.path is not None else self.source.builder,
                       self.materials, mesh_order=self.mesh_order, grading=self.grading, parameters=self.parameters)
        if not self.source.takes_parameters:
            m.source = self.source
        m.geometry = self.geometry
        m.add([_copy.copy(it) for it in self.regions + self.boundaries], [_copy.copy(c) for c in self.mesh_controls])
        return m

    def with_parameters(self, **values) -> "Model":
        """A copy of the model (unmeshed) with other parameter values."""
        m = self.copy()
        m.parameters = self.parameters.with_(**values)
        m.source = GeometrySource(self.source.path if self.source.path is not None else self.source.builder,
                                  m.parameters)
        return m

    def resolved(self) -> "Model":
        """A copy of the model whose items hold values instead of lp.P references."""
        m = self.copy()                                     # its own item copies, resolved below
        m.regions, m.boundaries, m.mesh_controls = m._resolved_items()
        return m

    def _resolved_items(self):
        p = self.parameters
        return ([resolve(r, p) for r in self.regions], [resolve(b, p) for b in self.boundaries],
                [resolve(c, p) for c in self.mesh_controls])

    class _Resolved:
        """Context: the model's items replaced by their resolved copies (meshing, checks), restored after."""
        def __init__(self, model):
            self.model = model

        def __enter__(self):
            m = self.model
            self.saved = (m.regions, m.boundaries, m.mesh_controls)
            m.regions, m.boundaries, m.mesh_controls = m._resolved_items()

        def __exit__(self, *exc):
            m = self.model
            m.regions, m.boundaries, m.mesh_controls = self.saved

    def __getitem__(self, name):
        """The region or boundary added with this name."""
        for it in self.regions + self.boundaries:
            if it.name == name:
                return it
        raise SelectionError(name)

    def groups(self, kind: str | None = None) -> list[str]:
        """Groups of the geometry: kind "domain" (curve loops, named surface groups, unnamed surfaces
        "surface_<tag>"), "boundary" (the other named curve groups) or None (all)."""
        index = GeometryIndex(self.source)
        bounds = [n for n in index.groups if n not in index.domains]
        if kind == "domain":
            return list(index.domains)
        if kind == "boundary":
            return bounds
        return list(dict.fromkeys(list(index.groups) + list(index.domains))) if index.kind == "loops" \
            else list(index.domains) + bounds

    def selection(self) -> dict[str, list[str]]:
        """
        {item name: what it selects in the geometry}, checked (module docstring): for a region the domain
        groups (curve loops, surface groups, unnamed surfaces) it covers; for a PML the curve groups its labels select, or its base curves
        ("curve 12") when its selector is not made of labels; for a boundary the named curve groups its
        labels select (kept in the mesh). The edges of a boundary are found on the mesh: model.select(…).
        """
        index = self._index = GeometryIndex(self.source)
        domains = list(index.domains)
        bounds = [n for n in index.groups if n not in index.domains]
        ctx = Domains.from_geometry(index, domains)
        out: dict = {}
        owner: dict = {}
        for r in self.regions:
            if isinstance(r, PML):
                continue
            sel = self._evaluate(r, lambda: r.selector.domains(ctx))
            if not sel:
                raise SelectionError(f"{type(r).__name__}({r.name!r}) selects no domain group of "
                                     f"{self.source.name}; domain groups: {domains}")
            for g in sel:                      # a curve loop is owned whole; a surface once, whatever its groups
                for key in ([g] if index.kind == "loops" else index.domains[g]):
                    if key in owner:
                        what = f"group {g!r}" if index.kind == "loops" else f"surface {key} (in {g!r})"
                        raise SelectionError(f"{what} is selected by both {owner[key]!r} and {r.name!r}")
                    owner[key] = r.name
            out[r.name] = [g for g in domains if g in sel]
        meshed = [g for g in domains if any(g in v for v in out.values())]
        self._pml_bases = self._select_pml(index, bounds, meshed, dict(out), out)
        for b in self.boundaries:
            sel = set()
            for label in b.selector.labels():
                found = [g for g in bounds if naming.matches(label, g)]
                if not found:
                    raise SelectionError(f"{type(b).__name__}({b.name!r}): {label!r} selects no boundary group of "
                                         f"{self.source.name}; boundary groups: {bounds}")
                sel |= set(found)
            out[b.name] = [g for g in bounds if g in sel]
        return {it.name: out[it.name] for it in self.regions + self.boundaries}

    def _select_pml(self, index, bounds, meshed, regions, out) -> dict:
        """
        The base curves of each PML region, selected on the curves of the geometry (geometry.selectors.Curves,
        sides = the meshed loops): {name: (curve tags in order, {tag: name of its layer surface})}. A selector
        made of labels keeps the order of the curve groups and names each layer after its group (as the
        meshes written before); any other selector names the layers after the region. Fills out[name].
        """
        pmls = [r for r in self.regions if isinstance(r, PML)]
        if not pmls:
            return {}
        curves = Curves(index, meshed, regions)
        bases: dict = {}
        taken: dict = {}
        for r in pmls:
            tags = self._evaluate(r, lambda: r.selector.edges(curves))
            if not tags:
                raise SelectionError(f"PML({r.name!r}) selects no curve of {self.source.name} "
                                     f"(selector {r.selector!r})")
            for t in sorted(tags):
                if t in taken:
                    raise SelectionError(f"curve {t} is the base of both PML {taken[t]!r} and {r.name!r}")
                taken[t] = r.name
            if r.selector.by_groups:
                groups = [g for g in bounds if any(naming.matches(lb, g) for lb in r.selector.labels())]
                order = list(dict.fromkeys(c for g in groups for c in index.groups[g] if c in tags))
                names = {c: next(g for g in groups if c in index.groups[g]) for c in order}
                out[r.name] = groups
            else:
                order = sorted(tags)
                names = {c: r.name for c in order}
                out[r.name] = [f"curve {c}" for c in order]
            bases[r.name] = (order, names)
        return bases

    @staticmethod
    def _evaluate(item, fn):
        try:
            return fn()
        except SelectionError as e:
            raise SelectionError(f"{type(item).__name__}({item.name!r}): {e}") from None

    def select(self, selector, kind: str = "boundary"):
        """
        Evaluate a selector (lp.Group, lp.Box, lp.Interface…, a label or an item) on the meshed model:
        kind "boundary" → (n, 3) edges [corner, corner, mid] (node tags), "domain" → mesh group names.
        """
        if self.meshed is None:
            raise StateError("call model.mesh() first")
        sel = as_selector(selector)
        if kind == "boundary":
            assert self._edges is not None                 # set by mesh()
            return self._edges.rows_of(sel.edges(self._edges))
        if kind == "domain":
            assert self._edges is not None
            names = sel.domains(self._edges.domains)
            return [n for n in self.meshed.names(2) if n in names]
        raise ValueError(f"kind must be 'boundary' or 'domain', got {kind!r}")

    # ── meshing ────────────────────────────────────────────────────────────
    def mesh(self, path=None, *, reuse: bool = True, deterministic: bool = True,
             show: bool = False, size_field=None) -> Mesh:
        """
        Mesh the model (only the selected domains and boundaries) and write it to `path`
        (default: next to the geometry) with its signature file.

        reuse         : load the saved mesh when it was built from the same geometry and settings (the
                        thread count is recorded but does not decide: a saved file is reproducible)
        deterministic : one meshing thread (default), so that the same model always gives the same mesh;
                        False: all threads (faster, the mesh may change from run to run)
        show          : open the gmsh window after meshing
        size_field    : a gmsh size view (.pos, written by loudpy.fem.adapt / lp.Adaptive): element sizes
                        anywhere in the geometry, the smaller of it and the regions' sizes wins
        Parameter references (lp.P) are resolved first.
        """
        with Model._Resolved(self):
            return self._mesh(path, reuse=reuse, deterministic=deterministic, show=show, size_field=size_field)

    def _mesh_manager(self, deterministic: bool = True, size_field=None):
        """The mesher of the model as it is (selection checked): (MeshManager, selection, kept boundary groups)."""
        from loudpy.geometry.meshing2d import MeshManager, PMLBase
        selection = self.selection()                            # checks the names first
        kept = sorted({g for b in self.boundaries for g in selection[b.name]})
        sizing = self._sizing()
        plan, entries, self._plan_names = self._plan(selection, {sz.name: sz for sz in sizing})
        sizing += [d.sizing for d in plan if not any(d.sizing is z for z in sizing)]   # lp.MeshSize overrides
        pml = [PMLBase(sz, *self._pml_bases[sz.name]) for sz in sizing if sz.kind == "pml"]
        mm = MeshManager(self.source, sizing, plan, mesh_order=self.mesh_order, grading=self.grading,
                         interfaces=kept, deterministic=deterministic,
                         groups={"domains": entries, "boundaries": kept}, pml=pml,
                         size_field=None if size_field is None else str(size_field))
        return mm, selection, kept

    def geometry_size(self) -> float:
        """The diagonal of the geometry's bounding box [m] (from gmsh, no mesh)."""
        from loudpy.geometry.gmsh_session import gmsh_session
        with gmsh_session() as gmsh:
            self.source.load(gmsh)
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(-1, -1)
        return float(np.linalg.norm([x1 - x0, y1 - y0, z1 - z0]))

    def preview_mesh(self, path=None, elements: int = 60) -> Mesh:
        """
        A display mesh of the geometry alone (no region needed): every domain group meshed, elements of about
        1/`elements` of the geometry's diagonal (thin parts keep a few elements across), on a copy of the model
        with placeholder solids — the model itself is not changed. path: default <geometry>_preview.msh.
            mesh = lp.Model("part.step").preview_mesh();  mesh.names(2)      # the domain groups
        """
        if path is None:
            if self.source.path is None:
                raise ModelError("the geometry is a builder function: give the mesh file, model.preview_mesh(path)")
            path = self.source.path.with_name(self.source.path.stem + "_preview.msh")
        return self.placeholder(elements).mesh(path)

    def placeholder(self, elements: int = 60, *, prefix: str = "", curves: bool = False) -> "Model":
        """
        An unmeshed copy of the model made for showing its geometry alone: a placeholder solid on every domain group
        (named prefix + the group, elements of about 1/`elements` of the geometry's diagonal); with curves=True every
        named curve group is kept in its mesh (a boundary "curve:<group>", never solved). The model is not changed.
        """
        domains = self.groups("domain")
        if not domains:
            raise ModelError("the geometry has no surface (domain) group to mesh")
        material = next((n for n, m in self.materials.items() if m.kind == "solid" and m.E), None)
        if material is None:
            raise ModelError("the material library has no solid material for the placeholder regions")
        h = self.geometry_size() / int(elements)
        twin = self.copy()
        twin.regions, twin.boundaries, twin.mesh_controls = [], [], []
        twin.add(*[Solid(prefix + g, material, on=g if prefix else None, mesh_size=h) for g in domains])
        if curves:
            from loudpy.model.boundaries import Clamped
            twin.add(*[Clamped(f"curve:{g}", on=g) for g in self.groups("boundary") if g not in domains])
        return twin

    def to_script(self, folder=None, title: str | None = None) -> str:
        """The model as Python code that builds it again (loudpy.model.script); folder: where the script will be
        (paths written relative to it), title: its first docstring line."""
        from loudpy.model.script import model_script
        return model_script(self, folder, title)

    def uses(self, name: str) -> list[str]:
        """What refers to parameter `name`: the items (lp.P(name) in their fields, found by resolving them without
        it) and "geometry" when it sets a DefineConstant of the .geo."""
        from loudpy.model.parameters import has_references
        rest = Parameters(**{k: float(v) for k, v in self.parameters.items() if k != name})
        out = []
        for it in list(self.regions) + list(self.boundaries) + list(self.mesh_controls):
            if not has_references(it):
                continue
            try:
                resolve(it, rest)
            except ModelError:
                out.append(it.name or repr(it))
        if name in self.source.geo_constants:
            out.append("geometry")
        return out

    def mesh_reusable(self, path, deterministic: bool = True, size_field=None) -> bool:
        """True when the saved mesh `path` was built from this model's geometry and mesh settings (model.mesh(path)
        would load it instead of meshing), e.g. a parameter that changes no mesh input."""
        if not Path(path).exists():
            return False
        with Model._Resolved(self):
            mm, _, _ = self._mesh_manager(deterministic, size_field)
            return bool(mm.can_reuse(str(path))[0])

    def _mesh(self, path=None, *, reuse: bool = True, deterministic: bool = True,
              show: bool = False, size_field=None) -> Mesh:
        """
        Mesh the model (only the selected domains and boundaries) and write it to `path`
        (default: next to the geometry) with its signature file.

        reuse         : load the saved mesh when it was built from the same geometry and settings (the
                        thread count is recorded but does not decide: a saved file is reproducible)
        deterministic : one meshing thread (default), so that the same model always gives the same mesh;
                        False: all threads (faster, the mesh may change from run to run)
        show          : open the gmsh window after meshing
        """
        if path is None and self.source.path is None:
            raise ModelError("the geometry is a builder function: give the mesh file, model.mesh(path)")
        if not self.regions:
            raise ModelError("the model has no region to mesh: add regions (lp.Solid, lp.Air, lp.PML …); to look at "
                             "the geometry alone, model.preview_mesh()")
        path = Path(path) if path is not None else self.source.path.with_suffix(".msh")
        mm, selection, kept = self._mesh_manager(deterministic, size_field)
        ok, why = mm.can_reuse(str(path)) if reuse else (False, "")
        if ok:
            log.info("mesh reused: %s", path)
            domains, boundaries, nodes = mm.load_mesh(str(path))
        else:
            if reuse:
                log.info("meshing %s (%s)", path, why)
            domains, boundaries, nodes = mm.generate_mesh(str(path), True, show_gui=show, extract_mesh_data=True)
        self._mesh_path = path
        self._selection = selection
        self.meshed = Mesh.from_axisymmetric(domains, boundaries, nodes, str(path))
        layers = mm.pml_layers          # {PML surface: geometry}; None for a mesh written before (in the names)
        self._pml_layer = {n: PMLLayer(**{k: v for k, v in m.items() if k != "base"})
                           for n, m in (layers or {}).items()}
        self._region_groups = {}
        for r in self.regions:
            if isinstance(r, PML) and layers is not None:
                tags = set(self._pml_bases[r.name][0])
                groups = [g.name for g in domains if g.name in layers and layers[g.name]["base"][0] in tags]
            elif isinstance(r, PML):                                    # mesh written before: the names
                groups = [g.name for g in domains if naming.source_group(g.name) in selection[r.name]]
            else:
                groups = [g.name for g in domains if g.name in self._plan_names[r.name]]
            if not groups:
                raise MeshError(f"{type(r).__name__}({r.name!r}) has no surface in the mesh {path.name} "
                                f"(remesh with reuse=False)")
            self._region_groups[r.name] = groups
        self._edges = Edges(self.meshed, regions=self._region_groups)
        self.parts = self._parts(domains, boundaries)
        return self.meshed

    def _plan(self, selection, sizing):
        """
        The domains to mesh, in the order of the geometry: [DomainPlan] (one mesh group each: the name of its
        group, or for unnamed surfaces the name of the region), the signature entries [group, kind, size, PML
        thickness] (an unnamed surface: "surface_7 as cone"; a PML: its curve groups or "curve 12"), and
        {region: its mesh group names}.
        """
        from loudpy.geometry.meshing2d import DomainPlan
        index = self._index
        owner = {g: r for r in self.regions if not isinstance(r, PML) for g in selection[r.name]}
        plan, entries, names = [], [], {r.name: [] for r in self.regions}
        used = {}
        control = self._control_sizes(index, owner)                    # {group: size} of lp.MeshSize
        overridden = {}
        for g, tags in index.domains.items():
            r = owner.get(g)
            if r is None:
                continue
            sz = sizing[r.name]
            if g in control and control[g] != sz.size:                 # one sizing per (region, size)
                sz = overridden.setdefault((r.name, control[g]), DomainSizing(sz.name, sz.kind, control[g], sz.t))
            name, key = (r.name, f"{g} as {r.name}") if g in index.unnamed else (g, g)
            if used.setdefault(name, r.name) != r.name:
                raise ModelError(f"mesh group {name!r} would hold domains of {used[name]!r} and {r.name!r}: "
                                 f"rename the region")
            plan.append(DomainPlan(sz, name, "loop" if index.kind == "loops" else "surfaces", list(tags)))
            entries.append([key, sz.kind, sz.size, sz.t])
            if name not in names[r.name]:
                names[r.name].append(name)
        for r in self.regions:
            if isinstance(r, PML):
                sz = sizing[r.name]
                entries += [[g, sz.kind, sz.size, sz.t] for g in selection[r.name]]
        return plan, entries, names

    def _control_sizes(self, index, owner) -> dict:
        """{domain group: size} from the mesh controls (the last control selecting a group wins)."""
        if not self.mesh_controls:
            return {}
        domains = list(index.domains)
        ctx = Domains.from_geometry(index, domains)
        out = {}
        for c in self.mesh_controls:
            try:
                sel = [g for g in c.selector.domains(ctx) if g in owner]
            except SelectionError as e:
                raise SelectionError(f"MeshSize({c.label!r}): {e}") from None
            if not sel:
                raise SelectionError(f"MeshSize({c.label!r}) selects no meshed domain group; meshed groups: "
                                     f"{[g for g in domains if g in owner]}")
            out.update({g: c.size for g in sel})
        return out

    def _sizing(self) -> list[DomainSizing]:
        out = []
        for r in self.regions:
            if isinstance(r, Solid):
                out.append(DomainSizing(r.name, "solid", r.mesh_size))
            elif isinstance(r, PML):
                out.append(DomainSizing(r.name, "pml", r.mesh_size, r.thickness))
            elif isinstance(r, Air):
                out.append(DomainSizing(r.name, "air", r.mesh_size))
            else:
                raise ModelError(f"cannot mesh {r!r}")
        return out

    def _parts(self, domains, boundaries):
        """
        The physics parts: one per (region, mesh group) in the order of the mesh; then for each boundary one
        part per named curve group when its selector is made of labels only (lp.Group), else one part named
        after the boundary with the selected edges.
        """
        from loudpy.geometry.meshing2d import MeshPart
        from loudpy.physics.parts import boundary_part, orient_couplings, region_part
        out = []
        for g in domains:
            for r in self.regions:
                if g.name in self._region_groups[r.name]:
                    out.append(region_part(r, g, self.materials, layer=self._pml_layer.get(g.name)))
        self._item_parts = {b.name: [] for b in self.boundaries}
        edges = {}
        for b in self.boundaries:
            codes = self._evaluate(b, lambda: b.selector.edges(self._edges))
            if not codes:
                raise SelectionError(f"{type(b).__name__}({b.name!r}) selects no edge of the mesh "
                                     f"(selector {b.selector!r})")
            edges[b.name] = codes
        for g in boundaries:
            for b in self.boundaries:
                if b.selector.by_groups and g.name.strip() in self._selection[b.name]:
                    p = boundary_part(b, g)
                    out.append(p)
                    self._item_parts[b.name].append(p)
        for b in self.boundaries:
            if b.selector.by_groups:
                continue
            rows = self._edges.rows_of(edges[b.name])
            tags = np.unique(rows)
            group = MeshPart(tags, self.meshed.coords_of(tags), rows, b.name, 1)
            p = boundary_part(b, group)
            out.append(p)
            self._item_parts[b.name].append(p)
        orient_couplings(out)                    # coupling normals from the mesh (outward of the air)
        return out

    def boundary_nodes(self) -> dict:
        """{boundary name: {"node_tags", "edges"}} of the meshed boundaries (stored with the results)."""
        if self.meshed is None:
            raise StateError("call model.mesh() first")
        out = {}
        for b in self.boundaries:
            groups = self._item_parts.get(b.name, [])
            tags = np.unique(np.concatenate([np.asarray(g.node_tags) for g in groups])) if groups \
                else np.zeros(0, np.int64)
            edges = np.vstack([np.asarray(g.edges) for g in groups]).astype(np.int64) if groups \
                else np.zeros((0, 3), np.int64)
            out[b.name] = {"node_tags": tags.astype(np.int64), "edges": edges}
        return out

    # ── overview and checks ────────────────────────────────────────────────
    def check(self) -> list[str]:
        """See _check (parameter references resolved first)."""
        try:
            with Model._Resolved(self):
                return self._check()
        except ModelError as e:
            if "unknown parameter" in str(e) and "the model has problems" not in str(e):
                raise ModelError(f"the model has problems:\n  - {e}") from None
            raise

    def _check(self) -> list[str]:
        """
        Check the model without meshing: every item selects something (and no domain group twice), every
        region's material is complete (rho, E, nu for a solid; rho, c for air; damping fits when the
        damping asks for them) and plausible, a PML has its tuning frequency, the mesh controls select
        meshed groups. Raises ModelError listing every problem; returns the notes (e.g. no coil).
        """
        from loudpy.physics.parts import region_parameters
        from loudpy.model.boundaries import Coil, FluidStructure
        problems, notes = [], []
        if not self.regions:
            problems.append("no region (add lp.Solid / lp.Air …)")
        selection = None
        try:
            selection = self.selection()
        except (SelectionError, ModelError) as e:
            problems.append(str(e))
        for r in self.regions:
            try:
                region_parameters(r, self.materials)
            except (ModelError, KeyError, ValueError) as e:
                problems.append(f"{type(r).__name__}({r.name!r}): {e}")
        if selection is not None and self.mesh_controls:
            owner = {g: r for r in self.regions if not isinstance(r, PML) for g in selection[r.name]}
            try:
                self._control_sizes(self._index, owner)
            except SelectionError as e:
                problems.append(str(e))
        if not any(isinstance(b, Coil) for b in self.boundaries):
            notes.append("no lp.Coil: modes only (a forced response needs a coil)")
        if any(isinstance(r, Air) for r in self.regions) and \
                not any(isinstance(b, FluidStructure) for b in self.boundaries) and \
                any(isinstance(r, Solid) for r in self.regions):
            notes.append("air and structure but no lp.FluidStructure: they are not coupled")
        if problems:
            raise ModelError("the model has problems:\n  - " + "\n  - ".join(problems))
        return notes

    def tree(self, selections: bool = False) -> str:
        """
        The model as a tree, like COMSOL's model builder: geometry, materials, regions, boundaries, mesh
        controls, mesh. selections=True evaluates what each item selects (reads the geometry).
        """
        sel = self._selection or (self.selection() if selections else {})

        def size(v):
            from loudpy.model.parameters import P
            return repr(v) if isinstance(v, P) else f"{v * 1e3:g} mm"

        def what(it):
            return f"  → {', '.join(sel[it.name])}" if it.name in sel else ""

        def material(r):
            m = r.material if isinstance(r.material, str) else r.material.name
            return m
        lines = [f"Model {self.source.name}  (2D axisymmetric, P{self.mesh_order} elements, grading {self.grading})",
                 f"├─ Geometry: {self.source.name}",
                 *([f"├─ Parameters: {', '.join(f'{k} = {v:g}' for k, v in self.parameters.items())}"]
                   if len(self.parameters) else []),
                 f"├─ Materials: {getattr(self.materials, 'path', None) or 'loudpy library'}"]
        lines.append(f"├─ Regions ({len(self.regions)})")
        for k, r in enumerate(self.regions):
            last = k == len(self.regions) - 1
            extra = []
            if isinstance(r, Solid):
                extra += [material(r), type(r.damping).__name__]
            elif isinstance(r, PML):
                extra += [f"thickness {size(r.thickness)}", f"tuned to {r.tuned_to} Hz"]
            else:
                extra += [material(r)]
            extra.append(f"h {size(getattr(r, 'mesh_size', None))}")
            on = "" if r.on is None else f" on {r.on!r}"
            lines.append(f"│  {'└' if last else '├'}─ {type(r).__name__} {r.name!r}{on}: {', '.join(extra)}{what(r)}")
        lines.append(f"├─ Boundaries ({len(self.boundaries)})")
        for k, b in enumerate(self.boundaries):
            last = k == len(self.boundaries) - 1
            detail = {"Clamped": f"dofs {getattr(b, 'dofs', '')}", "Coil": f"direction {getattr(b, 'direction', '')}"}
            extra_b = detail.get(type(b).__name__, "")
            on = "" if b.on is None else f" on {b.on!r}"
            lines.append(f"│  {'└' if last else '├'}─ {type(b).__name__} {b.name!r}{on}"
                         f"{': ' + extra_b if extra_b else ''}{what(b)}")
        if self.mesh_controls:
            lines.append(f"├─ Mesh controls ({len(self.mesh_controls)})")
            for k, c in enumerate(self.mesh_controls):
                lines.append(f"│  {'└' if k == len(self.mesh_controls) - 1 else '├'}─ MeshSize {c.label}: "
                             f"h {size(c.size)}")
        if self.meshed is None:
            lines.append("└─ Mesh: not meshed")
        else:
            cells: dict = {}
            for g in self.meshed.groups.values():
                cells[g.cell_type] = cells.get(g.cell_type, 0) + len(g)
            lines.append(f"└─ Mesh: {self._mesh_path.name if self._mesh_path else ''}, {self.meshed.n_nodes} nodes, "
                         + ", ".join(f"{n} {t}" for t, n in cells.items()))
        return "\n".join(lines)

    def __str__(self):
        return self.tree()

    def __repr__(self):
        return (f"Model({self.source.name}: {len(self.regions)} regions, {len(self.boundaries)} boundaries, "
                f"{'meshed' if self.meshed else 'not meshed'})")
