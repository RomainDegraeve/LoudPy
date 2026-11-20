"""
Model: a 2D axisymmetric model — a geometry (.geo of the loudpy convention), its regions and
boundaries, its materials.

    model = lp.Model("HPNEW-Sketch.geo")
    model.add(
        lp.Solid("membranne", "Paper", mesh_size=0.5e-3),
        lp.Solid("surround", "Rubber", damping=lp.Hysteretic()),
        lp.Clamped("interface_constrained_rz"), lp.Clamped("interface_constrained_r", dofs="r"),
        lp.Coil("interface_forced"),
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
from loudpy.geometry.gmsh_session import physical_groups
from loudpy.geometry.selectors import Domains, Edges, as_selector
from loudpy.geometry.mesh import Mesh
from loudpy.geometry.meshing2d import DEFAULT_MESH_GRADING, DomainSizing
from loudpy.model.boundaries import Boundary
from loudpy.model.materials import MaterialLibrary, load_materials
from loudpy.model.regions import PML, Air, Region, Solid
from loudpy.errors import ModelError, SelectionError, StateError

log = logging.getLogger("loudpy.model")

SUBDOMAINS_KEY = "sub"


def _is_domain_group(name: str) -> bool:
    t = naming.tokens(name)
    return bool(t) and (naming.is_prefix(t[0]) or t[0].upper() == "PML")


class Model:
    """An axisymmetric loudspeaker model: a geometry (.geo), regions, boundaries and materials.

        model = lp.Model("speaker.geo")
        model.add(lp.Solid("membrane", "Paper"), lp.Clamped("interface_constrained_rz"), lp.Clamped("interface_constrained_r", dofs="r"), lp.Coil("interface_forced"))
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
        self.parts: list = []                  # physics parts of the meshed model (loudpy.physics.parts)
        self._mesh_path: Path | None = None
        self._selection: dict = {}             # of the last meshing: {item: groups of the geometry}
        self._region_groups: dict = {}         # {region: mesh groups}
        self._item_parts: dict = {}            # {boundary: its parts}
        self._edges: Edges | None = None       # the boundary edges of the mesh (selector evaluation)

    # ── description ────────────────────────────────────────────────────────
    def add(self, *items) -> "Model":
        """Add regions and boundaries (or lists of them). Returns the model."""
        for it in items:
            if isinstance(it, (list, tuple)):
                self.add(*it)
                continue
            if not isinstance(it, (Region, Boundary)):
                raise ModelError(f"cannot add {it!r} to a Model (expected a region or a boundary)")
            if any(o.name == it.name for o in self.regions + self.boundaries):
                raise ModelError(f"the model already has an item named {it.name!r} (names are the labels of the "
                                 f"results: give each item its own)")
            (self.regions if isinstance(it, Region) else self.boundaries).append(it)
        self.meshed = None
        return self

    def __getitem__(self, name):
        """The region or boundary added with this name."""
        for it in self.regions + self.boundaries:
            if it.name == name:
                return it
        raise SelectionError(name)

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
        """
        {item name: groups of the geometry}, checked (module docstring): for a region the domain groups it
        covers; for a boundary the named curve groups its labels select (kept in the mesh). The edges of a
        boundary are found on the mesh: model.select(boundary).
        """
        domains, bounds = self.groups("domain"), self.groups("boundary")
        ctx = Domains.from_geometry(self.geometry, domains)
        out, owner = {}, {}
        for r in self.regions:
            sel = self._evaluate(r, lambda: r.selector.domains(ctx))
            if not sel:
                raise SelectionError(f"{type(r).__name__}({r.name!r}) selects no domain group of "
                                     f"{self.geometry.name}; domain groups: {domains}")
            for g in sel:
                if g in owner:
                    raise SelectionError(f"group {g!r} is selected by both {owner[g]!r} and {r.name!r}")
                owner[g] = r.name
            out[r.name] = [g for g in domains if g in sel]
        for b in self.boundaries:
            sel = set()
            for label in b.selector.labels():
                found = [g for g in bounds if naming.matches(label, g)]
                if not found:
                    raise SelectionError(f"{type(b).__name__}({b.name!r}): {label!r} selects no boundary group of "
                                         f"{self.geometry.name}; boundary groups: {bounds}")
                sel |= set(found)
            out[b.name] = [g for g in bounds if g in sel]
        return out

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
            return self._edges.rows_of(sel.edges(self._edges))
        if kind == "domain":
            names = sel.domains(self._edges.domains)
            return [n for n in self.meshed.names(2) if n in names]
        raise ValueError(f"kind must be 'boundary' or 'domain', got {kind!r}")

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
        selection = self.selection()                            # checks the names first
        path = Path(path) if path is not None else self.geometry.with_suffix(".msh")
        members = {r.name: set(selection[r.name]) for r in self.regions}
        kept = sorted({g for b in self.boundaries for g in selection[b.name]})
        mm = MeshManager(self._sizing(), SUBDOMAINS_KEY, mesh_order=self.mesh_order, grading=self.grading,
                         interfaces=kept, match=lambda spec, group: group.strip() in members.get(spec, ()),
                         deterministic=deterministic, groups=self._meshed_groups(selection))
        ok, why = mm.can_reuse(str(self.geometry), str(path)) if reuse else (False, "")
        if ok:
            log.info("mesh reused: %s", path)
            domains, boundaries, nodes = mm.load_mesh(str(path))
        else:
            if reuse:
                log.info("meshing %s (%s)", path, why)
            domains, boundaries, nodes = mm.generate_mesh(str(self.geometry), str(path), True,
                                                          show_gui=show, extract_mesh_data=True)
        self._mesh_path = path
        self._selection = selection
        self.meshed = Mesh.from_axisymmetric(domains, boundaries, nodes, str(path))
        self._region_groups = {r.name: [g.name for g in domains if naming.source_group(g.name) in selection[r.name]]
                               for r in self.regions}
        self._edges = Edges(self.meshed, regions=self._region_groups)
        self.parts = self._parts(domains, boundaries)
        return self.meshed

    def _meshed_groups(self, selection) -> dict:
        """The groups the mesh is built from (its signature): domain groups with kind, size, PML thickness;
        boundary groups."""
        sizing = {d.name: d for d in self._sizing()}
        domains = [[g, sizing[r.name].kind, sizing[r.name].size, sizing[r.name].t]
                   for r in self.regions for g in selection[r.name]]
        boundaries = sorted({g for b in self.boundaries for g in selection[b.name]})
        return {"domains": domains, "boundaries": boundaries}

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
                    out.append(region_part(r, g, self.materials))
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

    def __repr__(self):
        return (f"Model({self.geometry.name}: {len(self.regions)} regions, {len(self.boundaries)} boundaries, "
                f"{'meshed' if self.meshed else 'not meshed'})")
