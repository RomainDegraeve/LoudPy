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

Every name must select at least one physical group of the geometry, and no group may be claimed
by two regions; otherwise model.mesh() raises with the list of groups. model.groups() lists them.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from loudpy.geometry import naming
from loudpy.geometry.gmsh_session import physical_groups
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
                raise ModelError(f"cannot add {it!r} to a Model (expected a region or a boundary)")
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
        """{item name: physical groups it selects}, checked (see the module docstring)."""
        domains, bounds = self.groups("domain"), self.groups("boundary")
        out, owner = {}, {}
        for r in self.regions:
            sel = [g for g in domains if naming.matches(r.name, g)]
            if not sel:
                raise SelectionError(f"{type(r).__name__}({r.name!r}) selects no domain group of "
                                 f"{self.geometry.name}; domain groups: {domains}")
            for g in sel:
                if g in owner:
                    raise SelectionError(f"group {g!r} is selected by both {owner[g]!r} and {r.name!r}")
                owner[g] = r.name
            out[r.name] = sel
        for b in self.boundaries:
            sel = [g for g in bounds if naming.matches(b.name, g)]
            if not sel:
                raise SelectionError(f"{type(b).__name__}({b.name!r}) selects no boundary group of "
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
        selection = self.selection()                            # checks the names first
        path = Path(path) if path is not None else self.geometry.with_suffix(".msh")
        mm = MeshManager(self._sizing(), SUBDOMAINS_KEY, mesh_order=self.mesh_order, grading=self.grading,
                         interfaces=[b.name for b in self.boundaries], match=naming.matches,
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
        self.parts = self._parts(domains, boundaries)
        self._mesh_path = path
        self.meshed = Mesh.from_axisymmetric(domains, boundaries, nodes, str(path))
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
        """The physics parts: one per (item, selected mesh group), regions first."""
        from loudpy.physics.parts import boundary_part, orient_couplings, region_part
        out = []
        for g in domains:
            for r in self.regions:
                if naming.matches(r.name, g.name):
                    out.append(region_part(r, g, self.materials))
        for g in boundaries:
            for b in self.boundaries:
                if naming.matches(b.name, g.name):
                    out.append(boundary_part(b, g))
        orient_couplings(out)                    # coupling normals from the mesh (outward of the air)
        return out

    def boundary_nodes(self) -> dict:
        """{boundary name: {"node_tags", "edges"}} of the meshed boundaries (stored with the results)."""
        if self.meshed is None:
            raise StateError("call model.mesh() first")
        out = {}
        for b in self.boundaries:
            groups = [p.group for p in self.parts if hasattr(p, "edges") and naming.matches(b.name, p.name)]
            tags = np.unique(np.concatenate([np.asarray(g.node_tags) for g in groups])) if groups \
                else np.zeros(0, np.int64)
            edges = np.vstack([np.asarray(g.edges) for g in groups]).astype(np.int64) if groups \
                else np.zeros((0, 3), np.int64)
            out[b.name] = {"node_tags": tags.astype(np.int64), "edges": edges}
        return out

    def __repr__(self):
        return (f"Model({self.geometry.name}: {len(self.regions)} regions, {len(self.boundaries)} boundaries, "
                f"{'meshed' if self.meshed else 'not meshed'})")
