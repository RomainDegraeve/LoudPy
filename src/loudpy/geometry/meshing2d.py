"""
loudpy.geometry.meshing2d — the 2D (axisymmetric) mesher: MeshManager loads a geometry source into gmsh,
builds the planned domains and the PML layers, applies the size fields, meshes and extracts the mesh
groups (MeshPart, NodeSet); the mesh signature decides reuse.
"""
import gmsh
import json
import logging
import os
import multiprocessing
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from loudpy.errors import MeshError

log = logging.getLogger("loudpy.mesh")

DEFAULT_MESH_GRADING = 0.3

SIGNATURE_VERSION = 2
# recorded in the signature but not deciding reuse: a saved file is reproducible whatever the thread count
REUSE_IGNORES = ("deterministic",)


@dataclass(eq=False)
class DomainSizing:
    """What the mesher needs from a region: its name, kind ("solid", "air", "pml"), element size [m]
    and, for a PML, its thickness t [m]."""
    name: str
    kind: str
    size: float
    t: Optional[float] = None

    @property
    def is_structure(self) -> bool:
        """True for a solid domain."""
        return self.kind.startswith("solid")


@dataclass
class NodeSet:
    """All the nodes of a mesh: tags (n,) and coordinates (n, 3)."""
    node_tags: np.ndarray
    node_coords: np.ndarray


@dataclass
class MeshPart:
    """One physical group of the 2D mesh: its cells (TRI6 for a domain, LINE3 for a boundary)
    as node tags, and the tags / coordinates of its nodes."""
    node_tags: np.ndarray
    node_coords: np.ndarray
    cells: np.ndarray
    name: str
    dim: int

    @property
    def tri(self) -> np.ndarray:
        """Cells of a domain part (node tags)."""
        return self.cells

    @property
    def edges(self) -> np.ndarray:
        """Cells of an edge part (node tags)."""
        return self.cells


@dataclass
class DomainPlan:
    """One meshed domain: the mesh group `name` of the region `sizing`, from a curve loop (kind "loop": the
    curve tags, made a plane surface) or from existing surfaces (kind "surfaces": the surface tags)."""
    sizing: DomainSizing
    name: str
    kind: str
    tags: List[int]


@dataclass
class PMLBase:
    """The base of a PML region: the curves (gmsh tags) its layers grow from, in order, and the name of the
    layer surface of each curve (several curves with one name get _1, _2 …)."""
    sizing: DomainSizing
    curves: List[int]
    names: Dict[int, str]


class MeshManager:
    """
    Loads a geometry source (geometry.sources.GeometrySource), builds the domains of the plan (surfaces from
    curve loops, or the existing surfaces) and the PML layers, and meshes them.

    Mesh sizes: each domain (DomainSizing) has its own size (``size``, restricted to its surfaces).
    With ``grading`` (default ``DEFAULT_MESH_GRADING``), the element size also grows linearly
    with the distance d to every structural domain (kind "solid"),

        h = min( size of the region,  size of the structural domain + grading · d ),

    and gmsh does not extend the boundary sizes into the neighbouring domains
    (``Mesh.MeshSizeExtendFromBoundary = 0``). The fine structural size then stays local to
    the fluid–structure interfaces instead of spreading far into the air: for a loudspeaker,
    about half the air nodes of the default gmsh mesh for the same element quality.
    ``grading=None`` (or 0) gives the previous behaviour (gmsh extension from the boundary).

    Only what the simulation uses is meshed: the domains of ``plan`` (DomainPlan, one mesh group each),
    PML layers grown from the base curves of ``pml`` (PMLBase, see _build_pml_surfaces), the curve groups
    named in ``interfaces``; every other surface, curve and point is removed before meshing. E.g. a
    structural study on a .geo that also draws the air meshes and stores only the structure.

    ``deterministic=True`` meshes on one thread: gmsh with several threads does not give the
    same triangulation twice for the same .geo.
    """

    def __init__(self, source, domains: List[DomainSizing], plan: List[DomainPlan], mesh_order: int = 2,
                 grading: Optional[float] = DEFAULT_MESH_GRADING, interfaces: Optional[List[str]] = None,
                 deterministic: bool = False, groups: Optional[Dict[str, Any]] = None,
                 pml: Optional[List[PMLBase]] = None, size_field: Optional[str] = None):
        self.source = source          # geometry.sources.GeometrySource
        self.size_field = size_field  # a gmsh size view (.pos, loudpy.fem.adapt), combined (Min) with the sizes
        self.SimObjects = list(domains)
        self.plan = list(plan)
        self.mesh_order = mesh_order
        self.grading = grading
        self.interfaces = list(interfaces or [])
        self.deterministic = deterministic
        self.groups = groups          # what is meshed: {"domains": [[group, kind, size, t]], "boundaries": [group]}
        self.pml = list(pml or [])    # PML bases (PMLBase)
        self.pml_layers: Optional[Dict[str, Dict[str, Any]]] = None   # {PML surface: geometry}; None: not known

    # =========================================================================
    # CORE MESH GENERATION
    # =========================================================================

    def generate_mesh(self, msh_path: str, write_mesh_file: bool, show_gui: bool = False,
                      extract_mesh_data: bool = False):
        """Build and mesh the geometry with gmsh: write it to msh_path when write_mesh_file, open the gmsh
        window when show_gui, and return the extracted mesh data when extract_mesh_data."""
        self._initialize_gmsh(show_gui)
        try:
            self.source.load(gmsh)
            surfaces_by_obj = {obj: [] for obj in self.SimObjects}

            # Step 1: domains and PML layers
            normal_surfaces = self._build_domains(surfaces_by_obj)
            pml_surfaces = self._build_pml_surfaces(normal_surfaces, surfaces_by_obj)

            # Step 2: physical groups, and drop what the simulation does not use
            self._assign_physical_groups(normal_surfaces + pml_surfaces)
            self._prune_unused(normal_surfaces + pml_surfaces)

            # Step 3: mesh sizes
            self._apply_mesh_sizes(surfaces_by_obj)

            # Step 4: mesh and save
            self._configure_and_generate_mesh(msh_path, write_mesh_file)
            if write_mesh_file:
                self._write_signature(msh_path)

            if show_gui:
                gmsh.fltk.run()

            if extract_mesh_data:
                return self.extract_mesh_data()

        finally:
            if not extract_mesh_data and gmsh.isInitialized():
                gmsh.finalize()

    # =========================================================================
    # REUSING A SAVED MESH
    # =========================================================================
    # Every written mesh gets a signature file <msh_path>.json (version 2) describing what it was
    # built from: the geometry files (a .geo and the files it merges; for a builder function its code and
    # constants), every meshed domain group (kind, size, PML
    # thickness), every kept boundary group, the mesh order, the grading and the deterministic flag.
    # It records the groups, not the names that selected them: renaming a region or splitting a
    # boundary into two selections that cover the same groups keeps the mesh. A saved mesh is reused
    # only if its signature equals the one of the current model (version-1 signatures never do), except
    # that the saved mesh may keep more boundary groups than the model needs: the kept curve groups do not
    # change the triangulation (every curve bounding a meshed surface is meshed), only what is stored.

    @staticmethod
    def signature_path(msh_path: str) -> str:
        """Path of the signature file of a mesh (<mesh>.json)."""
        return msh_path + ".json"

    def mesh_signature(self) -> Dict[str, Any]:
        """What the mesh depends on (see the section comment)."""
        if self.groups is None:
            raise MeshError("MeshManager.mesh_signature needs the meshed groups (groups=)")
        sig = dict(version=SIGNATURE_VERSION, files=self.source.files(),
                   domains=sorted([list(d) for d in self.groups["domains"]]),
                   boundaries=sorted(self.groups["boundaries"]),
                   mesh_order=self.mesh_order, grading=self.grading, deterministic=bool(self.deterministic))
        if self.size_field is not None:                    # an adapted mesh depends on its size view
            import hashlib
            with open(self.size_field, "rb") as fh:
                sig["size_field"] = hashlib.sha256(fh.read()).hexdigest()
        return sig

    def _write_signature(self, msh_path: str) -> None:
        """The signature, plus what the mesh file cannot hold: the geometry of the PML layers (pml_layers)."""
        sig = self.mesh_signature()
        sig["pml_layers"] = self.pml_layers or {}
        with open(self.signature_path(msh_path), "w") as fh:
            json.dump(sig, fh, indent=1)

    def can_reuse(self, msh_path: str) -> Tuple[bool, str]:
        """(True, "") if msh_path was written from this geometry and these specs."""
        sig = self.signature_path(msh_path)
        if not os.path.exists(msh_path):
            return False, "no saved mesh"
        if not os.path.exists(sig):
            return False, f"no signature file {os.path.basename(sig)}"
        with open(sig) as fh:
            stored = json.load(fh)
        current = json.loads(json.dumps(self.mesh_signature()))       # same JSON types
        changed = [k for k in current if stored.get(k) != current[k] and k not in REUSE_IGNORES
                   and not (k == "boundaries" and set(current[k]) <= set(stored.get(k) or []))]
        return (not changed), ("" if not changed else "changed: " + ", ".join(changed))

    def load_mesh(self, msh_path: str) -> Tuple[List[MeshPart], List[MeshPart], NodeSet]:
        """Read a mesh written by generate_mesh (same data as generate_mesh(..., extract_mesh_data=True))."""
        if not os.path.exists(msh_path):
            raise FileNotFoundError(f"The file {msh_path} could not be found.")
        sig = self.signature_path(msh_path)
        self.pml_layers = None
        if os.path.exists(sig):
            with open(sig) as fh:
                self.pml_layers = json.load(fh).get("pml_layers")      # None: written before (names hold it)
        self._initialize_gmsh(False)
        gmsh.open(msh_path)
        return self.extract_mesh_data()          # finalizes gmsh

    # =========================================================================
    # DATA EXTRACTION
    # =========================================================================

    def extract_mesh_data(self) -> Tuple[List[MeshPart], List[MeshPart], NodeSet]:
        """Domain parts, edge parts and the node set of the current gmsh mesh."""
        try:
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            node_coords = node_coords.reshape(-1, 3)

            max_tag = int(np.max(node_tags))
            coords_map = np.zeros((max_tag + 1, 2))
            coords_map[node_tags] = node_coords[:, :2]

            domains, boundaries = [], []
            # cells of every order: 2D TRI3/6/10, QUAD4/9; 1D LINE2/3/4
            target_elem_types = {2: (2, 9, 21, 3, 10), 1: (1, 8, 26)}
            nodes_per_elem = {1: 2, 8: 3, 26: 4, 2: 3, 9: 6, 21: 10, 3: 4, 10: 9}

            groups = []
            for dim, phys_tag in gmsh.model.getPhysicalGroups():
                name = gmsh.model.getPhysicalName(dim, phys_tag)
                if not name or dim not in target_elem_types:
                    continue
                elements_array = self._get_elements_for_group(dim, phys_tag, target_elem_types, nodes_per_elem,
                                                              name)
                if elements_array is not None:
                    groups.append((dim, name, elements_array))

            # cubic triangles: gmsh's interior node of curved cells costs P3 half an order; put it where
            # the cubic map of the cell's edges places it (fem.reference.tri10_interior_nodes)
            from loudpy.fem.reference import tri10_interior_nodes
            for dim, _, cells in groups:
                if dim == 2 and cells.shape[1] == 10:
                    centre = cells[:, 9]
                    coords_map[centre] = tri10_interior_nodes(coords_map[cells])
                    rows = np.searchsorted(node_tags, centre) if np.all(np.diff(node_tags) > 0) else \
                        np.array([np.flatnonzero(node_tags == t)[0] for t in centre])
                    node_coords[rows, :2] = coords_map[centre]

            for dim, name, elements_array in groups:
                unique_nodes = np.unique(elements_array)
                group_coords = coords_map[unique_nodes]
                if dim == 2:
                    domains.append(MeshPart(unique_nodes, group_coords, elements_array, name, 2))
                else:
                    boundaries.append(MeshPart(unique_nodes, group_coords, elements_array, name, 1))

            return domains, boundaries, NodeSet(node_tags, node_coords)
        finally:
            gmsh.finalize()

    # =========================================================================
    # HELPER METHODS (PRIVATE)
    # =========================================================================

    def _initialize_gmsh(self, show_gui: bool):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1 if show_gui else 0)
        gmsh.option.setNumber("General.NumThreads", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)


    # -------------------------------------------------------------------------
    # FIX 1 — Ordered loop with hard closure validation (no silent fallback)
    # -------------------------------------------------------------------------

    def _build_ordered_loop(self, curve_tags: List[int]) -> List[int]:
        """
        Constructs an ordered, continuously-connected, closed loop of curves.

        Raises ValueError if the loop cannot be closed — this replaces the
        silent `or curve_tags` fallback that used to hide topology errors.
        """
        def get_endpoints(tag: int) -> Tuple[Optional[int], Optional[int]]:
            bnd = gmsh.model.getBoundary([(1, abs(tag))], oriented=True)
            return (bnd[0][1], bnd[1][1]) if len(bnd) >= 2 else (None, None)

        edges = {tag: get_endpoints(tag) for tag in curve_tags}

        # Check all curves have valid boundary data
        for tag, (s, e) in edges.items():
            if s is None or e is None:
                raise MeshError(
                    f"Curve {tag} has incomplete boundary data — cannot build ordered loop."
                )

        ordered = [curve_tags[0]]
        remaining = set(curve_tags[1:])

        for _ in range(len(curve_tags) - 1):
            last = ordered[-1]
            # Tail of the last oriented edge
            tail = edges[abs(last)][1 if last > 0 else 0]

            found_tag = next(
                (t for t in remaining if tail in edges[t]),
                None
            )
            if found_tag is None:
                from collections import defaultdict
                deg = defaultdict(list)
                for t in curve_tags:
                    n1, n2 = edges[t]
                    deg[n1].append(int(t))
                    deg[n2].append(int(t))
                odd = {n: tags for n, tags in sorted(deg.items()) if len(tags) != 2}
                raise MeshError(
                    f"Cannot continue ordered loop at node {tail} (ordered so far: {ordered}). "
                    f"Remaining curves: {remaining}; nodes not shared by exactly two curves: {odd}. "
                    "Check for disconnected or duplicate curves in the physical group."
                )

            s, e = edges[found_tag]
            ordered.append(found_tag if s == tail else -found_tag)
            remaining.remove(found_tag)

        # Verify the loop actually closes
        last = ordered[-1]
        tail_last  = edges[abs(last)][1 if last > 0 else 0]
        first = ordered[0]
        head_first = edges[abs(first)][0 if first > 0 else 1]

        if tail_last != head_first:
            raise MeshError(
                f"Ordered loop does not close: tail node {tail_last} != "
                f"head node {head_first}. Input curves: {curve_tags}"
            )

        return ordered

    # -------------------------------------------------------------------------
    # FIX 2 — Shoelace signed-area helper (single source of truth for winding)
    # -------------------------------------------------------------------------

    @staticmethod
    def _signed_area_2d(pts: List[Tuple[float, float]]) -> float:
        """
        Computes the signed area of a polygon via the Shoelace formula.
        Positive result  →  counter-clockwise (CCW) winding  →  correct for gmsh OCC.
        Negative result  →  clockwise (CW)                   →  must reverse the loop.
        """
        n = len(pts)
        return 0.5 * sum(
            pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
            for i in range(n)
        )

    # -------------------------------------------------------------------------

    def _build_domains(self, surfaces_by_obj: Dict) -> List[Tuple[int, str]]:
        """(surface, mesh group name) of every domain of the plan: a plane surface from each curve loop (its
        curves ordered into a closed loop), or the existing surfaces."""
        surfaces = []
        for d in self.plan:
            if d.kind == "loop":
                try:
                    ordered = self._build_ordered_loop(list(d.tags))
                    s = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop(ordered)])
                    gmsh.model.occ.synchronize()
                except MeshError:
                    raise
                except Exception as ex:
                    raise MeshError(f"cannot build the surface of {d.name!r} from its curves: {ex}") from None
                new = [s]
            else:
                new = list(d.tags)
            surfaces += [(s, d.name) for s in new]
            surfaces_by_obj[d.sizing] += new
        return surfaces

    # -------------------------------------------------------------------------
    # PML layers. Each base curve (PMLBase.curves) must be a straight line r = const or z = const bounding
    # exactly one meshed air surface; its layer is the quad grown by the thickness t along the outward
    # normal of that surface (found with gmsh.model.isInside, whatever the drawing direction of the curve).
    # Where two layers of perpendicular directions share an end point, a corner quad fills the gap. The
    # geometry of every layer goes to self.pml_layers {surface name: {"base": [curve tags], "use_r", "r0",
    # "tr", "sgn_r", "use_z", "z0", "tz", "sgn_z"}} (the fields of geometry.naming.PMLLayer), stored with the
    # mesh signature: nothing is read from the names.
    # -------------------------------------------------------------------------

    def _build_pml_surfaces(self, normal_surfaces: List[Tuple[int, str]],
                            surfaces_by_obj: Dict) -> List[Tuple[int, str]]:
        from collections import Counter
        surfaces: List[Tuple[int, str]] = []
        self.pml_layers = {}
        if not self.pml:
            return surfaces
        built = {s: name for s, name in normal_surfaces}
        kind_of = {s: obj.kind for obj, s_list in surfaces_by_obj.items() for s in s_list}
        corners: Dict[int, List[Dict[str, Any]]] = {}

        def unique(name):
            if name in self.pml_layers or name in built.values():
                raise MeshError(f"two surfaces would be named {name!r}: give the PML regions different names")
            return name

        for base in self.pml:
            obj = base.sizing
            t = obj.t if obj.t is not None else 0.1
            counts, seen = Counter(base.names[c] for c in base.curves), Counter()
            for c in base.curves:
                where = f"PML {obj.name!r}, curve {c}"
                bnd = gmsh.model.getBoundary([(1, c)], oriented=True)
                if len(bnd) < 2:
                    raise MeshError(f"{where}: the curve has no end points")
                pt0, pt1 = abs(bnd[0][1]), abs(bnd[1][1])
                p0 = np.asarray(gmsh.model.getValue(0, pt0, []))
                p1 = np.asarray(gmsh.model.getValue(0, pt1, []))
                d = (p1 - p0)[:2]
                length = float(np.hypot(*d))
                lo, hi = gmsh.model.getParametrizationBounds(1, c)
                middle = np.asarray(gmsh.model.getValue(1, c, [0.5 * (lo[0] + hi[0])]))[:2]
                chord = 0.5 * (p0 + p1)[:2]
                if length < 1e-14 or np.hypot(*(middle - chord)) > 1e-6 * length \
                        or min(abs(d[0]), abs(d[1])) > 1e-9 * length:
                    raise MeshError(f"{where}: a PML layer grows along r or z, its base must be a straight line "
                                    f"r = const or z = const (curve from {p0[:2]} to {p1[:2]})")
                n = np.array([-d[1], d[0]]) / length          # unit normal of the drawn line (the layer follows
                axis_n = np.round(n)                           # it exactly); its direction: (±1, 0) or (0, ±1)
                adjacent = [s for s in gmsh.model.getAdjacencies(1, c)[0] if s in built]
                if len(adjacent) != 1 or kind_of.get(adjacent[0]) != "air":
                    raise MeshError(f"{where}: the base of a PML must bound exactly one meshed air domain "
                                    f"(it bounds {[built[s] for s in adjacent] or 'none'})")
                probe = chord + 1e-4 * length * axis_n
                if gmsh.model.isInside(2, adjacent[0], [probe[0], probe[1], 0.0]):
                    n, axis_n = -n, -axis_n                                 # pointed into the air
                nx, ny = n * t

                p2 = gmsh.model.occ.addPoint(p1[0] + nx, p1[1] + ny, p1[2])
                p3 = gmsh.model.occ.addPoint(p0[0] + nx, p0[1] + ny, p0[2])
                side_a = gmsh.model.occ.addLine(pt1, p2)      # p1 → p2
                top = gmsh.model.occ.addLine(p2, p3)          # p2 → p3
                side_b = gmsh.model.occ.addLine(p3, pt0)      # p3 → p0
                quad = [(p0[0], p0[1]), (p1[0], p1[1]), (p1[0] + nx, p1[1] + ny), (p0[0] + nx, p0[1] + ny)]
                loop = [c, side_a, top, side_b] if self._signed_area_2d(quad) > 0 \
                    else [-c, -side_b, -top, -side_a]                       # counter-clockwise for OCC
                s = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop(loop)])
                gmsh.model.occ.synchronize()

                name = base.names[c]
                if counts[name] > 1:
                    seen[name] += 1
                    name = f"{name}_{seen[name]}"
                name = unique(name)
                axis = 0 if axis_n[0] != 0 else 1
                layer = dict(base=[int(c)], use_r=False, r0=0.0, tr=0.0, sgn_r=0.0,
                             use_z=False, z0=0.0, tz=0.0, sgn_z=0.0)
                key = "r" if axis == 0 else "z"
                layer.update({f"use_{key}": True, f"{key}0": float(chord[axis]), f"t{key}": float(t),
                              f"sgn_{key}": float(axis_n[axis])})
                self.pml_layers[name] = layer
                surfaces.append((s, name))
                surfaces_by_obj[obj].append(s)
                # side lines, traversed origin → outer as sign * line (for the corners)
                for orig, outer, line, sign in [(pt1, p2, side_a, 1), (pt0, p3, side_b, -1)]:
                    corners.setdefault(orig, []).append(dict(outer=outer, line=line, sign=sign, n=axis_n, v=(nx, ny),
                                                             t=t, name=name, obj=obj, curve=int(c)))

        for pt, infos in corners.items():
            if len(infos) != 2:
                continue
            i1, i2 = infos
            if np.array_equal(i1["n"], i2["n"]):
                continue                                   # two layers in line: they share a side, no gap
            if np.array_equal(i1["n"], -i2["n"]):
                raise MeshError(f"PML layers of curves {i1['curve']} and {i2['curve']} grow in opposite "
                                f"directions from one point")
            o = gmsh.model.getValue(0, pt, [])
            cx, cy = o[0] + i1["v"][0] + i2["v"][0], o[1] + i1["v"][1] + i2["v"][1]
            pc = gmsh.model.occ.addPoint(cx, cy, o[2])
            l1 = gmsh.model.occ.addLine(i1["outer"], pc)
            l2 = gmsh.model.occ.addLine(pc, i2["outer"])
            fwd = [i1["sign"] * i1["line"], l1, l2, -i2["sign"] * i2["line"]]
            quad = [(o[0], o[1]), (o[0] + i1["v"][0], o[1] + i1["v"][1]), (cx, cy),
                    (o[0] + i2["v"][0], o[1] + i2["v"][1])]
            loop = fwd if self._signed_area_2d(quad) > 0 else [-e for e in reversed(fwd)]
            s = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop(loop)])
            gmsh.model.occ.synchronize()
            ir, iz = (i1, i2) if i1["n"][0] != 0 else (i2, i1)
            name = unique(f"{i1['name']}_Corner")
            self.pml_layers[name] = dict(base=[i1["curve"], i2["curve"]],
                                         use_r=True, r0=float(o[0]), tr=float(ir["t"]), sgn_r=float(ir["n"][0]),
                                         use_z=True, z0=float(o[1]), tz=float(iz["t"]), sgn_z=float(iz["n"][1]))
            surfaces.append((s, name))
            surfaces_by_obj[i1["obj"]].append(s)
        return surfaces

    def _assign_physical_groups(self, surfaces_list: List[Tuple[int, str]]):
        """One physical surface group per name (with all its surfaces), in order of appearance."""
        from loudpy.geometry.sources import drop_physical_groups
        drop_physical_groups(gmsh, 2)
        by_name: Dict[str, List[int]] = {}
        for s, name in surfaces_list:
            by_name.setdefault(name, []).append(s)
        for name, tags in by_name.items():
            pg = gmsh.model.addPhysicalGroup(2, tags)
            gmsh.model.setPhysicalName(2, pg, name)

    def _prune_unused(self, surfaces_list: List[Tuple[int, str]]) -> None:
        """Keep only the built surfaces, the curve groups named in `interfaces`, and the curves / points
        bounding the built surfaces (see the class docstring)."""
        kept_surfaces = [s for s, _ in surfaces_list]
        kept_curves = {abs(c) for _, c in gmsh.model.getBoundary(
            [(2, s) for s in kept_surfaces], combined=False, oriented=False)} if kept_surfaces else set()
        kept_points = {abs(p) for _, p in gmsh.model.getBoundary(
            [(1, c) for c in kept_curves], combined=False, oriented=False)} if kept_curves else set()

        # curve groups: matched by an interface spec, restricted to curves that are kept
        for _, pg in gmsh.model.getPhysicalGroups(1):
            name = gmsh.model.getPhysicalName(1, pg)
            curves = [c for c in gmsh.model.getEntitiesForPhysicalGroup(1, pg) if c in kept_curves]
            matched = name.strip() in {i.strip() for i in self.interfaces}
            gmsh.model.removePhysicalGroups([(1, pg)])
            if matched and curves:
                gmsh.model.addPhysicalGroup(1, curves, pg, name)
        for _, pg in gmsh.model.getPhysicalGroups(0):
            gmsh.model.removePhysicalGroups([(0, pg)])

        # geometry not bounding a kept surface: never meshed nor stored
        unused = ([(2, s) for _, s in gmsh.model.getEntities(2) if s not in set(kept_surfaces)]
                  + [(1, c) for _, c in gmsh.model.getEntities(1) if c not in kept_curves]
                  + [(0, p) for _, p in gmsh.model.getEntities(0) if p not in kept_points])
        if unused:
            gmsh.model.occ.remove(unused)
            gmsh.model.occ.synchronize()

    @staticmethod
    def _is_structure(obj: DomainSizing) -> bool:
        return obj.is_structure

    def _grading_fields(self, surfaces_by_obj: Dict) -> List[int]:
        """One field per structural domain: its size + grading · (distance to its boundary)."""
        if not self.grading:
            return []
        fields = []
        for obj, s_list in surfaces_by_obj.items():
            if not s_list or not self._is_structure(obj):
                continue
            curves = sorted({abs(c) for _, c in gmsh.model.getBoundary(
                [(2, s) for s in s_list], combined=False, oriented=False)})
            f_dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", curves)
            gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)
            f_ramp = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(f_ramp, "F", f"{obj.size} + {self.grading}*F{f_dist}")
            fields.append(f_ramp)
        return fields

    def _apply_mesh_sizes(self, surfaces_by_obj: Dict):
        restrict_fields = []
        for obj, s_list in surfaces_by_obj.items():
            if not s_list:
                continue

            f_eval = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(f_eval, "F", str(obj.size))

            f_restrict = gmsh.model.mesh.field.add("Restrict")
            gmsh.model.mesh.field.setNumber(f_restrict, "IField", f_eval)
            gmsh.model.mesh.field.setNumbers(f_restrict, "SurfacesList", s_list)
            restrict_fields.append(f_restrict)

        size_fields = restrict_fields + self._grading_fields(surfaces_by_obj)
        if self.size_field is not None:
            from loudpy.fem.adapt import background_field
            size_fields.append(background_field(gmsh, self.size_field))
        if size_fields:
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", size_fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    def _configure_and_generate_mesh(self, msh_path: str, write_mesh_file:bool):
        gmsh.option.setNumber("General.NumThreads", 1 if self.deterministic else multiprocessing.cpu_count())
        gmsh.option.setNumber("Mesh.Binary", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", self.mesh_order)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        if self.grading or self.size_field is not None:
            # sizes come from the fields only (see the class docstring)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        if self.size_field is not None:
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        gmsh.model.mesh.generate(2)
        if write_mesh_file:
            if os.path.dirname(msh_path):
                os.makedirs(os.path.dirname(msh_path), exist_ok=True)
            gmsh.write(msh_path)
            log.info("mesh written: %s", msh_path)

    def _get_elements_for_group(
        self,
        dim: int,
        phys_tag: int,
        target_types: Dict[int, tuple],
        nodes_per_elem: Dict[int, int],
        name: str = "",
    ) -> Optional[np.ndarray]:
        """Cells of a physical group (one element type per group: mixed triangles and quadrangles are
        refused)."""
        group_elements, types = [], set()
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag):
            elem_types, _, node_tags_list = gmsh.model.mesh.getElements(dim, entity)
            for etype, enodes in zip(elem_types, node_tags_list):
                if etype in target_types.get(dim, ()):
                    types.add(int(etype))
                    group_elements.append(enodes.reshape(-1, nodes_per_elem[etype]))
        if len(types) > 1:
            raise MeshError(f"group {name!r} mixes element types {sorted(types)} (e.g. triangles and "
                            f"quadrangles after a partial recombination): one type per group")

        return np.vstack(group_elements) if group_elements else None
