import gmsh
import os
import multiprocessing
import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional
from loudpy.ObjectsGeo import Nodes, L3_Edges, TR6_Surfs


class MeshManager:
    def __init__(self, SimObjects: List[Any], subdomains_key: str, mesh_order: int = 2):
        self.SimObjects = SimObjects
        self.subdomains_key = subdomains_key
        self.mesh_order = mesh_order

    # =========================================================================
    # CORE MESH GENERATION
    # =========================================================================

    def generate_mesh(self, geo_path: str, msh_path: str,write_mesh_file:bool, show_gui: bool = False, extract_mesh_data: bool = False):
        if not os.path.exists(geo_path):
            raise FileNotFoundError(f"The file {geo_path} could not be found.")

        self._initialize_gmsh(show_gui)

        try:
            gmsh.open(geo_path)
            gmsh.model.occ.synchronize()

            phys_curves = gmsh.model.getPhysicalGroups(1)
            surfaces_by_obj = {obj: [] for obj in self.SimObjects}

            # Step 1: Create domains and PML layers
            normal_surfaces = self._build_normal_surfaces(phys_curves, surfaces_by_obj)
            pml_surfaces = self._build_pml_surfaces(phys_curves, surfaces_by_obj)

            # Step 2: Assign Physical Groups
            self._assign_physical_groups(normal_surfaces + pml_surfaces)

            # Step 3: Set mesh sizes
            self._apply_mesh_sizes(surfaces_by_obj)

            # Step 4: Generate & Save Mesh
            self._configure_and_generate_mesh(msh_path, write_mesh_file)

            if show_gui:
                gmsh.fltk.run()

            if extract_mesh_data:
                return self.extract_mesh_data()

        finally:
            # Ensures gmsh is closed properly even if an error occurs
            if not extract_mesh_data:
                gmsh.finalize()

    # =========================================================================
    # DATA EXTRACTION
    # =========================================================================

    def extract_mesh_data(self) -> Tuple[List[TR6_Surfs], List[L3_Edges], "Nodes"]:
        try:
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            node_coords = node_coords.reshape(-1, 3)

            max_tag = int(np.max(node_tags))
            coords_map = np.zeros((max_tag + 1, 2))
            coords_map[node_tags] = node_coords[:, :2]

            domains, boundaries = [], []
            target_elem_types = {2: 9, 1: 8, 0: 15}  # 2D: TR6 | 1D: L3 | 0D: Point
            nodes_per_elem = {9: 6, 8: 3, 15: 1}

            for dim, phys_tag in gmsh.model.getPhysicalGroups():
                name = gmsh.model.getPhysicalName(dim, phys_tag)
                if not name:
                    continue

                elements_array = self._get_elements_for_group(dim, phys_tag, target_elem_types, nodes_per_elem)
                if elements_array is None:
                    continue

                unique_nodes = np.unique(elements_array)
                group_coords = coords_map[unique_nodes]

                if dim == 2:
                    domains.append(TR6_Surfs(unique_nodes, group_coords, elements_array, name))
                elif dim == 1:
                    boundaries.append(L3_Edges(unique_nodes, group_coords, elements_array, name))

            return domains, boundaries, Nodes(node_tags, node_coords)
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

    def _get_matched_obj(self, name: str) -> Optional[Any]:
        return next((obj for obj in self.SimObjects if obj.name in name), None)

    def _get_pml_thickness(self, name: str) -> float:
        obj = self._get_matched_obj(name)
        return getattr(obj, 't', 0.1) if obj else 0.1

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
                raise ValueError(
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
                raise ValueError(
                    f"Cannot continue ordered loop at node {tail}. "
                    f"Remaining curves: {remaining}. "
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
            raise ValueError(
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

    def _build_normal_surfaces(
        self,
        phys_curves: List[Tuple[int, int]],
        surfaces_by_obj: Dict,
    ) -> List[Tuple[int, str]]:
        surfaces = []
        for _, pg_tag in phys_curves:
            name = gmsh.model.getPhysicalName(1, pg_tag)
            if self.subdomains_key not in name or "PML" in name.upper():
                continue

            gmsh.model.setPhysicalName(1, pg_tag, name.replace(self.subdomains_key, "cl"))
            curve_tags = list(gmsh.model.getEntitiesForPhysicalGroup(1, pg_tag))

            # _build_ordered_loop now raises on failure — no silent fallback
            ordered = self._build_ordered_loop(curve_tags)

            try:
                cl = gmsh.model.occ.addCurveLoop(ordered)
                s  = gmsh.model.occ.addPlaneSurface([cl])
                gmsh.model.occ.synchronize()
                surfaces.append((s, name))

                if matched_obj := self._get_matched_obj(name):
                    surfaces_by_obj[matched_obj].append(s)
            except Exception as ex:
                print(f"  ERROR building surface for '{name}': {ex}")

        return surfaces

    def _build_pml_surfaces(
        self,
        phys_curves: List[Tuple[int, int]],
        surfaces_by_obj: Dict,
    ) -> List[Tuple[int, str]]:
        surfaces = []

        # Reference "inside" point: centroid of all non-PML curve endpoints
        inside_ref = self._compute_domain_centroid(phys_curves)

        # Maps original node tag → list of side geometry info (for corner generation)
        pml_corners_info: Dict[int, List[Dict]] = {}

        for _, pg_tag in phys_curves:
            name = gmsh.model.getPhysicalName(1, pg_tag)
            if "PML" not in name.upper():
                continue

            thickness = self._get_pml_thickness(name)
            curve_tags = gmsh.model.getEntitiesForPhysicalGroup(1, pg_tag)

            for curve_tag in curve_tags:
                bnd = gmsh.model.getBoundary([(1, curve_tag)], oriented=True)
                if len(bnd) < 2:
                    continue

                pt0, pt1 = abs(bnd[0][1]), abs(bnd[1][1])
                p0 = gmsh.model.getValue(0, pt0, [])
                p1 = gmsh.model.getValue(0, pt1, [])

                dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                length = math.hypot(dx, dy)
                if length < 1e-14:
                    continue

                # Outward normal (perpendicular to edge, scaled by thickness)
                nx, ny = -dy / length, dx / length
                mx, my = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])
                ox, oy = mx - inside_ref[0], my - inside_ref[1]

                if nx * ox + ny * oy < 0:        # flip if pointing inward
                    nx, ny = -nx, -ny

                nx *= thickness
                ny *= thickness

                # Build the 4 vertices of the PML quad (in domain order: p0, p1, p2, p3)
                p2 = gmsh.model.occ.addPoint(p1[0] + nx, p1[1] + ny, p1[2])
                p3 = gmsh.model.occ.addPoint(p0[0] + nx, p0[1] + ny, p0[2])
                gmsh.model.occ.synchronize()

                side_a = gmsh.model.occ.addLine(pt1, p2)   # p1 → p2
                top    = gmsh.model.occ.addLine(p2,  p3)   # p2 → p3
                side_b = gmsh.model.occ.addLine(p3, pt0)   # p3 → p0
                gmsh.model.occ.synchronize()

                # ----------------------------------------------------------
                # FIX: Use Shoelace signed area to determine winding — no
                # cross-product heuristic, no blind try/except fallback.
                # Quad vertices in the order [pt0→pt1→p2→p3] (base edge first)
                # ----------------------------------------------------------
                quad_pts = [
                    (p0[0],       p0[1]),
                    (p1[0],       p1[1]),
                    (p1[0] + nx,  p1[1] + ny),   # p2
                    (p0[0] + nx,  p0[1] + ny),   # p3
                ]
                if self._signed_area_2d(quad_pts) > 0:
                    # Already CCW: traverse base edge forward
                    loop_edges = [curve_tag, side_a, top, side_b]
                else:
                    # CW: reverse all edge orientations to force CCW
                    loop_edges = [-curve_tag, -side_b, -top, -side_a]

                cl = gmsh.model.occ.addCurveLoop(loop_edges)
                s  = gmsh.model.occ.addPlaneSurface([cl])
                gmsh.model.occ.synchronize()

                # Save corner-generation info for each endpoint.
                #
                # side_sign encodes the direction of the stored line tag relative
                # to the (orig_pt → outer_pt) traversal direction:
                #   side_a = addLine(pt1, p2)  →  pt1→p2  = orig→outer  →  sign = +1
                #   side_b = addLine(p3, pt0)  →  p3→pt0  = outer→orig  →  sign = -1
                #
                # To traverse orig→outer use:  side_sign * side_line
                # To traverse outer→orig use: -side_sign * side_line
                for orig_pt, outer_pt, side_line, side_sign in [
                    (pt1, p2, side_a,  1),
                    (pt0, p3, side_b, -1),
                ]:
                    pml_corners_info.setdefault(orig_pt, []).append({
                        'outer_pt':    outer_pt,
                        'side_line':   side_line,
                        'side_sign':   side_sign,
                        'nx':          nx,
                        'ny':          ny,
                        'pml_name':    name,
                        'matched_obj': self._get_matched_obj(name),
                    })

                dir_label, coord_label, axis = self._classify_direction(nx, ny)
                if axis == "x":
                    coord_str = f"{coord_label}={mx:.8e}"
                elif axis == "y":
                    coord_str = f"{coord_label}={my:.8e}"
                else:
                    coord_str = f"x0={mx:.8e},{my:.8e}"

                pml_name = f"{name}__{coord_str}__t={thickness:.8e}"
                surfaces.append((s, pml_name))

                if matched_obj := self._get_matched_obj(name):
                    surfaces_by_obj[matched_obj].append(s)

        # ======================================================================
        # Corner generation
        # Each original node shared by exactly two PML edges needs a corner quad.
        # ======================================================================
        for pt_tag, infos in pml_corners_info.items():
            if len(infos) != 2:
                continue

            info1, info2 = infos[0], infos[1]
            orig_coords = gmsh.model.getValue(0, pt_tag, [])

            # Corner point = original point + both displacement vectors
            cx = orig_coords[0] + info1['nx'] + info2['nx']
            cy = orig_coords[1] + info1['ny'] + info2['ny']
            cz = orig_coords[2]

            p_corner = gmsh.model.occ.addPoint(cx, cy, cz)
            gmsh.model.occ.synchronize()

            line1 = gmsh.model.occ.addLine(info1['outer_pt'], p_corner)
            line2 = gmsh.model.occ.addLine(p_corner, info2['outer_pt'])
            gmsh.model.occ.synchronize()

            # ------------------------------------------------------------------
            # Build the corner loop in two steps:
            #
            # Step 1 - Topology: construct a head-to-tail closed loop using
            #   side_sign to traverse each side line in the correct direction.
            #
            #   Desired traversal:
            #     orig   -> outer1 :  info1['side_sign'] * info1['side_line']
            #     outer1 -> corner :  line1  (defined as outer1 -> corner)
            #     corner -> outer2 :  line2  (defined as corner -> outer2)
            #     outer2 -> orig   : -info2['side_sign'] * info2['side_line']
            #
            # Step 2 - Winding: Shoelace signed area on the actual vertex
            #   coordinates flips the whole loop if CW, so gmsh OCC always
            #   receives a CCW-oriented curve loop.
            # ------------------------------------------------------------------
            loop_fwd = [
                 info1['side_sign'] * info1['side_line'],
                 line1,
                 line2,
                -info2['side_sign'] * info2['side_line'],
            ]

            corner_quad_pts = [
                (orig_coords[0],                       orig_coords[1]),
                (orig_coords[0] + info1['nx'],         orig_coords[1] + info1['ny']),
                (cx,                                   cy),
                (orig_coords[0] + info2['nx'],         orig_coords[1] + info2['ny']),
            ]
            if self._signed_area_2d(corner_quad_pts) > 0:
                corner_cl = gmsh.model.occ.addCurveLoop(loop_fwd)
            else:
                corner_cl = gmsh.model.occ.addCurveLoop([-e for e in reversed(loop_fwd)])

            corner_surf = gmsh.model.occ.addPlaneSurface([corner_cl])
            gmsh.model.occ.synchronize()

            # Structured corner name
            dir1, _, ax1 = self._classify_direction(info1['nx'], info1['ny'])
            dir2, _, ax2 = self._classify_direction(info2['nx'], info2['ny'])

            # Always put the x (r) direction before the y (z) direction
            if ax1 == "y" and ax2 == "x":
                dir1, dir2 = dir2, dir1

            r0, z0     = orig_coords[0], orig_coords[1]
            thickness  = self._get_pml_thickness(info1['pml_name'])
            base_name  = info1['pml_name']

            corner_name = (
                f"{base_name}_Corner__{dir1}_{dir2}"
                f"__r0={r0:.8e}__z0={z0:.8e}__t={thickness:.8e}"
            )
            surfaces.append((corner_surf, corner_name))

            if info1['matched_obj']:
                surfaces_by_obj[info1['matched_obj']].append(corner_surf)

        return surfaces

    def _classify_direction(
        self, nx: float, ny: float, tol: float = 1e-6
    ) -> Tuple[str, str, str]:
        """
        Classifies a displacement vector into an axis-aligned PML direction.
        Returns (dir_label, coord_label, axis).
            dir_label   : 'r+', 'r-', 'z+', 'z-'  (or 'oblique')
            coord_label : 'r0' or 'z0'
            axis        : 'x' or 'y'
        Convention: r ↔ x,  z ↔ y  (axisymmetric).
        """
        if abs(ny) < tol and nx > 0:
            return "r+", "r0", "x"
        if abs(ny) < tol and nx < 0:
            return "r-", "r0", "x"
        if abs(nx) < tol and ny > 0:
            return "z+", "z0", "y"
        if abs(nx) < tol and ny < 0:
            return "z-", "z0", "y"
        return "oblique", "x0", "xy"

    def _compute_domain_centroid(
        self, phys_curves: List[Tuple[int, int]]
    ) -> Tuple[float, float]:
        """Centroid of all non-PML curve endpoints — used as the 'inside' reference."""
        xs, ys = [], []
        for _, pg_tag in phys_curves:
            name = gmsh.model.getPhysicalName(1, pg_tag)
            if "PML" in name.upper():
                continue
            for curve_tag in gmsh.model.getEntitiesForPhysicalGroup(1, pg_tag):
                bnd = gmsh.model.getBoundary([(1, curve_tag)], oriented=False)
                for _, pt in bnd:
                    coords = gmsh.model.getValue(0, abs(pt), [])
                    xs.append(coords[0])
                    ys.append(coords[1])
        if not xs:
            return (0.0, 0.0)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _assign_physical_groups(self, surfaces_list: List[Tuple[int, str]]):
        for s, name in surfaces_list:
            pg = gmsh.model.addPhysicalGroup(2, [s])
            gmsh.model.setPhysicalName(2, pg, name)

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

        if restrict_fields:
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", restrict_fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    def _configure_and_generate_mesh(self, msh_path: str, write_mesh_file:bool):
        gmsh.option.setNumber("General.NumThreads", multiprocessing.cpu_count())
        gmsh.option.setNumber("Mesh.Binary", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", self.mesh_order)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)

        gmsh.model.mesh.generate(2)
        if write_mesh_file:
            gmsh.write(msh_path)
            print(f"Mesh generated and saved: {msh_path}")

    def _get_elements_for_group(
        self,
        dim: int,
        phys_tag: int,
        target_types: Dict[int, int],
        nodes_per_elem: Dict[int, int],
    ) -> Optional[np.ndarray]:
        """Extracts and reshapes elements for a physical group, filtered by element type."""
        group_elements = []
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag):
            elem_types, _, node_tags_list = gmsh.model.mesh.getElements(dim, entity)
            for etype, enodes in zip(elem_types, node_tags_list):
                if etype == target_types.get(dim):
                    group_elements.append(enodes.reshape(-1, nodes_per_elem[etype]))

        return np.vstack(group_elements) if group_elements else None
