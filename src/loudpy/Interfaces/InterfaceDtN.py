from loudpy.ObjectsGeo import L3_Edges


class InterfaceDtN(L3_Edges):
    """Spherical exterior boundary carrying the exact non-reflecting (DtN)
    condition (see loudpy.assembly.dtn). Centre, truncation order and
    symmetry are given to the solver (EigenStudy.solve_fsi_eigen).

    orientation: +1 / -1 from the name ("front" / "rear", Problem.orientations_key),
    the same sign as the acoustic sub-domain the arc closes (the rear acoustic
    matrices are multiplied by -1, so are its DtN terms)."""

    def __init__(self, mesh_obj: L3_Edges, direction=None):
        super().__init__(
            node_tags=mesh_obj.node_tags,
            node_coords=mesh_obj.node_coords,
            edges=mesh_obj.edges,
            name=mesh_obj.name,
        )
        self.orientation = 1.0 if direction is None else float(direction)
