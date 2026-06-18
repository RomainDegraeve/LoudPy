from loudpy.ObjectsGeo import L3_Edges

class InterfaceForced(L3_Edges):      
    def __init__(self, mesh_obj: L3_Edges, direction : float):
        super().__init__(
            node_tags=mesh_obj.node_tags,
            node_coords=mesh_obj.node_coords,
            edges=mesh_obj.edges,
            name=mesh_obj.name
            )
        self.direction = direction

