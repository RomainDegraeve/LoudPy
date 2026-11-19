from typing import Optional, List
from loudpy.PhysicalEntitiesManager import PhysicalEntitiesManager
from loudpy.ObjectsGeo import Nodes  # adjust import path
import numpy as np
class Problem:
    def __init__(self, 
                 geo_path: str, 
                 msh_path: str, 
                 mat_path: str = "", 
                 subdomains_key: str = "sub", 
                 orientations_key: Optional[dict] = None,  
                 directions_key: Optional[dict] = None,
                 mesh_grading: Optional[float] = 0.3):
        
        # Specs
        self.specs_dom: list = []
        self.specs_interf: list = []
        
        # FEM objects
        self.fem_objects: Optional[List] = None
        self.nodes: Optional[Nodes] = None
        
        # Paths
        self.geo_path = geo_path
        self.msh_path = msh_path
        self.mat_path = mat_path
        
        # Mesh grading: element size = min(domain size, structural size + mesh_grading · distance
        # to the structure), gmsh boundary extension off (see MeshManager). None: gmsh default.
        self.mesh_grading = mesh_grading

        # Keys 
        self.subdomains_key = subdomains_key
        self.orientations_key = orientations_key if orientations_key is not None else {
            "front": +1.0, 
            "rear":  -1.0,
        }
        self.directions_key = directions_key if directions_key is not None else {
            "_r_":  (0,),
            "_z_":  (1,),
            "_rz_": (0, 1),
            "_zr_": (0, 1),
        }

    def add_sub_domain(self, *specs_dom):
        for d in specs_dom :
            if isinstance(d, (list, tuple)):
                self.specs_dom.extend(d)
            else :
                self.specs_dom.append(d) 

    def add_interface(self, *specs):
        for s in specs:
            if isinstance(s, (list, tuple)):
                self.specs_interf.extend(s)
            else:
                self.specs_interf.append(s)


    def set_mesh_grading(self, grading: Optional[float]):
        """Growth rate of the element size away from the structure (None or 0: gmsh default)."""
        self.mesh_grading = grading

    def set_mesh_sizes(self, sizes:dict):
    
        for domain in self.specs_dom:
            if domain.name in sizes:
                domain.size = sizes[domain.name]


    def mesh(self, show_mesh_gui: bool = True, write_mesh_file : bool = True,
             reuse_mesh: bool = False):
        """
        Mesh the geometry (loudpy MeshManager); the mesh is written to msh_path (with its
        signature msh_path + ".json") unless write_mesh_file=False.

        reuse_mesh=True: load the mesh saved at msh_path instead of remeshing, if it was built
        from the same .geo (and merged files), domain specs (sizes, PML thickness), mesh order
        and grading; otherwise remesh (and write) and print why.
        """
        entities = PhysicalEntitiesManager(
            self.specs_dom, 
            self.specs_interf,
            geo_path=self.geo_path, 
            msh_path=self.msh_path, 
            mat_path=self.mat_path, 
            subdomains_key = self.subdomains_key, 
            directions_key = self.directions_key, 
            orientations_key = self.orientations_key,
            mesh_grading = self.mesh_grading,
            )
        
        self.fem_objects, self.nodes = entities.create_meshed_objs(show_gui = show_mesh_gui, write_mesh_file = write_mesh_file, reuse_mesh = reuse_mesh)


    def specs_dict(self) -> dict:
        """Return all specs in the format expected by ResultStore.save_specs."""
        return {
            "subdomains": list(self.specs_dom),
            "interfaces": list(self.specs_interf),
        }


    def _get_interface_node_tags(self, spec) -> np.ndarray:
        tags = set()
        for fem_obj in self.fem_objects:
            if spec.name in fem_obj.name:
                if hasattr(fem_obj, "node_tags"):
                    tags.update(fem_obj.node_tags.tolist())
        return np.array(sorted(tags), dtype=np.int64)

    
    def _get_interface_edges(self, spec) -> np.ndarray:
        """Per-element node ordering for an interface, as row indices into
        the array returned by _get_interface_node_tags(spec)."""
        tags = self._get_interface_node_tags(spec)
        edges = []
        for fem_obj in self.fem_objects:
            if spec.name in fem_obj.name and hasattr(fem_obj, "edges"):
                rows = np.searchsorted(tags, fem_obj.edges.ravel())
                edges.append(rows.reshape(fem_obj.edges.shape))
        return np.vstack(edges).astype(np.int64) if edges else np.empty((0, 3), dtype=np.int64)
