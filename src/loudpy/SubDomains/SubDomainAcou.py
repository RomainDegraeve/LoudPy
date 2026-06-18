from loudpy.ObjectsGeo import TR6_Surfs
import numpy as np
class SubDomainAcou(TR6_Surfs): 
    def __init__(self, mesh_obj: TR6_Surfs, rho: float, c: float, orientation:float=1):
        self.dofs_per_node = 1
        super().__init__(
            node_tags=mesh_obj.node_tags,
            node_coords=mesh_obj.node_coords,
            tri=mesh_obj.tri,
            name=mesh_obj.name)
        
        self.rho = rho
        self.c = c
        self.orientation = orientation

class SubDomainAcou_PML(SubDomainAcou): 
    def __init__(self, mesh_obj: TR6_Surfs, orientation:float,rho: float, c: float, alpha: float, n:float, f_pml:float):
        super().__init__(mesh_obj, rho, c, orientation)
        self.alpha = alpha
        self.n = n 
        self.omega_pml = 2 * np.pi * f_pml
        
