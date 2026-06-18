from loudpy.ObjectsGeo import TR6_Surfs

class SubDomainMeca(TR6_Surfs): 
    def __init__(self, mesh_obj: TR6_Surfs, rho: float, E: float, nu: float):
        super().__init__(
            node_tags=mesh_obj.node_tags,
            node_coords=mesh_obj.node_coords,
            tri=mesh_obj.tri,
            name=mesh_obj.name,
        )
        self.rho, self.E, self.nu = rho, E, nu


class SubdomainMeca_Hysteretic(SubDomainMeca):
    def __init__(self, mesh_obj: TR6_Surfs, rho: float, E: float, nu: float, eta: float):
        super().__init__(mesh_obj, rho, E, nu)
        self.eta = eta

class SubdomainMeca_Rayleigh(SubDomainMeca):
    def __init__(self, mesh_obj: TR6_Surfs, rho: float, E: float, nu: float, alpha_ray: float, beta_ray: float):
        super().__init__(mesh_obj, rho, E, nu)
        self.alpha_ray, self.beta_ray = alpha_ray, beta_ray






