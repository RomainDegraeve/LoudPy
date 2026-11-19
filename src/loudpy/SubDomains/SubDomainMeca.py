"""Former structural region classes: SolidPart (loudpy.physics.parts) with their former constructors."""
from loudpy.physics.parts import SolidPart, material_parameters


class SubDomainMeca(SolidPart):
    def __init__(self, mesh_obj, rho: float, E: float, nu: float):
        SolidPart.__init__(self, mesh_obj, rho, E, nu)

    @classmethod
    def get_params(cls, mat_name: str, mat_path):
        return material_parameters(cls, mat_name, mat_path)


class SubdomainMeca_Hysteretic(SubDomainMeca):
    def __init__(self, mesh_obj, rho: float, E: float, nu: float, eta: float):
        SolidPart.__init__(self, mesh_obj, rho, E, nu, eta=eta)


class SubdomainMeca_Rayleigh(SubDomainMeca):
    def __init__(self, mesh_obj, rho: float, E: float, nu: float, alpha_ray: float, beta_ray: float):
        SolidPart.__init__(self, mesh_obj, rho, E, nu, alpha_ray=alpha_ray, beta_ray=beta_ray)
