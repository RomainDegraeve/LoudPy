"""Former acoustic region classes: AirPart / PMLPart (loudpy.physics.parts) with their former constructors."""
from loudpy.physics.parts import AirPart, PMLPart, material_parameters


class SubDomainAcou(AirPart):
    def __init__(self, mesh_obj, rho: float, c: float, orientation: float = 1):
        AirPart.__init__(self, mesh_obj, rho, c, orientation)

    @classmethod
    def get_params(cls, mat_name: str, mat_path):
        return material_parameters(cls, mat_name, mat_path)


class SubDomainAcou_PML(PMLPart, SubDomainAcou):
    def __init__(self, mesh_obj, orientation: float, rho: float, c: float, alpha: float, n: float,
                 f_pml: float):
        PMLPart.__init__(self, mesh_obj, orientation, rho, c, alpha, n, f_pml)
