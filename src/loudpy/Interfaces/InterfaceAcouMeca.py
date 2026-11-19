"""Former name of CouplingEdge (loudpy.physics.parts)."""
from loudpy.physics.parts import CouplingEdge


class InterfaceAcouMeca(CouplingEdge):
    def __init__(self, mesh_obj, direction: float):
        CouplingEdge.__init__(self, mesh_obj, direction)
