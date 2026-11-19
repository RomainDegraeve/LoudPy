"""Former name of ClampedEdge (loudpy.physics.parts)."""
from loudpy.physics.parts import ClampedEdge


class InterfaceConstrained(ClampedEdge):
    def __init__(self, mesh_obj, direction):
        ClampedEdge.__init__(self, mesh_obj, direction)
