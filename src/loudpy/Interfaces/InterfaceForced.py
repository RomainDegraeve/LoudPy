"""Former name of CoilEdge (loudpy.physics.parts)."""
from loudpy.physics.parts import CoilEdge


class InterfaceForced(CoilEdge):
    def __init__(self, mesh_obj, direction):
        CoilEdge.__init__(self, mesh_obj, direction)
