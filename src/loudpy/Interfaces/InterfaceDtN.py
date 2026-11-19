"""
Former name of RadiationEdge (loudpy.physics.parts): spherical exterior boundary carrying the exact
non-reflecting (DtN) condition (see loudpy.physics.radiation). Centre, truncation order and
symmetry are given to the solver (EigenStudy.solve_fsi_eigen).

orientation: +1 / -1 from the name ("front" / "rear"), the same sign as the acoustic sub-domain
the arc closes (the rear acoustic matrices are multiplied by -1, so are its DtN terms).
"""
from loudpy.physics.parts import RadiationEdge


class InterfaceDtN(RadiationEdge):
    def __init__(self, mesh_obj, direction=None):
        RadiationEdge.__init__(self, mesh_obj, direction)
