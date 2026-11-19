"""Moved to loudpy.physics (assemblers) and loudpy.fem (kernels); kept for existing imports."""
from loudpy.physics.structural import MecaAssembler, MecaAssemblerNL, MecaAssemblerFourier  # noqa: F401
from loudpy.physics.acoustic import AcouAssembler  # noqa: F401
from loudpy.physics.coupling import FsiAssembler  # noqa: F401
from loudpy.physics.forcing import ForceAssembler  # noqa: F401

__all__ = ["ForceAssembler", "FsiAssembler", "MecaAssembler", "AcouAssembler", "MecaAssemblerNL",
           "MecaAssemblerFourier"]
