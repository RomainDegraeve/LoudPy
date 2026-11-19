"""
loudpy.model — what the user declares: geometry, regions, boundaries, materials, damping, drives.
No numerics here.
"""
from loudpy.model.boundaries import Boundary, Clamped, Coil, FluidStructure, Radiation
from loudpy.model.damping import Hysteretic, Rayleigh, Undamped
from loudpy.model.drives import Force, Voltage
from loudpy.model.materials import Material, MaterialLibrary, load_materials, materials
from loudpy.model.model import Model
from loudpy.model.regions import PML, Air, Region, Solid

__all__ = [
    "Model",
    "Solid", "Air", "PML", "Region",
    "Clamped", "Coil", "FluidStructure", "Radiation", "Boundary",
    "Rayleigh", "Hysteretic", "Undamped",
    "Force", "Voltage",
    "Material", "MaterialLibrary", "load_materials", "materials",
]
