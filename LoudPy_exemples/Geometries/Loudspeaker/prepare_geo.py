"""
prepare_geo.py — Convert a CAD file to a GMSH-compatible geometry.

Run this script once before meshing.  The .brep / .step file contains the
CAD description of the loudspeaker and is imported into GMSH's native .geo
format.  All subsequent examples read from the .geo / .msh files.
"""
from loudpy.Meshing import prepare_geometry

prepare_geometry(
    brep_path="LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.brep",
    step_path="LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.step",
)
