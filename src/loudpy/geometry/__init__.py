"""
loudpy.geometry — meshes and geometry, no physics.

  Mesh, MeshGroup     one data model for 2D (axisymmetric) and 3D meshes
  naming              physical-group names of the .geo convention and how user names select them
  sources             where a 2D geometry comes from: .geo (curve loops or surfaces), CAD files, builders
  selectors           Group, Box, Near, Interface, BoundaryOf: what a region / boundary covers
  meshing2d           2D axisymmetric mesher (domains of a plan, PML layers, grading)
  meshing3d           3D builders: driver cut-outs, classification, meshing, P2 mesh reading
  Frame               placement and orientation of an axisymmetric part in 3D
  gmsh_session        gmsh access (context manager, quiet defaults)
"""
from loudpy.geometry.frame import Frame
from loudpy.geometry.mesh import Mesh, MeshGroup
from loudpy.geometry.gmsh_session import gmsh_session, physical_groups
from loudpy.geometry import naming
from loudpy.geometry.selectors import BoundaryOf, Box, Group, Interface, Near, Selector

__all__ = ["Frame", "Mesh", "MeshGroup", "gmsh_session", "physical_groups", "naming",
           "Selector", "Group", "Box", "Near", "Interface", "BoundaryOf"]
