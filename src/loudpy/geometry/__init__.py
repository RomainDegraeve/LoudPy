"""
loudpy.geometry — meshes and geometry, no physics.

  Mesh, MeshGroup     one data model for 2D (axisymmetric) and 3D meshes
  naming              physical-group names of the .geo convention and how user names select them
  meshing2d           2D axisymmetric mesher (surfaces from curve groups, PML layers, grading)
  meshing3d           3D builders: driver cut-outs, classification, meshing, P2 mesh reading
  Frame               placement and orientation of an axisymmetric part in 3D
  gmsh_session        gmsh access (context manager, quiet defaults)
"""
from loudpy.geometry.frame import Frame
from loudpy.geometry.mesh import Mesh, MeshGroup
from loudpy.geometry.gmsh_session import gmsh_session, physical_groups
from loudpy.geometry import naming

__all__ = ["Frame", "Mesh", "MeshGroup", "gmsh_session", "physical_groups", "naming"]
