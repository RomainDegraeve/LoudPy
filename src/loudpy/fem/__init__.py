"""
loudpy.fem — discretisation, no physics names.

  reference         reference cells, Lagrange elements P1–P3 / Q1–Q2 on gmsh's node layouts
  quadrature        rules exact to a degree on every cell; the former T6 / L3 rules (legacy)
  shape             shape functions, Jacobians, elasticity matrix
  elements          element matrices (tri6_axi: axisymmetric T6 / L3), numba kernels
  tet10             P2 tetrahedra and TRI6 faces (3D acoustics): stiffness / mass, surface quadrature
  batch             element kernels applied to every element of a region (parallel)
  sparse            COO element blocks → CSR with a precomputed scatter map
  assembly          ElementAssembler: node numbering + CSR pattern over T6 regions
  dofs              DirichletElimination: blocked DOFs by elimination, expand()
"""
