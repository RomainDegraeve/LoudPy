"""
loudpy.fem — discretisation, no physics names.

  quadrature        Gauss rules (T6: 7 points, L3: 3 / 4 points)
  shape             shape functions, Jacobians, elasticity matrix
  elements          element matrices (tri6_axi: axisymmetric T6 / L3), numba kernels
  batch             element kernels applied to every element of a region (parallel)
  sparse            COO element blocks → CSR with a precomputed scatter map
  assembly          ElementAssembler: node numbering + CSR pattern over T6 regions
  dofs              DirichletElimination: blocked DOFs by elimination, expand()
"""
