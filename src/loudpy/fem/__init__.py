"""
loudpy.fem — the generic finite-element core (discretisation, no physics names).

  reference         reference cells, Lagrange elements P1–P3 (lines, triangles, tetrahedra) and Q1–Q2
                    (quadrilaterals) on gmsh's node layouts; facet tables; cell type names
  quadrature        rules exact to a degree on every cell; the former T6 / L3 rules (legacy_*)
  space             Space: DOFs from the nodes of a high-order mesh, components interleaved
  kernels           generic Numba kernels (mass, Laplace, diagonal diffusion, loads, boundary mass /
                    load / normal mass; isoparametric map, 2πr weight) and the Assembler
  solid             axisymmetric elasticity, Fourier order m, total-Lagrangian kernels; elasticity matrices
  tet10             3D toolkit: P2 tetrahedra and TRI6 faces (curved), stiffness / mass, surface quadrature
  assembly          ElementAssembler (node numbering + CSR pattern over 2D regions), cell_tabulation
  sparse, indexing, sparse_kernels, batch_coupling
                    COO blocks → CSR with a precomputed scatter map; coupling block interleave
  dofs              DirichletElimination: blocked DOFs by elimination, expand()
"""
