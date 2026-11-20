Finite elements
===============

Every operator of loudpy — structure, Fourier orders, nonlinear structure, air with PML, coupling,
coil force, exact radiation, 3D air — is assembled by one generic core, ``loudpy.fem``. This page
states what it computes; the tests that check it are listed at the end.

Reference cells and elements
----------------------------

Reference cells follow gmsh: the line ``[-1, 1]``, the triangle ``ξ, η ≥ 0, ξ + η ≤ 1``, the
quadrilateral ``[-1, 1]²`` and the tetrahedron ``ξ, η, ζ ≥ 0, ξ + η + ζ ≤ 1``.

A Lagrange element of order *p* has one node per point of gmsh's layout (corners, then the edge
nodes in gmsh's edge order and direction, then face and interior nodes). Its basis is the nodal basis
of

* :math:`P_p` (polynomials of total degree ≤ *p*) on lines, triangles and tetrahedra, *p* = 1, 2, 3;
* :math:`Q_p` (degree ≤ *p* in each variable) on quadrilaterals, *p* = 1, 2,

obtained from the inverse of the Vandermonde matrix on monomials
(``loudpy.fem.reference.lagrange(cell, p)``). The gmsh types are 1, 8, 26 (lines), 2, 9, 21
(triangles), 3, 10 (quadrilaterals) and 4, 11, 29 (tetrahedra).

Quadrature
----------

``loudpy.fem.quadrature.quadrature(cell, q)`` is exact for polynomials of total degree ≤ *q*:
Gauss–Legendre (tensor products on quadrilaterals) and collapsed Gauss–Jacobi rules on simplices
(positive weights, interior points). Second-order triangles and lines keep the rules of the former
kernels (7-point triangle, 3-point Gauss line) so that their operators did not change; other
elements use degree 2·*p* + 2 (2·dim·*p* + 2 for :math:`Q_p`).

Geometry: isoparametric cells
-----------------------------

Each cell is mapped from the reference cell by its own nodes,
:math:`x(\xi) = \sum_a x_a N_a(\xi)`, :math:`J = \partial x / \partial \xi`. gmsh places the
high-order nodes on curved boundaries, so curved boundaries are followed with the accuracy of the
element (straight-sided cells only see a polygon: an error of order :math:`h^2` whatever *p*).
Physical gradients are :math:`\nabla N = J^{-T} \hat\nabla N`; the measure is
:math:`|\det J|` in the volume, :math:`|\partial x / \partial \xi|` on lines and
:math:`|\partial_\xi x \times \partial_\eta x|` on surfaces. Folded curved tetrahedra (``det J``
changing sign inside a cell) are refused.

Two placements of gmsh are corrected or chosen by loudpy:

* the interior node of cubic triangles on curved boundaries is moved to
  :math:`x_9 = \tfrac14 \sum \text{edge nodes} - \tfrac16 \sum \text{corners}` (exact for a cubic
  map with the given edges); with gmsh's position, P3 converges at order ≈ 3.5 instead of 4 in L2;
* the 3D meshes put the P2 mid-edge nodes on the CAD surfaces (``Mesh.SecondOrderLinear = 0``).

Axisymmetric forms
------------------

On the meridian plane (r, z) every volume integral carries :math:`2\pi r` (evaluated at the
quadrature points, never on the axis nodes):

* elasticity :math:`\int B^T D B \, 2\pi r \, dA` with the strains
  :math:`[\varepsilon_r, \varepsilon_z, \gamma_{rz}, \varepsilon_\theta = u_r / r]`;
* azimuthal order *m*: three components :math:`(U_r, U_z, U_\theta)`, six strains, and
  :math:`\int \dots r \, d\theta = 2\pi r` (*m* = 0) or :math:`\pi r` (*m* ≥ 1);
* the total-Lagrangian element: Green–Lagrange strains, tangent :math:`K_T = K_\text{mat} + K_\text{geo}`
  and internal force :math:`F_\text{int}`;
* acoustics: :math:`H = \int \nabla N \cdot \nabla N \, 2\pi r \, dA`,
  :math:`Q = \int N N / c^2 \, 2\pi r \, dA`.

A PML layer stretches the coordinates along *r* and / or *z*:
:math:`s(x) = 1 - i \alpha \lambda (n / t) \xi^{n-1}`, :math:`\tilde r = r - i \alpha \lambda \xi^n`
(:math:`\xi` the depth in the layer); then
:math:`H = \int (s_z / s_r \, \partial_r N \partial_r N + s_r / s_z \, \partial_z N \partial_z N)
\, 2\pi \tilde r \, dA` and :math:`Q = \int N N \, s_r s_z / c^2 \, 2\pi \tilde r \, dA`.

Boundary terms use the same cells of one dimension less: the coupling
:math:`\int N_i N_j \, n \, 2\pi r \, ds` with :math:`n\,ds = (\partial_\xi z, -\partial_\xi r)`
(oriented from the air), the coil force :math:`\int N \, 2\pi r \, ds`, the DtN vectors
:math:`\oint P_n(\cos\theta) N \, 2\pi r \, ds`.

What the tests check
--------------------

* ``tests/test_fem_reference.py``: gmsh layouts, nodal bases, partition of unity, reproduction of
  each element's polynomials, facet tables, quadrature exact to degree 8 on every cell.
* ``tests/test_fem_assembly.py``: textbook P1 matrices, measures, patch tests on distorted curved
  cells (P1–P3, Q1–Q2, axisymmetric, tetrahedra), manufactured solutions with L2 orders *p* + 1
  (Poisson, Helmholtz, axisymmetric with the natural axis condition, tetrahedra), curved cells,
  divergence theorem for the normals, the 3D toolkit on a ball.
* ``tests/test_fem_solid.py``: rigid modes, uniform-strain patch with the exact strain energy,
  Fourier rigid modes, the tangent of the total-Lagrangian element against finite differences, PML
  coefficients, the clamped plate through ``lp.Model(mesh_order=p)`` for *p* = 1, 2, 3.
* ``tests/test_operators.py`` and the verification tests: the operators and responses of the
  committed models (unchanged by the new core, see ROADMAP phase D).
