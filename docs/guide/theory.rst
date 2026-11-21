Theory manual
=============

The equations loudpy solves, their discrete form and where each one lives in the code. Time dependence
:math:`e^{j\omega t}` (:math:`s = j\omega` in the system builder), SI units; the elements and quadrature are in
:doc:`elements`, the sign and naming conventions in :doc:`conventions`.

Structure (axisymmetric)
------------------------

Linear elasticity on the meridian plane (r, z), displacement :math:`\mathbf u = (u_r, u_z)`, strains
:math:`\boldsymbol\varepsilon = (\partial_r u_r,\ \partial_z u_z,\ \partial_z u_r + \partial_r u_z,\ u_r / r)`,
isotropic stress :math:`\boldsymbol\sigma = \mathbf D(E, \nu)\, \boldsymbol\varepsilon`. With the volume measure
:math:`2\pi r\, dr\, dz`,

.. math::

   \mathbf K = \int \mathbf B^T \mathbf D \mathbf B\, 2\pi r, \qquad
   \mathbf M = \int \rho\, \mathbf N^T \mathbf N\, 2\pi r, \qquad
   (\mathbf K + s \mathbf C + s^2 \mathbf M)\, \mathbf u = \mathbf F .

Damping: Rayleigh :math:`\mathbf C = \alpha \mathbf M + \beta \mathbf K` (loss factor
:math:`\eta(\omega) = \alpha/\omega + \beta\omega`), or hysteretic :math:`\mathbf K (1 + j\eta)` (frequency domain
only). Azimuthal order *m* (``Modes(orders=[…])``): :math:`u_r, u_z \propto \cos m\theta`,
:math:`u_\theta \propto \sin m\theta`, three DOFs per node, the axis conditions of each order imposed on the axis
nodes. Geometric nonlinearity (``lp.Transient``): total Lagrangian form with Green–Lagrange strains, tangent
:math:`\mathbf K_T = \partial \mathbf F_\mathrm{int} / \partial \mathbf u`, generalized-α integration with Newton
iterations. Code: :mod:`loudpy.fem.solid`, :mod:`loudpy.physics.structural`.

Air
---

Helmholtz equation for the pressure, :math:`\nabla^2 p - (s/c)^2 p = 0`; weak form with test function *q*,

.. math::

   \mathbf H = \int \nabla \mathbf N \cdot \nabla \mathbf N^T\, 2\pi r, \qquad
   \mathbf Q = \int \frac{1}{c^2} \mathbf N \mathbf N^T\, 2\pi r, \qquad
   (\mathbf H + s^2 \mathbf Q)\, \mathbf p = \text{boundary terms}.

PML: complex coordinate stretching :math:`\tilde x = x + \frac{1}{j k}\int_0^{x} \sigma(\xi)\, d\xi` along *r* and /
or *z* inside each layer (strength and order of the profile :math:`\sigma` from ``lp.PML``), giving a diagonal
complex diffusion tensor and a complex mass weight. The air of every model is the same ideal-gas mixture of dry air
and water vapour (``lp.air(T, p, RH)``). Code: :mod:`loudpy.physics.acoustic`.

Fluid–structure coupling
------------------------

On a wet boundary, with :math:`\mathbf n` the outward normal of the air (taken from the adjacent air cell of every
edge, whatever the direction of the curves in the geometry),

.. math::

   \mathbf C_{am} = \int \mathbf N_p\, \mathbf n^T \mathbf N_u\, 2\pi r\, ds \quad (\text{air nodes} \times
   \text{structure DOFs}), \qquad
   \begin{bmatrix} \mathbf K + s\mathbf C + s^2\mathbf M & -\mathbf C_{am}^T \\
                   \rho s^2 \mathbf C_{am} & \mathbf H + s^2 \mathbf Q \end{bmatrix}
   \begin{bmatrix} \mathbf u \\ \mathbf p \end{bmatrix} = \begin{bmatrix} \mathbf F \\ 0 \end{bmatrix}

(the signs as assembled by the frequency form of :mod:`loudpy.physics.builder`). Checks: the volume swept by a rigid
translation is :math:`\pi r_\max^2` on every interface, and the power given by the coil equals the structural losses
plus the power radiated into each air region (both exact for the discrete system).

Voice coil and circuit
----------------------

A force drive puts :math:`F` axially on the coil nodes. A voltage drive adds the current *I* as an unknown:

.. math::

   U = (R_e + (j\omega)^n L_e)\, I + Bl\, \dot u_\mathrm{coil}, \qquad
   F_\mathrm{coil} = Bl\, I,

the semi-inductance :math:`(j\omega)^n L_e` being a non-polynomial term of the builder (no polynomial eigenproblem
then). Electrical impedance :math:`Z = U / I`; Thiele–Small parameters from Z and the volume velocity
(:mod:`loudpy.results.values`).

Radiation
---------

*2D, exact DtN* (``lp.Radiation``, coupled modes): on a sphere of radius *R* the outgoing field is expanded in
Legendre polynomials, :math:`p = \sum_n a_n h_n(kr) P_n(\cos\theta)`, and the boundary condition
:math:`\partial_r p = \sum_n k\, h_n'(kR)/h_n(kR)\, a_n P_n` is written with auxiliary unknowns in a symmetric
:math:`u`–:math:`\psi`–:math:`\phi` form whose eigenvalues are the coupled modes. *2D far field*: Kirchhoff–Helmholtz
integral of *p* and :math:`\partial_n p` over a meridian arc in the air with the ring Green's function (rigid baffle:
image source), from a result file alone. *3D, half-space DtN*: on a hemisphere centred on the rigid ground,

.. math::

   \mathbf D = \mathbf G\, \mathrm{diag}\!\left(\frac{2\gamma_n}{R^2}\right) \mathbf G^T, \quad
   G_{ih} = \int_\Gamma N_i Y_h\, dS, \quad \gamma_n = k\, \frac{h_n'(kR)}{h_n(kR)},

with real spherical harmonics :math:`Y_h` even about the ground (:math:`n + |m|` even); D is applied as a low-rank
operator, never assembled. Code: :mod:`loudpy.physics.radiation`, :mod:`loudpy.physics.exterior`.

3D scenes: modal coupling
-------------------------

A driver enters a 3D scene through its in-vacuo modes :math:`\boldsymbol\varphi_k` (all azimuthal orders, both
families, unit modal mass), the cavity through its rigid-wall modes :math:`\psi_n`
(:math:`\mathbf K\psi = \kappa^2 \mathbf M\psi`). With :math:`B_{ik} = \int_S N_i\, \boldsymbol\varphi_k\cdot\mathbf n\,
dS` on the wet surface and :math:`\mathbf S = \mathbf B^T\psi`,

.. math::

   (\omega_k^2 + 2j\zeta_k\omega_k\omega - \omega^2)\, q_k - \rho\omega^2 (\mathbf R q)_k + \sum_n S_{kn} P_n = F_k, \qquad
   (\omega_n^2 + j\eta\,\omega_n\omega - \omega^2)\, P_n + \rho c^2 \omega^2 \sum_k S_{kn} q_k = 0,

:math:`\mathbf R = \mathbf B^T \mathbf K^+ \mathbf B - \sum_{n \ge 1} \mathbf S_n \mathbf S_n^T / \kappa_n^2` being the
static correction of the truncated cavity modes. Eliminating *P* loads the drivers with the cavity's modal
impedance; the exterior air (P2 FE + half-space DtN) is solved by GMRES on its Schur complement with a sparse LU
preconditioner refreshed every 50 Hz. Code: :mod:`loudpy.physics.modal`, :mod:`loudpy.study.scene_response`.

Shells (elastic walls)
----------------------

Flat Reissner–Mindlin cells, six DOFs per node, in each cell's plane (rotations :math:`\theta_1, \theta_2` about the
in-plane axes, *h* the thickness):

.. math::

   \boldsymbol\varepsilon_m = (u_{,x},\ v_{,y},\ u_{,y} + v_{,x}), \quad
   \boldsymbol\kappa = (\theta_{2,x},\ -\theta_{1,y},\ \theta_{2,y} - \theta_{1,x}), \quad
   \boldsymbol\gamma = (w_{,x} + \theta_2,\ w_{,y} - \theta_1),

with :math:`\mathbf D_m = \frac{E h}{1-\nu^2}[\dots]`, :math:`\mathbf D_b = \frac{h^2}{12}\mathbf D_m`,
:math:`\mathbf D_s = k G h\, \mathbf I` (*k* = 5/6). The transverse shear strains are the MITC assumed strains
(MITC4: Bathe–Dvorkin; MITC3: Lee–Bathe) to avoid shear locking; the drilling rotation is tied to the in-plane
rotation by a penalty :math:`\alpha G h \int (\theta_3 - \omega)^2`. Walls on a P2 air mesh are coupled face by face,
:math:`A_{ai,j} = \int N_a n_i N_j\, dS`:

.. math::

   (\mathbf K + s\mathbf C + s^2\mathbf M)\, \mathbf d + \mathbf A \mathbf p = \mathbf F, \qquad
   (\mathbf H + s^2\mathbf Q)\, \mathbf p - \rho s^2 \mathbf A^T \mathbf d = 0,

directly (:mod:`loudpy.physics.shell`) or, in scenes, through the walls' in-vacuo modes appended to the drivers'
(the same modal coupling as above). Code: :mod:`loudpy.fem.shell`, :mod:`loudpy.physics.shell`.

Error estimation and adaptivity
-------------------------------

The recovered gradient :math:`G^*u_h` is the polynomial-preserving recovery: at each node, the gradient of the
least-squares polynomial of degree *p* + 1 fitted to the nodal values of its patch. Recovery indicators
:math:`\eta_K^2 = \int_K |G^*u_h - \nabla u_h|^2` (axisymmetric: × 2πr; elastic energy: :math:`(\varepsilon^* -
\varepsilon_h)^T \mathbf D (\varepsilon^* - \varepsilon_h)`). New sizes by equidistribution,
:math:`h_\mathrm{new} = h\,(T^2 / (N\eta_K^2))^{1/(2p + d)}`, change per step within [0.3, 2], written as a gmsh
size view. Expected orders: eigenvalues :math:`h^{2p}`, energy norm :math:`h^p`, L2 and point values
:math:`h^{p+1}`. Code: :mod:`loudpy.fem.estimate`, :mod:`loudpy.fem.adapt`, :mod:`loudpy.study.adaptive`.

How the systems are built
-------------------------

Every physics contributes terms :math:`f(s)\,\mathbf B` between named fields (u, p, I, auxiliaries, modal
coordinates) to a ``SystemBuilder``; a study asks the builder for :math:`\mathbf A(s)` (frequency sweep), for
:math:`(\mathbf K, \mathbf C, \mathbf M)` (eigenproblems) or solves it block-wise (``BlockSolver``: small dense fields
eliminated exactly, the big field by GMRES). See :doc:`architecture`.
