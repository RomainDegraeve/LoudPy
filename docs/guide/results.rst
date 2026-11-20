Results
=======

Every study returns a :class:`Results <loudpy.Results>` object; ``lp.load`` reads the same object
back from a file (or a folder, a glob or a list of files of one sweep).

.. code-block:: python

   res = lp.load("Results/Files/sweep.h5")
   res                                  # summary: kind, entries, fields, blocks
   res.frequencies                      # NumPy array (res.times for transients)
   res.names                            # fields: "u" (displacement), "p" (pressure), "v", "a"

Evaluating results returns **labelled data**: the values with their coordinates, their
quantity and their unit. They plot themselves in the loudpy style and convert to NumPy, pandas
and files (:doc:`postprocessing`).

Where: points, arcs, lines
--------------------------

Every ``at=`` argument takes a point or a *selection* (like COMSOL's cut points and cut lines):

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - ``at=``
     - evaluates at
   * - ``(0.0, 0.3)``
     - one point, (r, z) in 2D or (x, y, z) in 3D
   * - ``lp.Points([(0, .3), (.1, .3)], labels=[...])``
     - several points: a ``point`` dimension
   * - ``lp.Polar(0.3, 30)``
     - 0.3 m from the origin, 30° from the axis
   * - ``lp.Arc(0.3, range(0, 91, 2))``
     - the same for several angles: an ``angle`` dimension [°]
   * - ``lp.Line((0, .05), (0, .5), n=100)``
     - along a segment: an ``s`` dimension (distance [m])

In 3D, ``lp.Arc(R, angles, center=..., axis="z", toward="x")`` sets the plane of the angles (0° along
``axis``, 90° along ``toward``). Values are interpolated with the elements of the mesh themselves (any
order, triangles, quadrilaterals, tetrahedra, curved cells: exact for the FE field); a point outside
the mesh takes the value of the nearest node.

Evaluations
-----------

.. code-block:: python

   res.spl(at=(0.0, 0.3))                       # Data: SPL [dB re 20 µPa] over f (RMS)
   res.spl(at=lp.Arc(0.3, range(0, 91, 2)))     # Data over (f, angle): directivity
   res.pressure(at=lp.Line((0, .05), (0, .5)))  # complex pressure over (f, s)
   res.displacement(at=(0, 0), component="z")   # u_z over f (complex amplitude)
   res.velocity(at=..., component="z")            # jω u (harmonic) or the stored v (transient)
   res.acceleration(at=..., component="z")        # −ω² u, or the stored a
   res.probe("u", at=...)                         # any field; vector fields: a "component" dimension
   res.impedance()                              # U / I_coil over f (voltage drive) [Ω]
   res.scalar("I_coil")                         # any per-entry quantity as Data
   res.arrays()                                 # plain NumPy arrays (values, fields, meshes, per-entry)

Derived fields
--------------

.. code-block:: python

   res.field("strain", f=200)        # (ε_rr, ε_zz, γ_rz, ε_θθ) at the nodes (axisymmetric, m = 0)
   res.field("stress", f=200)        # σ = D(E, ν) ε of each region [Pa]; .component("tt") hoop stress
   res.field("von_mises", f=200)     # [Pa]; harmonic: the maximum over the cycle
   res.field("v", f=200)             # jω u (harmonic); "a": −ω² u
   res.field("particle_velocity", f=500), res.field("intensity", f=500)   # −∇p/(jωρ), ½ Re(p v*) [W/m²]

Gradients come from the elements (isoparametric, any order), averaged at the nodes shared by several
elements. The material data (E, ν, ρ of each mesh group) are stored in the result file (format 3), so a
reloaded file is enough; ``loudpy.results.derived`` has the same functions for any Field. Checked against
analytical fields: uniform and quadratic displacements (exact strain and stress), rigid motion, a
quadratic pressure (exact velocity), a plane wave (intensity ``|p|²/(2ρc)`` within 0.2 % with P2, kh = 0.08).
Inside a PML the air quantities are not physical.

Derived values
--------------

.. code-block:: python

   res.volume_velocity(on="front")                      # Q = ∫ jω u·n dA [m³/s], n out of the structure
   res.radiated_power(on="front")                       # ½ Re ∫ p* jω u·n dA [W] through a boundary
   res.radiated_power(radius=0.15, center=(0, z_b), baffle=z_b)   # flux of ½ Re(p v*) through a sphere
   ts = res.thiele_small(on="front")                    # voltage drive: fs, Mms, Cms, Rms, Qms, Qes, Qts, Sd, Vas

Boundary integrals use the curved edges with ``dA = 2πr ds``. The Thiele–Small parameters fit the
motional impedance ``Z − (Re + (jω)ⁿ Le)`` as one moving mass (Mms, Cms by least squares of its inverse
around the peak, Rms at fs); ``Sd = |Q| / |v_coil|`` at fs, ``Vas = ρc² Sd² Cms``. Checked: a rigid
translation sweeps exactly π r_max²; the power through the front interface equals the coupling
operator's (10⁻¹¹) and the flux through a sphere in the air (10⁻⁴); a lumped driver's parameters are
recovered to round-off.

ParaView: ``res.export("series.pvd")`` writes one ``.vtu`` per entry and per mesh, with the frequency,
time or mode as the time step.

Far field
---------

A 2D model computes the air in a finite domain. ``res.far_field`` gives the pressure **outside** it,
radiated into free space (or in front of an infinite rigid baffle), from the result alone: the
Kirchhoff–Helmholtz integral of p and ∂p/∂n over a sphere around the loudspeaker (a meridian arc of
the air mesh, revolved).

.. code-block:: python

   ff = res.far_field(lp.Arc(1.0, range(0, 91, 5), center=(0, z_b)),   # 1 m, directivity
                      radius=0.2, center=(0, z_b), baffle=z_b)          # sphere in the air; baffle plane
   ff.spl()                                                            # Data over (f, angle)

The sphere must lie in the physical air (not in a PML) and enclose the loudspeaker; the points must be
outside it. Checked against analytical monopoles (free field and baffled: 3·10⁻⁴ at 1 kHz with 1 cm
P2 elements) and against the FE field inside the mesh. 3D scenes: ``res.far_field(at)`` uses the exact
DtN expansion stored in the file.

SPL is **RMS** everywhere in loudpy (``|p| / √2``; ``rms=False`` for peak amplitudes).
``lp.spl(p)`` converts any complex amplitudes.

Fields on the mesh
------------------

.. code-block:: python

   p = res.field("p", f=1000)           # Field: values on the mesh at the nearest frequency
   p.values, p.mesh, p.coordinate       # NumPy arrays, geometry.Mesh, 991.9 (Hz)
   p.spl(), p.magnitude(), p.real, p.phase()
   u = res.field("u", f=1000)
   u.component("z")                     # one component (r, z, theta)
   u.at_phase(0.3)                      # instantaneous real field Re(u e^{iφ})
   u.restrict("surround")               # the field on some regions
   u.at((0.0, 0.03)), u.at(lp.Line(...))  # values at a point / along a selection
   coords, values = u.at_nodes(res.boundary("interface_acou_meca_front"))

Modes
-----

.. code-block:: python

   modes = lp.load("modes.h5")
   modes.frequencies, modes.zeta        # NumPy arrays
   modes.table()                        # pandas DataFrame: set, order m, f, ζ
   modes.mode(3)                        # Field: the shape of mode 3
   modes.order(1)                       # the modes of azimuthal order 1 (Fourier modes)
   modes.set("fsi")                     # one mode set of a file holding several
   modes.mode(3).revolve()              # 3D surface (pyvista) of the mode, its order m applied

Saving
------

``study.save(path, **metadata)`` or ``res.save(path)`` write the
:doc:`loudpy result format 2 <file_format>`; the metadata come back in ``res.meta``. Large 3D
fields are read lazily (only the rows asked for are read from disk).
