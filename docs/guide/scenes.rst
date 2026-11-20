3D scenes: drivers in enclosures
================================

A driver is cheap in 2D (axisymmetric, a few thousand nodes) but its enclosure is not
axisymmetric. loudpy couples the driver's **azimuthal-order modes** (``m = 0 … M``, both
``cos`` and ``sin`` families, from :class:`lp.Modes(…, orders=…) <loudpy.Modes>`) to 3D air by
modal superposition. Only the air is 3D.

The pieces
----------

.. code-block:: python

   hp = lp.Driver.from_modes("eigen_fourier.h5", orders=[0, 1, 2, 3], f_max=10000,
                             geometry="HPNEW-Sketch.geo",
                             rear=lp.Profile([14, 7, 8, 33, 32], start=(0.05435, 0.031)),
                             front=lp.Profile([5, 4, 3, 2, 1, 113, 22, 21, 20], start=(0.05735, 0.032)))

   def cabinet(occ):                     # gmsh OCC; called each time a mesh is built
       inner = occ.addBox(0, 0, -0.25, 0.2, 0.3, 0.281)
       outer = occ.addBox(0, 0, -0.25, 0.2, 0.3, 0.282)
       return lp.Cabinet(inner=inner, outer=outer)

   scene = lp.Scene(cabinet, [hp.place("main", at=(0.1, 0.18))])

* :class:`lp.Driver <loudpy.Driver>` — the modes, the coil (the ``interface_forced`` boundary of
  the modal file) and the wet **profiles** (:class:`lp.Profile <loudpy.Profile>`: chained curves
  of the 2D geometry, from the baffle to the axis), which shape the cut-outs of the driver in the
  3D air.
* :meth:`Driver.place(name, at, axis=None) <loudpy.Driver.place>` — one placed copy (axis +z
  through ``at`` by default; any axis for slanted baffles). Several placements of the same driver
  share its modal basis.
* the **cabinet builder** — a function of the gmsh OCC kernel returning an
  :class:`lp.Cabinet <loudpy.Cabinet>`: the inside air (``inner``), solids removed from it
  (``cuts``: braces, internal panels, port tube walls), the outer envelope (``outer``) and the
  solids joining inside and outside (``openings``: ports, slots; ``opening_size`` refines the
  mesh around them).

Three air configurations
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 58 20

   * - configuration
     - air
     - for
   * - ``scene.cavity(...)``
     - rigid-wall **cavity modes** of the inside air; the front in vacuo
     - "A": closed boxes, fast
   * - ``scene.cavity(...)`` + ``scene.exterior(...)``
     - cavity modes (rear) + **exterior** P2 FE with the exact half-space DtN on a hemisphere (front)
     - "B": closed box radiating
   * - ``scene.merged(...)``
     - **one air domain**: inside + openings + outside, FE + DtN; every driver wetted on both faces
     - "C": vented boxes, transmission lines

The exterior and merged air are meshed per frequency **band** (``bands=[(20, 1000), (1000, 2000)]``,
``points_per_wavelength``, ``max_size``, ``driver_size``); the ground is a rigid plane
(``ground="y"``, ``level``) and the DtN hemisphere is sized automatically (``margin``).

.. code-block:: python

   scene.merged(ground="y", bands=[(20, 1000)], points_per_wavelength=4, max_size=0.07, driver_size=0.004)
   scene.mesh("Results/Files")           # built once, reused while the scene is unchanged

   res = lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve(freqs, store="Results/Files/box.h5")

``store=`` streams the 3D pressure field to the result file frequency by frequency (the field
never has to fit in memory). Several drivers are driven together or separately:
``drive={"top": lp.Force(1), "bottom": lp.Force(1j)}``.

Scene results
-------------

``lp.load`` gives back a ``SceneResults``:

.. code-block:: python

   res = lp.load("Results/Files/box.h5")
   res.drivers                               # ["main"]
   d = res.driver("main")                    # DriverResponse
   d.excursion()                             # Data: axial displacement of the dust-cap centre over f
   d.coil_excursion(), d.displacement(at=(0.04, 0.026), theta=np.pi / 2)
   d.kinetic_energy(by_order=True)           # Data over (f, order): non-axisymmetric motion
   d.in_vacuo()                              # the same driver and drive without air (DriverResponse)
   d.surface(f=500)                          # deformed 3D surface (pyvista), placed in the scene
   res.driver_surfaces(f=500)                # every driver at one instant and one scale
   res.spl(at=(0.1, 0.2, 1.0))               # far field outside the hemisphere (exact DtN expansion),
                                             # FE field inside, cavity modes otherwise
   res.spl(at=lp.Arc(1.0, range(-180, 181, 5), center=c, axis="z", toward="x"))   # horizontal directivity
   res.far_field(at)                         # outside the hemisphere only
   res.field("p", f=500)                     # Field: FE pressure on the band mesh (read lazily)
   res.cavity_field(f=500), res.cavity_mode(3), res.cavity_frequencies   # cavity configurations

Plots: ``lp.plot.directivity(res, f, radius, center=…)`` and ``lp.plot.scene_view`` (cut-away views
of the SPL and the drivers, :doc:`plotting`).

Coupled modes of a cavity scene: ``lp.Modes(scene).solve(f_max=4500)`` (complex ``s``, driver and
cavity components in ``extra("q")``, ``extra("P")``).

Low-level toolkit
-----------------

The scene layer is built on :mod:`loudpy.physics.modal` (``DriverModes``, ``MultiDriver``,
``CavityModes``, ``DriverCavityCoupling``) and :mod:`loudpy.physics.exterior` (``HalfSpaceDtN``,
``ExteriorFE``, ``coupled_sweep``, ``port_block``), with the meshing helpers of
:mod:`loudpy.geometry.meshing3d`. Models outside the three configurations — e.g. the hybrid vented
box of ``Driver_in_cavity`` (modal inside + 1D port + exterior FE) — use them directly.
