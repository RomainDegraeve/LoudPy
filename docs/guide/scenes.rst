3D scenes: drivers in enclosures
================================

A driver is cheap in 2D (axisymmetric, a few thousand nodes) but its enclosure is not
axisymmetric. loudpy couples the driver's **azimuthal-order modes** (``m = 0 … M``, both
``cos`` and ``sin`` families, from :class:`lp.Modes(…, orders=…) <loudpy.Modes>`) to 3D air by
modal superposition. Only the air is 3D.

The pieces
----------

.. code-block:: python

   cone = lp.Group("cone", "dustcap", "surround")
   hp = lp.Driver.from_modes("eigen_fourier.h5", orders=[0, 1, 2, 3], f_max=10000,
                             geometry="HPNEW-Sketch.geo",
                             rear=lp.Profile(lp.BoundaryOf("air_rear"), start=(0.05435, 0.031),
                                             end=(0.00633, 0.0052)),
                             front=lp.Profile(lp.Interface(cone, "air_front")))

   def cabinet(occ):                     # gmsh OCC; called each time a mesh is built
       inner = occ.addBox(0, 0, -0.25, 0.2, 0.3, 0.281)
       outer = occ.addBox(0, 0, -0.25, 0.2, 0.3, 0.282)
       return lp.Cabinet(inner=inner, outer=outer)

   scene = lp.Scene(cabinet, [hp.place("main", at=(0.1, 0.18))])

The air is the library's ``Air`` (20 °C, 101.325 kPa, dry: the same air as the 2D models); give
other conditions with ``lp.Scene(cabinet, drivers, air=lp.air(temperature=…, humidity=…))``
(see :doc:`models`). ``scene.rho`` and ``scene.c`` are its density and speed of sound.

* :class:`lp.Driver <loudpy.Driver>` — the modes, the coil (the ``coil_drive`` boundary of
  the modal file) and the wet **profiles** (:class:`lp.Profile <loudpy.Profile>`: chained curves
  of the 2D geometry, from the baffle towards the axis), which shape the cut-outs of the driver in
  the 3D air. A profile is selected like a boundary (see :doc:`models`), on the curves of the
  geometry: ``lp.Profile(selector)`` when the selected curves form one chain (it starts at the end
  farthest from the axis), ``lp.Profile(selector, start=…, end=…)`` for the shortest way along them
  between two points (e.g. the rear air's boundary from the baffle to the cone foot). ``geometry=``
  is a path, a builder or the driver's ``lp.Model`` (then its region names can be used). gmsh curve
  tags (``lp.Profile([14, 7, 8, 33, 32], start=…)``) are still accepted.
* :meth:`Driver.place(name, at, axis=None) <loudpy.Driver.place>` — one placed copy (axis +z
  through ``at`` by default; any axis for slanted baffles). Several placements of the same driver
  share its modal basis.
* the **cabinet builder** — a function of the gmsh OCC kernel returning an
  :class:`lp.Cabinet <loudpy.Cabinet>`: the inside air (``inner``), solids removed from it
  (``cuts``: braces, internal panels, port tube walls), the outer envelope (``outer``) and the
  solids joining inside and outside (``openings``: ports, slots; ``opening_size`` refines the
  mesh around them).

Ready-made cabinets and scenes
------------------------------

For a box, no builder is needed: :mod:`lp.cabinets <loudpy.scene.cabinets>` has a parametric box and an
import of your own CAD file, and ``lp.Scene.sealed_box`` / ``lp.Scene.vented_box`` choose the air
configuration and its settings from ``f_max``:

.. code-block:: python

   port = lp.cabinets.Port(at=(0.06, 0.06), radius=0.02, length=0.062)          # tube wall 3 mm inside
   box = lp.cabinets.Box(0.20, 0.30, 0.18, depth_top=0.11, wall=0.015,            # inner sizes, slanted back
                         braces=[lp.cabinets.Brace(y=0.08, thickness=0.015, depth=0.08)], ports=[port])
   scene = lp.Scene.vented_box(box, hp, at=(0.125, 0.19), f_max=3000)            # one air domain, 1 kHz bands
   scene.mesh("meshes")
   res = lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve(freqs, store="vented.h5")

   sealed = lp.Scene.sealed_box(lp.cabinets.Box(0.2, 0.3, 0.18), hp, at=(0.1, 0.19))                 # cavity modes
   radiating = lp.Scene.sealed_box(lp.cabinets.Box(0.2, 0.3, 0.18), hp, at=(0.1, 0.19), radiating=True)
   lp.cabinets.volume(box, scene.drivers), port.tuning(lp.cabinets.volume(box, scene.drivers))

The frame is the one of the examples: the driver axis is +z, y is up and the box stands on the ground
``y = 0``. The driver sets the front (the inside ends on its rear baffle plane, the front face is its front
plane), so a box has no front coordinate. Sizes may be parameters (``lp.cabinets.Box(lp.P("w"), …)`` with
``lp.Scene(…, parameters=lp.Parameters(w=0.2))``).

**Your own CAD file.** ``lp.cabinets.Step("box.step")`` reads the cabinet as built (STEP, IGES or BREP, any
unit: read in metres): walls, panels, braces, port tubes, as one or several solids, in the frame above
(``translate=``, ``scale=`` otherwise). loudpy finds the inside air itself (the closed void enclosed by the
solids, and by the ground when the box has no bottom), closes the driver holes at the drivers' positions,
and you name the port bores:

.. code-block:: python

   cad = lp.cabinets.Step("my_box.step", openings=[lp.cabinets.Port(at=(0.06, 0.06), radius=0.02,
                                                                    length=0.062, wall=0)])
   scene = lp.Scene.vented_box(cad, hp, at=(0.125, 0.19))

It fails loudly rather than guessing: an unnamed port or slot ("no closed inside air — the cabinet is
open"), several separate inside voids (give ``inner=(x, y, z)``, a point of the right one), a driver whose
baffle is not on the inside air or whose front is inside a panel (the file is not where the drivers are).
Tested: a box written to STEP and read back has the inside volume of the parametric box to 1e-9 and the same
cavity modes and driver response (to the discretisation, 2e-3).

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

Each mesh is rebuilt only when what it depends on changes (the cabinet, the drivers, its own settings), and
``scene.mesh_files(folder)`` tells which are up to date without meshing. A mesh can be used as it is even when out
of date with ``scene.keep(["cavity"])`` (``"walls"``, ``"air"`` for every band, ``"air 2000"`` for one band,
``"all"``), for example a fine cavity mesh kept while the air bands change. With one merged air domain and elastic
walls, ``scene.walls(…, mesh_size=(near, far))`` sets the inside-air mesh the shells lie on. ``scene.settings``
gives the air, walls and kept meshes as given and ``scene.restore(settings)`` sets them back.
``scene.air_hemisphere()`` measures the outside air's hemisphere without meshing.

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

Plots: ``res.plot(f)`` (cut-away views of the SPL and the drivers, six default cameras), ``res.view(f)``
(one view to save as HTML / glTF or turn) and ``lp.plot.directivity(res, f, radius, center=…)`` (:doc:`plotting`).

Coupled modes of a cavity scene: ``lp.Modes(scene).solve(f_max=4500)`` (complex ``s``, driver and
cavity components in ``extra("q")``, ``extra("P")``).

Low-level toolkit
-----------------

The scene layer is built on :mod:`loudpy.physics.modal` (``DriverModes``, ``MultiDriver``,
``CavityModes``, ``DriverCavityCoupling``) and :mod:`loudpy.physics.exterior` (``HalfSpaceDtN``,
``ExteriorFE``, ``coupled_sweep``, ``port_block``), with the meshing helpers of
:mod:`loudpy.geometry.meshing3d`. Models outside the three configurations — e.g. the hybrid vented
box in ``examples/legacy/vented_hybrid`` (modal inside + 1D port + exterior FE) — use them directly.

The model pattern in 3D
-----------------------

A scene follows the pattern of :class:`lp.Model <loudpy.Model>`: ``lp.Model(cabinet, dim=3, drivers=[…])``
builds it, ``print(scene)`` shows its tree (parameters, air, drivers, cavity / exterior / merged settings,
meshes), ``scene.check()`` checks it without meshing, ``scene.copy()`` and ``scene.with_parameters(…)`` give
unmeshed copies. Its settings take ``lp.P(name)`` references, and a cabinet builder whose second argument
is named ``p`` receives the parameters, which then enter the mesh signatures:

.. code-block:: python

    def box(occ, p):
        inner = occ.addBox(0, 0, -p["depth"], 0.26, 0.20, p["depth"])
        return lp.Cabinet(inner=inner)

    scene = lp.Model(box, dim=3, drivers=[hp.place("main", at=(0.13, 0.10))],
                     parameters=lp.Parameters(depth=0.15, h=0.012))
    scene.cavity(mesh_size=(0.004, lp.P("h")), f_max=3000)
    scene.mesh("Results/box")
    sweep = lp.Sweep(lp.Modes(scene), "depth", [0.12, 0.15, 0.20]).solve(f_max=1000)

Each value is meshed in ``<folder>/<name>_<value>/``. The 3D meshes are built on one thread
(reproducible) with the P2 mid-edge nodes on the CAD surfaces; the few cells a coarse mesh folds near a
strongly curved surface (det J changing sign inside) get their mid-edge nodes moved back to the straight
edges when the mesh is read. The signatures record the node placement: meshes saved before (straight
edges) are rebuilt once.
