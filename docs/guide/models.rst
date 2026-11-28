Models
======

An :class:`lp.Model <loudpy.Model>` is an axisymmetric loudspeaker: a gmsh geometry of the
meridian half-plane, the **regions** that fill its surfaces, the **boundaries** on its curves,
and a material library. Everything is declared with small value objects passed to
:meth:`Model.add <loudpy.Model.add>`.

.. code-block:: python

   model = lp.Model("speaker.geo")                     # materials: loudpy's library by default
   model.add(
       lp.Solid("cone", "Paper", mesh_size=5e-4),
       lp.Solid("surround", "Rubber", damping=lp.Hysteretic(eta=0.1)),            # … one item per solid region
       lp.Air("air", mesh_size=5e-3),
       lp.PML("PML", thickness=0.1, tuned_to=1000.0, order=2, strength=1.5),
       lp.Clamped("rims", on=lp.Group("surround_rim", "spider_rim")), lp.Clamped("axis", dofs="r"),
       lp.Coil("coil_drive"),
       lp.FluidStructure("wet_front"),
       lp.FluidStructure("wet_rear"),
   )
   model.mesh("Results/Files/mesh.msh")

Geometry
--------

The first argument of ``lp.Model`` is the meridian half-plane (r ≥ 0 horizontal, z the axis), in
metres:

================================= ===============================================================
``"speaker.geo"``                 a gmsh ``.geo``. Without surfaces (the curve-loop convention), every
                                  **closed physical curve group** is a domain, made a surface by the
                                  mesher; with surfaces (``Plane Surface``, ``Physical Surface`` …), its
                                  physical surface groups are the domains.
``"section.step"`` / ``.iges``    a CAD file: its faces are the domains; a **sketch** (curves only)
                                  gives the closed regions of its curves. STEP / IGES lengths are
                                  converted to metres from the file's unit; ``.brep`` is read as metres.
``builder``                       a function ``builder(occ)`` drawing with gmsh's OpenCASCADE kernel and
                                  returning the named parts ``{label: surface tag(s)}``
================================= ===============================================================

Surfaces are fragmented together, so that the mesh is conforming whatever the drawing. A surface in
no named group (a STEP face, a region of a sketch) is a domain ``surface_<tag>``: select it by
position (``lp.Near``, ``lp.Box``), and its mesh group then takes the name of the region.

.. code-block:: python

   # the loudspeaker straight from the CAD sketch (curves in mm, no names): regions by position
   model = lp.Model("HPNEW-Sketch.step")
   model.add(lp.Solid("dustcap", "Polypropylene", on=lp.Near((0.0089, 0.0105))),
             lp.Air("front_air", on=lp.Near((0.02, 0.02))))     # and so on

   # a builder: parts named on the fly
   def plate(occ):
       return {"plate": occ.addRectangle(0, 0, 0, 0.1, 0.002)}
   model = lp.Model(plate)              # give the mesh file: model.mesh("plate.msh")

A builder's mesh is reused while its code, its default arguments and the simple constants it reads
are unchanged.

Regions
-------

========================================= ==================================================
:class:`lp.Solid <loudpy.Solid>`          elastic structure (Lagrange P1–P3 / Q1–Q2, P2 by default; axisymmetric or azimuthal order *m*)
:class:`lp.Air <loudpy.Air>`              linear acoustic fluid
:class:`lp.PML <loudpy.PML>`              perfectly matched layer, grown by the mesher outward from its base curves
========================================= ==================================================

The first argument is the region's **name**, its label in the results. By default it also selects
the physical groups (see :doc:`conventions`): ``"glue"`` selects ``glue_1`` and
``glue_2``, ``"coil"`` selects ``coil`` but not ``coil_former``, and a glob
(``"glue_*"``) is allowed. ``on=`` selects them otherwise (see `Selections`_ below). Material
properties can be overridden on the region (``lp.Solid("cone", "Paper", E=3e9)``).

Boundaries
----------

========================================================= =========================================
:class:`lp.Clamped <loudpy.Clamped>`                       fixed displacement (``dofs="rz"`` by default, ``"r"`` or ``"z"``)
:class:`lp.Coil <loudpy.Coil>`                             where the drive acts (``direction="z"`` by default)
:class:`lp.FluidStructure <loudpy.FluidStructure>`         structure ↔ air interface (normal from the mesh)
:class:`lp.Radiation <loudpy.Radiation>`                   exact DtN radiation condition on a spherical boundary
========================================================= =========================================

As for the regions, the name is the label and selects named curve groups by default; ``on=``
selects edges of the mesh otherwise (an interface then needs no curve drawn in the geometry).

Selections
----------

Every region and boundary takes a selector as ``on=``:

========================================================= =================================================
:class:`lp.Group("cone", "dustcap") <loudpy.Group>`        named groups, by label (naming rules, globs)
:class:`lp.Box(r=(0, 0.01), z=0.032) <loudpy.Box>`         what lies entirely inside a box; a number is a
                                                           line (``lp.Box(r=0)``: on the axis)
:class:`lp.Near((r, z)) <loudpy.Near>`                     the domain containing the point, or the
                                                           boundary curve nearest to it
:class:`lp.Interface(a, b) <loudpy.Interface>`             the edges shared by the domains of ``a`` and ``b``
:class:`lp.BoundaryOf(a) <loudpy.BoundaryOf>`              the edges around the domains of ``a``
``a | b``, ``a & b``, ``a - b``                            union, intersection, difference
========================================================= =================================================

A string stands for ``lp.Group(string)``, or for a region when it is the name of a region of the
model (``lp.Interface("cone", "front_air")``); a region or boundary stands for what it covers, and
items combine like selectors (``lp.Interface(cone | dustcap, front_air)``). Python's precedence applies (``-`` before ``&`` before ``|``): use parentheses when mixing.

.. code-block:: python

   solids = lp.Group("cone", "dustcap", "surround", "spider", "former", "coil", "glue")
   model.add(
       lp.Solid("cone", "Paper", on=lp.Group("cone") | "dustcap"),
       lp.Clamped("spider_rim", on=lp.Near((0.0447, -0.00075))),
       lp.Clamped("surround_rim", on=lp.Box(z=0.032) & lp.BoundaryOf("surround")),
       lp.Clamped("axis", on=lp.BoundaryOf("dustcap") & lp.Box(r=0), dofs="r"),
       lp.Coil("drive", on=lp.Interface("coil", "former")),
       lp.FluidStructure("front", on=lp.Interface(solids, "air_front")),
       lp.FluidStructure("rear", on=lp.Interface(solids, "air_rear")),
   )

**Where they are evaluated.** A region covers whole domain groups: its selector is evaluated
before meshing, on the curve loops of the geometry. A PML grows from curves of the geometry: its
selector is evaluated before meshing, on the curves (see `PML layers`_). A boundary covers edges of the mesh: its selector is evaluated after meshing, on the
edges between two different domain groups, on the outside of the meshed domains and on the named
curve groups kept in the mesh. A boundary made of labels only (``lp.Group``) keeps one part per
named curve group, as before; any other selector gives one part named after the boundary.
``model.select(selector)`` returns the selected edges ``[corner, corner, mid]`` of the meshed model
(``kind="domain"``: the mesh groups), and ``field.restrict(selector)`` restricts a result field.

A "boundary curve" of :class:`lp.Near <loudpy.Near>` is a chain of edges that stops at corners
(a turn of more than 30°), at junctions and where the adjacent domains change (on the mesh), or a
curve of the ``.geo`` (for a PML).

PML layers
~~~~~~~~~~

``lp.PML(name, on=…)`` selects the **base curves** of the layers: straight lines r = const or
z = const, each bounding one air region. Each layer grows by ``thickness`` along the outward normal
of that air (found from the geometry, whatever the direction in which the curve is drawn), and a
corner layer fills the gap where two layers meet at a right angle. The absorption grows from the base
across the layer (along r, z, or both in a corner).

.. code-block:: python

   lp.PML("PML", thickness=0.1, tuned_to=1000.0)                      # the PML_* curve groups (labels)
   lp.PML("absorber", on=lp.Box(r=0.3) | lp.Box(z=0.532), thickness=0.1, tuned_to=1000.0)

The geometry of every layer (start, thickness, direction) is stored with the mesh
(``<mesh>.msh.json``, ``pml_layers``), not in the names of its surfaces. Layers built from named
curve groups are named after them (``PML_front_r``, ``PML_front_r_Corner``), the others after the
region (``absorber_1``, ``absorber_1_Corner`` …). Meshes written by earlier versions held the layer
geometry in their surface names (``PML_front_r__r0=…__t=…``): they are still read.

Materials and damping
---------------------

A :class:`lp.Material <loudpy.Material>` holds SI values: a **solid** has ``rho`` [kg/m³], ``E``
[Pa] and ``nu``; a **fluid** has ``rho`` and ``c`` [m/s]. Each material also records the
``temperature`` [K] its values refer to and their ``source``. Values are checked on creation
(errors outside the valid range, a ``PlausibilityWarning`` outside the usual one).

``lp.materials`` is the default library (``Paper``, ``Rubber``, ``Copper``, ``Kapton``,
``PhenolicCloth``, ``Polypropylene``, ``SolidGlue``, ``Air``). The solid values are **typical
values, not verified against measurements**: give your own for quantitative work, in the script
(``lp.Material("Steel", rho=7850, E=2.1e11, nu=0.3)``, ``lp.materials["Paper"].with_(E=1.2e9)``)
or in a library loaded with :func:`lp.load_materials <loudpy.load_materials>` and passed to the
model (``lp.Model(geo, materials=lib)``):

.. code-block:: json

   {"format": "loudpy-materials/2",
    "materials": {
      "Paper": {"kind": "solid", "rho": 1200, "E": 1.0e9, "nu": 0.30,
                "damping": {"rayleigh": [18.0, 4.6e-6], "loss_factor": 0.03},
                "temperature": 293.15, "source": "datasheet XYZ"},
      "Air":   {"kind": "fluid", "model": "air", "temperature": 293.15, "pressure": 101325.0,
                "humidity": 0.0}}}

Libraries of the earlier flat format (``"type"``, ``"eta"``, ``"alpha_ray"``, ``"beta_ray"``) are
still read; ``MaterialLibrary.save`` writes the current format.

**Damping is not a material constant.** It is chosen on the region:
:class:`lp.Rayleigh <loudpy.Rayleigh>` (default; ``C = αM + βK``, viscous: every study),
:class:`lp.Hysteretic <loudpy.Hysteretic>` (complex stiffness ``K(1 + iη)``, a constant loss factor:
modes and frequency responses only; a transient study refuses it) or
:class:`lp.Undamped <loudpy.Undamped>`. A coefficient that is not given comes from the
material's *default damping fit* (``Material.rayleigh = (α [1/s], β [s])``,
``Material.loss_factor = η``). ``lp.Rayleigh.from_loss_factor(η, band=(f1, f2))`` gives the α, β
whose loss factor is η at f1 and f2 (``η(ω) = α/ω + βω``, lower in between); the library fits use
100 Hz – 1 kHz. PML settings belong to the region (``lp.PML(strength=3, order=1)`` by default), not
to the air.

Air
~~~

The library's ``Air`` is computed by :func:`lp.air <loudpy.air>` at 20 °C, 101.325 kPa, dry:
ρ = 1.2041 kg/m³, c = 343.24 m/s. 2D models, 3D scenes (``lp.Scene(…, air=…)``) and the low-level
3D toolkit all use these values. Other conditions:

.. code-block:: python

   warm = lp.air(temperature=308.15, humidity=0.8)     # 35 °C, 80 % RH: ρ = 1.126, c = 354.45
   model.add(lp.Air("air", material=warm), lp.PML("PML", thickness=0.1, tuned_to=1000.0, material=warm))
   scene = lp.Scene(cabinet, drivers, air=warm)

``lp.air`` treats air as an ideal-gas mixture of dry air and water vapour (saturation pressure
from the Buck equation). Its speed of sound is within 0.05 % of Cramer's correlation (1993) over
0–40 °C, 80–101 kPa and 0–100 % RH (checked in the verification tests).

Drives
------

* :class:`lp.Force(amplitude) <loudpy.Force>` — a force on the coil [N]: a complex amplitude for
  harmonic studies, or a sampled signal with ``fs`` for transients.
* :class:`lp.Voltage(amplitude, Re, Bl, Le, n) <loudpy.Voltage>` — an electro-mechanical coil
  (semi-inductance model ``Z_e = Re + (jω)^n Le``); the frequency results then hold the coil
  current ``I_coil`` per frequency.

Overview and checks
-------------------

``print(model)`` shows the model as a tree, like the model builder of commercial FE software: geometry, materials, regions
(material, damping, element size), boundaries, mesh controls and the mesh; ``model.tree(selections=True)``
adds what each item selects in the geometry. ``model.check()`` checks everything that can be checked
without meshing — every item selects something and no domain group twice, every material is complete and
plausible, a PML has its tuning frequency, the mesh controls select meshed groups — raises one error
listing every problem, and returns notes (no coil, air not coupled to the structure).

Element sizes can also be given by selection, next to each item's ``mesh_size``:

.. code-block:: python

    model.add(lp.MeshSize(on=lp.Group("surround", "spider"), size=0.3e-3, name="suspension"))

A control overrides the regions' size on the domain groups it selects (the last control wins).

``lp.Model(geometry, mesh_order=1 | 2 | 3)`` chooses the element order (P2 by default); every order is
assembled by the same finite-element core (:doc:`elements`). Studies never change the model: a
frequency study that remeshes per band (``remesh=lp.PerWavelength(…)``) meshes copies.

Meshing
-------

:meth:`Model.mesh(path) <loudpy.Model.mesh>` calls gmsh (P2 triangles) and **reuses** the file
when the geometry, the regions' mesh sizes and the settings are unchanged (a signature is
stored next to the mesh). Meshing is deterministic by default (one thread: the same model always gives
the same mesh); ``deterministic=False`` uses every thread (faster, the mesh may change from run to run).
The meshed model is an :class:`lp.Mesh <loudpy.Mesh>` with named groups
(``model.groups()``, ``model.selection()`` show what each item selected in the geometry). A saved
mesh that keeps more named curve groups than the model needs is reused (they do not change the
triangulation).

``model.mesh(path, size_field="sizes.pos")`` adds a gmsh size view (element sizes anywhere in the geometry,
written by ``loudpy.fem.adapt`` from error indicators): the smaller of the view and the regions' sizes
wins, and the view is part of the signature (the mesh is rebuilt when it changes).


A simulation as a document
--------------------------

``lp.Document`` holds a model (or a 3D scene) and its studies as plain data — the nodes of
:doc:`building_blocks` — in one JSON file: what the interface edits and saves. The script is generated from it.

.. code-block:: python

   studies = [{"name": "modes", "type": "Modes", "params": {"f_max": 3000.0}},
              {"name": "fr", "type": "FrequencyResponse", "params": {
                  "drive": {"type": "Force", "params": {"amplitude": 1.0}},
                  "freqs": {"from": 20.0, "to": 20000.0, "points": 200, "spacing": "log"}}}]
   doc = lp.Document.of(model, studies, name="speaker")
   doc.save("speaker/speaker.loudpy.json")       # paths relative to its folder
   doc = lp.Document.load("speaker/speaker.loudpy.json")
   model = doc.build()                            # the same model again
   study, kwargs = doc.study("fr", model)         # lp.FrequencyResponse(model, drive=…), {"freqs": …}
   Path("speaker/speaker_ui.py").write_text(doc.to_script())   # python3 speaker_ui.py meshes and runs both

Its parts are reached by **paths** (``lp.docpath``: a list element by its index or its name):

.. code-block:: python

   doc.get("studies/fr/params/freqs/to")                 # 20000.0
   doc.set("model/items/cone/params/mesh_size", 1e-3)    # the value before
   doc.remove("model/items/rims"); doc.insert("model/items", node, at=2)
   doc.problems()       # [(path, message)]: each problem with the node it is about
   doc.readiness()      # what the simulation still needs: [Step(key, title, state, message, path)] — done / todo /
                        # warning / optional: geometry, domains, materials, held, coil, motor, coupling, radiation,
                        # studies, the circuits they name (a scene: drivers, cabinet, air, motors, walls, studies,
                        # circuits); lp.readiness(model, studies, circuits) too
   doc.fingerprint("study:fr")   # a digest of what that study's results depend on (the study, the circuit it
                                 # names, the model and its files): it changes with any of them

What each group of the geometry could be — read from the geometry only (what lies on each side of its curves, whether
it is on the symmetry axis, a straight line or an arc), never from its name:

.. code-block:: python

   for s in lp.suggest(model):        # per group: Suggestion(group, kind, place, shape, types, usual, why, params, items)
       print(s.group, s.place, s.best, s.why)
   # wet_front   solid|air   FluidStructure  between the structure and the air: a fluid–structure interface
   # axis        solid|axis  Clamped         the symmetry axis: no radial displacement there   (params: dofs = "r")
   # PML_front_z air|-       PML             an outer edge of the air: a layer grown outward from it
   # dtn_front   air|-       Radiation       an outer arc of the air: exact radiation on it

A geometry without named curves (a STEP sketch of lines only: its regions are the closed areas of the sketch) gets
rows for its unnamed pieces: ``"cone ↔ air_front"`` (the curves two regions share; ``s.selector`` is
``lp.Interface(lp.Near(…), lp.Near(…))``, a point inside each region) and ``"cone · edge near (r, z)"`` (one outer
curve, ``lp.Near`` its middle). Give such a selector as an item's ``on=`` (the interface's Physics tables do).

Each item type declares what it is next to its class (``lp.Solid.facts``: what it selects — domain groups or curves
—, the medium it fills, its physics, its role for the checklist — holding the structure, the drive, the coupling, the
absorbing end of the air —, its key parameter and where it fits); ``lp.schema.block("Solid").facts`` and
``.key`` show them to programs, and ``lp.readiness`` / ``lp.suggest`` / the interface's Physics tables read only
these. A new item type is one class with its facts plus ``lp.schema.register(cls, params)``: it then appears in the
tables, the suggestions and the checklist.

A study built by ``doc.study(name)`` stamps its results with what made them: ``res.provenance`` →
``{"study", "simulation", "fingerprint"}`` (kept in the result file), so a result knows its study and whether the
simulation changed since (its fingerprint now differs).

A simulation's **workspace** (``lp.Workspace``) is its folder and its document, edited by commands — what the
interface does, and what a script can do too:

.. code-block:: python

   ws = lp.Workspace.open("speaker")                   # a folder (its <name>.loudpy.json) or the document's path
   ws.set("studies/fr/params/freqs/to", 5000.0)        # checked on its node (lp.schema): refused, the document as
                                                       # it was, when the node does not hold
   with ws.batch("crossover at 400 Hz"):               # several changes, one undo step (one refused: none made)
       ws.set("circuits/xo/params/parts/hi/params/H/0/params/fc", 400.0)
       ws.set("circuits/xo/params/parts/lo/params/H/0/params/fc", 400.0)
   ws.undo(); ws.redo()
   ws.on_change(lambda paths: print(paths))            # the paths every change touched (undo / redo too)
   ws.target                                           # the model or scene: built again only when it changed
   ws.readiness()                                      # what it still needs
   study, kwargs = ws.study("fr")                      # its results stamped (provenance)
   ws.results()                                        # results/*.h5: [{"name", "study", "state": current /
                                                       # out of date / unknown}]
   ws.save()                                           # <name>.loudpy.json and <name>_ui.py

