Architecture
============

loudpy is organised in layers; dependencies only point down, and each layer is importable and
testable on its own.

.. code-block:: text

                 ┌──────────────────────────────────────────────────────┐
   public API    │  loudpy  (import loudpy as lp)                       │  re-exports only
                 ├──────────────────────────────────────────────────────┤
   studies       │  loudpy.study: Modes · FrequencyResponse · Transient │  orchestration
                 ├───────────────────┬──────────────────────────────────┤
   results/plot  │  loudpy.results   │  loudpy.plot                     │  Results → Data / Field (quantities,
                 │  (Data, Field,    │  (curves, fields, view3d,        │  selections, exports), HDF5 format 2;
                 │  selections)      │  scene3d, animate, style)        │  plots read Data / Field only
                 ├───────────────────┴──────────────────────────────────┤
   3D scenes     │  loudpy.scene: Driver · Profile · Scene · Cabinet    │  drivers placed in enclosures
                 ├──────────────────────────────────────────────────────┤
   physics       │  loudpy.physics: parts + role registry, assembled    │  a Model → operator blocks
                 │  systems (structure, air, DtN), modal (3D cavity),   │  (K, C, M, coupling, F)
                 │  exterior (half-space DtN FE)                        │
                 ├──────────────────────────────────────────────────────┤
   fem           │  loudpy.fem: Lagrange P1–P3 / Q1–Q2 · quadrature ·   │  discretisation, no physics
                 │  spaces · generic kernels · assembly · sparse        │  names
                 ├──────────────────────────────────────────────────────┤
   geometry      │  loudpy.geometry: Mesh · naming · meshing2d ·        │
                 │  meshing3d (cut-outs, classification) · Frame        │
                 ├──────────────────────────────────────────────────────┤
   model         │  loudpy.model: Model · regions · boundaries ·        │  declarative, no numerics
                 │  materials · damping · drives                        │
                 └──────────────────────────────────────────────────────┘
   solvers (side layer, NumPy/SciPy only): linear (sweeps) · eigen (ARPACK, multishift, QEP) · time

Rules
-----

* ``model`` imports nothing numerical; ``solvers`` know nothing about physics (matrices in,
  vectors out); ``physics`` never touches files; ``plot`` only reads results.
* **Composition, not inheritance**: a study holds a model, a drive and solver options.
* **Explicit selectors**: regions and boundaries select mesh groups by token rules
  (:doc:`conventions`); orientation and clamped directions are parameters.
* **A registry of roles**: each region / boundary type declares what it contributes to the system
  (see below); a new physics is one new contribution.
* **Values everywhere**: materials, damping and drives are small immutable objects.
* **One result model** and one versioned file format (:doc:`file_format`).
* **Reproducible**: fixed ARPACK start vectors, deterministic meshing by default, mesh reuse by
  signature.

Systems: fields, terms and contributions
----------------------------------------

Every study solves a system declared on a ``loudpy.physics.builder.SystemBuilder``;
no study assembles blocks itself.

* **Fields** are the unknowns: structural displacement ``u``, air pressure ``p``, coil current ``I``,
  the auxiliaries of the exact radiation ``phi``, driver modal coordinates ``q`` and cavity modes ``a``
  in 3D scenes.
* **Terms** are blocks f(s)·B between two fields, with s = jω: B times s⁰, s¹ or s², or any function of
  s (the semi-inductance sⁿ Le, a cavity eliminated through its modes), and ``LowRank`` operator blocks
  that are dense on a boundary only and never assembled (the 3D half-space DtN).
* **Contributions**: each role (solid, air, fluid–structure, radiation, coil, …) adds its own terms, in
  the frequency form (forced responses) or the symmetric form (coupled modes). A new physics is one
  contribution and its role; nothing else changes.

The builder gives A(s) for a frequency sweep, the polynomial form (K, C, M) for eigenproblems (refused if
a term is not polynomial), and a ``BlockSolver`` that eliminates
small dense fields exactly and runs GMRES on the Schur complement of the large one with a preconditioner
— the scheme of the 3D scenes (driver and cavity modes eliminated, exterior FE air iterated).

Results and plots
-----------------

Results are evaluated into labelled data — :class:`Data <loudpy.Data>` (values over named
coordinates) and :class:`Field <loudpy.Field>` (values on a mesh) — that carry their quantity
(:mod:`loudpy.results.quantities`: one name, label, symbol, unit and colormap per quantity). The
plots read only those objects, so every figure gets the same labels, units and colormaps, and every
evaluated quantity exports the same way (NumPy, pandas, CSV, MATLAB, VTU). Where to evaluate is a
selection (point, points, arc, line), like COMSOL's datasets.

The 3D toolkit (:mod:`loudpy.physics.modal`, :mod:`loudpy.physics.exterior`) is importable from its
modules but is not part of the top-level namespace.

Tests
-----

``python3 -m pytest`` runs the fast tests; ``python3 -m pytest --slow`` adds the 3D and long
cases. References were frozen from the original code before the refactor (``tests/reference``),
matched by node coordinates and, for mode shapes, by MAC; meshes are committed
(``tests/data/meshes``) because gmsh's multithreaded meshing is not bit-reproducible.

The full design notes and the migration history are in ``docs/ARCHITECTURE.md``.
