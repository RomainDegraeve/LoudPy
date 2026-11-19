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
   results/plot  │  loudpy.results   │  loudpy.plot (+ style)           │  Results, HDF5 format 2, load()
                 ├───────────────────┴──────────────────────────────────┤
   3D scenes     │  loudpy.scene: Driver · Profile · Scene · Cabinet    │  drivers placed in enclosures
                 ├──────────────────────────────────────────────────────┤
   physics       │  loudpy.physics: role registry (solid, air, pml,     │  a Model → operator blocks
                 │  clamped, coil, fluid_structure, radiation)          │  (K, C, M, coupling, F)
                 ├──────────────────────────────────────────────────────┤
   fem           │  loudpy.fem: quadrature · shape functions · element  │  discretisation, no physics
                 │  kernels · batched assembly · sparse patterns        │  names
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
* **A registry of roles**: each region / boundary type declares what it contributes; a new
  physics is one new class.
* **Values everywhere**: materials, damping and drives are small immutable objects.
* **One result model** and one versioned file format (:doc:`file_format`).
* **Reproducible**: fixed ARPACK start vectors, deterministic meshing option, mesh reuse by
  signature.

The low-level 3D toolkit (``loudpy.Studies.Cavity_coupling``, ``loudpy.Studies.Exterior_dtn``)
and the internal engines of the studies (``loudpy.Studies``) remain importable from their modules
but are not part of the top-level namespace.

Tests
-----

``python3 -m pytest`` runs the fast tests; ``python3 -m pytest --slow`` adds the 3D and long
cases. References were frozen from the original code before the refactor (``tests/reference``),
matched by node coordinates and, for mode shapes, by MAC; meshes are committed
(``tests/data/meshes``) because gmsh's multithreaded meshing is not bit-reproducible.

The full design notes and the migration history are in ``docs/ARCHITECTURE.md``.
