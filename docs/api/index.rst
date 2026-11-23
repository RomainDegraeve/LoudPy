API reference
=============

Everything below is available from ``import loudpy as lp``.

.. currentmodule:: loudpy

Model
-----

.. autosummary::
   :toctree: generated
   :nosignatures:

   Model
   Solid
   Air
   PML
   Clamped
   Coil
   FluidStructure
   Radiation
   Rayleigh
   Hysteretic
   Undamped
   Force
   Voltage
   Material
   MaterialLibrary
   load_materials

Geometry and selections
-----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Mesh
   Frame
   Group
   Box
   Near
   Interface
   BoundaryOf
   mesh_info
   Circuit

Studies
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Modes
   FrequencyResponse
   PerWavelength
   Transient
   Sweep
   SweepResults
   Adaptive
   convergence

3D scenes
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Driver
   Profile
   Scene
   Cabinet

Ready-made cabinets (``lp.cabinets``):

.. automodule:: loudpy.scene.cabinets
   :members: Box, Brace, Port, Step, volume

.. currentmodule:: loudpy

Results
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   load
   spl
   Results
   FrequencyResults
   ModeResults
   TimeResults
   results.scene.SceneResults
   results.scene.DriverResponse

Evaluated data and selections
-----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Data
   Field
   Point
   Points
   Polar
   Arc
   Line
   results.Quantity

Describing a simulation (lp.schema)
-----------------------------------

Every building block with its parameters, for programs (the interface's forms): :doc:`../guide/building_blocks`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   schema.catalog
   schema.block
   schema.blocks
   schema.defaults
   schema.describe
   schema.check
   schema.build
   schema.to_value
   schema.from_value
   schema.reference_rst
   Document
   schema.Param

Plotting
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot.figure
   plot.save
   plot.pdf_pages
   plot.data
   plot.line
   plot.map
   plot.polar
   plot.field
   plot.boundary
   plot.modes
   plot.mode_chart
   plot.coupled_modes
   plot.directivity
   plot.animate
   plot.View3D
   plot.Camera
   plot.default_cameras

.. automodule:: loudpy.plot.view3d
   :members: tet_grid, tri_surface, clip, isolines, edges, dashed, colorbars_for

.. automodule:: loudpy.plot.style
   :members: use, reset, context, figure, frequency_axis, tidy, label_panels, save, pdf_pages

Errors, warnings and input checks
---------------------------------

.. automodule:: loudpy.errors
   :members:
   :show-inheritance:

.. automodule:: loudpy.checks
   :members: check

.. currentmodule:: loudpy

Low-level toolkit
-----------------

.. currentmodule:: loudpy

.. autosummary::
   :toctree: generated
   :nosignatures:

   physics.systems
   physics.modal
   physics.exterior
   results.surfaces
   solvers.eigen
   solvers.linear
   solvers.time
