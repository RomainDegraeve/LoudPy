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

Geometry
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Mesh
   Frame

Studies
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Modes
   FrequencyResponse
   PerWavelength
   Transient

3D scenes
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Driver
   Profile
   Scene
   Cabinet

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
   plot.scene_view

.. automodule:: loudpy.plot.view3d
   :members: tet_grid, tri_surface, clip, isolines, edges, dashed, colorbars_for

.. automodule:: loudpy.plot.style
   :members: use, reset, context, figure, frequency_axis, tidy, label_panels, save, pdf_pages

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
