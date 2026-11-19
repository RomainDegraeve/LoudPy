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
   results.results.FieldView

Plotting
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot.field
   plot.mode
   plot.modes_grid
   plot.spl
   plot.response
   plot.history
   plot.directivity
   plot.deformed_boundary
   plot.coupled_modes_grid

.. automodule:: loudpy.plot.style
   :members: use, reset, context, figure, frequency_axis, label_panels, save, pdf_pages
