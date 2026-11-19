"""
loudpy.scene — 3D scenes: axisymmetric drivers (from their modes) placed in a cabinet, with the air.

  Driver, Profile      a driver from an eigen result file + its wet profiles; Driver.place(...)
  Scene, Cabinet       cabinet builder + placed drivers; cavity(), exterior(), merged(); mesh()

Solve with lp.FrequencyResponse(scene, drive=...) (lp.Modes(scene) for drivers + cavity modes).
"""
from loudpy.scene.driver import Driver, PlacedDriver, Profile
from loudpy.scene.scene import Cabinet, Scene

__all__ = ["Driver", "PlacedDriver", "Profile", "Scene", "Cabinet"]
