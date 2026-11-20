"""
loudpy.study — the questions asked of a model, each returning Results.

  Modes               eigenmodes: structure (axisymmetric or azimuthal orders), coupled, acoustic
  FrequencyResponse   harmonic response to lp.Force / lp.Voltage, with PerWavelength remeshing
  Transient           nonlinear time response to an lp.Force signal
  Sweep               a study run for every value of a parameter (copies of the model)
  Adaptive            a study on meshes adapted to its error estimate; convergence: on refined meshes

They run on a meshed lp.Model (or an lp.Scene): the systems come from loudpy.physics.systems,
the algorithms from loudpy.solvers, and the results are built in memory (save them with
study.save(path)). The package is named `study`, singular (on a case-insensitive file system
"studies" was the former "Studies" folder).
"""
from loudpy.study.adaptive import Adaptive, Convergence, convergence
from loudpy.study.frequency import FrequencyResponse, PerWavelength
from loudpy.study.modes import Modes
from loudpy.study.sweep import Sweep, SweepResults
from loudpy.study.transient import Transient

__all__ = ["Adaptive", "Convergence", "convergence", "Modes", "FrequencyResponse", "PerWavelength", "Transient", "Sweep", "SweepResults"]
