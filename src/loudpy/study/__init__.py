"""
loudpy.study — the questions asked of a model, each returning Results.

  Modes               eigenmodes: structure (axisymmetric or azimuthal orders), coupled, acoustic
  FrequencyResponse   harmonic response to lp.Force / lp.Voltage, with PerWavelength remeshing
  Transient           nonlinear time response to an lp.Force signal

They run on a meshed lp.Model. The former Problem-based classes (loudpy.Studies) do the assembly
and the solving underneath until they are folded in. (The package is named `study`, singular:
on a case-insensitive file system "studies" would be the former "Studies" folder.)
"""
from loudpy.study.frequency import FrequencyResponse, PerWavelength
from loudpy.study.modes import Modes
from loudpy.study.transient import Transient

__all__ = ["Modes", "FrequencyResponse", "PerWavelength", "Transient"]
