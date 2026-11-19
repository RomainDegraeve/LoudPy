"""
loudpy.results — the outcome of a study, independent of how it was computed.

  Results           FrequencyResults, ModeResults, TimeResults: one interface (fields on meshes,
                    probes, SPL, mode sets / azimuthal orders, multi-block sweeps)
  load              lp.load(file | folder | glob | list): format 2, and format 1 of the former studies
  schema            result format 2 (HDF5), written by Results.save
  legacy            reader of format 1
"""
from loudpy.results.io import load
from loudpy.results.results import (Block, FieldView, FrequencyResults, ModeResults, Results,
                                    TimeResults, spl)

__all__ = ["load", "Results", "FrequencyResults", "ModeResults", "TimeResults", "FieldView", "Block", "spl"]
