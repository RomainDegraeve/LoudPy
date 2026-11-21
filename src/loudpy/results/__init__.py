"""
loudpy.results — the outcome of a study, independent of how it was computed.

  Results           FrequencyResults, ModeResults, TimeResults (and SceneResults): fields on meshes,
                    evaluations at points / arcs / lines, mode sets, multi-block sweeps
  Data, Field       evaluated results: labelled arrays (Data) and fields on a mesh (Field) that
                    know their quantity and units, plot themselves and export (.csv, .mat, .npz, .vtu)
  Point … Line      where to evaluate (selection)
  quantities        names, labels, symbols and units of every quantity
  load              lp.load(file | folder | glob | list): format 2, and format 1 of the former studies
  schema / legacy   result format 3 (HDF5, written by Results.save; format 2 read) / reader of format 1
  description       the model description and study settings stored with the results
  SweepResults      the results of a parametric sweep (one file, one run per value)
"""
from loudpy.results.comsol import ComsolTable, Comparison, compare, read_comsol
from loudpy.results.data import Data, Field, spl
from loudpy.results.io import load
from loudpy.results.quantities import Quantity, quantity
from loudpy.results.results import Block, FrequencyResults, ModeResults, Results, TimeResults
from loudpy.results.selection import Arc, Line, Point, Points, Polar
from loudpy.results.sweep import SweepResults

__all__ = ["read_comsol", "compare", "ComsolTable", "Comparison", "load", "SweepResults", "Results", "FrequencyResults", "ModeResults", "TimeResults", "Block", "Data", "Field",
           "Quantity", "quantity", "Point", "Points", "Polar", "Arc", "Line", "spl"]
