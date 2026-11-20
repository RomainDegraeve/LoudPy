"""
Errors and warnings of loudpy.

Every error raised by loudpy is a :class:`LoudpyError`; each one also derives from the built-in
exception it replaces (ValueError, KeyError, RuntimeError), so ``except ValueError`` keeps working.

    LoudpyError
     ├─ ModelError          (ValueError)    invalid model description: regions, boundaries, drives, settings
     │   └─ MaterialError                   missing or invalid material properties
     ├─ SelectionError      (KeyError, ValueError)  a name / selection matches nothing, or is ambiguous
     ├─ MeshError           (ValueError)    geometry or mesh that cannot be built or used
     ├─ StateError          (RuntimeError)  call order: mesh before solving, solve before saving …
     ├─ SolverError         (RuntimeError)  a solver failed
     │   └─ ConvergenceError                an iterative solver did not converge (when asked to be strict)
     ├─ ResultsError        (KeyError, ValueError)  something the results do not contain
     └─ ResultsFormatError  (ValueError)    a result file that cannot be read

    LoudpyWarning (UserWarning)
     ├─ ConvergenceWarning                  an iterative solver did not reach its tolerance (results kept)
     └─ PlausibilityWarning                 a value is valid but unlikely in SI units (e.g. a size in mm)
"""
from __future__ import annotations


class LoudpyError(Exception):
    """Base class of every loudpy error."""

    def __str__(self):                       # KeyError subclasses would otherwise quote the message
        return str(self.args[0]) if len(self.args) == 1 else super().__str__()


class ModelError(LoudpyError, ValueError):
    """The model description is invalid (regions, boundaries, drives, study or mesh settings)."""


class MaterialError(ModelError):
    """A material is missing, or lacks a property the model needs."""


class SelectionError(LoudpyError, KeyError, ValueError):
    """A name or selection matches nothing (or matches what it must not)."""


class MeshError(LoudpyError, ValueError):
    """The geometry or the mesh cannot be built, read or used."""


class StateError(LoudpyError, RuntimeError):
    """An operation was called too early (mesh the model first, solve before saving …)."""


class SolverError(LoudpyError, RuntimeError):
    """A solver failed."""


class ConvergenceError(SolverError):
    """An iterative solver did not converge."""


class ResultsError(LoudpyError, KeyError, ValueError):
    """The results do not contain what was asked for (a field, a driver, a mode set …)."""


class ResultsFormatError(LoudpyError, ValueError):
    """A result file cannot be read (unknown layout or version)."""


class LoudpyWarning(UserWarning):
    """Base class of loudpy warnings."""


class ConvergenceWarning(LoudpyWarning, RuntimeWarning):
    """An iterative solver did not reach its tolerance; the results are kept but less accurate."""


class PlausibilityWarning(LoudpyWarning):
    """A value is valid but unlikely in SI units (for example a mesh size given in millimetres)."""


__all__ = ["LoudpyError", "ModelError", "MaterialError", "SelectionError", "MeshError", "StateError",
           "SolverError", "ConvergenceError", "ResultsError", "ResultsFormatError",
           "LoudpyWarning", "ConvergenceWarning", "PlausibilityWarning"]
