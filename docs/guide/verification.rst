Verification manual
===================

How loudpy checks that its numbers are right, not only unchanged. The measured numbers of the last run are in the
:doc:`verification_report`; this page explains the methods behind them.

Four kinds of tests
-------------------

**Verification against exact solutions.** Problems whose solution is known in closed form, solved through the public
API, with a tolerance stated and justified in the test:

* *Manufactured solutions* (``tests/test_fem_assembly.py``): Poisson and Helmholtz problems with a prescribed
  smooth solution, on structured triangles and quadrilaterals, curved cells, axisymmetric meshes and unstructured
  tetrahedra. The observed L2 order must reach p + 1 (within 0.15, or 0.3 on unstructured tetrahedra) for P1–P3
  and Q1–Q2. Patch tests: linear fields reproduced exactly on distorted curved cells; a uniform-strain patch with
  its exact strain energy.
* *Analytical benchmarks* (``tests/test_verification.py``, ``tests/test_shell.py``, ``tests/test_shell_air.py``,
  ``tests/test_interpolation_farfield.py``, ``tests/test_derived_*.py``): clamped circular plate (3D elasticity
  against Kirchhoff), rigid rectangular cavity (exact modes, order 2p = 4), pulsating hemisphere on a rigid ground
  (exact radiated field), simply supported and clamped plates with MITC shells (exact Mindlin and Kirchhoff
  frequencies), the air spring of a cavity-backed panel, the rigid-wall limit of elastic walls, monopole far
  fields from a file, plane-wave intensity, Thiele–Small parameters of a lumped driver, the speed of sound of the
  air model against Cramer (1993).

**Physical invariants.** Properties that hold exactly for the discrete system, so that any sign or assembly error
shows at the percent level: the volume swept by a rigid translation through each fluid–structure interface
(π r²), the energy balance of the coupled speaker (coil power = structural losses + radiated power), the
antiphase of front and rear pressures of a piston, the radiated power through the wet boundary against the
intensity flux through a sphere.

**Regression.** Operators and responses frozen in ``tests/reference/*.npz`` and matched at tolerances measured from
the floating-point spread between machines (``tests/support/reference.py``): operators 1e-10, forced responses 1e-6,
damped coupled modes in near-degenerate pairs 1e-4. A reference is regenerated only on purpose, and every change is
explained by the physics in the roadmap (``docs/ROADMAP.md``): the air model (phase B), the coupling-normal fix
(C1), the damping refits. The committed meshes are fixed inputs: a run in which a test rewrites one fails.

**Plots.** Images of the plotting functions compared with committed baselines (texts hidden, axes on their grid
cells, RMS tolerance), and a test that no plotting function changes the global matplotlib settings.

The verification report
-----------------------

A verification test records each number it measures with the ``verify`` fixture (``tests/support/verification.py``):
the area, the case, the quantity, the value, the test's own bound and the reference. The outcome of the test is
attached at the end of the run, and every run writes ``tests/.cache/verification.json``. The report is generated
from the same run that gates the change:

.. code-block:: bash

   python3 -m pytest -q --slow --verification-report docs/guide/verification_report.rst

so a number in the report is always one that a passing test measured. Regenerate it for each release.

Comparison with reference results
---------------------------------

``lp.read_reference(path)`` reads the text exports of commercial FE software (tables, point graphs; ``%`` headers, columns with units,
complex values written ``a+bi``) and ``lp.compare(ours, reference, rtol=…, metric=…)`` compares two 1D ``lp.Data``
after interpolating the reference onto loudpy's points (log frequency; complex parts separately), returning the
largest error and where it occurs. The reference cases are registered in ``tests/test_reference.py`` with the export
each one needs (``tests/data/reference/README.md``); a case runs, and enters the report, as soon as its export is in
``tests/data/reference``.

What is not verified
--------------------

The material values of the library are typical values, not measurements; the far-field accuracy at low frequency
depends on the PML tuning of the model (a coarse PML tuned to 1 kHz leaves a reflected part of ~26 % at 100 Hz in
the test model); MITC3 shells keep some shear locking for very thin walls (h/L ≈ 1e-3); the walls of 3D scenes
radiate into the cavity only. Each limit is listed at the end of its phase in ``docs/ROADMAP.md``.
