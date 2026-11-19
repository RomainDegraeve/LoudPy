"""
loudpy.solvers — algebra on assembled matrices, no physics, no files.

  linear    sparse factorisation (factorize), SweepSolver (frequency sweeps, factorisation reuse),
            QEPShiftInvert (shift-invert of a quadratic eigenproblem from one n × n LU)
  eigen     GEP / QEP shift-invert (ARPACK, reproducible start vector), multishift, covering,
            MAC deduplication, filter_modes, normalisation
  time      generalised-α with Newton–Raphson (generalized_alpha_newton), free-DOF reduction
"""
