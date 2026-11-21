Studies
=======

A study takes a meshed model (or a 3D :doc:`scene <scenes>`) and returns results. All studies
share the same three steps:

.. code-block:: python

   study = lp.<Study>(model, ...)          # what to compute
   res = study.solve(...)                  # Results (FrequencyResults, ModeResults, TimeResults)
   study.save("Results/Files/name.h5", note="anything")   # loudpy result format 3 (+ metadata)

Modes
-----

.. code-block:: python

   modes = lp.Modes(model).solve(f_max=20000)                       # structure (in vacuo)
   fourier = lp.Modes(model).solve(f_max=10000, orders=[0, 1, 2, 3])  # azimuthal orders m
   fsi = lp.Modes(model, physics="coupled").solve(f_max=5000, dtn_center=(0, 0), dtn_half=0.3)

``physics`` is ``"structure"`` (default), ``"acoustic"`` or ``"coupled"`` (structure + air, with
the exact radiation condition: complex modes). The structural band is covered by shift-invert
Lanczos at several shifts (``method="multishift"``) or one ARPACK call (``method="arpack"``);
the start vector is fixed, so the solves are reproducible.

Frequency response
------------------

.. code-block:: python

   study = lp.FrequencyResponse(model, drive=lp.Force(0.1))
   res = study.solve(np.geomspace(20, 20000, 300))

With air, the problem is coupled automatically. For wide bands the air and PML meshes can follow
the wavelength:

.. code-block:: python

   remesh = lp.PerWavelength(every=30, air_elements=6, air_max_size=0.005,
                             pml_thickness=0.25, pml_elements=4, folder="Results/Files")
   res = study.solve(freqs, remesh=remesh)      # one result block per mesh

Transient
---------

.. code-block:: python

   study = lp.Transient(model, drive=lp.Force(signal, fs=48000), rho_inf=0.98)
   res = study.solve()                          # TimeResults: u, v, a over time

Generalised-α time integration (``rho_inf``: high-frequency damping), Newton–Raphson iterations
for the nonlinear coil (``nonlinear=True``).

What a study keeps: lp.Output
-----------------------------

By default a study keeps every field at every entry (time step, frequency, mode). ``output=lp.Output(…)`` on
``solve`` (2D models) chooses what the results and the file hold:

.. code-block:: python

   out = lp.Output(fields=["u"], every=40, probes={"tip": (0.0, 0.0)})
   res = lp.Transient(model, drive=lp.Force(signal, fs=48000)).solve(output=out)

   res.times                              # every 40th step: the stored fields
   res.probe_signal("tip", "u", "z")      # u_z at the tip at every step (full rate), an lp.Data
   res.probe_signal("tip", "a")           # every component ("component" dimension)
   res.probe_names                        # ["tip"]

``fields`` keeps only the named fields, ``every`` keeps every n-th entry (per mesh block), and ``probes`` evaluates
every field at the given points for **every** entry before the fields are thinned (the probes are saved with the
results). The settings are recorded in ``res.meta["output"]``. The multitone example stores the cone tip at 48 kHz
and the displacement field at 1.2 kHz: about 70 MB instead of 8.7 GB. A 3D scene streams its field to a file
instead (``store=``).

Studies on a 3D scene
---------------------

``lp.FrequencyResponse(scene, drive=…)`` and ``lp.Modes(scene)`` accept an
:class:`lp.Scene <loudpy.Scene>`; see :doc:`scenes`.

Parameters and sweeps
---------------------

A model can carry named parameters (COMSOL's global parameters). ``lp.P(name)`` stands for a parameter
in any numeric field of an item — sizes, PML settings, material overrides, damping coefficients — and
``a * lp.P(name) + b`` is allowed. A geometry builder whose second argument is named ``p`` receives the
parameters; their values then enter the mesh signature.

.. code-block:: python

    def disc(occ, p):
        return {"plate": occ.addRectangle(0, 0, 0, 0.1, p["t"])}

    model = lp.Model(disc, parameters=lp.Parameters(t=2e-3, E=2.1e11))
    model.add(lp.Solid("plate", steel, E=lp.P("E"), mesh_size=0.5 * lp.P("t")), ...)
    model.mesh("disc.msh")

    sweep = lp.Sweep(lp.Modes(model), "t", [1.5e-3, 2e-3, 3e-3]).solve(f_max=4000)
    sweep.values                              # the parameter values
    sweep[2].frequencies                      # results of the third run
    sweep.evaluate(lambda r: r.frequencies[:3], name="f")   # lp.Data over t (× mode)
    sweep.save("Results/sweep.h5")            # one result file; lp.load gives the sweep back

Several parameters, as COMSOL's Parametric Sweep: **all combinations** (``combine="grid"``, the default) or
**specified combinations** (``combine="zip"``, the lists taken pairwise):

.. code-block:: python

    sweep = lp.Sweep(lp.FrequencyResponse(model, drive=lp.Force(1.0)),
                     {"E": [1.9e11, 2.1e11], "t": [1.5e-3, 2e-3, 3e-3]}).solve(freqs)     # 6 runs
    sweep.table()                                       # the runs (pandas)
    spl = sweep.evaluate(lambda r: r.spl(at=(0.0, 0.3)))   # lp.Data over E × t × f
    spl.sel(E=2.1e11).plot()                            # one curve per t
    spl.export("spl_sweep.nc")                          # netCDF for xarray / MATLAB / ParaView
    sweep.at(E=2.1e11, t=2e-3)                          # the results of one run

``evaluate(fn)`` stacks what ``fn`` returns for every run: an ``lp.Data`` keeps its own axes after the parameter
axes; an array or a number becomes a quantity named ``name``. Zipped runs give a ``"run"`` axis, the parameter
values of every run in ``.meta["runs"]``.

**Geometry parameters**: the arguments of a builder ``builder(occ, p)``, and the ``DefineConstant[…]`` entries of a
``.geo`` file — a model parameter of the same name replaces the value given in the file:

.. code-block:: text

    // plate.geo
    SetFactory("OpenCASCADE");
    DefineConstant[ a = 0.1, h = 0.002 ];     // radius and thickness: model parameters may set them
    Point(2) = {a, 0, 0}; …

.. code-block:: python

    model = lp.Model("plate.geo", parameters=lp.Parameters(a=0.1))
    lp.Sweep(lp.Modes(model), "a", [0.08, 0.1, 0.12])

Each run is the study on ``model.with_parameters(…)``, a copy: the model itself is not changed. A run that changes
no mesh input (a material value, a damping coefficient …) uses the model's own mesh; the others are meshed next to
it (``disc_t_0.0015.msh``, ``plate_E_2.1e+11_a_0.08.msh`` …, reused while unchanged; the parameter values are part of
the mesh signature). A sweep gives exactly the results of the separate
runs. Runs are computed one after the other.

Adaptive meshing
----------------

``lp.Adaptive`` runs a study on meshes adapted to its own error estimate, on copies of the model:

.. code-block:: python

    ad = lp.Adaptive(lp.FrequencyResponse(model), target=0.03, max_iterations=4)
    res = ad.solve(freqs)            # the results on the last mesh
    print(ad)                        # per mesh: DOFs, cells, estimate (solid, air)

After each solve the error is estimated per cell from the results (``loudpy.fem.estimate``): in the solids
the energy norm of the difference between a recovered strain (polynomial-preserving recovery) and the FE
strain, in the air outside the PML the same for the pressure gradient, each relative to the norm of the
solution, for every frequency or mode; a cell keeps its largest value. While the estimate is above
``target`` (a relative error, e.g. 0.03), new element sizes are computed (equidistribution of the error) and
the model is remeshed with them as a gmsh size view (``<mesh>_adapt1.msh`` / ``.pos`` …, next to the
model's mesh). The regions' mesh sizes stay the coarsest sizes. The history is kept in
``res.meta["adaptive"]`` (JSON) and saved with the results. Axisymmetric (m = 0) Modes and
FrequencyResponse; not with ``remesh=lp.PerWavelength``.

Convergence studies
-------------------

``lp.convergence`` solves a study on meshes with every mesh size multiplied by each factor and tabulates
a quantity (default for ``lp.Modes``: the mode frequencies):

.. code-block:: python

    table = lp.convergence(lp.Modes(model), factors=(1, 0.7, 0.5, 0.35), f_max=4000)
    print(table)
    table.n_dofs, table.values       # per mesh
    table.order, table.extrapolated  # fit q ≈ q∞ + C h^r over all meshes (h = DOFs^(-1/2))
    table.errors                     # |q − q∞| / |q∞| per mesh
    table.expected                   # smooth-solution orders for the element order p

    lp.convergence(lp.FrequencyResponse(model), quantity=lambda r: abs(r.probe("p", lp.Point(0, 1)).values),
                   freqs=[1000.0])

For elements of order p the expected orders in h are 2p for eigenfrequencies, p for the energy norm and
p + 1 for the L2 norm and point values, on smooth solutions; re-entrant corners, clamped edges and
material corners lower them (the clamped plate converges at about 2 instead of 4 on fine meshes).
