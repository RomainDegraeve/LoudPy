Post-processing and export
==========================

Every evaluated result is a :class:`Data <loudpy.Data>` (values over named coordinates) or a
:class:`Field <loudpy.Field>` (values on a mesh). Both are plain NumPy underneath, convert to
pandas, and write files that other software reads directly.

In Python: NumPy, pandas
------------------------

.. code-block:: python

   d = res.spl(at=lp.Arc(0.3, range(0, 91, 5)))   # Data over (f, angle)
   d.values                       # (n_f, n_angle) ndarray — no copy
   np.asarray(d)                  # the same; any NumPy function takes a Data
   d.coords["f"], d.coords["angle"], d.dims
   d.quantity.unit                # "dB re 20 µPa"
   d.sel(angle=30)                # nearest angle → Data over f
   d.sel(f=slice(100, 1000))      # a range
   d.isel(angle=0)                # by position
   d.normalize(angle=0)           # re on axis (dB subtracted, others divided)
   d.to_pandas()                  # DataFrame: index f, one column per angle
   d.to_xarray()                  # xarray.DataArray (if xarray is installed)

Complex amplitudes stay complex: ``abs()``, ``phase(deg=True)``, ``real``, ``imag``, ``db()``,
``spl()``, ``velocity()`` (×jω), ``acceleration()`` (×−ω²). Time data have ``spectrum()``,
frequency responses ``impulse_response()``. Arithmetic works with numbers, arrays and other Data
(``u / u.sel(f=100)``); ``rename(quantity, label=…)`` changes the quantity or the legend label.

The raw arrays of a result are always reachable: ``res.frequencies``, ``res.blocks[k].fields["p"]``
(entries × nodes), ``res.blocks[k].meshes["air"].coords``.

Files for other software
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - call
     - writes
   * - ``d.export("spl.csv")``
     - text table: ``#`` header lines (quantity, unit, source, where, then the column names),
       one row per value of the first coordinate; 2D data: one column per value of the second;
       complex values: ``re`` and ``im`` columns. Opens in Excel, MATLAB (``readmatrix``,
       ``readtable``), Origin, ``numpy.loadtxt(…, delimiter=",")``, ``pandas.read_csv(…, comment="#")``
   * - ``d.export("spl.mat")``
     - MATLAB / Octave: the values (complex kept), one variable per coordinate, an ``info`` struct
   * - ``d.export("spl.nc")``
     - **netCDF** (the standard for labelled arrays): one variable per coordinate, the values (``<name>_real``
       / ``_imag`` for complex), units and labels as attributes, the metadata (JSON). Opens with
       ``xarray.open_dataset``, MATLAB ``ncread``, ParaView, Panoply, ``ncdump``; ``lp.load_data("spl.nc")`` gives
       the ``lp.Data`` back. Sweeps: ``sweep.evaluate(fn).export("x.nc")``
   * - ``d.export("spl.npz")``
     - NumPy: ``values``, ``coord_<name>``, a JSON ``header`` (``lp.load_data`` reads it back)
   * - ``field.export("p.vtu")``
     - ParaView / VisIt: the mesh (linear sub-cells through the mid-side nodes) with the values
       (re / im for complex)
   * - ``field.export("p.csv")``
     - one row per node: coordinates, values
   * - ``field.export("p.mat")``
     - coordinates, triangles (1-based), node tags, values
   * - ``modes.to_csv("modes.csv")``
     - the mode table
   * - ``res.export("sweep.mat")``
     - every block: coordinate, fields (entries × nodes [× components]), meshes, per-entry quantities
   * - ``res.export("sweep.npz")``
     - the same as NumPy arrays (keys ``block<i>/<name>``)
   * - ``res.export("u.vtkhdf")``
     - **VTKHDF** time series (one HDF5 file): every entry is a time step (time, frequency or mode frequency) on
       the mesh stored once; opens in ParaView ≥ 5.12 (play the steps), VTK and pyvista. One file per block and
       mesh (``u_<block>_<mesh>.vtkhdf``) when there are several
   * - ``res.export("u.pvd")``
     - ParaView series of one ``.vtu`` per entry (older ParaView)

The HDF5 result file itself (``res.save``, :doc:`file_format`) is self-describing and opens in
HDFView, MATLAB (``h5read``) or h5py.

Example: a directivity map for another tool
-------------------------------------------

.. code-block:: python

   res = lp.load("Results/Files/sweep.h5")
   d = res.spl(at=lp.Arc(0.3, np.linspace(0, 90, 46)))
   d.export("directivity.csv")          # 1 + 46 columns: f, then SPL at each angle

.. code-block:: matlab

   T = readmatrix("directivity.csv", "CommentStyle", "#");   % the header lines all start with #
   f = T(:, 1); spl = T(:, 2:end);
