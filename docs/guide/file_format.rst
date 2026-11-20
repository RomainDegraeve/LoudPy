Result file format
==================

loudpy writes results as **format 3**, a self-describing, self-contained HDF5 layout that does not
depend on Python class names (any HDF5 reader can open it). Besides the fields it stores what produced
them: the model description (with the geometry files) and the study settings.

.. code-block:: text

   /                       attrs: loudpy_format = "3.0", kind ("frequency" | "modes" | "time" | "sweep"),
                                  created, meta (JSON), results_class (optional, e.g. "SceneResults")
   /model/                 attrs: description (JSON): loudpy version, geometry (name, file digests),
                                  parameters, regions (material, damping, mesh size), boundaries, mesh
                                  controls, mesh (path, signature), the model tree, and "groups": the
                                  physical data of every mesh group (region, role, ρ, E, ν, c, damping,
                                  PML layer) — what derived quantities need
       files/<i>           the geometry files (uint8; attrs name: .geo and merged files, CAD file) or
                           the builder's code (up to 2 MB)
   /study/                 attrs: settings (JSON: study type, drive, options)
       arrays/<name>       large settings (e.g. the drive signal of a transient)
   /blocks/<i>/            attrs: label, meta (JSON)           one block per mesh or mode set
       values              (n,) coordinate of the entries (frequency [Hz] or time [s])
       meshes/<physics>/   physics = "solid" | "air"
           node_tags       (n_nodes,)
           coords          (n_nodes, dim)
           groups/<name>   (n_cells, k) node tags;  attrs: dim, cell_type
       fields/<name>       (n, n_nodes[, dpn]), real or complex     "u", "p", "v", "a"
       extra/<name>        (n, …) per-entry quantities (zeta, I_coil, q, dtn …)
   /boundaries/<name>/     node_tags, edges
   /attachments/<name>/    arrays + attrs (JSON "meta"): data a result needs besides its fields
                           (driver modal bases, cavity modes, scene description, drive)

A **parametric sweep** (``lp.Sweep(…).solve().save("sweep.h5")``) is one file of kind ``"sweep"`` (attrs
``parameter``, ``values``) holding one complete result per value in ``/runs/<i>/``, with the layout of the
root above; ``lp.load`` returns a ``SweepResults``.

In Python, ``res.model`` is the description (``.tree``, ``.parameters``, ``.groups``, ``.files``,
``.extract(folder)`` writes the geometry files back) and ``res.study`` the settings.

* Field members keep their order (an ``order`` attribute: HDF5 sorts members alphabetically).
* Datasets larger than 5·10⁶ values are read **lazily**: 3D pressure fields stay on disk and only
  the rows asked for are read.
* 3D sweeps are written frequency by frequency (``StreamWriter``): an interrupted run keeps the
  fields of the frequencies already computed.
* Files of format 2 (the same without ``/model`` and ``/study``) and of the former format (version 1,
  ``ResultStore`` / ``StudySaver``) are still read by ``lp.load``.
