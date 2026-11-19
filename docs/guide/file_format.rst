Result file format
==================

loudpy writes results as **format 2**, a self-describing HDF5 layout that does not depend on
Python class names (any HDF5 reader can open it).

.. code-block:: text

   /                       attrs: loudpy_format = "2.0", kind ("frequency" | "modes" | "time"),
                                  created, meta (JSON), results_class (optional, e.g. "SceneResults")
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

* Field members keep their order (an ``order`` attribute: HDF5 sorts members alphabetically).
* Datasets larger than 5·10⁶ values are read **lazily**: 3D pressure fields stay on disk and only
  the rows asked for are read.
* 3D sweeps are written frequency by frequency (``StreamWriter``): an interrupted run keeps the
  fields of the frequencies already computed.
* Files of the former format (version 1, ``ResultStore`` / ``StudySaver``) are still read by
  ``lp.load``.
