Elastic walls (shells)
======================

Cabinet walls are not rigid. loudpy models them as shells — flat Reissner–Mindlin cells, MITC3 triangles and
MITC4 quadrilaterals, six degrees of freedom per node — laid directly on the faces of the 3D air mesh, so the
wall nodes are air nodes and the coupling is exact face by face.

In a scene
----------

.. code-block:: python

    scene = lp.Scene(cabinet, [hp.place("main", at=(x0, y0))])
    scene.cavity(f_max=6000, eta=0.02)
    scene.walls("MDF", thickness=0.018)            # lp.Rayleigh() damping from the MDF fit, panel edges clamped
    scene.mesh("Results/Files")

    walls = lp.Modes(scene, physics="walls").solve(f_max=3000)    # in-vacuo panel modes
    res = lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve(freqs)
    res.wall_frequencies, res.wall_field(f=400), res.wall_mode(0)
    res.wall_volume_velocity(), res.driver_volume_velocity()       # into the cavity [m³/s]

``scene.walls(material, thickness, damping=None, fixed="edges", f_max=None)``: the shells cover the faces of the
cavity mesh that are not driver surfaces (its "walls" group). ``fixed="edges"`` clamps the panel edges (folds
sharper than ``angle``, 30° by default) and the rims of the driver cut-outs; a function of the node coordinates
returning a mask clamps other nodes. The walls enter the cavity coupling through their in-vacuo modes up to
``f_max`` (default: the cavity's), exactly like the driver modes: the coordinates ``q`` of the results hold the
driver modes first, then the wall modes. ``lp.Modes(scene)`` gives the coupled driver + wall + cavity modes.

Limits (phase I): the walls are coupled to the cavity only (configuration ``cavity``; the outside of the walls
and the driver fronts are in vacuo), the shells lie on the inner faces of the walls (no offset), and the
materials are isotropic (plywood's two directions are averaged).

The elements
------------

:mod:`loudpy.fem.shell` (``shell_matrices``, ``modes``) builds K and M of a shell mesh in 3D. In each cell's
plane: membrane (Q1 / constant strain), bending κ, transverse shear γ with the MITC assumed strains (MITC4:
Bathe–Dvorkin, covariant shear tied at the edge mid-points; MITC3: Lee–Bathe), and the drilling rotation tied to
the in-plane rotation by a small penalty. Verified (``tests/test_shell.py``): six rigid motions with zero energy,
patch tests, simply supported plates against the exact Mindlin frequencies (MITC4 0.43 % at 32 × 32, order 2,
no shear locking), a clamped disk against Kirchhoff (0.02 %). MITC3 keeps some shear locking when the wall is
very thin (h/L ≈ 1e-3: a few %); cabinet walls (h/L ≈ 0.02–0.05) are well inside its range.

Direct coupling
---------------

:mod:`loudpy.physics.shell` solves the walls and the air as one system (no modal truncation), for checks and
small models:

.. code-block:: python

    from loudpy.physics.shell import ShellStructure, wall_shells, direct_modes, direct_response

    walls = wall_shells(nodes, tets, faces)              # MITC3 cells on TRI6 air faces, normals into the air
    shell = ShellStructure(walls.nodes, walls.cells, "MDF", 0.018, fixed=walls.feature_nodes())
    A = walls.coupling(len(nodes))                       # ∫ N_a n_i N_j dS
    f, d, p = direct_modes(shell, A, H, Mair, rho, c, n=10)

with (K + sC + s²M) d + A p = F and (H + s²Q) p − ρ s² Aᵀ d = 0. With stiff walls the coupled frequencies tend
to the rigid-wall cavity modes as 1/E; the scene's modal coupling converges to this direct solution as wall and
cavity modes are added (``tests/test_shell_air.py``).
