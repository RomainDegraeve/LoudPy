"""Shells on the faces of a P2 air mesh, coupled directly (loudpy.physics.shell): coupling identities, the air
spring of a cavity-backed panel, the rigid-wall limit."""
import dataclasses

import numpy as np
import scipy.linalg as sla

import loudpy as lp
from loudpy.fem.shell import fixed_dofs
from loudpy.fem.tet10 import assemble_p2
from loudpy.physics.shell import ShellStructure, direct_modes, wall_shells
from support.fem_meshes import box_cavity

RHO, C = 1.2041, 343.235


def test_coupling_identities():
    """d = rigid translation: ∫ d·n = 0 on a closed box; d = x (dilation): ∫ x·n_into_air dS = −3V."""
    nodes, tets, faces = box_cavity(0.3, 0.25, 0.2, 0.08)
    walls = wall_shells(nodes, tets, np.vstack(list(faces.values())))
    A = walls.coupling(len(nodes))
    one = np.ones(len(nodes))
    d = np.zeros((len(walls.nodes), 6))
    d[:, :3] = walls.nodes
    V = 0.3 * 0.25 * 0.2
    assert np.isclose(d.ravel() @ (A @ one), -3 * V, rtol=1e-12)
    for i in range(3):
        t = np.zeros((len(walls.nodes), 6))
        t[:, i] = 1.0
        assert abs(t.ravel() @ (A @ one)) < 1e-14
    assert np.isclose(walls.area(), 2 * (0.3 * 0.25 + 0.3 * 0.2 + 0.25 * 0.2), rtol=1e-12)
    # the P2 pressure basis integrates exactly: Σ_j A[·, j] x_j = ∫ N_a n x dS for the linear field p = x
    px = A @ nodes[:, 0]
    f = px.reshape(-1, 6)[:, 0].sum()                    # Σ ∫ N_a n_x x dS = ∫ n_x x dS = −V
    assert np.isclose(f, -V, rtol=1e-12)


def test_cavity_backed_panel_air_spring():
    """Simply supported steel panel (2 mm, 0.3 × 0.3 m) on a rigid 0.1 m deep box: the first coupled mode is the
    in-vacuo panel mode stiffened by the air, f² ≈ f_p² + f_air², f_air² = 64 ρc² / (π⁴ ρ_s h L) / (2π)²
    (uniform pressure, ≪ the first cavity mode at 572 Hz); the remaining −0.15 % is the air's added mass."""
    a, L, h = 0.3, 0.1, 0.002
    steel = lp.materials["Steel"]
    nodes, tets, faces = box_cavity(a, a, L, 0.025)
    H, Ma, _ = assemble_p2(nodes, tets)
    walls = wall_shells(nodes, tets, faces["z1"])
    shell = ShellStructure(walls.nodes, walls.cells, steel, h, damping=lp.Undamped(),
                           fixed_dofs=fixed_dofs(walls.feature_nodes(), "xyz"))
    f_vac = shell.modes(2)[0][0]
    f_air = np.sqrt(64 * RHO * C ** 2 / (np.pi ** 4 * steel.rho * h * L)) / (2 * np.pi)
    f, d, p = direct_modes(shell, walls.coupling(len(nodes)), H, Ma, RHO, C, 3)
    expected = np.hypot(f_vac, f_air)
    assert -0.004 < f[0] / expected - 1 < 0.0, (f[0], expected)
    # the panel moves into the air where the pressure is high: the pressure is uniform to a few % at 117 Hz
    pm = p[0] * np.sign(p[0][np.argmax(np.abs(p[0]))])
    assert pm.min() > 0.9 * pm.max()


def test_rigid_wall_limit():
    """Gate I: all six walls elastic (MDF, 18 mm, clamped panel edges, first panel mode 1.08 kHz): the four
    lowest non-uniform cavity modes (572–894 Hz) move by up to 3.3e-3; with E × 10 and × 100 they approach the
    rigid-wall modes of the same air mesh as 1/E (1.4e-4, 1.3e-5)."""
    a, b, c = 0.3, 0.25, 0.2
    nodes, tets, faces = box_cavity(a, b, c, 0.06)
    H, Ma, _ = assemble_p2(nodes, tets)
    lam = sla.eigh(H.toarray(), Ma.toarray() / C ** 2, eigvals_only=True, subset_by_index=[1, 4])
    f_rigid = np.sqrt(lam) / (2 * np.pi)
    walls = wall_shells(nodes, tets, np.vstack(list(faces.values())))
    A = walls.coupling(len(nodes))
    dev = []
    for scale in (1.0, 10.0, 100.0):
        mat = dataclasses.replace(lp.materials["MDF"], E=lp.materials["MDF"].E * scale)
        shell = ShellStructure(walls.nodes, walls.cells, mat, 0.018, damping=lp.Undamped(),
                               fixed=walls.feature_nodes())
        f, _, _ = direct_modes(shell, A, H, Ma, RHO, C, 12, f_shift=0.5 * f_rigid[0])
        near = np.array([f[np.argmin(np.abs(f - fr))] for fr in f_rigid])
        dev.append(np.abs(near / f_rigid - 1).max())
    assert 1e-3 < dev[0] < 1e-2 and dev[2] < 2e-5, dev
    assert 9 < dev[1] / dev[2] < 11, dev


def test_wall_modes_converge_to_direct():
    """Elastic walls by modes (in-vacuo wall modes + rigid-wall cavity modes + static correction, as in the
    scenes) against the direct shell–air system: one MDF wall (6 mm) of a box under a uniform pressure load,
    60–700 Hz. With the wall modes to 1.5 / 6 kHz (cavity modes to 6 kHz) the wall displacement differs by
    ≤ 4 % / 1.1e-4 and the cavity pressure by ≤ 3.5 % / 7e-4: the wall modes' truncation dominates."""
    from loudpy.physics.modal import CavityModes, solve_forced, static_correction
    from loudpy.physics.shell import cavity_walls, direct_response
    nodes, tets, faces = box_cavity(0.3, 0.25, 0.2, 0.04)
    H, Ma, _ = assemble_p2(nodes, tets)
    freqs = np.array([60.0, 150.0, 300.0, 450.0, 700.0])
    probe = np.argmin(np.linalg.norm(nodes - [0.05, 0.05, 0.05], axis=1))
    cav = CavityModes(nodes, tets, np.zeros((0, 6), np.int64))
    cav.solve(6000.0, c=C)
    err = []
    for f_walls in (1500.0, 6000.0):
        w = cavity_walls(nodes, tets, faces["z1"], "MDF", 0.006, lp.Rayleigh(), "edges", f_walls)
        A = w.walls.coupling(len(nodes))
        F = -np.asarray(A @ np.ones(len(nodes)))
        k = 6 * np.argmin(np.linalg.norm(w.walls.nodes - [0.1, 0.08, 0.2], axis=1)) + 2
        S = w.B.T @ cav.psi[w.rows]
        q, P = solve_forced(freqs, (w.phi.T @ F)[:, None], 2 * np.pi * w.freq, w.zeta, S,
                            static_correction(cav, w.rows, w.B, S), cav.kappa2, RHO, C)
        d_m, p_m = (w.phi @ q[:, :, 0].T)[k], (cav.psi @ P[:, :, 0].T)[probe]
        if not err:
            d, p = direct_response(w.shell, A, H, Ma, RHO, C, freqs, F)
            d_d, p_d = d[:, k], p[:, probe]
        err.append((np.abs(d_m / d_d - 1).max(), np.abs(p_m / p_d - 1).max()))
    assert err[1][0] < 2e-4 and err[1][1] < 1.5e-3, err
    assert err[0][0] > 10 * err[1][0] and err[0][1] > 10 * err[1][1], err


def test_scene_wall_settings():
    """scene.walls needs the cavity and is refused with the exterior / merged air (check() lists it)."""
    import pytest
    scene = lp.Scene(lambda occ: None, []).walls("Plywood", thickness=lp.P("t"))
    with pytest.raises(lp.errors.ModelError, match="elastic walls need the cavity"):
        scene.check()
    scene.cavity().exterior()
    with pytest.raises(lp.errors.ModelError, match="coupled to the cavity only"):
        scene.check()
    assert "Walls: Plywood" in scene.tree()
    assert scene.copy()._walls_spec == scene._walls_spec
