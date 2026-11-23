"""
3D scenes (phase 6, slow): lp.Driver + lp.Scene + lp.FrequencyResponse give the same numbers as
the low-level coupling code.

  cavity (A)            on the committed boxes: same references as test_coupling_3d
  cavity+exterior (B)   small coarse scene, committed meshes: same as ExteriorFE + coupled_sweep
  merged (C)            idem, the driver wetted on both faces by one air domain
"""
from pathlib import Path

import numpy as np
import pytest

import loudpy as lp
from support import current_api as api
from support.reference import RTOL_COUPLED_MODES, RTOL_RESPONSE
from support.reference import ReferenceStore

pytestmark = pytest.mark.slow
FREQS = np.array([60.0, 120.0, 300.0, 900.0, 1500.0])
REAR, FRONT = api.REAR_PROFILE, api.FRONT_PROFILE
W, H, D = api.BOX


@pytest.fixture(scope="module")
def driver(mesh_dir, cache_dir):
    api.driver_modes(mesh_dir, cache_dir)                         # writes tests/.cache/driver_modes.h5
    return lp.Driver.from_modes(cache_dir / "driver_modes.h5", [0, 1, 2], f_max=5000.0,
                                geometry=api.SPEAKER_GEO, rear=REAR, front=FRONT)


def box_cabinet(occ):
    """The test box (inside air), and an outer box 1 mm larger at the front."""
    inner = occ.addBox(0.0, 0.0, api.Z_B - D, W, H, D)
    outer = occ.addBox(0.0, 0.0, api.Z_B - D, W, H, D + (api.Z_F - api.Z_B))
    return lp.Cabinet(inner=inner, outer=outer)


def _ref(name):
    from pathlib import Path
    return ReferenceStore(Path(__file__).parent / "reference", name, update=False)


# ── A: drivers + cavity modes, same numbers as the low-level code ─────────────
def test_cavity_one_driver(driver, mesh_dir):
    scene = lp.Scene(box_cabinet, [driver.place("main", at=(W / 2, H / 2))])
    scene.cavity(f_max=3000.0, eta=0.02, file="box_one_driver.msh", cache_modes=False)
    scene.mesh(mesh_dir, reuse="always")
    res = lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve(FREQS)
    _ref("test_driver_in_box").check(q=res.q, rtol=RTOL_RESPONSE)
    assert res.driver("main").excursion().shape == (len(FREQS),)
    assert res.cavity_pressure((0.03, 0.03, api.Z_B - 0.12)).shape == (len(FREQS),)

    # evaluated data: quantities, coordinates, in vacuo, energy by order, cavity fields
    hp = res.driver("main")
    x = hp.excursion()
    assert isinstance(x, lp.Data) and x.dims == ("f",) and x.quantity.name == "u_z"
    np.testing.assert_allclose(x.velocity().values, 2j * np.pi * FREQS * x.values)
    vac = hp.in_vacuo()
    assert vac.label.endswith("(in vacuo)") and vac.q.shape == hp.q.shape
    E = hp.kinetic_energy(by_order=True)
    assert E.dims == ("f", "order")
    np.testing.assert_allclose(E.values.sum(axis=1), hp.kinetic_energy().values, rtol=1e-12)
    probes = lp.Points([(0.03, 0.03, api.Z_B - 0.12), (0.05, 0.05, api.Z_B - 0.10)])
    assert res.spl(at=probes).dims == ("f", "point")
    p = res.cavity_field(FREQS[1])
    assert p.mesh.dim == 3 and p.values.shape == (p.mesh.n_nodes,)
    k = p.mesh.nearest((0.03, 0.03, api.Z_B - 0.12))
    np.testing.assert_allclose(p.values[k], res.cavity_pressure((0.03, 0.03, api.Z_B - 0.12)).values[1], rtol=1e-6)
    assert res.cavity_mode(1).coordinate == pytest.approx(res.cavity_frequencies[1])


def test_cavity_two_drivers_transfer(driver, mesh_dir):
    """Coil displacement of each driver for a unit force on each: the reference transfer matrix."""
    centers = [(0.07, H / 2), (W - 0.07, H / 2)]
    scene = lp.Scene(box_cabinet, [driver.place("left", at=centers[0]), driver.place("right", at=centers[1])])
    scene.cavity(f_max=3000.0, eta=0.02, file="box_two_drivers.msh", cache_modes=False)
    scene.mesh(mesh_dir, reuse="always")
    cols = [lp.FrequencyResponse(scene, drive={name: lp.Force(1.0)}).solve(FREQS).q for name in ("left", "right")]
    from loudpy.study.scene_response import multidriver
    md = multidriver(scene)
    Fs = md.coil_forces([driver.coil_nodes, driver.coil_nodes])
    u = np.stack([np.einsum("ni,fn->fi", Fs, q) for q in cols], axis=2)        # (f, i, j)
    _ref("test_two_drivers_transfer").check(u=u, rtol=RTOL_RESPONSE)


def test_scene_coupled_modes(driver, mesh_dir):
    scene = lp.Scene(box_cabinet, [driver.place("main", at=(W / 2, H / 2))])
    scene.cavity(f_max=3000.0, eta=0.02, file="box_one_driver.msh", cache_modes=False)
    scene.mesh(mesh_dir, reuse="always")
    modes = lp.Modes(scene).solve(f_max=1500.0)
    s_ref = _ref("test_driver_in_box")._get("coupled")
    np.testing.assert_allclose(np.sort(modes.extra("s").imag), np.sort(s_ref.imag), rtol=RTOL_COUPLED_MODES)


# ── A with elastic walls (phase I) ──────────────────────────────────────────────
def test_cavity_elastic_walls(driver, mesh_dir, tmp_path):
    """MDF walls (18 mm, panel edges clamped) on the committed box: wall modes as extra coordinates, the driver
    response close to the rigid box away from the wall resonances, results saved and read back."""
    scene = lp.Scene(box_cabinet, [driver.place("main", at=(W / 2, H / 2))])
    scene.cavity(f_max=3000.0, eta=0.02, file="box_one_driver.msh", cache_modes=False)
    scene.walls("MDF", thickness=0.018)
    assert "Walls: MDF" in scene.tree()
    scene.mesh(mesh_dir, reuse="always")
    walls = lp.Modes(scene, physics="walls").solve(f_max=3000.0)
    f_w = walls.frequencies
    assert len(f_w) >= 3 and f_w[0] > 500.0
    assert walls.blocks[0].fields["u_n"].shape[0] == len(f_w)
    res = lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve(FREQS, store=tmp_path / "walls.h5")
    rigid = _ref("test_driver_in_box")._get("q")
    n_d = rigid.shape[1]
    np.testing.assert_allclose(res.wall_frequencies, f_w, rtol=1e-12)
    assert res.q.shape[1] == n_d + len(f_w)
    import copy
    hp = res.driver("main")
    hp_rigid = copy.copy(hp)
    hp_rigid.q = rigid
    dev = np.abs(hp.excursion().values / hp_rigid.excursion().values - 1)
    assert dev.max() < 5e-3, dev                     # 0.16 % at 120 Hz, ≤ 1.3e-4 elsewhere
    Q = res.wall_volume_velocity()
    assert Q.dims == ("f",) and np.all(np.abs(Q.values) > 0)
    u = res.wall_field(FREQS[2])
    assert u.values.shape == (u.mesh.n_nodes,)
    assert res.wall_panels == ["walls"]                       # uniform walls: one panel
    Qp = res.wall_volume_velocity(panel="walls")              # from the shell (P1 normals) vs the coupling (P2)
    np.testing.assert_allclose(Qp.values, Q.values, rtol=0.05)
    u_rms = res.wall_displacement()
    assert u_rms.dims == ("f",) and np.all(u_rms.values > 0) and np.isrealobj(u_rms.values)
    with pytest.raises(lp.errors.SelectionError, match="no wall panel"):
        res.wall_displacement(panel="back")
    from loudpy.plot.scene3d import wall_mode_panels, wall_mode_view
    v = res.wall_view(FREQS[2], deformation=0.05)                # deformed walls, coloured by u_n re its maximum
    surf = next(ds for ds, kw, _ in v.items if kw.get("scalars") == "u_n")
    rest = np.asarray(res.wall_mesh().coords, float)
    size = np.linalg.norm(rest.max(axis=0) - rest.min(axis=0))
    assert np.abs(surf["u_n"]).max() == pytest.approx(1.0)
    assert np.linalg.norm(surf.points - rest, axis=1).max() == pytest.approx(0.05 * size)
    assert wall_mode_panels(walls) == ["walls"]
    m = wall_mode_view(walls, 0, show="abs")
    assert next(ds for ds, kw, _ in m.items if kw.get("scalars") == "u_n")["u_n"].min() >= 0
    back = lp.load(tmp_path / "walls.h5")
    np.testing.assert_allclose(back.wall_volume_velocity().values, Q.values)
    np.testing.assert_allclose(back.wall_displacement().values, u_rms.values)
    assert back.attachments["walls"]["thickness"] == pytest.approx(0.018)
    modes = lp.Modes(scene).solve(f_max=1500.0)
    assert len(modes.frequencies) >= len(_ref("test_driver_in_box")._get("coupled"))


# ── B and C: against ExteriorFE + coupled_sweep on the same meshes ─────────────
def _small(kind, driver, mesh_dir):
    scene = lp.Scene(box_cabinet if kind != "merged" else slotted_cabinet, [driver.place("main", at=(W / 2, H / 2))])
    if kind == "cavity+exterior":
        scene.cavity(f_max=3000.0, eta=0.02, mesh_size=(0.006, 0.03), file="scene_b_cavity.msh", cache_modes=False)
        scene.exterior(bands=[(20.0, 700.0)], driver_size=0.006, max_size=0.07, margin=0.04,
                       file="scene_b_exterior.msh")
    else:
        scene.merged(bands=[(20.0, 700.0)], driver_size=0.004, max_size=0.07, margin=0.04,
                     file="scene_c_air.msh")                  # 4 mm: the 1 mm front wall needs it
    # committed meshes are fixed test inputs: used as they are (they predate the curved mid-edge nodes of
    # phase D4b, which the signature now records, so reuse=True would rebuild them)
    scene.mesh(mesh_dir, reuse="always")
    return scene


def slotted_cabinet(occ):
    """The test box with a mouth: a slot through the front at the bottom (merged air)."""
    cab = box_cabinet(occ)
    cab.openings = [occ.addBox(0.02, 0.0, api.Z_B, W - 0.04, 0.03, api.Z_F - api.Z_B)]
    return cab


@pytest.mark.parametrize("kind", ["cavity+exterior", "merged"])
def test_open_scene_matches_toolkit(driver, mesh_dir, tmp_path, kind, reference):
    from loudpy.physics.exterior import ExteriorFE, coupled_sweep
    from loudpy.study.scene_response import cavity_coupling, cavity_modes, coil_force, multidriver
    scene = _small(kind, driver, mesh_dir)
    freqs = np.array([100.0, 400.0])
    res = lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve(freqs, store=tmp_path / "s.h5")
    assert res.model["kind"] == "scene" and res.model.tree                 # format 3: the scene described
    assert res.study["type"] == "FrequencyResponse"

    # the same computation, by hand with the low-level code
    md = multidriver(scene)
    F = coil_force(scene, md, lp.Force(1.0))
    dtn = scene.dtn()
    b = scene.band_meshes[0]
    front = "interface_driver" if kind == "merged" else "interface_front"
    prof = [(driver.front, driver.rear)] if kind == "merged" else [driver.front]
    ext = ExteriorFE(str(b["path"]), dtn, freqs.max(), front=front).couple_driver(md, prof, patch=[front])
    if kind == "merged":
        S, R, k2, eta = np.zeros((len(md), 0)), np.zeros((len(md), len(md))), np.zeros(0), 0.0
    else:
        cpl = cavity_coupling(scene, md, cavity_modes(scene))
        S, R, k2, eta = cpl.S, cpl.R, cpl.kappa2 if hasattr(cpl, "kappa2") else cpl.cav.kappa2, 0.02
    fields = {}
    q, coefs, _ = coupled_sweep([(b["f_lo"], b["f_hi"], ext)], freqs, F, 2 * np.pi * md.freq, md.zeta, S, R, k2,
                                eta, scene.rho, scene.c, on_frequency=lambda i, bi, p, c, qq: fields.setdefault(i, p),
                                log=lambda *a, **k: None)
    np.testing.assert_allclose(res.q, q, rtol=1e-10, atol=1e-14 * np.abs(q).max())
    for i in range(len(freqs)):
        np.testing.assert_allclose(res.field("p", index=i).values, fields[i].astype(np.complex64), rtol=1e-6)
        far = np.array([[W / 2, H / 2, 2.0]])
        np.testing.assert_allclose(res.far_field(far)[i], dtn.pressure(far, coefs[i], 2 * np.pi * freqs[i] / scene.c),
                                   rtol=1e-10)
    reference.check(q=res.q, rtol=1e-7)
    assert res.spl(at=(W / 2, H / 2, 2.0)).shape == (2,)
    arc = res.spl(at=lp.Arc(2.0, [0, 45, 90], center=(W / 2, H / 2, 0.0), axis="z", toward="x"))
    assert arc.dims == ("f", "angle") and arc.shape == (2, 3)
    assert set(res.driver_surfaces(400.0)) == set(res.drivers)
    # gate G: the far field from the file alone, in a process that has nothing but the file
    import subprocess
    import sys
    code = ("import sys, numpy as np, loudpy as lp; r = lp.load(sys.argv[1]); "
            "a = lp.Arc(2.0, [0, 45, 90], center=(%r, %r, 0.0), axis='z', toward='x'); "
            "np.save(sys.argv[2], np.asarray(r.far_field(a)))" % (W / 2, H / 2))
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "s.h5"), str(tmp_path / "far.npy")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    alone = np.load(tmp_path / "far.npy")
    live = np.asarray(res.far_field(lp.Arc(2.0, [0, 45, 90], center=(W / 2, H / 2, 0.0), axis="z", toward="x")))
    np.testing.assert_array_equal(alone, live)


# ── F3: the scene follows the model pattern ─────────────────────────────────
@pytest.mark.slow
def test_cavity_without_an_acoustic_mode_is_refused(driver, tmp_path):
    """A cavity whose f_max is below its first acoustic mode keeps only the constant mode: loudpy says to raise the
    cavity's f_max (it raised an IndexError in the static correction)."""
    scene = lp.Scene(box_cabinet, [driver.place("main", at=(W / 2, H / 2))])
    scene.cavity(f_max=300.0, eta=0.02, mesh_size=(0.012, 0.03), cache_modes=False)       # first mode ≈ 660 Hz
    scene.mesh(tmp_path)
    with pytest.raises(lp.errors.ModelError, match="constant mode"):
        lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve([100.0])


@pytest.mark.slow
def test_scene_to_script_round_trip(driver, tmp_path):
    """Driver.from_modes remembers its profiles; a scene with an lp.cabinets.Box is written as a script that builds
    the same scene (cabinet, drivers and their modes, air, parameters)."""
    import runpy
    assert driver.source["rear"] is REAR and driver.source["front"] is FRONT
    scene = lp.Scene(lp.cabinets.Box(W, H, D), [driver.place("main", at=(W / 2, H / 2))],
                     parameters=lp.Parameters(k=1.0))
    scene.cavity(f_max=1500.0, eta=0.02, mesh_size=(0.012, 0.03), cache_modes=False)
    (tmp_path / "box.py").write_text(scene.to_script(tmp_path, title="box"))
    again = runpy.run_path(str(tmp_path / "box.py"))["scene"]
    assert again.cabinet == scene.cabinet and dict(again.parameters) == {"k": 1.0}
    assert [tuple(d.frame.origin) for d in again.drivers] == [tuple(d.frame.origin) for d in scene.drivers]
    assert len(again.drivers[0].driver.modes) == len(driver.modes) and again._cavity_spec == scene._cavity_spec
    text = lp.Scene(box_cabinet, [driver.place("main", at=(W / 2, H / 2))]).cavity().to_script()
    assert "from test_scene import box_cabinet as cabinet" in text          # a builder in a module: imported
    with pytest.raises(lp.errors.ModelError, match="builder"):              # a lambda cannot be written out
        lp.Scene(lambda occ: box_cabinet(occ), [driver.place("main", at=(W / 2, H / 2))]).cavity().to_script()


@pytest.mark.slow
def test_scene_document_round_trip(driver, tmp_path):
    """A scene (Box cabinet, a driver from its modes file, cavity + exterior air) as a document: saved, read back and
    built the same; a cabinet builder of a module file is an import; a lambda is refused."""
    scene = lp.Scene(lp.cabinets.Box(W, H, D, ports=(lp.cabinets.Port((0.03, 0.03), 0.01, 0.02),)),
                     [driver.place("main", at=(W / 2, H / 2))], parameters=lp.Parameters(k=1.0))
    scene.cavity(f_max=1500.0, eta=0.02, mesh_size=(0.012, 0.03), cache_modes=False).exterior(bands=[(20, 500)])
    doc = lp.Document.of(scene, [{"name": "fr", "type": "FrequencyResponse", "params": {
        "drive": {"type": "Force", "params": {"amplitude": 1.0}}, "freqs": [100.0, 200.0]}}], name="box")
    assert doc.check() == []
    back = lp.Document.load(doc.save(tmp_path / "box.loudpy.json"))
    again = back.build()
    assert again.to_script() == scene.to_script()
    assert "scene.cavity(" in back.to_script() and "store=HERE / 'results' / 'fr.h5'" in back.to_script()
    built = lp.Document.of(lp.Scene(box_cabinet, [driver.place("main", at=(W / 2, H / 2))]).cavity())
    assert built.scene["params"]["cabinet"]["import"].endswith("test_scene.py:box_cabinet")
    with pytest.raises(lp.errors.ModelError, match="builder"):
        lp.Document.of(lp.Scene(lambda occ: box_cabinet(occ), [driver.place("main", at=(W / 2, H / 2))]))


def deep_box(occ, p):
    """The test box with a parameterised depth p["D"] (cabinet builder taking the parameters)."""
    inner = occ.addBox(0.0, 0.0, api.Z_B - p["D"], W, H, p["D"])
    outer = occ.addBox(0.0, 0.0, api.Z_B - p["D"], W, H, p["D"] + (api.Z_F - api.Z_B))
    return lp.Cabinet(inner=inner, outer=outer)


def test_scene_follows_the_model_pattern(driver, tmp_path):
    scene = lp.Model(deep_box, dim=3, drivers=[driver.place("main", at=(W / 2, H / 2))],
                     parameters=lp.Parameters(D=D, h=0.03))
    assert isinstance(scene, lp.Scene)
    with pytest.raises(lp.errors.ModelError, match="configure the scene"):
        scene.check()
    scene.cavity(f_max=1500.0, eta=0.02, mesh_size=(0.012, lp.P("h")), cache_modes=False)
    assert scene.check() == ["cavity only: the driver fronts radiate in vacuo"]
    tree = str(scene)
    assert "configuration cavity" in tree and "Parameters: D = " in tree and "'main'" in tree
    twin = scene.with_parameters(D=0.20)
    assert twin.parameters["D"] == 0.20 and scene.parameters["D"] == D and twin.folder is None

    scene.mesh(tmp_path / "base")
    sig = __import__("json").loads((tmp_path / "base" / "cavity.msh.json").read_text())
    assert sig["p2_nodes"] == "curved" and sig["parameters"] == {"D": D, "h": 0.03}
    assert sig["settings"]["cavity"]["mesh_size"] == [0.012, 0.03]          # lp.P resolved

    values = [D, 0.20]                                                     # D = 0.15 m
    sweep = lp.Sweep(lp.Modes(scene), "D", values).solve(f_max=600.0)
    for v, res in zip(values, sweep):
        alone = scene.with_parameters(D=v)
        alone.mesh(tmp_path / f"alone_{v:g}")
        ref = lp.Modes(alone).solve(f_max=600.0)
        np.testing.assert_array_equal(res.extra("s"), ref.extra("s"))
    s0, s1 = sweep[0].extra("s"), sweep[1].extra("s")
    assert len(s0) != len(s1) or not np.allclose(s0, s1, rtol=1e-6)      # the depth changes the coupled modes
    assert (tmp_path / "base" / "D_0.2" / "cavity.msh").exists()
    assert scene.parameters["D"] == D


def test_scene_results_plot_and_view(driver, mesh_dir, tmp_path):
    """One way to plot a scene result: res.plot(f) (six default cameras, cut through the driver axis, dashed
    outline of the removed part), a subset of the default cameras without a cut, res.view(f) saved as HTML."""
    pytest.importorskip("pyvista")
    from loudpy.plot.scene3d import _outline, resolve_cut, scene_bounds
    from test_plots import _can_render
    scene = _small("merged", driver, mesh_dir)
    res = lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve([300.0], store=tmp_path / "s.h5")
    cut = resolve_cut(res, "driver")
    assert cut == ((-1.0, 0.0, 0.0), (W / 2, 0.0, 0.0))
    b = scene_bounds(res)                                   # the walls (here the 1 mm front slab) and the driver
    mesh = res.field("p", index=0).mesh
    walls = mesh.coords[mesh.rows_of(np.unique(mesh.groups["walls"].cells))]
    pts = np.vstack([walls, [[W / 2, H / 2, 0.0]]])
    np.testing.assert_allclose(b, [pts.min(axis=0), pts.max(axis=0)], atol=1e-12)
    line = _outline(res, cut)
    assert line is not None and line.n_points and np.all(line.points[:, 0] <= W / 2 + 1e-6)   # the removed half
    with pytest.raises(ValueError, match="no default camera"):
        res.plot(300.0, cameras=["nowhere"])
    if not _can_render():
        pytest.skip("no off-screen OpenGL here")
    panels = lambda fig: [ax.get_title() for ax in fig.axes if ax.images and ax.get_title()]   # noqa: E731
    assert panels(res.plot(300.0)) == list(lp.plot.default_cameras(b, [[W / 2, H / 2, 0.0]]))
    assert panels(res.plot(300.0, cameras=["from above", "side, facing the cut"], cut=None)) == \
        ["from above", "side, facing the cut"]
    assert res.view(300.0).save(tmp_path / "s.html", title="scene").stat().st_size > 10_000


def test_scene_view_choices(driver, mesh_dir, tmp_path):
    """What a scene's 3D view shows: a cut across x, y or z at a value (its camera on the removed side), the air
    surfaces coloured, observation points and arcs drawn; the directivity balloon on the half sphere above the
    ground (the DtN expansion at one frequency, as for all)."""
    pytest.importorskip("pyvista")
    from loudpy.plot.scene3d import resolve_cut
    scene = _small("merged", driver, mesh_dir)
    res = lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve([300.0, 600.0], store=tmp_path / "s.h5")
    assert resolve_cut(res, ("y", 0.1)) == ((0.0, 1.0, 0.0), (0.0, 0.1, 0.0))
    assert resolve_cut(res, ("-z", 0.02)) == ((-0.0, -0.0, -1.0), (0.0, 0.0, 0.02))
    with pytest.raises(ValueError, match="axis cut"):
        resolve_cut(res, ("w", 0.0))
    assert {"ground", "walls"} <= set(res.air_surfaces)
    top = max(b["f_hi"] for b in scene.band_meshes)                 # beyond the air's meshes: refused, said why
    with pytest.raises(lp.errors.ModelError, match="outside the air's meshes"):
        lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve([300.0, 2.0 * top])
    arc = lp.Arc(1.0, np.arange(-90.0, 91.0, 5.0), (W / 2, H / 2, 0.0), axis="z", toward="x")
    v = res.view(600.0, cut=("y", H / 2), surfaces=["walls"], points=[(W / 2, H / 2, 1.0)], arcs=[arc])
    pos, focus = np.asarray(v.cam[0]), np.asarray(v.cam[1])
    assert pos[1] - focus[1] > np.abs(pos - focus)[[0, 2]].max()          # seen from above: the top is cut off
    ball, tube = v.items[-2][0], v.items[-1][0]                       # a sphere on the point, a tube on the arc
    np.testing.assert_allclose(ball.center, (W / 2, H / 2, 1.0), atol=1e-9)
    along = arc.points(3)
    d = np.min(np.linalg.norm(tube.points[:, None, :] - along[None, :, :], axis=2), axis=1)
    assert d.max() < 0.01 and len(np.unique(np.round(tube.points[:, 0], 3))) > 20
    spl = [ds for ds, kw, _ in v.items if kw.get("scalars") == "spl"]
    assert spl and all(ds.points[:, 1].max() <= H / 2 + 1e-9 for ds in spl)   # nothing above the cut
    h = res.scene["hemisphere"]
    pts = np.asarray(h["center"]) + 1.3 * h["radius"] * np.array([[0.0, 0.6, 0.8], [0.6, 0.8, 0.0]])
    np.testing.assert_allclose(res._far_at(pts, 1), res._far(pts)[1], rtol=1e-12)
    b = res.balloon_view(600.0, range_db=30.0)
    grid = b.items[0][0]
    L = np.asarray(grid["spl"])
    r = np.linalg.norm(grid.points, axis=1)
    assert r.max() == pytest.approx(1.0) and grid.points[:, h["up"]].min() >= -1e-12   # above the ground
    np.testing.assert_allclose(r, np.clip((L - L.max() + 30.0) / 30.0, 0, 1), atol=1e-9)


def _open_with_walls(driver, mesh_dir, walls=None):
    """The committed cavity + exterior scene (band 20–700 Hz), the cavity lossless, elastic walls when given."""
    scene = lp.Scene(box_cabinet, [driver.place("main", at=(W / 2, H / 2))])
    scene.cavity(f_max=3000.0, eta=0.0, mesh_size=(0.006, 0.03), file="scene_b_cavity.msh", cache_modes=False)
    scene.exterior(bands=[(20.0, 700.0)], driver_size=0.006, max_size=0.07, margin=0.04, file="scene_b_exterior.msh")
    if walls:
        scene.walls(**walls)
    scene.mesh(mesh_dir, reuse="always")
    return scene


def _radiated_power(res, scene):
    """Power through the DtN hemisphere [W] from the exact expansion: Σ |c_h|² / (4ρc k² |h_n(kR)|²)."""
    from loudpy.physics.exterior import HalfSpaceDtN, _h2
    out = []
    for blk in res.blocks:
        for f, c, nh in zip(blk.values, blk.extra["dtn"], blk.extra["dtn_n"]):
            n_max = int(round((np.sqrt(8 * nh + 1) - 3) / 2))
            n = HalfSpaceDtN.harmonics(n_max)[:, 0]
            k = 2 * np.pi * f / scene.c
            out.append(np.sum(np.abs(c[:nh]) ** 2 / np.abs(_h2(n, k * scene.dtn().R)) ** 2) / (4 * scene.rho * scene.c * k * k))
    return np.array(out)


@pytest.mark.slow
def test_walls_radiate_into_the_outside_air(driver, mesh_dir):
    """Elastic walls with the outside air (cavity + exterior, half-space DtN): the outer faces move with the walls
    (rigid through the thickness), so each wall mode displaces the outside air by minus what it displaces in the
    cavity (the bottom stands on the ground: left out); stiffer walls bring the response to the rigid box; the power
    put in at the coil is what the modes dissipate plus what the hemisphere radiates."""
    from loudpy.fem import shell as fe
    from loudpy.physics.exterior import ExteriorFE
    from loudpy.study import scene_response as sr
    freqs = np.array([150.0, 400.0, 650.0])
    scene = _open_with_walls(driver, mesh_dir, dict(material="MDF", thickness=0.018, damping=lp.Undamped()))
    cav = sr.cavity_modes(scene)
    walls = sr.scene_walls(scene, cav)
    drivers = sr.multidriver(scene)
    band = scene.band_meshes[0]
    ext = ExteriorFE(str(band["path"]), scene.dtn(), band["f_hi"], front="interface_front", surfaces=["openings"])
    ext.couple_driver(drivers, [d.driver.front for d in scene.drivers], patch=["interface_front"])
    n_d = ext.Bf.shape[1]
    ext.couple_walls(walls)
    assert ext.Bf.shape[1] == n_d + len(walls) and len(walls) >= 3
    ws = walls.walls                                              # inner volume displacement, the bottom left out
    nrm, area = fe.cell_normals(ws.nodes, ws.cells)
    up = ws.nodes[ws.cells].mean(axis=1)[:, 1] > 1e-6
    U = walls.translations()[:, ws.cells].mean(axis=2)            # (k, cells, 3)
    inner = np.einsum("kcd,cd,c->k", U[:, up], nrm[up], area[up])
    big = np.abs(inner) > 0.2 * np.abs(inner).max()
    ratio = ext.wall_volume_disp[big] / inner[big]
    assert np.all(np.abs(ratio + 1) < 0.1), ratio

    rigid = lp.FrequencyResponse(_open_with_walls(driver, mesh_dir), drive=lp.Force(1.0)).solve(freqs)
    soft = lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve(freqs)
    stiff_mdf = lp.Material("stiff MDF", rho=lp.materials["MDF"].rho, E=100 * lp.materials["MDF"].E, nu=0.3)
    stiff = lp.FrequencyResponse(_open_with_walls(driver, mesh_dir, dict(material=stiff_mdf, thickness=0.018,
                                                                          damping=lp.Undamped(), f_max=40000.0)),
                                 drive=lp.Force(1.0)).solve(freqs)
    n = rigid.q.shape[1]
    dev = [np.abs(r.q[:, :n] - rigid.q).max() / np.abs(rigid.q).max() for r in (soft, stiff)]
    assert stiff.q.shape[1] > n and dev[1] < dev[0] / 10 and dev[1] < 5e-3, dev

    cpl = sr.cavity_coupling(scene, drivers, cav)                 # power: coil in = modes dissipate + radiated
    F = np.concatenate([soft.attachments["drive"]["F"], np.zeros(len(cpl.w_s) - len(drivers))])
    w = 2 * np.pi * freqs
    p_in = 0.5 * np.real(np.conj(F)[None, :] * 1j * w[:, None] * soft.q).sum(axis=1)
    p_d = 0.5 * np.sum(2 * cpl.zeta * cpl.w_s * (w[:, None] * np.abs(soft.q)) ** 2, axis=1)
    p_rad = _radiated_power(soft, scene)
    np.testing.assert_allclose(p_in, p_d + p_rad, rtol=1e-4)
    assert np.all(p_rad > 0)


@pytest.mark.slow
def test_box_wall_panels_of_their_own(driver, tmp_path):
    """Walls of an lp.cabinets.Box, each side a panel: their thicknesses measured on the box (wall, the front between
    the drivers' planes), a Panel of its own for the back; the bottom on the ground is no wall."""
    from loudpy.study import scene_response as sr
    box = lp.cabinets.Box(W, H, D, wall=0.012)
    scene = lp.Scene(box, [driver.place("main", at=(W / 2, H / 2))])
    scene.cavity(f_max=1500.0, mesh_size=(0.012, 0.03), cache_modes=False)
    scene.walls("MDF", thickness=None, damping=lp.Undamped(), f_max=1500.0,
                panels={"back": lp.cabinets.Panel("Plywood", 0.03), "front": lp.cabinets.Panel(thickness=0.018)})
    assert "measured" in scene.tree() and "2 panel(s)" in scene.tree()
    scene.mesh(tmp_path)
    walls = sr.scene_walls(scene, sr.cavity_modes(scene))
    p = walls.shell.panels
    assert set(p) == {"left", "right", "top", "back", "front"}
    assert p["left"]["thickness"] == pytest.approx(0.012) and p["back"]["thickness"] == pytest.approx(0.03)
    assert p["back"]["material"].name == "Plywood" and p["front"]["thickness"] == pytest.approx(0.018)
    with pytest.raises(lp.errors.SelectionError, match="not panels"):
        scene.walls("MDF", thickness=None, panels={"side": lp.cabinets.Panel(thickness=0.01)})
        sr.scene_walls(scene, sr.cavity_modes(scene))


@pytest.mark.slow
def test_walls_in_a_merged_air_domain(driver, tmp_path):
    """Elastic walls in ONE air domain (inside, a port, outside; lp.cabinets.Box, 12 mm walls): every wet face of a
    wall moves with its shell — each wall mode displaces the outside air by minus what it displaces the inside; walls
    100 × stiffer bring the response to rigid walls; coil power in = modal dissipation + radiated."""
    from loudpy.physics.exterior import ExteriorFE
    from loudpy.physics.shell import wall_face_coupling
    from loudpy.study import scene_response as sr
    freqs = np.array([120.0, 300.0, 480.0])

    def scene_of(walls=None, mesh=True):
        box = lp.cabinets.Box(W, H, D, wall=0.012, ports=(lp.cabinets.Port((0.05, 0.05), 0.015, 0.03, wall=0.0),))
        s = lp.Scene(box, [driver.place("main", at=(W / 2 + 0.03, H / 2))])
        s.merged(bands=[(20.0, 500.0)], driver_size=0.004, max_size=0.07, margin=0.04,
                 file="merged_walls_{f_hi:.0f}Hz.msh")                # 4 mm: the 1 mm front wall needs it
        if walls:
            s.walls(**walls)
        return s.mesh(tmp_path) if mesh else s
    mdf = dict(material="MDF", thickness=None, damping=lp.Undamped(), f_max=3000.0,
               panels={"front": lp.cabinets.Panel(thickness=0.012)})
    scene = scene_of(mdf)
    assert scene.configuration == "merged" and scene.walls_mesh.exists()
    files = scene.mesh_files()                                    # every mesh it builds, and their state
    assert [f["what"] for f in files] == ["walls", "merged_500"] and {f["state"] for f in files} == {"built"}
    sig = Path(str(files[1]["path"]) + ".json")
    kept = sig.read_text()
    sig.write_text("{}")                                          # other settings than these
    assert [f["state"] for f in scene.mesh_files()] == ["built", "out of date"]
    sig.write_text(kept)
    assert [f["state"] for f in scene.mesh_files(tmp_path / "elsewhere")] == ["missing", "missing"]
    info = lp.mesh_info(scene.walls_mesh)                         # what a mesh is for, and its counts
    assert info["role"] == "inside air of the walls" and info["mesh_size"] == [0.003, 0.012] and info["nodes"] > 0
    assert lp.mesh_info(files[1]["path"])["band"] == [20.0, 500.0] and "Tetrahedron 10" in info["elements"]
    assert [f["state"] for f in scene.mesh_files()] == ["built", "built"]     # its counts kept: still up to date
    finer = scene_of({**mdf, "mesh_size": (0.003, 0.010)}, mesh=False)        # the shells' mesh of its own sizes
    assert [f["state"] for f in finer.mesh_files(tmp_path)] == ["out of date", "built"]
    finer.keep(["walls"])                                         # used as it is, though out of date
    assert [f["state"] for f in finer.mesh_files(tmp_path)] == ["kept", "built"]
    assert finer.mesh(tmp_path).walls_mesh == scene.walls_mesh and "scene.keep(['walls'])" in finer.to_script()
    lp.schema.build({"type": "Scene.keep", "params": {"meshes": ["air 500"]}}, scene=finer)   # the document's node
    assert finer._keep == ("air 500",) and finer._kept("merged_500") and not finer._kept("walls")
    before = finer.settings                                       # the settings as given, and back
    assert before["air"].kind == "merged" and before["walls"].mesh_size == (0.003, 0.010)
    assert before["keep"] == ("air 500",) and finer.restore({"air": before["air"]}).settings["walls"] is None
    assert finer.restore(before).settings == before
    with pytest.raises(lp.errors.ModelError, match="merged"):
        finer.restore({"cavity": scene_of(mesh=False).cavity().settings["cavity"], "air": before["air"]})
    fresh = scene_of(mdf, mesh=False)
    h = fresh.air_hemisphere()                                    # measured as mesh() will, nothing meshed
    assert h["radius"] == pytest.approx(scene.hemisphere["radius"]) and fresh.hemisphere is None
    one = scene_of(mdf, mesh=False).mesh(tmp_path / "one", only=["walls"])  # one mesh built, alone
    assert [p.name for p in (tmp_path / "one").glob("*.msh")] == ["walls_merged_walls_500Hz.msh"]
    assert [f["state"] for f in one.mesh_files()] == ["built", "missing"] and not one.band_meshes
    with pytest.raises(lp.errors.ModelError, match="keep"):
        finer.keep(["cabinet"])
    cav = lp.Scene(lp.cabinets.Box(W, H, D), [driver.place("main", at=(W / 2, H / 2))]).cavity()
    with pytest.raises(lp.errors.ModelError, match="merged air"):
        cav.walls(mesh_size=(0.003, 0.01)).mesh(tmp_path / "cavity")
    walls = sr.scene_walls(scene, None)
    assert len(walls) >= 3 and set(walls.shell.panels) == {"left", "right", "top", "back", "front"}
    band = scene.band_meshes[0]
    ext = ExteriorFE(str(band["path"]), scene.dtn(), band["f_hi"], front="interface_driver", surfaces=["openings"])
    faces = np.vstack([ext.faces[n] for n in ("walls", "openings") if len(ext.faces.get(n, []))])
    rows, B, dist = wall_face_coupling(walls, ext.nodes, ext.tets, faces)
    inside = B[dist < 1e-6].sum(axis=0)                           # faces on the shells: the inside of the walls
    outside = B[(dist >= 1e-6) & (dist < 0.02)].sum(axis=0)       # their outer faces, 12 mm away
    big = np.abs(inside) > 0.2 * np.abs(inside).max()
    np.testing.assert_allclose(outside[big] / inside[big], -1.0, atol=0.12)
    np.testing.assert_allclose(inside[big] / walls.volume_disp[big], 1.0, atol=0.05)   # the two meshes agree

    rigid = lp.FrequencyResponse(scene_of(), drive=lp.Force(1.0)).solve(freqs)
    soft = lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve(freqs)
    stiff_mdf = lp.Material("stiff MDF", rho=lp.materials["MDF"].rho, E=100 * lp.materials["MDF"].E, nu=0.3)
    stiff = lp.FrequencyResponse(scene_of({**mdf, "material": stiff_mdf, "f_max": 20000.0}),
                                 drive=lp.Force(1.0)).solve(freqs)
    n = rigid.q.shape[1]
    dev = [np.abs(r.q[:, :n] - rigid.q).max() / np.abs(rigid.q).max() for r in (soft, stiff)]
    assert soft.q.shape[1] == n + len(walls) and dev[1] < dev[0] / 10, dev

    drivers = sr.multidriver(scene)                               # power: coil in = modes dissipate + radiated
    w_s = np.concatenate([2 * np.pi * drivers.freq, 2 * np.pi * walls.freq])
    zeta = np.concatenate([drivers.zeta, walls.zeta])
    F = np.concatenate([soft.attachments["drive"]["F"], np.zeros(len(walls))])
    w = 2 * np.pi * freqs
    p_in = 0.5 * np.real(np.conj(F)[None, :] * 1j * w[:, None] * soft.q).sum(axis=1)
    p_d = 0.5 * np.sum(2 * zeta * w_s * (w[:, None] * np.abs(soft.q)) ** 2, axis=1)
    np.testing.assert_allclose(p_in, p_d + _radiated_power(soft, scene), rtol=1e-4)

    soft.save(tmp_path / "soft.h5")                               # the panels in the results, read back
    back = lp.load(tmp_path / "soft.h5")
    assert back.wall_panels == ["back", "front", "left", "right", "top"]
    assert {f"panel {p}" for p in back.wall_panels} <= set(back.wall_mesh().groups)
    th = dict(zip(back._wall_labels(), back.attachments["walls"]["thickness"]))
    assert th["front"] == pytest.approx(0.012) and th["left"] == pytest.approx(0.012, abs=1e-6)
    Q = back.wall_volume_velocity()                              # the panels' shells add up to the walls (no board)
    Qp = sum(back.wall_volume_velocity(panel=p).values for p in back.wall_panels)
    np.testing.assert_allclose(Qp, Q.values, rtol=0.05)
    u2 = sum(back.wall_displacement(panel=p).values ** 2 * back._wall_cells(p)[1].sum() for p in back.wall_panels)
    np.testing.assert_allclose(u2 / back._wall_cells()[1].sum(), back.wall_displacement().values ** 2, rtol=1e-10)
    from loudpy.plot.scene3d import wall_mode_panels
    assert wall_mode_panels(lp.Modes(scene, physics="walls").solve(f_max=3000.0)) == back.wall_panels   # the modes' panels
    v = back.wall_view(freqs[1], panels=["back", "top"])
    assert sum(kw.get("scalars") == "u_n" for _, kw, _ in v.items) == 2
    Qd = back.driver_volume_velocity()                           # merged air: the drivers' rear patches
    assert np.all(np.abs(Qd.values) > 0) and np.all(np.abs(Q.values / Qd.values) < 0.5)


@pytest.mark.parametrize("kind", ["cavity", "cavity+exterior", "merged"])
def test_voltage_drive(driver, mesh_dir, kind):
    """A scene driven by a voltage: one coil current solved with the modes, its back-EMF included. Linear, so it is
    the force drive with F = Bl·I at each frequency; the impedance is the blocked Re + (jω)ⁿ Le plus Bl² times the
    coil mobility (air and walls included); one air domain: the power put in is the coil's losses, what the modes
    dissipate and what radiates."""
    from loudpy.study import scene_response as sr
    if kind == "cavity":
        scene = lp.Scene(box_cabinet, [driver.place("main", at=(W / 2, H / 2))])
        scene.cavity(f_max=3000.0, eta=0.02, file="box_one_driver.msh", cache_modes=False)
        scene.mesh(mesh_dir, reuse="always")
    else:
        scene = _small(kind, driver, mesh_dir)
    freqs = np.array([60.0, 250.0, 600.0])
    U, Re, Bl, Le, n = 2.83, 6.0, 5.0, 3e-4, 0.8
    res = lp.FrequencyResponse(scene, drive=lp.Voltage(U, Re=Re, Bl=Bl, Le=Le, n=n)).solve(freqs)
    unit = lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve(freqs)          # the force drive, F = 1 N
    I = res.coil_current().values
    Z = res.impedance().values
    np.testing.assert_allclose(Z, U / I, rtol=1e-12)
    np.testing.assert_allclose(res.q, unit.q * (Bl * I)[:, None], rtol=1e-6, atol=1e-9 * np.abs(res.q).max())
    w = 2 * np.pi * freqs
    Fc = sr.multidriver(scene).coil_forces([driver.coil_nodes])[:, 0]
    Y = 1j * w * (unit.q[:, :len(Fc)] @ Fc)                                    # coil velocity for 1 N
    Ze = Re + (1j * w) ** n * Le
    np.testing.assert_allclose(Z, Ze + Bl ** 2 * Y, rtol=1e-6)
    assert np.all(Z.real > Re)                                                # the motional part adds losses
    if kind == "merged":                                                      # power: in = losses + radiated
        md = sr.multidriver(scene)
        p_in = 0.5 * np.real(U * np.conj(I))
        p_coil = 0.5 * Ze.real * np.abs(I) ** 2
        p_d = 0.5 * np.sum(2 * md.zeta * 2 * np.pi * md.freq * (w[:, None] * np.abs(res.q[:, :len(md.freq)])) ** 2,
                           axis=1)
        np.testing.assert_allclose(p_in, p_coil + p_d + _radiated_power(res, scene), rtol=1e-4)


def test_voltage_drive_two_drivers(driver, mesh_dir):
    """Two drivers in one box, each its own circuit: driven alike they draw the same current (symmetry); the
    current of one depends on what the other plays (their mutual impedance through the cavity air); a driver left
    out of a {name: Voltage} drive is refused (its coil must be given)."""
    centers = [(0.07, H / 2), (W - 0.07, H / 2)]
    scene = lp.Scene(box_cabinet, [driver.place("left", at=centers[0]), driver.place("right", at=centers[1])])
    scene.cavity(f_max=3000.0, eta=0.02, file="box_two_drivers.msh", cache_modes=False)
    scene.mesh(mesh_dir, reuse="always")
    freqs = np.array([80.0, 300.0])
    v = lp.Voltage(2.0, Re=6.0, Bl=5.0, Le=3e-4)
    both = lp.FrequencyResponse(scene, drive={"left": v, "right": v}).solve(freqs)
    alone = lp.FrequencyResponse(scene, drive={"left": v, "right": lp.Voltage(0.0, Re=6.0, Bl=5.0, Le=3e-4)}
                                 ).solve(freqs)
    np.testing.assert_allclose(both.coil_current("left").values, both.coil_current("right").values,
                               rtol=1e-4)                              # (the cavity mesh: symmetric to ~1e-5)
    assert np.all(np.abs(both.impedance("left").values - alone.impedance("left").values) > 1e-3)
    with pytest.raises(lp.errors.ModelError, match="names every driver"):
        lp.FrequencyResponse(scene, drive={"left": v}).solve(freqs)
    with pytest.raises(lp.errors.ResultsError, match="name one"):
        both.impedance()
