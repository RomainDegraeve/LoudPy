"""
3D scenes (phase 6, slow): lp.Driver + lp.Scene + lp.FrequencyResponse give the same numbers as
the low-level coupling code.

  cavity (A)            on the committed boxes: same references as test_coupling_3d
  cavity+exterior (B)   small coarse scene, committed meshes: same as ExteriorFE + coupled_sweep
  merged (C)            idem, the driver wetted on both faces by one air domain
"""
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


# ── F3: the scene follows the model pattern ─────────────────────────────────
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
