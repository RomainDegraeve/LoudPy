"""
Ready-made cabinets (lp.cabinets) and scenes (lp.Scene.sealed_box / vented_box), default 3D views.

The geometry tests use a driver with profiles only (no modes); the slow test meshes and solves a scene from a
STEP file against the same box built by lp.cabinets.Box.
"""
import numpy as np
import pytest

import loudpy as lp
from loudpy.errors import MeshError, ModelError
from loudpy.geometry import gmsh_session
from support import current_api as api

W, H, D = api.BOX


def profile_driver():
    """A driver with straight wet profiles only (enough for the geometry): baffle radius 5 cm at Z_B, front 5.3 cm
    at Z_F, cone foot on the axis 2 cm behind the baffle."""
    rear = [np.array([[0.050, api.Z_B], [0.010, api.Z_B - 0.02]]), np.array([[0.010, api.Z_B - 0.02], [0.0, api.Z_B - 0.02]])]
    front = [np.array([[0.053, api.Z_F], [0.0, api.Z_F - 0.015]])]
    return lp.Driver(list(range(3)), rear, front, coil_nodes=[0], name="profiles")


def hand_box(occ):
    """The cabinet of tests/test_scene.py, written by hand."""
    inner = occ.addBox(0.0, 0.0, api.Z_B - D, W, H, D)
    outer = occ.addBox(0.0, 0.0, api.Z_B - D, W, H, D + (api.Z_F - api.Z_B))
    return lp.Cabinet(inner=inner, outer=outer)


def masses(cabinet):
    with gmsh_session() as g:
        g.model.add("m")
        occ = g.model.occ
        cab = cabinet(occ)
        occ.synchronize()
        return (occ.getMass(3, cab.inner), occ.getMass(3, cab.outer),
                np.array(occ.getBoundingBox(3, cab.outer)).reshape(2, 3))


def test_box_is_the_hand_written_cabinet():
    placed = [profile_driver().place("main", at=(W / 2, H / 2))]
    box = lp.cabinets.Box(W, H, D).for_drivers(placed)
    assert (box.front, box.front_face) == (api.Z_B, api.Z_F)                # the drivers set the front
    a, b = masses(box), masses(hand_box)
    assert a[0] == pytest.approx(b[0], rel=1e-12) and a[1] == pytest.approx(b[1], rel=1e-12)
    np.testing.assert_allclose(a[2], b[2], atol=1e-6)
    assert lp.cabinets.volume(lp.cabinets.Box(W, H, D), placed) == pytest.approx(W * H * D, rel=1e-12)
    np.testing.assert_allclose(box.corners().min(axis=0), [0, 0, api.Z_B - D], atol=1e-12)


def test_box_parts_and_checks():
    placed = [profile_driver().place("main", at=(0.15, 0.12))]
    port = lp.cabinets.Port(at=(0.05, 0.05), radius=0.02, length=0.06, wall=0.003)
    brace = lp.cabinets.Brace(y=0.08, thickness=0.015, depth=0.05)
    box = lp.cabinets.Box(W, H, D, depth_top=0.10, wall=0.015, braces=[brace], ports=[port])
    v = lp.cabinets.volume(box, placed)
    plain = lp.cabinets.volume(lp.cabinets.Box(W, H, D, depth_top=0.10), placed)
    assert plain == pytest.approx(W * H * 0.5 * (D + 0.10), rel=1e-12)       # slanted back
    wedge = W * 0.5 * 0.015 ** 2 * (D - 0.10) / H                          # the brace meets the slanted back
    removed = W * 0.015 * 0.05 + wedge + np.pi * ((0.023 ** 2 - 0.02 ** 2) * (0.06 - (api.Z_F - api.Z_B)) +
                                          0.02 ** 2 * (0.06 - (api.Z_F - api.Z_B)))
    assert v == pytest.approx(plain - removed, rel=1e-9)                     # brace, tube wall, bore
    assert 30 < port.tuning(v) < 300
    assert "1 port(s)" in box.describe() and "walls 0.015 m" in box.describe()
    with pytest.raises(ModelError, match="not on the front face"):
        lp.cabinets.Box(W, H, D, ports=[lp.cabinets.Port(at=(0.01, 0.05), radius=0.02, length=0.05)])
    with pytest.raises(ModelError, match="outside the inner height"):
        lp.cabinets.Box(W, H, D, braces=[lp.cabinets.Brace(y=H - 0.005, thickness=0.015)])
    with pytest.raises(ModelError, match="width must be > 0"):
        lp.cabinets.Box(0.0, H, D)
    with pytest.raises(ModelError, match="axis \\+z"):
        lp.cabinets.Box(W, H, D).for_drivers([profile_driver().place("side", at=(0.1, 0.1, 0.0), axis=(1, 0, 0))])
    with pytest.raises(ModelError, match="no front plane"):
        lp.cabinets.volume(lp.cabinets.Box(W, H, D))


def write_cad(path, box, placed, hole=True, inner_chamber=False):
    """The walls of `box` as a STEP file (metres), with the driver hole and the port bores cut through."""
    box = box.for_drivers(placed)
    with gmsh_session() as g:
        g.model.add("cad")
        occ = g.model.occ
        cab = box(occ)
        air = occ.cut([(3, cab.inner)], [(3, t) for t in cab.cuts])[0] if cab.cuts else [(3, cab.inner)]
        tools = air + [(3, t) for t in cab.openings]
        if hole:
            for d in placed:
                x, y, _ = d.frame.origin
                tools.append((3, occ.addCylinder(x, y, d.driver.z_baffle - 1e-3, 0, 0, 0.01, d.driver.r_rear)))
        walls, _ = occ.cut([(3, cab.outer)], tools)
        if inner_chamber:                                    # a sealed hollow brace: a second closed void
            shell = occ.addBox(0.02, 0.15, api.Z_B - 0.10, 0.06, 0.04, 0.06)
            hollow = occ.addBox(0.025, 0.155, api.Z_B - 0.095, 0.05, 0.03, 0.05)
            sh, _ = occ.cut([(3, shell)], [(3, hollow)])
            walls, _ = occ.fuse(walls, sh)
        occ.synchronize()
        g.option.setString("Geometry.OCCTargetUnit", "M")
        g.write(str(path))
    return path


def test_step_cabinet_finds_the_inside(tmp_path):
    placed = [profile_driver().place("main", at=(0.15, 0.12))]
    port = lp.cabinets.Port(at=(0.05, 0.05), radius=0.02, length=0.06, wall=0.003)
    box = lp.cabinets.Box(W, H, D, depth_top=0.10, wall=0.015, braces=[lp.cabinets.Brace(0.08, 0.015, 0.05)],
                          ports=[port])
    cad = write_cad(tmp_path / "box.step", box, placed)
    bore = [lp.cabinets.Port(port.at, port.radius, port.length, wall=0)]
    step = lp.cabinets.Step(cad, openings=bore)
    assert lp.cabinets.volume(step, placed) == pytest.approx(lp.cabinets.volume(box, placed), rel=1e-9)
    assert lp.cabinets.volume(lp.cabinets.Step(write_cad(tmp_path / "sealed.step", lp.cabinets.Box(W, H, D, wall=0.01),
                                                         placed)), placed) == pytest.approx(W * H * D, rel=1e-9)
    with pytest.raises(MeshError, match="no closed inside air"):              # the port is not named
        lp.cabinets.volume(lp.cabinets.Step(cad), placed)
    with pytest.raises(MeshError, match="no closed inside air"):              # the bottom needs the ground
        lp.cabinets.volume(lp.cabinets.Step(cad, openings=bore, ground=None), placed)
    with pytest.raises(MeshError, match="no closed inside air"):              # the file is 1 cm off the drivers
        lp.cabinets.volume(lp.cabinets.Step(cad, openings=bore, translate=(0, 0, 0.01)), placed)
    with pytest.raises(MeshError, match="the inside air|inside the cabinet"):  # moved back: hole closed by the panel
        lp.cabinets.volume(lp.cabinets.Step(write_cad(tmp_path / "nohole.step", box, placed, hole=False),
                                            openings=bore, translate=(0, 0, 0.0005)), placed)
    two = write_cad(tmp_path / "two.step", lp.cabinets.Box(W, H, D, wall=0.01), placed, inner_chamber=True)
    with pytest.raises(MeshError, match="2 separate inside voids"):
        lp.cabinets.volume(lp.cabinets.Step(two), placed)
    v = lp.cabinets.volume(lp.cabinets.Step(two, inner=(0.2, 0.05, api.Z_B - 0.05)), placed)
    assert v == pytest.approx(W * H * D - 0.06 * 0.04 * 0.06, rel=1e-9)
    with pytest.raises(ModelError, match="not found"):
        lp.cabinets.Step(tmp_path / "missing.step")
    a, b = lp.cabinets.Step(cad, openings=bore).signature(), lp.cabinets.Step(cad, openings=bore, scale=1.0).signature()
    assert a == b and a != lp.cabinets.Step(cad, openings=bore, translate=(0, 0, 1e-3)).signature()


def test_ready_made_scenes_and_parameters():
    hp = profile_driver()
    sealed = lp.Scene.sealed_box(lp.cabinets.Box(W, H, D), hp, at=(0.15, 0.12), f_max=2500)
    assert sealed.configuration == "cavity" and sealed._cavity_spec.f_max == 5000.0
    radiating = lp.Scene.sealed_box(lp.cabinets.Box(W, H, D), hp, at=(0.15, 0.12), f_max=2500, radiating=True)
    assert radiating.configuration == "cavity+exterior"
    assert radiating._open_spec.bands == [(20.0, 1000.0), (1000.0, 2000.0), (2000.0, 2500.0)]
    vented = lp.Scene.vented_box(lp.cabinets.Box(W, H, D, ports=[lp.cabinets.Port((0.05, 0.05), 0.02, 0.06)]), hp,
                                 at=(0.15, 0.12))
    assert vented.configuration == "merged" and "Box 0.26 × 0.2 × 0.15 m inside" in vented.tree()
    with pytest.raises(ModelError, match="at=\\(x, y\\)"):
        lp.Scene.sealed_box(lp.cabinets.Box(W, H, D), hp)
    # a parameter in the box: resolved for meshing, part of the mesh signature
    scene = lp.Scene(lp.cabinets.Box(lp.P("w"), H, D), [hp.place("main", at=(0.15, 0.12))],
                     parameters=lp.Parameters(w=W)).cavity()
    other = scene.with_parameters(w=W + 0.02)
    assert scene._cabinet().width == W and other._cabinet().width == W + 0.02
    assert scene._signature("cavity") != other._signature("cavity")
    assert "0.26 × 0.2" in scene.tree()


def test_default_cameras():
    cams = lp.plot.default_cameras([[0, 0, -0.2], [0.3, 0.4, 0.03]], drivers=[(0.15, 0.2, 0.0)])
    assert len(cams) == 6 and all(isinstance(c, lp.plot.Camera) for c in cams.values())
    left = cams["front left (cut side)"]
    assert left.position[0] < 0.0 and left.position[2] > 0.03 and left.focus == pytest.approx((0.15, 0.2, -0.085))
    assert cams["close-up on the driver (cut side)"].focus == pytest.approx((0.15, 0.2, 0.0))


@pytest.mark.slow
def test_step_scene_like_the_box(tmp_path, mesh_dir, cache_dir):
    """A real driver in a sealed box from a STEP file: same cavity and response as the lp.cabinets.Box (other
    mesh: within the discretisation)."""
    api.driver_modes(mesh_dir, cache_dir)
    hp = lp.Driver.from_modes(cache_dir / "driver_modes.h5", [0, 1, 2], f_max=5000.0, geometry=api.SPEAKER_GEO,
                              rear=api.REAR_PROFILE, front=api.FRONT_PROFILE)
    placed = [hp.place("main", at=(W / 2, H / 2))]
    box = lp.cabinets.Box(W, H, D, wall=0.01)
    cad = write_cad(tmp_path / "box.step", box, placed)
    out = {}
    for name, cab in (("box", box), ("step", lp.cabinets.Step(cad))):
        scene = lp.Scene.sealed_box(cab, hp, at=(W / 2, H / 2), f_max=1500)
        scene.cavity(mesh_size=(0.006, 0.03), f_max=3000.0, eta=0.02, cache_modes=False)
        scene.mesh(tmp_path / name)
        out[name] = lp.FrequencyResponse(scene, drive=lp.Force(1.0)).solve([100.0, 400.0])
    a, b = out["box"], out["step"]
    np.testing.assert_allclose(b.cavity_frequencies[:6], a.cavity_frequencies[:6], rtol=2e-3)
    np.testing.assert_allclose(np.abs(b.driver("main").excursion().values),
                               np.abs(a.driver("main").excursion().values), rtol=2e-3)
