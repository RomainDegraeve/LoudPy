"""
Phase G4: derived values (volume velocity, radiated power, Thiele–Small parameters) vs analytical cases and
independent computations, and ParaView series.
"""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import loudpy as lp
from loudpy.results.results import Block, FrequencyResults
from support import current_api as api

FRONT, REAR = "interface_acou_meca_front", "interface_acou_meca_rear"


@pytest.fixture(scope="module")
def coupled(mesh_dir):
    return lp.FrequencyResponse(api.speaker_problem(mesh_dir, acoustic=True), drive=lp.Force(1.0)).solve([300.0, 1000.0])


def test_volume_velocity_of_a_rigid_translation(coupled):
    """u = (0, 1) everywhere: Q / (jω) = the area swept by each interface, ±π r_max² (curved edges, exact)."""
    b = coupled.blocks[0]
    mesh = b.mesh_of("u")
    U = np.zeros((1, mesh.n_nodes, 2), complex)
    U[..., 1] = 1.0
    rigid = FrequencyResults([Block(b.meshes, np.array([100.0]), {"u": U})], {}, coupled.boundaries)
    for name, sign in [(FRONT, 1.0), (REAR, -1.0)]:
        r_max = mesh.coords_of(np.unique(coupled.boundaries[name]["node_tags"]))[:, 0].max()
        Q = rigid.volume_velocity(name).values[0] / (2j * np.pi * 100.0)
        assert Q == pytest.approx(sign * np.pi * r_max ** 2, rel=1e-12)


def test_radiated_power_two_ways(coupled, mesh_dir, verify):
    """Power into the front air: through the fluid–structure boundary (u and p) = the coupling operator's
    power of the C1 energy check (round-off) = the intensity flux through a sphere in the front air (p only)."""
    on_boundary = coupled.radiated_power(on=FRONT).values
    energy = api.speaker_fsi_energy(mesh_dir, [300.0, 1000.0])
    np.testing.assert_allclose(on_boundary, energy["P_air"][FRONT], rtol=1e-9)
    sphere = coupled.radiated_power(radius=0.15, center=(0.0, 0.032), baffle=0.032).values
    verify("Results", "Radiated power of the speaker, two ways", "flux through a sphere vs through the boundary",
           np.abs(sphere / on_boundary - 1).max(), bound=1e-3, reference="power through the fluid–structure boundary")
    np.testing.assert_allclose(sphere, on_boundary, rtol=1e-3)
    assert (on_boundary > 0).all()


def test_thiele_small_of_a_lumped_driver(verify):
    """A driver that is exactly one moving mass: the parameters are recovered from its impedance."""
    Re, Le, Bl, Mms, Cms, Rms = 5.6, 2e-4, 6.5, 0.012, 2.4e-4, 0.9
    f = np.linspace(20.0, 400.0, 400)
    w = 2 * np.pi * f
    Z = Re + 1j * w * Le + Bl ** 2 / (Rms + 1j * w * Mms + 1 / (1j * w * Cms))
    U = np.full(len(f), 2.83 + 0j)
    res = FrequencyResults([Block({}, f, {}, {"U": U, "I_coil": U / Z})])
    res.study = {"type": "FrequencyResponse", "drive": dict(type="Voltage", amplitude=2.83, Re=Re, Bl=Bl, Le=Le, n=1.0)}
    ts = res.thiele_small()
    ws = 1 / np.sqrt(Mms * Cms)
    exact = dict(fs=ws / (2 * np.pi), Mms=Mms, Cms=Cms, Rms=Rms, Qms=ws * Mms / Rms, Qes=ws * Mms * Re / Bl ** 2)
    verify("Results", "Thiele–Small parameters of a lumped driver", "max relative error (fs, Mms, Cms, Rms, Qms, Qes)",
           max(abs(ts[k] / v - 1) for k, v in exact.items()), bound=1e-9, reference="the lumped model's own values")
    for k, v in exact.items():
        assert ts[k] == pytest.approx(v, rel=1e-9), k


def test_thiele_small_of_the_speaker(mesh_dir):
    res = lp.FrequencyResponse(api.speaker_problem(mesh_dir, acoustic=True),
                               drive=lp.Voltage(2.83, Re=6.0, Bl=7.0, Le=0.0, n=1.0)).solve(np.linspace(60, 220, 65))
    ts = res.thiele_small(on=FRONT)
    Z = np.abs(res.impedance().values)
    assert ts["fs"] == pytest.approx(res.frequencies[np.argmax(Z)], abs=2.5)          # the impedance peak
    assert ts["Re"] + ts["Bl"] ** 2 / ts["Rms"] == pytest.approx(Z.max(), rel=0.05)
    assert 0.007 < ts["Sd"] < 0.0093 and ts["Vas"] > 0                                 # π (5 cm)² … π r_max²


@pytest.mark.slow
def test_thiele_small_of_a_remeshed_sweep(mesh_dir, tmp_path):
    """One mesh per band: every block keeps its own boundaries (saved and read back); files without them (written
    before) find each boundary again on the other meshes from the geometry, edge for edge."""
    rule = lp.PerWavelength(bands=[60.0, 140.0], air_max_size=0.02, folder=tmp_path)
    res = lp.FrequencyResponse(api.speaker_problem(mesh_dir, acoustic=True),
                               drive=lp.Voltage(2.83, Re=6.0, Bl=7.0, Le=0.0, n=1.0)).solve(
        np.linspace(60, 220, 33), remesh=rule)
    assert len(res.blocks) == 2
    assert not np.array_equal(res.blocks[0].meshes["air"].coords.shape, res.blocks[1].meshes["air"].coords.shape)
    ts = res.thiele_small(on=FRONT)
    assert 0.007 < ts["Sd"] < 0.0093 and ts["Vas"] > 0
    assert res.model.groups                                          # physical data from a meshed copy

    back = lp.load(res.save(tmp_path / "remeshed.h5"))
    for a, b in zip(res.blocks, back.blocks):
        for name, bd in a.boundaries.items():
            np.testing.assert_array_equal(b.boundaries[name]["edges"], bd["edges"])
    assert back.thiele_small(on=FRONT) == pytest.approx(ts, rel=1e-12)

    def corners(block, edges):
        mesh = block.mesh_of("u")
        pts = mesh.coords[mesh._rows(np.asarray(edges)[:, :2])].round(12)
        return sorted(map(tuple, np.sort(pts.reshape(len(pts), -1), axis=1).tolist()))

    for b in back.blocks:
        stored = dict(b.boundaries)
        b.boundaries = {}                                            # as written before 0.2.1
        for name in (FRONT, REAR):
            found = back.boundaries_of(b)[name]["edges"]
            assert corners(b, found) == corners(b, stored[name]["edges"]), (b.label, name)
    assert back.thiele_small(on=FRONT) == pytest.approx(ts, rel=1e-12)


def test_paraview_series(coupled, tmp_path):
    pyvista = pytest.importorskip("pyvista")
    path = coupled.export(tmp_path / "series.pvd")
    sets = ET.parse(path).getroot().find("Collection").findall("DataSet")
    assert [(float(d.get("timestep")), d.get("part")) for d in sets] == \
        [(300.0, "0"), (300.0, "1"), (1000.0, "0"), (1000.0, "1")]
    grid = pyvista.read(tmp_path / sets[2].get("file"))
    u = coupled.field("u", f=1000.0)
    np.testing.assert_allclose(grid.point_data["u_z_re"], u.values[:, 1].real, rtol=1e-12)
    air = pyvista.read(tmp_path / sets[3].get("file"))
    np.testing.assert_allclose(air.point_data["p_im"], coupled.field("p", f=1000.0).values.imag, rtol=1e-12)
