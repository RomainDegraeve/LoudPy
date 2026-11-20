"""
Adapter between the regression tests and the loudpy API they exercise.

Every test calls loudpy only through the functions of this module, which return plain numpy
data (frequencies, nodal fields with their node coordinates, probes). When the public API
changes (docs/ARCHITECTURE.md), only this module is rewritten; the tests and their reference
numbers stay the same.

All cases use coarse meshes so that the whole suite runs in a few minutes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import loudpy as lp

SPEAKER_GEO = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
PLATE_GEO = "LoudPy_exemples/Geometries/Plate/clamped_plate.geo"
MATERIALS = "src/loudpy/data/materials.json"
SPEAKER_PARTS = [("membranne", "Paper"), ("coil", "Copper"), ("surround", "Rubber"),
                 ("spider", "PhenolicCloth"), ("former", "Kapton"), ("glue", "SolidGlue"),
                 ("dustcap", "Polypropylene")]
PLATE = dict(E=2.1e11, nu=0.3, rho=7850.0, a=0.1, h=0.002)


# ════════════════════════════════════════════════════════════════════════════
#  Models
# ════════════════════════════════════════════════════════════════════════════
def plate_model():
    """Clamped steel plate (radius 0.1 m, 2 mm), axisymmetric solid, no damping."""
    steel = lp.Material("Steel", rho=PLATE["rho"], E=PLATE["E"], nu=PLATE["nu"])
    return lp.Model(PLATE_GEO).add(
        lp.Solid("plate", steel, damping=lp.Undamped(), mesh_size=PLATE["h"]),
        lp.Clamped("interface_constrained_rz"), lp.Clamped("interface_constrained_r", dofs="r"),
    )


def speaker_model(size=1e-3, acoustic=False, rear=False):
    """HPNEW loudspeaker (structure only, or with the air and a PML for the FSI case)."""
    model = lp.Model(SPEAKER_GEO, MATERIALS)
    model.add([lp.Solid(n, m, mesh_size=size) for n, m in SPEAKER_PARTS],
              lp.Clamped("interface_constrained_rz"), lp.Clamped("interface_constrained_r", dofs="r"), lp.Coil("interface_forced"),
              lp.FluidStructure("interface_acou_meca_front"))
    if acoustic or rear:
        model.add(lp.FluidStructure("interface_acou_meca_rear"))
    if acoustic:
        model.add(lp.Air("subacou", mesh_size=0.01),
                  lp.PML("PML", thickness=0.1, mesh_size=0.025, tuned_to=1000.0, order=2, strength=1.5))
    return model


def speaker_model_by_selectors(size=1e-3):
    """The FSI speaker of speaker_model(acoustic=True) with every boundary given by a selector instead of
    a named curve group of the .geo (interfaces from the mesh, clamps by position)."""
    solids = lp.Group(*[n for n, _ in SPEAKER_PARTS])
    model = lp.Model(SPEAKER_GEO, MATERIALS)
    model.add([lp.Solid(n, m, mesh_size=size) for n, m in SPEAKER_PARTS],
              lp.Air("subacou", mesh_size=0.01),
              lp.PML("PML", thickness=0.1, mesh_size=0.025, tuned_to=1000.0, order=2, strength=1.5),
              lp.Clamped("spider_rim", on=lp.Near((0.04471635, -0.00075))),
              lp.Clamped("surround_rim", on=lp.Box(z=0.032) & lp.BoundaryOf("surround")),
              lp.Clamped("axis", on=lp.BoundaryOf("dustcap") & lp.Box(r=0), dofs="r"),
              lp.Coil("drive", on=lp.Interface("coil", "former")),
              lp.FluidStructure("front", on=lp.Interface(solids, "subacou_front")),
              lp.FluidStructure("rear", on=lp.Interface(solids, "subacou_rear")))
    return model


def speaker_fsi_response(model, mesh_path, freqs):
    """u and p (by node coordinates) of a meshed FSI speaker model, 1 N on the coil."""
    model.mesh(mesh_path)
    b = lp.FrequencyResponse(model, drive=lp.Force(1.0)).solve(list(freqs)).blocks[0]
    return dict(coords_u=b.mesh_of("u").coords, u=b.fields["u"], coords_p=b.mesh_of("p").coords, p=b.fields["p"])


def _speaker_tag(size, acoustic, rear):
    return f"speaker_{size:g}_{'fsi' if acoustic else 'meca'}{'_rear' if rear else ''}"


def plate_problem(mesh_dir: Path):
    """The meshed plate model (its physics parts: model.parts)."""
    model = plate_model()
    model.mesh(mesh_dir / "plate.msh")
    return model


def speaker_problem(mesh_dir: Path, size=1e-3, acoustic=False, rear=False):
    """The meshed speaker model (its physics parts: model.parts)."""
    model = speaker_model(size, acoustic, rear)
    model.mesh(mesh_dir / f"{_speaker_tag(size, acoustic, rear)}.msh")
    return model


# ════════════════════════════════════════════════════════════════════════════
#  Nodal mapping (full DOF vectors → per-node values with coordinates)
# ════════════════════════════════════════════════════════════════════════════
def _coords_of(model, tags):
    mesh = model.meshed
    return mesh.coords[mesh.rows_of(np.asarray(tags))][:, :2]


def nearest_row(coords, x):
    return int(np.argmin(np.linalg.norm(coords - np.asarray(x, float), axis=1)))


# ════════════════════════════════════════════════════════════════════════════
#  Eigen studies
# ════════════════════════════════════════════════════════════════════════════
def _meshed(model, mesh_dir, name):
    model.mesh(mesh_dir / f"{name}.msh")
    return model


def _modes_by_order(res, orders):
    out = {}
    for m in orders:
        rm = res.order(m)
        b = rm.blocks[0]
        out[m] = dict(freqs=rm.frequencies, zeta=rm.zeta, coords=b.mesh_of("u").coords, shapes=b.fields["u"])
    return out


def plate_fourier_modes(mesh_dir: Path, orders, f_max=4000.0):
    """{m: dict(freqs, coords, shapes (n, nodes, dpn))} of the clamped plate (lp.Modes)."""
    model = _meshed(plate_model(), mesh_dir, "plate")
    res = lp.Modes(model).solve(f_max=f_max, f_min=10.0, orders=list(orders), n_per_shift=10)
    return _modes_by_order(res, orders)


def speaker_modes(mesh_dir: Path, f_max=3000.0, orders=(0,)):
    """In-vacuo modes of the coarse loudspeaker, azimuthal orders `orders` (lp.Modes)."""
    model = _meshed(speaker_model(), mesh_dir, _speaker_tag(1e-3, False, False))
    study = lp.Modes(model)
    res = study.solve(f_max=f_max, f_min=20.0, orders=list(orders), n_per_shift=20)
    return _modes_by_order(res, orders), study


# ════════════════════════════════════════════════════════════════════════════
#  Frequency studies
# ════════════════════════════════════════════════════════════════════════════
def speaker_freq_meca(mesh_dir: Path, freqs, force=1.0):
    """In-vacuo forced response, force on the coil (lp.FrequencyResponse): coords, u (n_f, nodes, 2)."""
    model = _meshed(speaker_model(), mesh_dir, _speaker_tag(1e-3, False, False))
    study = lp.FrequencyResponse(model, drive=lp.Force(force))
    res = study.solve(freqs)
    b = res.blocks[0]
    return dict(coords=b.mesh_of("u").coords, u=b.fields["u"]), study


def speaker_fsi(mesh_dir: Path, freq, voltage=None):
    """
    Coupled structure + air + PML at one frequency (lp.FrequencyResponse). voltage None: 1 N on
    the coil; else dict(Re, Le, n, Bl, u): voice-coil circuit. Nodal u, p and the coil current.
    """
    model = _meshed(speaker_model(acoustic=True), mesh_dir, _speaker_tag(1e-3, True, False))
    drive = lp.Force(1.0) if voltage is None else lp.Voltage(voltage["u"], Re=voltage["Re"], Bl=voltage["Bl"],
                                                             Le=voltage["Le"], n=voltage["n"])
    res = lp.FrequencyResponse(model, drive=drive).solve([freq])
    b = res.blocks[0]
    I = res.extra("I_coil")[0] if "I_coil" in b.extra else 0j
    return dict(coords_u=b.mesh_of("u").coords, u=b.fields["u"][0], coords_p=b.mesh_of("p").coords,
                p=b.fields["p"][0][:, None], I_coil=np.array(I))


def speaker_fsi_energy(mesh_dir: Path, freqs):
    """
    Physical checks of the structure–air coupling of the speaker (front and rear air, PML), per
    fluid–structure boundary: the volume swept by a unit rigid axial translation, and at each
    frequency the power given by the coil, dissipated in the structure and radiated into each air
    region (P = ½ Re ∫ p* iω u·n_s dA, n_s the outward normal of the structure), and the pressure on
    the axis just in front of and behind the dust cap.
    """
    from loudpy.physics.acoustic import AcouAssembler
    from loudpy.physics.coupling import FsiAssembler
    from loudpy.physics.forcing import ForceAssembler
    from loudpy.physics.structural import MecaAssembler
    model = _meshed(speaker_model(acoustic=True), mesh_dir, _speaker_tag(1e-3, True, False))
    meca, acou = MecaAssembler(model.parts), AcouAssembler(model.parts)
    K, M, C = meca.assemble()
    F = ForceAssembler(model.parts, meca).assemble()
    mesh = model.meshed
    key = lambda c: tuple(np.round(np.asarray(c)[:2], 10))
    tag_of = {key(mesh.coords[mesh.rows_of(np.array([t]))[0]]): t
              for t in np.concatenate([meca.unique_tags, acou.unique_tags])}
    couplings = {p.name: FsiAssembler([p], meca, acou).assemble() for p in model.parts if p.role == "fluid_structure"}
    r_max = {p.name: float(np.abs(p.node_coords[:, 0]).max()) for p in model.parts if p.role == "fluid_structure"}
    uz = np.zeros(meca.matrix_size)
    uz[1::2] = 1.0
    out = dict(swept={n: -(Cn @ uz).sum() for n, Cn in couplings.items()}, r_max=r_max, freqs=np.asarray(freqs),
               P_in=[], P_diss=[], P_air={n: [] for n in couplings}, p_front=[], p_rear=[])
    for f in freqs:
        b = lp.FrequencyResponse(model, drive=lp.Force(1.0)).solve([f]).blocks[0]
        u = np.zeros(meca.matrix_size, complex)
        p = np.zeros(acou.matrix_size, complex)
        for c, val in zip(b.mesh_of("u").coords, b.fields["u"][0]):
            i = np.searchsorted(meca.unique_tags, tag_of[key(c)])
            u[2 * i:2 * i + 2] = val
        for c, val in zip(b.mesh_of("p").coords, b.fields["p"][0]):
            p[np.searchsorted(acou.unique_tags, tag_of[key(c)])] = val
        w = 2 * np.pi * f
        out["P_in"].append(0.5 * np.real(np.conj(F) @ (1j * w * u)))
        out["P_diss"].append(0.5 * w ** 2 * np.real(np.conj(u) @ (C @ u)))
        for n, Cn in couplings.items():
            out["P_air"][n].append(-0.5 * np.real(np.conj(p) @ (1j * w * (Cn @ u))))
        xy, pa = b.mesh_of("p").coords, b.fields["p"][0]
        axis = np.abs(xy[:, 0]) < 1e-9
        out["p_front"].append(pa[axis][np.argmin(np.abs(xy[axis, 1] - 0.033))])
        out["p_rear"].append(pa[axis][np.argmin(np.abs(xy[axis, 1] + 0.008))])
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Time study
# ════════════════════════════════════════════════════════════════════════════
def speaker_transient(mesh_dir: Path, n_steps=200, fs=48_000.0, f0=200.0, amplitude=1.0):
    """Nonlinear transient, sine force on the coil (lp.Transient): time, coords, U (n_t, nodes, 2)."""
    model = _meshed(speaker_model(), mesh_dir, _speaker_tag(1e-3, False, False))
    sig = amplitude * np.sin(2 * np.pi * f0 * np.arange(n_steps + 1) / fs)
    study = lp.Transient(model, drive=lp.Force(sig, fs=fs))
    res = study.solve(n_steps=n_steps)
    b = res.blocks[0]
    return dict(time=res.times, coords=b.mesh_of("u").coords, U=np.real(b.fields["u"])), study


# ════════════════════════════════════════════════════════════════════════════
#  3D coupling (physics.modal, physics.exterior)
# ════════════════════════════════════════════════════════════════════════════
Z_B, Z_F = 0.031, 0.032                                 # rear baffle plane, front face [m]
REAR_CURVES, REAR_START = [14, 7, 8, 33, 32], (0.05435, Z_B)
FRONT_CURVES, FRONT_START = [5, 4, 3, 2, 1, 113, 22, 21, 20], (0.05735, Z_F)
COIL_CURVE = 37
BOX = (0.26, 0.20, 0.15)                                # inner box W × H × D [m] (front at z = Z_B)


def rect_cavity_modes(mesh_dir: Path, size=(0.30, 0.20, 0.14), h=0.03, f_max=2000.0):
    """Rigid rectangular box: CavityModes frequencies."""
    from loudpy.physics.modal import CavityModes
    from loudpy.geometry.gmsh_session import gmsh_module as _gmsh
    msh = mesh_dir / "rect_box.msh"
    if not msh.exists():
        g = _gmsh()
        g.clear()
        v = g.model.occ.addBox(0, 0, 0, *size)
        g.model.occ.synchronize()
        faces = [t for _, t in g.model.getBoundary([(3, v)], oriented=False)]
        g.model.addPhysicalGroup(3, [v], 1, name="cavity")
        g.model.addPhysicalGroup(2, faces[:1], 2, name="interface_driver")
        for k, val in (("Mesh.MeshSizeMax", h), ("Mesh.MeshSizeMin", h), ("Mesh.ElementOrder", 2),
                       ("Mesh.SecondOrderLinear", 1)):
            g.option.setNumber(k, val)
        g.model.mesh.generate(3)
        g.write(str(msh))
        g.clear()
    cav = CavityModes.from_msh(msh)
    return cav.solve(f_max), cav


def hemisphere_radiation(mesh_dir: Path, freqs, a=0.10, R=0.25, h=0.04):
    """
    Pulsating hemisphere (radius a) on a rigid ground, exterior FE + half-space DtN (radius R):
    FE pressure at 7 points at 1 m and the exact one, per frequency.
    """
    from loudpy.geometry.gmsh_session import gmsh_module as _gmsh
    from loudpy.fem.tet10 import surface_quadrature
    from loudpy.model.materials import RHO_AIR, C_AIR
    from loudpy.physics.exterior import HalfSpaceDtN, ExteriorFE, _h2
    msh = mesh_dir / "dtn_shell.msh"
    if not msh.exists():
        g = _gmsh()
        g.clear()
        occ = g.model.occ
        hemi, _ = occ.intersect([(3, occ.addSphere(0, 0, 0, R))], [(3, occ.addBox(-1, 0, -1, 2, 1, 2))])
        shell, _ = occ.cut(hemi, [(3, occ.addSphere(0, 0, 0, a))])
        occ.synchronize()
        vol = shell[0][1]
        groups = {"dtn": [], "inner": [], "ground": []}
        for _, t in g.model.getBoundary([(3, vol)], oriented=False):
            b = g.model.getBoundingBox(2, t)
            if g.model.getType(2, t) == "Sphere":
                groups["dtn" if max(abs(v) for v in b) > 0.5 * (a + R) else "inner"].append(t)
            else:
                groups["ground"].append(t)
        g.model.addPhysicalGroup(3, [vol], 1, name="air")
        for k, (n, ts) in enumerate(groups.items(), start=2):
            g.model.addPhysicalGroup(2, ts, k, name=n)
        for k, v in (("Mesh.MeshSizeMax", h), ("Mesh.MeshSizeMin", h * 0.5), ("Mesh.ElementOrder", 2),
                     ("Mesh.SecondOrderLinear", 1)):
            g.option.setNumber(k, v)
        g.model.mesh.generate(3)
        g.write(str(msh))
        g.clear()
    dtn = HalfSpaceDtN((0, 0, 0), R, up=1)
    ext = ExteriorFE(msh, dtn, max(freqs), front="inner")
    xq, wq, inodes, W = surface_quadrature(ext.nodes, ext.faces["inner"])
    far = np.array([[np.cos(t), np.sin(t) * 0.6, np.sin(t) * 0.8] for t in np.linspace(0, np.pi, 7)])
    fe, exact = [], []
    for f in freqs:
        w, k = 2 * np.pi * f, 2 * np.pi * f / C_AIR
        rhs = np.zeros(len(ext.nodes), complex)
        rhs[inodes] = -RHO_AIR * w ** 2 * (W.T @ wq)
        p, _, info = ext.solve(k, rhs, rtol=1e-12)
        assert info == 0
        A = RHO_AIR * w ** 2 / (k * _h2(0, k * a, True))
        fe.append(dtn.pressure(far, ext.coefficients(p, k), k))
        exact.append(A * _h2(0, k * 1.0) * np.ones(len(far)))
    return np.array(fe), np.array(exact)


def driver_modes(mesh_dir: Path, cache_dir: Path, orders=(0, 1, 2), f_max=5000.0):
    """Fourier modes of the coarse loudspeaker, saved and read back as DriverModes."""
    from loudpy.physics.modal import DriverModes
    h5 = cache_dir / "driver_modes.h5"
    study = lp.Modes(speaker_problem(mesh_dir))
    study.solve(f_max=f_max, f_min=20.0, orders=list(orders), n_per_shift=20)
    study.save(h5)
    return DriverModes.from_eigen_h5(h5, list(orders), f_max=f_max)


def driver_profiles():
    from loudpy.geometry.meshing3d import sample_profile
    rear = sample_profile(SPEAKER_GEO, REAR_CURVES, REAR_START)
    front = sample_profile(SPEAKER_GEO, FRONT_CURVES, FRONT_START)
    return rear, front


def coil_nodes(drv):
    from loudpy.geometry.gmsh_session import gmsh_module as _gmsh
    g = _gmsh()
    g.clear()
    g.open(SPEAKER_GEO)
    a, b = g.model.getParametrizationBounds(1, COIL_CURVE)
    ends = np.array(g.model.getValue(1, COIL_CURVE, [a[0], b[0]])).reshape(2, 3)[:, :2]
    g.clear()
    t = np.clip(((drv.xy - ends[0]) @ (ends[1] - ends[0])) / np.sum((ends[1] - ends[0]) ** 2), 0, 1)
    d = np.linalg.norm(drv.xy - (ends[0] + t[:, None] * (ends[1] - ends[0])), axis=1)
    return np.where(d < 1e-6)[0]


def box_cavity(mesh_dir: Path, centers, name, f_max=3000.0, h_near=0.006, h_far=0.03):
    """
    Box W × H × D behind the baffle z = Z_B with drivers cut out at `centers` (2D centers or
    DriverFrames); one wet patch per driver when there are several. Returns solved CavityModes.
    """
    from loudpy.geometry.gmsh_session import gmsh_module as _gmsh
    from loudpy.geometry.meshing3d import add_driver_cutout, carve_drivers, mesh_cavity
    from loudpy.physics.modal import CavityModes
    msh = mesh_dir / f"{name}.msh"
    if not msh.exists():
        rear, _ = driver_profiles()
        rear = rear + [np.array([rear[-1][-1], [0.0, rear[-1][-1][1]]])]        # lid at the cone foot
        g = _gmsh()
        g.clear()
        occ = g.model.occ
        W, H, D = BOX
        box = occ.addBox(0.0, 0.0, Z_B - D, W, H, D)
        cuts = [add_driver_cutout(rear, center=c) for c in centers]
        vols = carve_drivers(box, cuts)
        assert len(vols) == 1, vols
        mesh_cavity(msh, vols[0], centers[0] if len(centers) == 1 else list(centers),
                    cuts[0][2], cuts[0][3], h_near, h_far)
        g.clear()
    cav = CavityModes.from_msh(msh)
    cav.solve(f_max)
    return cav


def coupled_box(cav, drivers_or_drv, center=None, eta_air=0.02):
    """DriverCavityCoupling for one DriverModes (with its center) or a MultiDriver."""
    from loudpy.physics.modal import DriverCavityCoupling, MultiDriver
    rear, _ = driver_profiles()
    if isinstance(drivers_or_drv, MultiDriver):
        return DriverCavityCoupling(cav, drivers_or_drv, [rear] * drivers_or_drv.n_drivers, eta_air=eta_air)
    return DriverCavityCoupling(cav, drivers_or_drv, rear, center=center, eta_air=eta_air)


# ════════════════════════════════════════════════════════════════════════════
#  Assembled operators (phase 2 gate: matrices unchanged by the fem/physics refactor)
# ════════════════════════════════════════════════════════════════════════════
def _probes(coords, dpn):
    """Three smooth nodal fields built from the coordinates (n_nodes, dpn) → flat DOF vectors."""
    r, z = coords[:, 0], coords[:, 1]
    comp = np.arange(dpn)[None, :]
    fields = [1.0 + 3 * r[:, None] + 2 * z[:, None] + 0.5 * comp,
              np.sin(40 * r[:, None] + comp) * np.cos(30 * z[:, None]),
              r[:, None] ** 2 - z[:, None] + 0.3 * comp]
    return [f.ravel() for f in fields]


def operator_fingerprint(problem, A, row_map, col_map):
    """
    Order-independent fingerprint of a (sparse) operator: the 3 × 3 forms yᵢᵀ A xⱼ with probe
    vectors from the coordinates, the product A x₁ per row node, nnz and Σ|A|.
    row_map / col_map: (unique node tags, dofs per node).
    """
    rc, cc = _coords_of(problem, row_map[0]), _coords_of(problem, col_map[0])
    ys, xs = _probes(rc, row_map[1]), _probes(cc, col_map[1])
    forms = np.array([[y @ (A @ x) for x in xs] for y in ys])
    Ax = np.asarray(A @ xs[0]).reshape(len(row_map[0]), row_map[1])
    return dict(forms=forms, Ax=Ax, coords=rc, nnz=np.array(A.nnz), abs_sum=np.array(abs(A).sum()))


def assembled_operators(mesh_dir: Path):
    """All the operators the studies assemble, on the committed meshes, as fingerprints."""
    from loudpy.physics.acoustic import AcouAssembler
    from loudpy.physics.coupling import FsiAssembler
    from loudpy.physics.forcing import ForceAssembler
    from loudpy.physics.structural import MecaAssembler, MecaAssemblerFourier, MecaAssemblerNL
    import scipy.sparse as sp
    out = {}
    p = speaker_problem(mesh_dir, acoustic=True)
    fo = p.parts
    meca, acou = MecaAssembler(fo), AcouAssembler(fo)
    mm, am = (meca.unique_tags, meca.dofs_per_node), (acou.unique_tags, acou.dofs_per_node)
    K, M, C = meca.assemble()
    F = ForceAssembler(fo, meca).assemble()
    H, Q, rho = acou.assemble()
    Cam = FsiAssembler(fo, meca, acou).assemble()
    for name, A, rmap, cmap in (("K", K, mm, mm), ("M", M, mm, mm), ("C", C, mm, mm),
                                ("H", H, am, am), ("Q", Q, am, am), ("Cam", Cam, am, mm),
                                ("F", sp.csr_matrix(F.reshape(-1, 1)), mm, (np.array([0]), 1))):
        out[name] = operator_fingerprint(p, A, rmap, cmap) if name != "F" else \
            dict(forms=np.array([[y @ F for y in _probes(_coords_of(p, mm[0]), 2)]]),
                 Ax=F.reshape(-1, 2), coords=_coords_of(p, mm[0]), nnz=np.array(np.count_nonzero(F)),
                 abs_sum=np.array(np.abs(F).sum()))
    out["rho"] = dict(value=np.array(rho))
    Kr, Mr, Cr, Fr = meca.apply_blocked_boundaries(fo, K, M, C, F)
    out["free"] = dict(value=np.array([len(meca._free), Kr.shape[0], abs(Kr).sum()]))

    four = MecaAssemblerFourier(fo, 1)
    fm = (four.unique_tags, four.dofs_per_node)
    K1, M1, C1 = four.assemble()
    for name, A in (("K_m1", K1), ("M_m1", M1), ("C_m1", C1)):
        out[name] = operator_fingerprint(p, A, fm, fm)
    T = four.constraint_matrix(fo)
    out["T_m1"] = dict(value=np.array([T.shape[0], T.shape[1], T.nnz, T.sum()]))

    nl = MecaAssemblerNL(fo)
    u = 1e-4 * _probes(_coords_of(p, nl.unique_tags), 2)[1]
    KT, Mn, Cn, Fint = nl.assemble_tangent(u)
    for name, A in (("KT", KT), ("M_nl", Mn), ("C_nl", Cn)):
        out[name] = operator_fingerprint(p, A, (nl.unique_tags, 2), (nl.unique_tags, 2))
    out["Fint"] = dict(Ax=Fint.reshape(-1, 2), coords=_coords_of(p, nl.unique_tags))
    return out


DTN_GEO = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch_front_rear_dtn.geo"


def dtn_system(mesh_dir: Path, f_max=3000.0, n_max=8):
    """
    Coupled structure + air + exact DtN radiation (front and rear hemispheres) of the coarse
    speaker: the symmetric system (Ks, Ms, Cs) of EigenStudy, as order-independent invariants.
    """
    model = lp.Model(DTN_GEO, MATERIALS)
    model.add([lp.Solid(n, m, mesh_size=1e-3) for n, m in SPEAKER_PARTS],
              lp.Air("subacou", mesh_size=0.02),
              lp.Clamped("interface_constrained_rz"), lp.Clamped("interface_constrained_r", dofs="r"), lp.Coil("interface_forced"),
              lp.FluidStructure("interface_acou_meca_front"), lp.FluidStructure("interface_acou_meca_rear"),
              lp.Radiation("dtn_boundary"))
    model.mesh(mesh_dir / "speaker_dtn.msh")
    from loudpy.physics.systems import assemble, dtn_system
    centers = {"dtn_boundary_front": (0.0, 0.032), "dtn_boundary_rear": (0.0, 0.031)}
    Ks, Ms, Cs, info = dtn_system(assemble(model.parts, air=True), model.parts, centers, f_max, half=True,
                                  n_max=n_max)
    nm, na = info["nm"], info["na"]

    def invariants(A):
        A = A.tocsr()
        return np.concatenate([[A.shape[0], A.nnz, abs(A).sum(), np.sqrt(abs(A.multiply(A.conj())).sum())],
                               np.sort(A.diagonal().real), np.sort(np.asarray(abs(A).sum(axis=1)).ravel())])
    return dict(Ks=invariants(Ks), Ms=invariants(Ms), Cs=invariants(Cs), sizes=np.array([nm, na]))


# ── the same files through lp.load (results layer, phase 4) ────────────────────
def load_freq_file(path):
    """[(f, coords, u (nodes, 2))] through lp.load."""
    r = lp.load(path)
    return [(f, r.field("u", index=i).mesh.coords, r.field("u", index=i).values)
            for i, f in enumerate(r.frequencies)]


def load_eigen_file(path):
    """{order m: (freqs, coords, shapes (n, nodes, dpn))} through lp.load."""
    r = lp.load(path)
    out = {}
    for m in r.orders:
        rm = r.order(m)
        b = rm.blocks[0]
        out[m] = (rm.frequencies, b.mesh_of("u").coords, b.fields["u"])
    return out


def load_time_file(path):
    """(time, coords, U (n_t, nodes, 2)) through lp.load."""
    r = lp.load(path)
    b = r.blocks[0]
    return r.times, b.mesh_of("u").coords, np.real(b.fields["u"])
