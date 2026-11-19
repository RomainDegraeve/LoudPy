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
from loudpy import (DomainSpecMecaRayleigh, DomainSpecAcou, DomainSpecPML, InterfaceSpecClamped,
                    InterfaceSpecForced, InterfaceSpecAcouMeca)
from loudpy.Studies import Problem, EigenStudy, FreqStudy, TimeStudy

SPEAKER_GEO = "LoudPy_exemples/Geometries/Loudspeaker/HPNEW-Sketch.geo"
PLATE_GEO = "LoudPy_exemples/Geometries/Plate/clamped_plate.geo"
MATERIALS = "src/loudpy/Materials_Bank/materials.json"
SPEAKER_PARTS = [("membranne", "Paper"), ("coil", "Copper"), ("surround", "Rubber"),
                 ("spider", "PhenolicCloth"), ("former", "Kapton"), ("glue", "SolidGlue"),
                 ("dustcap", "Polypropylene")]
PLATE = dict(E=2.1e11, nu=0.3, rho=7850.0, a=0.1, h=0.002)


# ════════════════════════════════════════════════════════════════════════════
#  Models (new API: lp.Model; the studies still take the equivalent Problem)
# ════════════════════════════════════════════════════════════════════════════
def plate_model():
    """Clamped steel plate (radius 0.1 m, 2 mm), axisymmetric solid, no damping."""
    steel = lp.Material("Steel", rho=PLATE["rho"], E=PLATE["E"], nu=PLATE["nu"])
    return lp.Model(PLATE_GEO).add(
        lp.Solid("plate", steel, damping=lp.Undamped(), mesh_size=PLATE["h"]),
        lp.Clamped("interface_constrained"),
    )


def speaker_model(size=1e-3, acoustic=False, rear=False):
    """HPNEW loudspeaker (structure only, or with the air and a PML for the FSI case)."""
    model = lp.Model(SPEAKER_GEO, MATERIALS)
    model.add([lp.Solid(n, m, mesh_size=size) for n, m in SPEAKER_PARTS],
              lp.Clamped("interface_constrained"), lp.Coil("interface_forced"),
              lp.FluidStructure("interface_acou_meca_front"))
    if acoustic or rear:
        model.add(lp.FluidStructure("interface_acou_meca_rear"))
    if acoustic:
        model.add(lp.Air("subacou", mesh_size=0.01),
                  lp.PML("PML", thickness=0.1, mesh_size=0.025, tuned_to=1000.0, order=2, strength=1.5))
    return model


def _speaker_tag(size, acoustic, rear):
    return f"speaker_{size:g}_{'fsi' if acoustic else 'meca'}{'_rear' if rear else ''}"


def plate_problem(mesh_dir: Path):
    model = plate_model()
    model.mesh(mesh_dir / "plate.msh")
    return model.as_problem()


def speaker_problem(mesh_dir: Path, size=1e-3, acoustic=False, rear=False):
    model = speaker_model(size, acoustic, rear)
    model.mesh(mesh_dir / f"{_speaker_tag(size, acoustic, rear)}.msh")
    return model.as_problem()


# ── former API (kept to check that Model builds the same objects) ──────────────
def legacy_plate_problem(mesh_dir: Path):
    """Clamped steel plate (radius 0.1 m, 2 mm), axisymmetric solid, no damping."""
    p = Problem(geo_path=PLATE_GEO, msh_path=str(mesh_dir / "plate.msh"))
    p.add_sub_domain(DomainSpecMecaRayleigh("plate", material="Steel", E=PLATE["E"], nu=PLATE["nu"],
                                            rho=PLATE["rho"], alpha_ray=0.0, beta_ray=0.0,
                                            size=PLATE["h"]))
    p.add_interface(InterfaceSpecClamped("interface_constrained"))
    p.mesh(show_mesh_gui=False, reuse_mesh=True)
    return p


def legacy_speaker_problem(mesh_dir: Path, size=1e-3, acoustic=False, rear=False):
    """HPNEW loudspeaker through Problem + DomainSpec* (former API)."""
    tag = _speaker_tag(size, acoustic, rear)
    p = Problem(geo_path=SPEAKER_GEO, msh_path=str(mesh_dir / f"{tag}.msh"), mat_path=MATERIALS)
    p.add_sub_domain(*[DomainSpecMecaRayleigh(n, material=m, size=size) for n, m in SPEAKER_PARTS])
    p.add_interface(InterfaceSpecClamped("interface_constrained"), InterfaceSpecForced("interface_forced"),
                    InterfaceSpecAcouMeca("interface_acou_meca_front"))
    if acoustic or rear:
        p.add_interface(InterfaceSpecAcouMeca("interface_acou_meca_rear"))
    if acoustic:
        p.add_sub_domain(DomainSpecAcou("subacou", "Air", size=0.01))
        pml = DomainSpecPML("PML", "Air")
        pml.n, pml.alpha, pml.t, pml.size, pml.f_pml = 2, 1.5, 0.1, 0.025, 1000.0
        p.add_sub_domain(pml)
    p.mesh(show_mesh_gui=False, reuse_mesh=True)
    return p


# ════════════════════════════════════════════════════════════════════════════
#  Nodal mapping (full DOF vectors → per-node values with coordinates)
# ════════════════════════════════════════════════════════════════════════════
def _coords_of(problem, tags):
    pos = {int(t): i for i, t in enumerate(problem.nodes.node_tags)}
    xyz = np.asarray(problem.nodes.node_coords)
    return xyz[[pos[int(t)] for t in tags], :2]


def nodal(problem, dof_map, vec):
    """Full DOF vector(s) (..., n_nodes·dpn) → (coords (n, 2), values (..., n, dpn))."""
    tags, dpn = dof_map["unique_tags"], int(dof_map["dofs_per_node"])
    vec = np.asarray(vec)
    return _coords_of(problem, tags), vec.reshape(vec.shape[:-1] + (len(tags), dpn))


def nearest_row(coords, x):
    return int(np.argmin(np.linalg.norm(coords - np.asarray(x, float), axis=1)))


# ════════════════════════════════════════════════════════════════════════════
#  Eigen studies
# ════════════════════════════════════════════════════════════════════════════
def plate_fourier_modes(mesh_dir: Path, orders, f_max=4000.0):
    """{m: dict(freqs, coords, shapes (n, nodes, dpn))} of the clamped plate."""
    p = plate_problem(mesh_dir)
    study = EigenStudy(p)
    out = {}
    for m in orders:
        f, _, modes = study.solve_meca_eigen_fourier(m, f_min=10.0, f_max=f_max, n_modes_per_shift=10)
        key = "u_meca" if m == 0 else f"u_meca_m{m}"
        coords, shapes = nodal(p, study.dof_maps()[key], np.asarray(modes))
        out[m] = dict(freqs=np.asarray(f), coords=coords, shapes=shapes)
    return out


def speaker_modes(mesh_dir: Path, f_max=3000.0, orders=(0,)):
    """In-vacuo modes of the coarse loudspeaker: multishift (m = 0) and Fourier orders m ≥ 1."""
    p = speaker_problem(mesh_dir)
    study = EigenStudy(p)
    out = {}
    for m in orders:
        if m == 0:
            f, z, modes = study.solve_meca_eigen_multishift(f_min=20.0, f_max=f_max, n_modes_per_shift=20)
            key = "u_meca"
        else:
            f, z, modes = study.solve_meca_eigen_fourier(m, f_min=20.0, f_max=f_max, n_modes_per_shift=20)
            key = f"u_meca_m{m}"
        coords, shapes = nodal(p, study.dof_maps()[key], np.asarray(modes))
        out[m] = dict(freqs=np.asarray(f), zeta=np.asarray(z), coords=coords, shapes=shapes)
    return out, study


# ════════════════════════════════════════════════════════════════════════════
#  Frequency studies
# ════════════════════════════════════════════════════════════════════════════
def speaker_freq_meca(mesh_dir: Path, freqs, force=1.0):
    """In-vacuo forced response (unit force on the coil): coords, u (n_f, nodes, 2)."""
    p = speaker_problem(mesh_dir)
    study = FreqStudy(p)
    study.assemble_meca()
    u = [study.solve_meca(freq=f, force=force) for f in freqs]
    coords, U = nodal(p, study.dof_maps()["u_meca"], np.array(u))
    return dict(coords=coords, u=U), study


def speaker_fsi(mesh_dir: Path, freq, voltage=None):
    """
    Coupled structure + air + PML at one frequency. voltage None: unit force on the coil;
    else dict(Re, Le, n, Bl, u): voice-coil circuit. Returns nodal u, p and the coil current.
    """
    p = speaker_problem(mesh_dir, acoustic=True)
    study = FreqStudy(p)
    study.assemble_domains()
    if voltage is None:
        u, pa, I = study.solve_fsi(freq=freq, force=True)
    else:
        u, pa, I = study.solve_fsi(freq=freq, **voltage)
    dm = study.dof_maps()
    cu, U = nodal(p, dm["u_meca"], u)
    cp, P = nodal(p, dm["p_acou"], pa)
    return dict(coords_u=cu, u=U, coords_p=cp, p=P, I_coil=np.array(0j if I is None else I))


# ════════════════════════════════════════════════════════════════════════════
#  Time study
# ════════════════════════════════════════════════════════════════════════════
def speaker_transient(mesh_dir: Path, n_steps=200, fs=48_000.0, f0=200.0, amplitude=1.0):
    """Nonlinear (total Lagrangian) transient, sine force on the coil: time, coords, U (n_t, nodes, 2)."""
    p = speaker_problem(mesh_dir)
    study = TimeStudy(p)
    study.assemble_init()
    sig = np.sin(2 * np.pi * f0 * np.arange(n_steps + 1) / fs)
    t, U, _, _ = study.solve_time_domain_rayleigh_nl(force_amplitude=amplitude, force_direction="_z_",
                                                     force_signal=sig, dt=1.0 / fs, n_steps=n_steps)
    coords, Un = nodal(p, study.dof_maps()["u_meca"], np.asarray(U))
    return dict(time=np.asarray(t), coords=coords, U=Un), study


# ════════════════════════════════════════════════════════════════════════════
#  Files: save with the study, read back with the public readers
# ════════════════════════════════════════════════════════════════════════════
def read_freq_file(path):
    """[(f, coords, u (nodes, 2))] of a frequency-study file, through the public reader."""
    from loudpy.Files_Loader import FreqReader, Domain
    out = []
    with FreqReader(str(path)) as r:
        for s in r.snapshots():
            mesh = r.mesh(s.mesh_id, Domain.MECA)
            out.append((s.f, mesh.coords[:, :2], s.complex_nodal("u_meca", mesh, dpn=2)))
    return out


def read_eigen_file(path):
    """{order m: (freqs, coords, shapes (n, nodes, dpn))} of an eigen-study file."""
    from loudpy.Files_Loader import EigenReader, Domain
    out = {}
    with EigenReader(str(path)) as r:
        for m in r.fourier_orders():
            snaps, mesh, dpn = r.load(Domain.MECA) if m == 0 else r.load_fourier(m)
            field = next(iter(snaps[0].fields))
            shapes = np.array([s.complex_nodal(field, mesh, dpn=dpn) for s in snaps])
            out[m] = (np.array([s.freq for s in snaps]), mesh.coords[:, :2], shapes)
    return out


def read_time_file(path):
    """(time, coords, U (n_t, nodes, 2)) of a time-study file."""
    from loudpy.Files_Loader import TimeReader, Domain
    with TimeReader(str(path)) as r:
        kind = r.kinds()[0]
        run = r.load_run(kind, 0)
        mesh = r.mesh(run.mesh_id, Domain.MECA)
        U = np.array([run.snapshot_at(i).complex_nodal("u_meca", mesh, dpn=2).real
                      for i in range(len(run.time))])
        return run.time, mesh.coords[:, :2], U


# ════════════════════════════════════════════════════════════════════════════
#  3D coupling (Cavity_coupling, Exterior_dtn)
# ════════════════════════════════════════════════════════════════════════════
Z_B, Z_F = 0.031, 0.032                                 # rear baffle plane, front face [m]
REAR_CURVES, REAR_START = [14, 7, 8, 33, 32], (0.05435, Z_B)
FRONT_CURVES, FRONT_START = [5, 4, 3, 2, 1, 113, 22, 21, 20], (0.05735, Z_F)
COIL_CURVE = 37
BOX = (0.26, 0.20, 0.15)                                # inner box W × H × D [m] (front at z = Z_B)


def rect_cavity_modes(mesh_dir: Path, size=(0.30, 0.20, 0.14), h=0.03, f_max=2000.0):
    """Rigid rectangular box: CavityModes frequencies."""
    import gmsh
    from loudpy.Studies.Cavity_coupling import CavityModes, _gmsh
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
    from loudpy.Studies.Cavity_coupling import _gmsh, surface_quadrature, RHO_AIR, C_AIR
    from loudpy.Studies.Exterior_dtn import HalfSpaceDtN, ExteriorFE, _h2
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
    from loudpy.Studies.Cavity_coupling import DriverModes
    h5 = cache_dir / "driver_modes.h5"
    p = speaker_problem(mesh_dir)
    study = EigenStudy(p)
    for m in orders:
        study.solve_meca_eigen_fourier(m, f_min=20.0, f_max=f_max, n_modes_per_shift=20)
    study.save(h5)
    return DriverModes.from_eigen_h5(h5, list(orders), f_max=f_max)


def driver_profiles():
    from loudpy.Studies.Cavity_coupling import sample_profile
    rear = sample_profile(SPEAKER_GEO, REAR_CURVES, REAR_START)
    front = sample_profile(SPEAKER_GEO, FRONT_CURVES, FRONT_START)
    return rear, front


def coil_nodes(drv):
    from loudpy.Studies.Cavity_coupling import _gmsh
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
    from loudpy.Studies.Cavity_coupling import (_gmsh, add_driver_cutout, carve_drivers, mesh_cavity,
                                                CavityModes)
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
    from loudpy.Studies.Cavity_coupling import DriverCavityCoupling, MultiDriver
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
    from loudpy.assembly import MecaAssembler, AcouAssembler, ForceAssembler, FsiAssembler, MecaAssemblerNL
    from loudpy.assembly.assemblers.assembly_tri import MecaAssemblerFourier
    import scipy.sparse as sp
    out = {}
    p = speaker_problem(mesh_dir, acoustic=True)
    fo = p.fem_objects
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
              lp.Clamped("interface_constrained"), lp.Coil("interface_forced"),
              lp.FluidStructure("interface_acou_meca_front"), lp.FluidStructure("interface_acou_meca_rear"),
              lp.Radiation("dtn_boundary"))
    model.mesh(mesh_dir / "speaker_dtn.msh")
    study = EigenStudy(model.as_problem())
    centers = {"dtn_boundary_front": (0.0, 0.032), "dtn_boundary_rear": (0.0, 0.031)}
    Ks, Ms, Cs, nm, na = study._assemble_dtn_system(centers, f_max, half=True, n_max=n_max)

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
