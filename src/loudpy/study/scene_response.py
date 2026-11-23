"""
Frequency response and coupled modes of a 3D scene (lp.FrequencyResponse / lp.Modes on lp.Scene).

  cavity            DriverCavityCoupling: drivers + cavity modes, the driver fronts in vacuo
  cavity+exterior   coupled_sweep: drivers + cavity modes (rear) + exterior FE / DtN (front)
  merged            coupled_sweep: drivers wetted on both faces by one FE air domain

The FE pressure field is streamed to the result file (store=path), one block per band mesh.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from loudpy.geometry.mesh import Mesh, MeshGroup
from loudpy.model.drives import Force
from loudpy.errors import ModelError, SelectionError, StateError


def multidriver(scene):
    """The scene's drivers as one MultiDriver (stacked modal coordinates)."""
    from loudpy.physics.modal import MultiDriver
    return MultiDriver([d.driver.modes for d in scene.drivers], [d.frame for d in scene.drivers],
                       names=[d.name for d in scene.drivers])


def coil_circuits(scene, drivers, drive) -> dict | None:
    """
    A voltage drive's coil circuits (modal.add_coils): dict(Fc, Bl, Re, Le, n, U) with Fc the unit coil forces
    (n_modes, n_drivers); None for a force drive. drive: an lp.Voltage (every driver, each with these coil
    values) or {driver name: lp.Voltage} naming every driver (each its own coil).
    """
    from loudpy.model.drives import Voltage
    names = [d.name for d in scene.drivers]
    per = drive if isinstance(drive, dict) else {n: drive for n in names}
    if not any(isinstance(v, Voltage) for v in per.values()):
        return None
    if not all(isinstance(v, Voltage) for v in per.values()):
        raise ModelError("a scene's drivers are all driven by a Force or all by a Voltage")
    unknown = set(per) - set(names)
    if unknown:
        raise ModelError(f"drive: no driver {sorted(unknown)} (drivers: {names})")
    if set(per) != set(names):
        raise ModelError(f"a Voltage drive names every driver (each has its own coil: Re, Bl, Le): {names}")
    v = [per[n] for n in names]
    Fc = drivers.coil_forces([d.driver.coil_nodes for d in scene.drivers])
    return dict(Fc=np.asarray(Fc, float).reshape(len(Fc), -1), Bl=np.array([x.Bl for x in v], float),
                Re=np.array([x.Re for x in v], float), Le=np.array([x.Le for x in v], float),
                n=np.array([x.n for x in v], float), U=np.array([complex(x.amplitude) for x in v], complex))


def circuit_drive(scene, drivers, drive):
    """A circuit drive: (Circuit, {coil name: its driver's unit coil force on the modes}); None otherwise. Each coil
    names a driver of the scene, each driver at most one coil (a driver with none is not driven)."""
    from loudpy.circuit import Circuit
    if not isinstance(drive, Circuit):
        return None
    drive.check()
    names = [d.name for d in scene.drivers]
    used = [c.driver for c in drive.coils]
    bad = [c.name for c in drive.coils if c.driver not in names]
    if bad:
        raise ModelError(f"coil(s) {bad}: name the driver each one belongs to (driver=…; drivers: {names})")
    twice = sorted({n for n in used if used.count(n) > 1})
    if twice:
        raise ModelError(f"driver(s) {twice} have several coils in the circuit: one coil per driver")
    Fc = np.asarray(drivers.coil_forces([d.driver.coil_nodes for d in scene.drivers]), float)
    return drive, {c.name: Fc[:, names.index(c.driver)] for c in drive.coils}


def _padded_circuit(circuit, n_y: int):
    if circuit is None:
        return None
    return circuit[0], {k: np.concatenate([v, np.zeros(n_y - len(v))]) for k, v in circuit[1].items()}


def _padded(coils: dict | None, n_y: int) -> dict | None:
    """The coil circuits with zero rows for the modes after the drivers' (walls, extra coordinates)."""
    if coils is None or coils["Fc"].shape[0] >= n_y:
        return coils
    pad = np.zeros((n_y - coils["Fc"].shape[0], coils["Fc"].shape[1]))
    return dict(coils, Fc=np.vstack([coils["Fc"], pad]))


def coil_force(scene, drivers, drive):
    """Stacked modal force from a Force (every driver) or {driver name: Force}."""
    names = [d.name for d in scene.drivers]
    if isinstance(drive, Force):
        amps = [drive.amplitude] * len(names)
    else:
        unknown = set(drive) - set(names)
        if unknown:
            raise SelectionError(f"no driver {sorted(unknown)} (drivers: {names})")
        amps = [drive[n].amplitude if n in drive else 0.0 for n in names]
    return drivers.coil_forces([d.driver.coil_nodes for d in scene.drivers], np.asarray(amps, complex))


def cavity_modes(scene):
    """Cavity modes of the scene's cavity mesh (cached next to it, rebuilt when the mesh is newer)."""
    from loudpy.physics.modal import CavityModes
    msh = Path(scene.cavity_mesh)
    h5 = msh.with_name(msh.stem + "_modes.h5")
    if not scene._cavity.cache_modes:
        cav = CavityModes.from_msh(msh)
        cav.solve(scene._cavity.f_max, c=scene.c)
        return cav
    if h5.exists() and h5.stat().st_mtime > msh.stat().st_mtime:
        cav = CavityModes.load(h5)
        if np.isclose(cav.freqs[-1], 0) or cav.c != scene.c:
            cav = None
    else:
        cav = None
    if cav is None:
        cav = CavityModes.from_msh(msh)
        cav.solve(scene._cavity.f_max, c=scene.c)
        cav.save(h5)
    return cav


def scene_walls(scene, cav):
    """The scene's elastic walls (physics.shell.WallModes) on the "walls" faces of its cavity mesh, or None."""
    w = scene._walls
    if w is None:
        return None
    from loudpy.geometry.meshing3d import read_p2_mesh
    from loudpy.physics.shell import cavity_walls
    merged = scene.configuration == "merged"
    if merged:                                                   # the shells on a mesh of the inside air
        if scene.walls_mesh is None:
            raise StateError("mesh the scene first: scene.mesh(folder)")
        nodes, tets, faces = read_p2_mesh(scene.walls_mesh, ["walls"])
    else:
        nodes, tets, faces = read_p2_mesh(scene.cavity_mesh, ["walls"])
        if len(nodes) != len(cav.nodes) or not np.allclose(nodes, cav.nodes):
            raise StateError(f"the cavity modes do not belong to {scene.cavity_mesh}: remesh the scene")
    if not len(faces["walls"]):
        raise SelectionError("the inside air has no 'walls' faces")
    panels = wall_panels(scene, nodes, tets, faces["walls"]) if (w.panels or w.thickness is None or merged) \
        else None
    return cavity_walls(nodes, tets, faces["walls"], w.material, w.thickness, w.damping, w.fixed,
                        w.f_max or (scene._cavity.f_max if scene._cavity is not None else
                                    2 * max(b[1] for b in scene._open.bands)), w.angle, panels=panels)


def wall_panels(scene, nodes, tets, faces):
    """(panel of each wall face, {panel: (material, thickness, damping)}) of the scene's walls: the cabinet's panels
    (lp.cabinets.Box sides, lp.cabinets.Step solids) with their measured thickness, each Panel's own values over
    the walls' (faces on no panel: None, rigid)."""
    from loudpy.fem import shell as fe
    from loudpy.physics.shell import wall_shells
    from loudpy.scene.cabinets import Box, Step
    from loudpy._log import info as log
    w = scene._walls
    cab = scene._cabinet()
    if not isinstance(cab, (Box, Step)):
        raise ModelError("walls panels, measured thicknesses and walls in a merged air domain need an "
                         "lp.cabinets.Box or lp.cabinets.Step cabinet (a cabinet builder has no panels): give "
                         "thickness= (cavity air only)")
    ws = wall_shells(nodes, tets, faces)                         # the faces' order kept, normals into the air
    normals, _ = fe.cell_normals(ws.nodes, ws.cells)
    centres = ws.nodes[ws.cells].mean(axis=1)
    labels, measured = cab.wall_panels(centres, normals)
    own = {str(k): v for k, v in (w.panels or {}).items()}
    unknown = set(own) - {str(lab) for lab in set(labels) if lab is not None}
    if unknown:
        raise SelectionError(f"walls panels {sorted(unknown)}: not panels of the cabinet "
                             f"(its panels: {sorted({str(x) for x in labels if x is not None})})")
    table = {}
    for lab in sorted({x for x in labels if x is not None}, key=str):
        p = own.get(str(lab))
        material = p.material if p is not None and p.material is not None else w.material
        thickness = p.thickness if p is not None and p.thickness is not None else \
            (w.thickness if w.thickness is not None else measured.get(lab))
        if thickness is None:
            raise ModelError(f"walls panel {lab!r}: no thickness measured on the cabinet: give it "
                             "(lp.cabinets.Panel(thickness=…) or walls(thickness=…))")
        damping = p.damping if p is not None and p.damping is not None else w.damping
        table[lab] = (material, float(thickness), damping)
        log(f"  wall panel {lab}: {getattr(material, 'name', material)}, {float(thickness) * 1e3:.1f} mm"
            + (" (measured)" if (p is None or p.thickness is None) and w.thickness is None else ""))
    return labels, table


def cavity_coupling(scene, drivers, cav):
    """The cavity coupling of the scene's drivers (and walls) with its cavity modes."""
    from loudpy.physics.modal import DriverCavityCoupling
    patch = ["interface_driver"] if len(scene.drivers) == 1 else None
    return DriverCavityCoupling(cav, drivers, [d.driver.rear for d in scene.drivers], rho=scene.rho, c=scene.c,
                                eta_air=scene._cavity.eta, patch=patch, walls=scene_walls(scene, cav))


def _wall_attachment(walls, offset):
    """What the results keep of the walls: shell mesh, modal basis (translations), offset in q."""
    extra = {}
    if walls.labels is not None:
        extra["panel"] = np.asarray(walls.labels).astype("S")
    if not isinstance(walls.shell.thickness, dict):
        extra["thickness"] = np.broadcast_to(np.asarray(walls.shell.thickness, float),
                                             (len(walls.walls.cells),)).copy()
    return dict(**extra, nodes=walls.walls.nodes, cells=walls.walls.cells, rows=walls.walls.rows, freq=walls.freq,
                zeta=walls.zeta, U=walls.translations().astype(np.float32),
                U_n=walls.normal().astype(np.float32), volume_disp=walls.volume_disp,
                meta=dict(offset=int(offset), material=walls.shell.material.name,
                          thickness=float(np.mean(extra["thickness"])) if "thickness" in extra else None))


def _attachments(scene, drivers, extra=None):
    """Driver modal bases (one per distinct DriverModes) and the scene description."""
    att, keys, desc = {}, {}, []
    for d, off in zip(scene.drivers, drivers.offsets[:-1]):
        m = d.driver.modes
        key = keys.setdefault(id(m), f"modes{len(keys)}")
        if key not in att:
            att[key] = dict(xy=m.xy, tris=m.tris, U=m.U, m=m.m, family=np.asarray(m.family).astype("S1"),
                            freq=m.freq, zeta=m.zeta, coil_nodes=d.driver.coil_nodes,
                            meta=dict(source=d.driver.source))
        desc.append(dict(name=d.name, modes=key, offset=int(off), origin=d.frame.origin.tolist(),
                         rotation=d.frame.rotation.tolist()))
    att["scene"] = {"meta": dict(drivers=desc, configuration=scene.configuration, rho=scene.rho, c=scene.c,
                                 air=dict(name=scene.air.name, **scene.air.to_dict()),
                                 hemisphere=scene.hemisphere, **(extra or {}))}
    return att


def _fe_mesh(ext) -> Mesh:
    """geometry.Mesh of an exterior FE domain (node tags = rows, faces as named groups)."""
    groups = {"air": MeshGroup("air", 3, "tet10", np.asarray(ext.tets))}
    for name, tris in ext.faces.items():
        if len(tris):
            groups[name] = MeshGroup(name, 2, "tri6", np.asarray(tris))
    return Mesh(np.arange(len(ext.nodes)), np.asarray(ext.nodes), groups)


def _driver_volume_disp(scene, drivers, mesh_path):
    """∫ φ·n dS of the driver modes on their rear patches of an inside-air mesh (merged air: the walls mesh)."""
    from loudpy.geometry.meshing3d import read_p2_mesh
    from loudpy.physics.modal import couple_drivers
    names = ["interface_driver"] if len(scene.drivers) == 1 else \
        [f"interface_driver_{j}" for j in range(len(scene.drivers))]
    nodes, _, faces = read_p2_mesh(mesh_path, names)
    _, Bc, _, _ = couple_drivers(nodes, [faces[n] for n in names], drivers, [d.driver.rear for d in scene.drivers],
                                 5e-4)
    return np.asarray(Bc).sum(axis=0)


def _mesh_meta(path) -> dict:
    """A block's mesh: its file and the fingerprint of its signature (a result knows on which mesh it was computed)."""
    from loudpy.geometry.mesh_files import fingerprint
    return {"mesh": str(path), "mesh_fingerprint": fingerprint(path) or ""}


def solve(study, freqs, store=None, chunk_hz=50.0, rtol=1e-7, log=None):
    """Frequency response of a scene in its configuration (cavity, cavity + exterior, merged); the 3D
    pressure field is streamed to `store` when given. Returns SceneResults."""
    if log is None:                                   # progress messages: logger "loudpy"
        from loudpy._log import info as log
    from loudpy.results import load
    from loudpy.results.results import Block
    from loudpy.results.scene import SceneResults
    from loudpy.results.schema import StreamWriter
    from loudpy.study.base import finish
    from loudpy.physics.exterior import ExteriorFE, coupled_sweep
    scene = study.model
    if scene.folder is None:
        raise StateError("mesh the scene first: scene.mesh(folder)")
    freqs = np.atleast_1d(np.asarray(freqs, float))
    if scene.configuration != "cavity" and scene.band_meshes:        # every frequency on an air mesh made for it
        lo, hi = min(b["f_lo"] for b in scene.band_meshes), max(b["f_hi"] for b in scene.band_meshes)
        out = freqs[(freqs < lo * (1 - 1e-9)) | (freqs > hi * (1 + 1e-9))]
        if len(out):
            raise ModelError(f"{len(out)} frequencies ({out.min():g}–{out.max():g} Hz) outside the air's meshes "
                             f"({lo:g}–{hi:g} Hz): give the air bands up to {freqs.max():g} Hz "
                             f"(scene.merged / exterior(bands=lp.Scene.bands({freqs.max():g})), or f_max= of "
                             f"lp.Scene.sealed_box / vented_box) and mesh again")
    drivers = multidriver(scene)
    circuit = circuit_drive(scene, drivers, study.drive)               # a circuit: its MNA unknowns
    coils = None if circuit is not None else coil_circuits(scene, drivers, study.drive)   # a voltage drive
    electric = coils is not None or circuit is not None
    n_s = sum(len(d.driver.modes.freq) for d in scene.drivers)
    F = np.zeros(n_s, complex) if electric else coil_force(scene, drivers, study.drive)
    config = scene.configuration

    meta = dict(study="FrequencyResponse", scene=True, configuration=config, drive=repr(study.drive))
    t0 = time.time()

    S = np.zeros((len(drivers), 0))
    R = np.zeros((len(drivers), len(drivers)))
    kappa2, eta, att_extra, cav, cpl = np.zeros(0), 0.0, {}, None, None
    if scene._cavity is not None:
        cav = cavity_modes(scene)
        cpl = cavity_coupling(scene, drivers, cav)
        S, R, kappa2, eta = cpl.S, cpl.R, cav.kappa2, scene._cavity.eta
        log(f"  cavity: {len(cav.nodes)} nodes, {len(kappa2)} modes, wet areas "
            f"{np.round(np.array(cpl.wet_areas) * 1e4, 1)} cm² ({time.time() - t0:.0f} s)")
    walls_m = None
    if config == "merged" and scene._walls is not None:             # walls in the one air domain (no cavity)
        walls_m = scene_walls(scene, None)
        log(f"  walls: {len(walls_m.walls.nodes)} shell nodes, {len(walls_m)} modes up to "
            f"{walls_m.freq[-1] if len(walls_m) else 0:.0f} Hz")
    attachments = _attachments(scene, drivers, att_extra)
    if walls_m is not None:
        attachments["walls"] = _wall_attachment(walls_m, len(drivers))
        try:
            attachments["drive_volume_disp"] = dict(V=_driver_volume_disp(scene, drivers, scene.walls_mesh))
        except Exception as e:                                       # noqa: BLE001  (a diagnostic only)
            log(f"  (no driver volume displacement: {e})")
    attachments["drive"] = dict(F=np.asarray(F, complex))                # modal force, stacked drivers
    if coils is not None:                                              # a voltage drive: the coil circuits
        attachments["drive"].update({k: np.asarray(v) for k, v in coils.items()})
    if circuit is not None:                                            # a circuit: itself, as data (JSON)
        attachments["circuit"] = dict(meta=circuit[0].to_dict())
    if cav is not None:
        attachments["cavity"] = dict(nodes=cav.nodes, tets=cav.tets, psi=cav.psi.astype(np.float32), kappa2=cav.kappa2,
                                     meta=dict(f_max=scene._cavity.f_max, volume=float(cav.volume)))
        attachments["cavity"]["volume_disp"] = np.asarray(cpl.volume_disp)        # ∫ φ·n dS per coordinate
        if cpl.walls is not None:
            attachments["walls"] = _wall_attachment(cpl.walls, len(drivers))
            log(f"  walls: {len(cpl.walls.walls.nodes)} shell nodes, {len(cpl.walls)} modes up to "
                f"{cpl.walls.freq[-1] if len(cpl.walls) else 0:.0f} Hz")

    if config == "cavity":                                          # A: no FE air
        extra = {}
        if not electric:
            q, P = cpl.forced(freqs, F)
        else:
            q, P, E = cpl.forced(freqs, F, coils=coils, circuit=circuit)
            extra["I" if coils is not None else "e"] = E
        block = Block({}, freqs, {}, {"q": q, "P_cavity": P, **extra}, {"configuration": config})
        res = finish(study, SceneResults([block], meta, {}))
        res.attachments = attachments
        if store:
            res.save(store)
        return res

    # B / C: FE air per band (+ cavity modes for B), streamed
    if config == "merged":
        profiles = [(d.driver.front, d.driver.rear) for d in scene.drivers]
        front_name = "interface_driver"
    else:
        profiles = [d.driver.front for d in scene.drivers]
        front_name = "interface_front"
    dtn = scene.dtn()
    f_top = float(freqs.max())
    bands, writer, bmap = [], None, []
    for bi, b in enumerate(scene.band_meshes):
        sel = [i for i, f in enumerate(freqs) if (b["f_lo"] if bi == 0 else b["f_lo"] + 1e-6) <= f <= b["f_hi"]]
        if not sel:
            continue
        ext = ExteriorFE(str(b["path"]), dtn, min(b["f_hi"], f_top), front=front_name, surfaces=["openings"])
        if len(scene.drivers) == 1:
            ext.couple_driver(drivers, profiles, patch=[front_name])
        else:
            ext.couple_driver(drivers, profiles)
        wm = cpl.walls if cpl is not None else walls_m
        if wm is not None and len(wm):
            ext.couple_walls(wm)                                     # every wet face of the walls in this air
        bands.append((b["f_lo"] if bi == 0 else b["f_lo"] + 1e-6, b["f_hi"], ext))
        bmap.append((bi, sel))
        log(f"  band {b['f_lo']:.0f}–{b['f_hi']:.0f} Hz: {len(ext.nodes)} nodes, wet area per driver "
            f"{np.round(np.array(ext.wet_areas) * 1e4, 1)} cm²")
    if store:
        writer = StreamWriter(store, "frequency", meta, results_class="SceneResults")
        wblocks = [writer.block(f"{scene.band_meshes[bi]['f_hi']:.0f}Hz", freqs[sel], {"air": _fe_mesh(ext)},
                                {"p": (len(ext.nodes),)}, meta=_mesh_meta(scene.band_meshes[bi]["path"]))
                   for (bi, sel), (_, _, ext) in zip(bmap, bands)]
    row_of = {i: (k, j) for k, (_, sel) in enumerate(bmap) for j, i in enumerate(sel)}

    def on_frequency(i, b, p, c_h, q):
        if writer is not None:
            k, j = row_of[i]
            writer.row(wblocks[k], "p", j, p.astype(np.complex64))

    w_s = cpl.w_s if cpl is not None else 2 * np.pi * drivers.freq        # driver (and wall) modes
    zeta = cpl.zeta if cpl is not None else np.asarray(drivers.zeta, float)
    if walls_m is not None and len(walls_m):
        w_s = np.concatenate([w_s, 2 * np.pi * walls_m.freq])
        zeta = np.concatenate([zeta, walls_m.zeta])
        S, R = np.zeros((len(w_s), 0)), np.zeros((len(w_s), len(w_s)))
    Fq = np.concatenate([np.asarray(F), np.zeros((len(w_s) - len(F),) + np.shape(F)[1:], complex)])
    out = coupled_sweep(bands, freqs, Fq, w_s, zeta, S, R, kappa2, eta, scene.rho, scene.c, chunk_hz, rtol,
                        on_frequency, log=log, coils=_padded(coils, len(w_s)),
                        circuit=_padded_circuit(circuit, len(w_s)))
    q, coefs, iters = out[:3]
    currents = out[3] if electric else None
    n_h = np.array([len(c) for c in coefs])
    C = np.zeros((len(freqs), n_h.max()), complex)
    for i, c in enumerate(coefs):
        C[i, :len(c)] = c
    P = None
    if cpl is not None:                                               # cavity modal pressures (B)
        k = 2 * np.pi * freqs / scene.c
        D = 1.0 / (kappa2[None, :] - k[:, None] ** 2 + 1j * eta * np.sqrt(kappa2)[None, :] * k[:, None])
        P = -scene.rho * (2 * np.pi * freqs[:, None]) ** 2 * D * (q[:, :S.shape[0]] @ S)
    blocks = []
    for k_, (bi, sel) in enumerate(bmap):
        extra = {"q": q[sel], "dtn": C[sel], "dtn_n": n_h[sel], "iters": iters[sel]}
        if currents is not None:
            extra["I" if coils is not None else "e"] = currents[sel]
        if P is not None:
            extra["P_cavity"] = P[sel]
        if writer is not None:
            for name, v in extra.items():
                writer.extra(wblocks[k_], name, v)
        blocks.append(Block({"air": _fe_mesh(bands[k_][2])}, freqs[sel], {}, extra,
                            _mesh_meta(scene.band_meshes[bi]["path"]), f"{scene.band_meshes[bi]['f_hi']:.0f}Hz"))
    log(f"  sweep: {len(freqs)} frequencies in {(time.time() - t0) / 60:.1f} min "
        f"(GMRES iterations: mean {iters.mean():.0f}, max {iters.max()})")
    if writer is not None:
        writer.close(attachments=attachments, meta=meta, described=finish(study, SceneResults([], meta, {})))
        return load(store)                                            # fields read lazily from the file
    res = finish(study, SceneResults(blocks, meta, {}))
    res.attachments = attachments
    return res


def coupled_modes(study, f_max=None):
    """Coupled driver + cavity modes (configuration cavity): ModeResults with q and P per mode."""
    from loudpy.results.results import Block, ModeResults
    scene = study.model
    if scene.configuration != "cavity":
        raise NotImplementedError("lp.Modes on a scene: the cavity configuration (drivers + cavity modes)")
    drivers = multidriver(scene)
    cav = cavity_modes(scene)
    cpl = cavity_coupling(scene, drivers, cav)
    s, Xq, XP = cpl.coupled_modes(f_max=f_max)
    f = s.imag / (2 * np.pi)
    zeta = -s.real / np.abs(s)
    block = Block({}, f, {}, {"zeta": zeta, "s": s, "q": Xq.T, "P": XP.T}, {"configuration": "cavity"}, "coupled")
    res = ModeResults([block], dict(study="Modes", scene=True), {})
    res.attachments = _attachments(scene, drivers)
    if cpl.walls is not None:
        res.attachments["walls"] = _wall_attachment(cpl.walls, len(drivers))
    return res


def wall_modes(study, f_max=None):
    """In-vacuo modes of the scene's elastic walls: ModeResults with the normal displacement "u_n" and the
    translations "u" on the shell mesh (group "walls", TRI3)."""
    from loudpy.results.results import Block, ModeResults
    scene = study.model
    if scene._walls is None:
        raise ModelError("the scene has no elastic walls: call scene.walls(…)")
    w = scene_walls(scene, cavity_modes(scene) if scene._cavity is not None else None)
    keep = w.freq <= (f_max if f_max is not None else np.inf)
    groups = {"walls": MeshGroup("walls", 2, "tri3", w.walls.cells)}
    if w.labels is not None:                                      # one group per panel
        lab = np.asarray(w.labels).astype(str)
        for p in sorted(set(lab.tolist()), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x)):
            groups[f"panel {p}"] = MeshGroup(f"panel {p}", 2, "tri3", w.walls.cells[lab == p])
    mesh = Mesh(np.arange(len(w.walls.nodes)), w.walls.nodes, groups)
    U = w.translations()[keep]
    block = Block({"walls": mesh}, w.freq[keep], {"u": U, "u_n": w.normal()[keep]},
                  {"zeta": w.zeta[keep], "volume_disp": w.volume_disp[keep]}, {"physics": "walls"}, "walls")
    return ModeResults([block], dict(study="Modes", scene=True, physics="walls"), {})
