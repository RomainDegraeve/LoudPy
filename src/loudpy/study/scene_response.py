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
from loudpy.errors import SelectionError, StateError


def multidriver(scene):
    from loudpy.physics.modal import MultiDriver
    return MultiDriver([d.driver.modes for d in scene.drivers], [d.frame for d in scene.drivers],
                       names=[d.name for d in scene.drivers])


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


def cavity_coupling(scene, drivers, cav):
    from loudpy.physics.modal import DriverCavityCoupling
    patch = ["interface_driver"] if len(scene.drivers) == 1 else None
    return DriverCavityCoupling(cav, drivers, [d.driver.rear for d in scene.drivers], rho=scene.rho, c=scene.c,
                                eta_air=scene._cavity.eta, patch=patch)


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


def solve(study, freqs, store=None, chunk_hz=50.0, rtol=1e-7, log=None):
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
    drivers = multidriver(scene)
    F = coil_force(scene, drivers, study.drive)
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
    attachments = _attachments(scene, drivers, att_extra)
    attachments["drive"] = dict(F=np.asarray(F, complex))                # modal force, stacked drivers
    if cav is not None:
        attachments["cavity"] = dict(nodes=cav.nodes, tets=cav.tets, psi=cav.psi.astype(np.float32), kappa2=cav.kappa2,
                                     meta=dict(f_max=scene._cavity.f_max, volume=float(cav.volume)))

    if config == "cavity":                                          # A: no FE air
        q, P = cpl.forced(freqs, F)
        block = Block({}, freqs, {}, {"q": q, "P_cavity": P}, {"configuration": config})
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
        bands.append((b["f_lo"] if bi == 0 else b["f_lo"] + 1e-6, b["f_hi"], ext))
        bmap.append((bi, sel))
        log(f"  band {b['f_lo']:.0f}–{b['f_hi']:.0f} Hz: {len(ext.nodes)} nodes, wet area per driver "
            f"{np.round(np.array(ext.wet_areas) * 1e4, 1)} cm²")
    if store:
        writer = StreamWriter(store, "frequency", meta, results_class="SceneResults")
        wblocks = [writer.block(f"{scene.band_meshes[bi]['f_hi']:.0f}Hz", freqs[sel], {"air": _fe_mesh(ext)},
                                {"p": (len(ext.nodes),)}, meta={"mesh": str(scene.band_meshes[bi]["path"])})
                   for (bi, sel), (_, _, ext) in zip(bmap, bands)]
    row_of = {i: (k, j) for k, (_, sel) in enumerate(bmap) for j, i in enumerate(sel)}

    def on_frequency(i, b, p, c_h, q):
        if writer is not None:
            k, j = row_of[i]
            writer.row(wblocks[k], "p", j, p.astype(np.complex64))

    q, coefs, iters = coupled_sweep(bands, freqs, F, 2 * np.pi * drivers.freq, drivers.zeta, S, R, kappa2, eta,
                                    scene.rho, scene.c, chunk_hz, rtol, on_frequency, log=log)
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
        if P is not None:
            extra["P_cavity"] = P[sel]
        if writer is not None:
            for name, v in extra.items():
                writer.extra(wblocks[k_], name, v)
        blocks.append(Block({"air": _fe_mesh(bands[k_][2])}, freqs[sel], {}, extra,
                            {"mesh": str(scene.band_meshes[bi]["path"])}, f"{scene.band_meshes[bi]['f_hi']:.0f}Hz"))
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
    return res
