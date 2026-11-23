"""
The files of the meshes. Every mesh <name>.msh that loudpy writes has a sidecar <name>.msh.json: the signature of
what it was built from (a 2D model's geometry files, groups and sizes; a scene's cabinet, drivers and settings —
reuse compares it) and, under "info", what the mesh is for (its role, its frequency band, its sizes) and its
statistics, measured the first time they are asked for and kept while the file is unchanged.

    lp.mesh_info(path)          {"nodes", "elements": {type: count}, "groups": {dim: [names]}, "bounds", "bytes",
                                 "built", "role", …}
    same_signature(stored, now)   a stored sidecar matches the signature of now ("info" left out)
    fingerprint(path)           a short hash of a mesh's signature (results keep it: on which mesh they were computed)
    listing(folder)             every mesh of a folder: [{"path", "role", "signature", "info"}]
    annotate(path, **info)      add to a mesh's "info" (what it is for: its role, its band …)
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path

INFO = "info"


def sidecar(path) -> Path:
    """The sidecar of a mesh file: <mesh>.json."""
    return Path(str(path) + ".json")


def _read(path) -> dict:
    sc = sidecar(path)
    try:
        data = json.loads(sc.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(path, data: dict) -> None:
    """Write the sidecar (to a temporary file first, then renamed: never half written)."""
    sc = sidecar(path)
    tmp = sc.with_name(sc.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str))
    os.replace(tmp, sc)


def signature_of(stored: dict) -> dict:
    """The signature part of a sidecar (its "info" left out)."""
    return {k: v for k, v in stored.items() if k != INFO}


def same_signature(stored: dict, current: dict) -> bool:
    """A stored sidecar was written for this signature (what the mesh says about itself is not compared)."""
    return signature_of(stored) == json.loads(json.dumps(current, default=str))


def write_sidecar(path, signature: dict, **info) -> None:
    """The sidecar of a mesh just built: its signature and what it is for (info)."""
    data = dict(json.loads(json.dumps(signature, default=str)))
    if info:
        data[INFO] = json.loads(json.dumps(info, default=str))
    _write(path, data)


def annotate(path, **info) -> None:
    """Add to the "info" of a mesh's sidecar (nothing when the mesh has no sidecar)."""
    data = _read(path)
    if not data:
        return
    merged = dict(data.get(INFO) or {})
    merged.update(json.loads(json.dumps(info, default=str)))
    if merged != data.get(INFO):
        data[INFO] = merged
        _write(path, data)


def fingerprint(path) -> str | None:
    """A short hash of the signature of a mesh (None: no sidecar): it changes when the mesh is built again from
    other settings."""
    data = _read(path) if path is not None else {}
    if not data:
        return None
    text = json.dumps(signature_of(data), sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def role_of(stored: dict) -> str:
    """What a mesh is for, from its sidecar: its "info" role, a scene mesh's "what", or a 2D model's mesh."""
    info = stored.get(INFO) or {}
    if info.get("role"):
        return str(info["role"])
    what = str(stored.get("what") or "")
    if what == "cavity":
        return "cavity"
    if what == "walls":
        return "inside air of the walls"
    if what:
        kind, _, f_hi = what.partition("_")
        return f"{kind} air up to {f_hi} Hz" if f_hi else what
    if "domains" in stored:
        return "2D model"
    return "mesh"


def _stats(path: Path) -> dict:
    """Counts of a mesh file read with gmsh (nodes, elements per type, physical groups, bounding box)."""
    from loudpy.geometry.gmsh_session import gmsh_session
    with gmsh_session() as gmsh:
        gmsh.open(str(path))
        tags, coords, _ = gmsh.model.mesh.getNodes()
        elements: dict[str, int] = {}
        for t in gmsh.model.mesh.getElementTypes():
            name = gmsh.model.mesh.getElementProperties(t)[0]
            elements[name] = elements.get(name, 0) + len(gmsh.model.mesh.getElementsByType(t)[0])
        groups: dict[str, list[str]] = {}
        for dim, tag in gmsh.model.getPhysicalGroups():
            groups.setdefault(str(dim), []).append(gmsh.model.getPhysicalName(dim, tag) or f"group {tag}")
        xyz = coords.reshape(-1, 3) if len(coords) else None
        gmsh.clear()
    bounds = None if xyz is None else [xyz.min(axis=0).round(9).tolist(), xyz.max(axis=0).round(9).tolist()]
    return {"nodes": int(len(tags)), "elements": elements, "groups": groups, "bounds": bounds}


def mesh_info(path) -> dict:
    """
    What a mesh file is and holds: {"nodes", "elements": {gmsh element type: count}, "groups": {dimension: [names]},
    "bounds": [min xyz, max xyz], "bytes", "built" (the file's date), "role" (what it is for), and what was noted
    when it was built (a band's frequencies, its element sizes …)}. The counts are read once and kept in the
    sidecar while the file is unchanged.
        lp.mesh_info("meshes/cavity.msh")["nodes"]
    """
    path = Path(path)
    st = path.stat()
    data = _read(path)
    info = dict(data.get(INFO) or {})
    stamp = [st.st_size, round(st.st_mtime, 3)]
    if info.get("stamp") != stamp or "nodes" not in info:
        info.update(_stats(path), stamp=stamp)
        if data:                                            # kept for the next time (a mesh loudpy wrote)
            data[INFO] = info
            try:
                _write(path, data)
            except OSError:
                pass
    out = dict(info)
    out.pop("stamp", None)
    out.update(bytes=int(st.st_size), built=_dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
               role=role_of(data) if data else "mesh")
    return out


def listing(folder) -> list[dict]:
    """Every mesh file of a folder (not its subfolders), by name: [{"path", "role", "signature", "info"}] — what the
    sidecars say, nothing read from the meshes (lp.mesh_info for their counts)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    out = []
    for p in sorted(folder.glob("*.msh")):
        data = _read(p)
        out.append({"path": p, "role": role_of(data) if data else "mesh", "signature": signature_of(data),
                    "info": dict(data.get(INFO) or {})})
    return out


__all__ = ["mesh_info", "listing", "fingerprint", "annotate", "same_signature", "write_sidecar", "sidecar"]
