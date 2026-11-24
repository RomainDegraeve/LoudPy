"""
What a simulation still needs, step by step — the interface's Setup checklist, and a script's too:

    for step in lp.readiness(model, studies, circuits):      # or doc.readiness()
        print(step.state, step.title, step.message)

A step is "done", "todo" (needed before solving), "warning" (it runs, but probably not as meant) or "optional";
step.path points at the part of the document to change (lp.Document paths: loudpy.docpath). Meshing and results are
the workspace's (their files); everything here reads the model or the scene and the studies only — names are labels:
nothing physical is read from a group's name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

STATES = ("done", "todo", "warning", "optional")


@dataclass
class Step:
    """One step of the checklist."""
    key: str
    title: str
    state: str
    message: str = ""
    path: str = ""
    items: list = field(default_factory=list)          # what it is about (group names, items …)

    def to_dict(self) -> dict:
        """As plain data."""
        return {"key": self.key, "title": self.title, "state": self.state, "message": self.message,
                "path": self.path, "items": list(self.items)}


def _electrical(studies, circuits) -> bool:
    """Whether something drives the coils electrically: a circuit, or a study driven by a voltage on the motors or by
    a circuit."""
    if circuits:
        return True
    for s in studies or []:
        for holder in (s, s.get("study") or {}):
            node = (holder.get("params") or {}).get("drive") if isinstance(holder, dict) else None
            if isinstance(node, dict) and node.get("type") in ("Circuit", "CircuitRef"):
                return True
            if isinstance(node, dict) and node.get("type") == "Voltage" and \
                    (node.get("params") or {}).get("Re") is None:
                return True
    return False


def _forced(studies) -> bool:
    return any((s.get("type") in ("FrequencyResponse", "Transient")) or
               ((s.get("study") or {}).get("type") in ("FrequencyResponse", "Transient")) for s in studies or [])


def of_model(model, studies=None, circuits=None) -> list[Step]:
    """The steps of a 2D model: geometry, domains, materials, held, coil, motor, coupling, radiation, studies."""
    from loudpy.model.boundaries import Clamped, Coil, FluidStructure, Radiation
    from loudpy.model.regions import PML, Air, Solid
    from loudpy.physics.parts import region_parameters
    out: list[Step] = []
    path = getattr(model.source, "path", None)
    if path is not None and not Path(path).exists():
        out.append(Step("geometry", "Geometry", "todo", f"{Path(path).name} not found", "model/params/geometry"))
        return out
    out.append(Step("geometry", "Geometry", "done", model.source.name, "model/params/geometry"))
    regions = [r for r in model.regions if not isinstance(r, PML)]
    try:
        groups = list(model.groups("domain"))
        selection = model.selection() if model.regions else {}
        covered = {g for r in regions for g in selection.get(r.name, [])}
        missing = [g for g in groups if g not in covered]
        if not regions:
            out.append(Step("domains", "Domains", "todo", f"{len(groups)} domain group(s): give each one its physics "
                            "(Solid with a material, Air, PML)", "model/items", groups))
        elif missing:
            out.append(Step("domains", "Domains", "todo", f"{len(missing)} domain group(s) without physics: "
                            f"{', '.join(missing[:6])}{' …' if len(missing) > 6 else ''}", "model/items", missing))
        else:
            out.append(Step("domains", "Domains", "done", f"{len(groups)} group(s), {len(model.regions)} region(s)",
                            "model/items"))
    except Exception as e:                                           # noqa: BLE001  (said, never raised)
        out.append(Step("domains", "Domains", "todo", str(e), "model/items"))
    bad = []
    for r in model.regions:
        try:
            region_parameters(r, model.materials)
        except Exception as e:                                       # noqa: BLE001
            bad.append(f"{r.name}: {e}")
    out.append(Step("materials", "Materials", "todo" if bad else ("done" if regions else "todo"),
                    "; ".join(bad) if bad else ("every region has its material" if regions else "no region yet"),
                    "model/items"))
    solids = [r for r in model.regions if isinstance(r, Solid)]
    airs = [r for r in model.regions if isinstance(r, Air) and not isinstance(r, PML)]
    if solids:
        held = any(isinstance(b, Clamped) for b in model.boundaries)
        out.append(Step("held", "Held edges", "done" if held else "warning",
                        "clamped" if held else "the structure is free: no lp.Clamped (its rims, the axis)",
                        "model/items"))
    coil = next((b for b in model.boundaries if isinstance(b, Coil)), None)
    forced = _forced(studies)
    out.append(Step("coil", "Coil", "done" if coil is not None else ("todo" if forced else "optional"),
                    coil.name if coil is not None else "where the drive acts: a boundary lp.Coil (needed by a "
                    "forced response)", "model/items"))
    if coil is not None:
        electrical = _electrical(studies, circuits)
        has = coil.motor is not None
        out.append(Step("motor", "Motor", "done" if has else ("todo" if electrical else "optional"),
                        type(coil.motor).__name__ if has else "its blocked coil and Bl (lp.Coil(motor=…)): what a "
                        "voltage or a circuit drives", f"model/items/{coil.name}/params/motor"))
    if solids and airs:
        fsi = any(isinstance(b, FluidStructure) for b in model.boundaries)
        out.append(Step("coupling", "Air ↔ structure", "done" if fsi else "warning",
                        "coupled" if fsi else "air and structure but no lp.FluidStructure: not coupled",
                        "model/items"))
    if airs:
        ends = any(isinstance(r, PML) for r in model.regions) or any(isinstance(b, Radiation) for b in model.boundaries)
        out.append(Step("radiation", "Radiation", "done" if ends else "warning",
                        "absorbed (PML or radiation boundary)" if ends else "the air ends on a rigid wall: add a "
                        "PML or an lp.Radiation boundary", "model/items"))
    out.append(_studies_step(studies))
    return out


def of_scene(scene, studies=None, circuits=None) -> list[Step]:
    """The steps of a 3D scene: drivers, cabinet, air, motors, studies."""
    out: list[Step] = []
    out.append(Step("drivers", "Drivers", "done" if scene.drivers else "todo",
                    ", ".join(d.name for d in scene.drivers) or "no driver yet", "scene/drivers"))
    out.append(Step("cabinet", "Cabinet", "done" if scene.cabinet is not None else "todo",
                    getattr(scene.cabinet, "describe", lambda: "")() if scene.cabinet is not None else "no cabinet",
                    "scene/params/cabinet"))
    try:
        config = scene.configuration
        out.append(Step("air", "Air", "done", str(config), "scene/settings"))
    except Exception as e:                                           # noqa: BLE001
        out.append(Step("air", "Air", "todo", str(e), "scene/settings"))
    electrical = _electrical(studies, circuits)
    bare = [d.name for d in scene.drivers if getattr(d, "motor", None) is None]
    out.append(Step("motor", "Motors", "todo" if bare and electrical else ("optional" if bare else "done"),
                    f"no motor: {', '.join(bare)}" if bare else "every driver has its motor", "scene/drivers", bare))
    out.append(_studies_step(studies))
    return out


def _studies_step(studies) -> Step:
    n = len(studies or [])
    return Step("studies", "Studies", "done" if n else "todo", f"{n} stud{'y' if n == 1 else 'ies'}" if n else
                "no study yet (Modes, Frequency response …)", "studies")


def readiness(target, studies=None, circuits=None) -> list[Step]:
    """The checklist of an lp.Model or an lp.Scene with its studies (document nodes) and circuits."""
    if hasattr(target, "drivers") and not hasattr(target, "regions"):
        return of_scene(target, studies, circuits)
    return of_model(target, studies, circuits)
