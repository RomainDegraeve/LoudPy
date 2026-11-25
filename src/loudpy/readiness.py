"""
What a simulation still needs, step by step — the interface's Setup checklist, and a script's too — and what each
group of its geometry could be:

    for step in lp.readiness(model, studies, circuits):      # or doc.readiness()
        print(step.state, step.title, step.message)
    for s in lp.suggest(model):                               # per geometry group: the item types that fit it
        print(s.group, s.place, s.usual, s.types, s.why)

A step is "done", "todo" (needed before solving), "warning" (it runs, but probably not as meant) or "optional";
step.path points at the part of the document to change (lp.Document paths: loudpy.docpath). Meshing and results are
the workspace's (their files); everything here reads the model or the scene and the studies only — names are labels:
nothing physical is read from a group's name. What an item does (its medium, its role: holding the structure, the
drive, the coupling, the absorbing end of the air) is what its type declares (loudpy.model.facts): a new item type
takes part without a change here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

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


def _facts(item):
    from loudpy.model.facts import facts_of
    return facts_of(item)


def _media(model, selection: dict) -> dict:
    """{domain group: "solid" | "air"} from the regions as they are (what their types declare)."""
    out = {}
    for r in model.regions:
        f = _facts(r)
        if f.selects == "domains" and f.medium:
            for g in selection.get(r.name, []):
                out[g] = f.medium
    return out


def of_model(model, studies=None, circuits=None) -> list[Step]:
    """The steps of a 2D model: geometry, domains, materials, held, coil, motor, coupling, radiation, studies."""
    from loudpy.model.facts import facts_of, item_types, names, of_role
    from loudpy.physics.parts import region_parameters
    out: list[Step] = []
    path = getattr(model.source, "path", None)
    if path is not None and not Path(path).exists():
        out.append(Step("geometry", "Geometry", "todo", f"{Path(path).name} not found", "model/params/geometry"))
        return out
    out.append(Step("geometry", "Geometry", "done", model.source.name, "model/params/geometry"))
    items = list(model.regions) + list(model.boundaries)
    regions = [r for r in model.regions if facts_of(r).selects == "domains"]
    domain_types = names([n for n, c in item_types().items() if c.facts.table and c.facts.selects == "domains"])
    selection: dict = {}
    try:
        groups = list(model.groups("domain"))
        selection = model.selection() if model.regions else {}
        covered = {g for r in regions for g in selection.get(r.name, [])}
        missing = [g for g in groups if g not in covered]
        if not regions:
            out.append(Step("domains", "Domains", "todo", f"{len(groups)} domain group(s): give each one its physics "
                            f"({domain_types})", "model/items", groups))
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
    media = {facts_of(r).medium for r in regions}

    def role(name):
        return [it for it in items if facts_of(it).role == name]
    if "solid" in media:
        held = role("held")
        out.append(Step("held", "Held edges", "done" if held else "warning",
                        ", ".join(b.name for b in held) if held else
                        f"the structure is free: nothing holds it ({names(of_role('held'))} on its rims, the axis)",
                        "model/items"))
    coil = next(iter(role("drive")), None)
    forced = _forced(studies)
    out.append(Step("coil", "Coil", "done" if coil is not None else ("todo" if forced else "optional"),
                    coil.name if coil is not None else f"where the drive acts: {names(of_role('drive'))} (needed by "
                    "a forced response)", "model/items"))
    if coil is not None and hasattr(coil, "motor"):
        electrical = _electrical(studies, circuits)
        has = coil.motor is not None
        out.append(Step("motor", "Motor", "done" if has else ("todo" if electrical else "optional"),
                        type(coil.motor).__name__ if has else "its blocked coil and Bl (the coil's motor=…): what a "
                        "voltage or a circuit drives", f"model/items/{coil.name}/params/motor"))
    if {"solid", "air"} <= media:
        out.append(_coupling(model, role("coupling"), selection, names(of_role("coupling"))))
    if "air" in media:
        ends = role("absorbing")
        out.append(Step("radiation", "Radiation", "done" if ends else "warning",
                        "absorbed (" + ", ".join(e.name for e in ends) + ")" if ends else
                        f"the air ends on a rigid wall: add {names(of_role('absorbing'))}", "model/items"))
    out.append(_studies_step(studies))
    out += _circuits_step(studies, circuits)
    return out


def _coupling(model, couplings, selection, kinds) -> Step:
    """The structure ↔ air step: done when every group between the structure and the air is coupled or held (a
    warning names the others; a coupling selected otherwise than by groups is taken as it is)."""
    if not couplings:
        return Step("coupling", "Air ↔ structure", "warning",
                    f"air and structure but nothing couples them ({kinds})", "model/items")
    if not all(c.selector.by_groups for c in couplings):
        return Step("coupling", "Air ↔ structure", "done", ", ".join(c.name for c in couplings), "model/items")
    try:
        wet = [s for s in suggest(model, selection) if s.place == "solid|air"]
    except Exception:                                                # noqa: BLE001  (no geometry to read: as it is)
        wet = []
    held = [it for it in list(model.regions) + list(model.boundaries) if _facts(it).role == "held"]
    covered = {g for c in couplings + held for g in selection.get(c.name, [])}       # (a held edge does not move)
    holding = {c.name for c in couplings + held}
    loose = [s.group for s in wet if (not set(s.items) & holding if s.selector is not None else s.group not in covered)]
    if loose:
        return Step("coupling", "Air ↔ structure", "warning",
                    f"{len(loose)} group(s) between the structure and the air not coupled (a rigid wall for the air): "
                    f"{', '.join(loose[:6])}{' …' if len(loose) > 6 else ''}", "model/items", loose)
    return Step("coupling", "Air ↔ structure", "done", ", ".join(c.name for c in couplings), "model/items")


# ── what each geometry group could be (from the geometry only) ─────────────────────────────────────────────────────
@dataclass
class Suggestion:
    """What a geometry group could be: kind "domain" or "boundary"; place (loudpy.model.facts.PLACES: what lies on
    each side of its curves, the same for all of them; "" when unknown or mixed) and shape ("straight": lines r or z
    = const, "arc": arcs centred on the axis, ""); types: the item types that fit it (usual ones first), usual: those
    usual there; why: said to the user; params: {type: values it takes there}; items: the items selecting it now.
    A geometry without named curves (a STEP sketch) still gets boundary rows: its unnamed pieces — the curves two
    domains share (group "a ↔ b", selector lp.Interface(a, b)) and each outer curve off the axis (group "a · edge
    near (r, z)", selector lp.Near(its middle); the air's edges on the axis need nothing and are left out)."""
    group: str
    kind: str
    place: str = ""
    shape: str = ""
    types: list = field(default_factory=list)
    usual: list = field(default_factory=list)
    why: str = ""
    params: dict = field(default_factory=dict)
    items: list = field(default_factory=list)
    selector: Any = None                     # an unnamed piece of the geometry: what selects it (named groups: None)

    @property
    def best(self) -> str | None:
        """The usual type there (None when none is usual)."""
        return self.usual[0] if self.usual else None

    def to_dict(self) -> dict:
        """As plain data."""
        return {"group": self.group, "kind": self.kind, "place": self.place, "shape": self.shape,
                "types": list(self.types), "usual": list(self.usual), "why": self.why, "params": dict(self.params),
                "items": list(self.items), "selector": None if self.selector is None else repr(self.selector)}


def _shape(xy: np.ndarray, tol: float) -> str:
    """"straight" (r = const or z = const), "arc" (a circular arc centred on the axis r = 0) or ""."""
    r, z = xy[:, 0], xy[:, 1]
    if np.ptp(r) <= tol or np.ptp(z) <= tol:
        return "straight"
    A = np.column_stack([2.0 * z, np.ones_like(z)])                  # r² + z² = 2 zc z + (R² − zc²)
    sol = np.linalg.lstsq(A, r ** 2 + z ** 2, rcond=None)[0]
    R2 = sol[1] + sol[0] ** 2
    if R2 <= 0:
        return ""
    dev = np.abs(np.hypot(r, z - sol[0]) - np.sqrt(R2))
    return "arc" if dev.max() <= max(tol, 1e-6 * np.sqrt(R2)) else ""


def suggest(model, selection: dict | None = None) -> list[Suggestion]:
    """Per group of a 2D model's geometry, the item types that fit it, read from the geometry only (the domains on
    each side of its curves, the media of the regions there as their types declare them, whether it lies on the
    axis, its shape) — never from its name. Domain groups: every type that fills domains. selection: the model's
    selection when already known. A scene: []."""
    from loudpy.geometry.sources import GeometryIndex
    from loudpy.model.facts import PLACES, item_types
    if not hasattr(model, "regions"):
        return []
    index = GeometryIndex(model.source)
    if selection is None:
        try:
            selection = model.selection() if (model.regions or model.boundaries) else {}
        except Exception:                                            # noqa: BLE001  (a selection to fix: no media)
            selection = {}
    media = _media(model, selection)
    holders: dict = {}
    for it in list(model.regions) + list(model.boundaries):
        for g in selection.get(it.name, []):
            holders.setdefault(g, []).append(it.name)
    types = {n: c.facts for n, c in item_types().items() if c.facts.table}
    out = [Suggestion(g, "domain", types=[n for n, f in types.items() if f.selects == "domains"],
                      items=holders.get(g, [])) for g in index.domains]
    pts = np.concatenate(list(index.curves.values())) if index.curves else np.zeros((1, 2))
    tol = 1e-9 * max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1e-12)
    curve_types = {n: f for n, f in types.items() if f.selects == "curves"}
    def classify(tags) -> tuple[str, str]:
        places, shapes = set(), set()
        for c in tags:
            sides = index.sides.get(c, [])
            m = [media.get(d) for d in sides]
            xy = index.curves.get(int(c))
            on_axis = xy is not None and float(np.abs(xy[:, 0]).max()) <= tol
            if not sides or None in m or len(sides) > 2:
                places.add("")
            elif len(sides) == 1:
                places.add(f"{m[0]}|axis" if on_axis else f"{m[0]}|-")
            else:
                a, b = sorted(m, key=lambda x: ("solid", "air").index(x))
                places.add(f"{a}|{b}")
            shapes.add(_shape(xy, tol) if xy is not None else "")
        return (places.pop() if len(places) == 1 else ""), (shapes.pop() if len(shapes) == 1 else "")

    def fitted(s: Suggestion) -> Suggestion:
        fits = [(n, f.fit(s.place, s.shape)) for n, f in curve_types.items()] if s.place else []
        fits = [(n, x) for n, x in fits if x is not None]
        s.usual = [n for n, x in fits if x.usual]
        s.types = s.usual + [n for n, x in fits if not x.usual]
        s.params = {n: dict(x.params) for n, x in fits if x.params}
        if s.place:
            first = next((x for n, x in fits if x.usual), None)
            s.why = first.why if first is not None else PLACES[s.place] + (
                " — " + "; ".join(f"{n}: {x.why}" for n, x in fits) if fits else "")
        return s

    for g, tags in index.groups.items():
        if g in index.domains:
            continue
        place, shape = classify(tags)
        out.append(fitted(Suggestion(g, "boundary", place, shape, items=holders.get(g, []))))
    out += [fitted(s) for s in _pieces(model, index, selection, classify, tol)]
    return out


def _pieces(model, index, selection: dict, classify, tol: float) -> list:
    """The unnamed pieces of a geometry's curves (in no named curve group): the curves two domains share, together
    (lp.Interface of a point inside each domain), and each outer curve (lp.Near its middle; a solid's edge on the axis
    too — it is held radially there —, not the air's); the items selecting each one (evaluated on the curves)."""
    import loudpy as lp
    from loudpy.geometry.selectors import Curves
    named = {int(t) for g, tags in index.groups.items() if g not in index.domains for t in tags}
    pairs: dict = {}
    outer: list = []
    for c, sides in index.sides.items():
        c = int(c)
        if c in named or c not in index.curves:
            continue
        if len(sides) == 2:
            pairs.setdefault(tuple(sorted(sides, key=list(index.domains).index)), []).append(c)
        elif len(sides) == 1:
            outer.append((sides[0], c))
    if not pairs and not outer:
        return []
    shown = {g: r.name for r in model.regions if _facts(r).selects == "domains"
             for g in selection.get(r.name, [])}                     # (labels: the regions' names when given)
    found = []
    for (a, b), tags in pairs.items():
        inner = _inner_points(index, a, b, tags)
        sel = lp.Interface(lp.Near(inner[0]), lp.Near(inner[1])) if inner is not None else lp.Interface(a, b)
        found.append((f"{shown.get(a, a)} ↔ {shown.get(b, b)}", sel, tags))
    for d, c in sorted(outer, key=lambda x: (list(index.domains).index(x[0]), x[1])):
        xy = index.curves[c]
        mid = xy[len(xy) // 2]
        found.append((f"{shown.get(d, d)} · edge near ({mid[0]:.4g}, {mid[1]:.4g})",
                      lp.Near((float(mid[0]), float(mid[1]))), [c]))
    seen: dict = {}
    for i, (label, sel, tags) in enumerate(found):                  # (two regions with the same name: kept apart)
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            found[i] = (f"{label} ({seen[label]})", sel, tags)
    regions = {r.name: list(selection.get(r.name, [])) for r in model.regions if r.name in selection}
    try:
        ctx = Curves(index, list(index.domains), regions)
    except Exception:                                                # noqa: BLE001  (no holders: rows stay)
        ctx = None
    picked: dict = {}
    for it in list(model.boundaries) + [r for r in model.regions if _facts(r).selects == "curves"]:
        try:
            picked[it.name] = {int(t) for t in it.selector.edges(ctx)} if ctx is not None else set()
        except Exception:                                            # noqa: BLE001  (a label of no curve here)
            picked[it.name] = set()
    out = []
    for label, sel, tags in found:
        place, shape = classify(tags)
        if place == "air|axis":
            continue                                                 # (the air on the axis needs nothing)
        holders = [n for n, got in picked.items() if set(tags) <= got]
        out.append(Suggestion(label, "boundary", place, shape, items=holders, selector=sel))
    return out


def _inner_points(index, a: str, b: str, tags) -> tuple | None:
    """A point just inside domain a and one just inside domain b, on either side of the middle of their longest shared
    curve: lp.Interface(lp.Near(pa), lp.Near(pb)) selects their shared curves on the geometry and on the mesh alike
    (an unnamed surface's mesh group takes its region's name) and survives renaming the regions. None when no such
    points are found."""
    from loudpy.geometry.selectors import inside_outline
    length = {int(t): float(np.linalg.norm(np.diff(index.curves[int(t)], axis=0), axis=1).sum()) for t in tags}
    c = max(length, key=length.get)
    xy = index.curves[c]
    i = len(xy) // 2
    tangent = xy[min(i + 1, len(xy) - 1)] - xy[max(i - 1, 0)]
    if not np.linalg.norm(tangent):
        return None
    normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
    seg_a, seg_b = index.segments(index.outline[a]), index.segments(index.outline[b])
    for f in (1e-2, 1e-3, 1e-4, 1e-5):
        for sign in (1.0, -1.0):
            pa, pb = xy[i] + sign * f * length[c] * normal, xy[i] - sign * f * length[c] * normal
            if inside_outline(seg_a, pa) and inside_outline(seg_b, pb):
                return tuple(round(float(v), 9) for v in pa), tuple(round(float(v), 9) for v in pb)
    return None


def of_scene(scene, studies=None, circuits=None) -> list[Step]:
    """The steps of a 3D scene: drivers, cabinet, air, motors, walls, studies, the circuits they name."""
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
    walls = getattr(scene, "_walls", None)
    out.append(Step("walls", "Walls", "done" if walls is not None else "optional",
                    f"elastic: {walls.material}" + (f", {len(walls.panels)} panel(s) of their own" if walls.panels
                                                    else "") if walls is not None else
                    "rigid (Scene.walls: elastic walls, their modes computed)", "scene/settings/walls"))
    out.append(_studies_step(studies))
    out += _circuits_step(studies, circuits)
    return out


def _circuits_step(studies, circuits) -> list[Step]:
    """The circuits the studies name (CircuitRef): each one of the simulation's; no step when none is named."""
    named = []
    for s in studies or []:
        for holder in (s, s.get("study") or {}):
            node = (holder.get("params") or {}).get("drive") if isinstance(holder, dict) else None
            if isinstance(node, dict) and node.get("type") == "CircuitRef":
                named.append(((node.get("params") or {}).get("name"), s.get("name")))
    if not named:
        return []
    missing = [f"{c!r} (study {st})" for c, st in named if c not in (circuits or {})]
    return [Step("circuits", "Circuits", "todo" if missing else "done",
                 f"no such circuit: {', '.join(missing)}" if missing else
                 ", ".join(sorted({str(c) for c, _ in named})) + " (named by the studies)", "circuits",
                 [c for c, _ in named])]


def _studies_step(studies) -> Step:
    n = len(studies or [])
    return Step("studies", "Studies", "done" if n else "todo", f"{n} stud{'y' if n == 1 else 'ies'}" if n else
                "no study yet (Modes, Frequency response …)", "studies")


def readiness(target, studies=None, circuits=None) -> list[Step]:
    """The checklist of an lp.Model or an lp.Scene with its studies (document nodes) and circuits."""
    if hasattr(target, "drivers") and not hasattr(target, "regions"):
        return of_scene(target, studies, circuits)
    return of_model(target, studies, circuits)
