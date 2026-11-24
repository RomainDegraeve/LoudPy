"""
Paths into a simulation's plain data (lp.Document): what an edit, a check or a dependency points at.

    "studies/fr/params/freqs/to"            the "to" of the frequencies of the study named fr
    "model/items/cone/params/mesh_size"     an item by its name
    "scene/drivers/main/place/params/at"    a placed driver by its name
    "circuits/xo/params/parts/amp/params/U" a circuit's part by its name

A segment in a list is an index (0, 1 …) or a name: the element whose "name", params "name" or placement name it is.
The functions work on nested dicts and lists (a document's data): get, put (replace or add, returns the value before),
remove (returns it), insert (into a list, at an index or the end); diff(a, b) the paths where two of them differ.
"""
from __future__ import annotations

import copy
from typing import Any

from loudpy.errors import ModelError

MISSING = object()                 # "nothing there" (a value before, or a default asked for)
REQUIRED = object()                # get's default: raise when the path is not there


def split(path) -> list:
    """The segments of a path ("a/b/0" or a list)."""
    if isinstance(path, (list, tuple)):
        return list(path)
    return [s for s in str(path).strip("/").split("/") if s != ""]


def join(*parts) -> str:
    """A path from segments (and paths)."""
    return "/".join(str(s) for p in parts for s in split(p))


def _name(element) -> str | None:
    if not isinstance(element, dict):
        return None
    for d in (element, element.get("params") or {}, (element.get("place") or {}).get("params") or {}):
        if isinstance(d, dict) and isinstance(d.get("name"), str):
            return d["name"]
    return None


def _index(container: list, seg) -> int | None:
    """The place of a segment in a list: an index, or the element of that name (None if absent)."""
    if isinstance(seg, int) or (isinstance(seg, str) and seg.lstrip("-").isdigit()):
        i = int(seg)
        return i if -len(container) <= i < len(container) else None
    return next((i for i, e in enumerate(container) if _name(e) == seg), None)


def _step(node, seg, path):
    if isinstance(node, dict):
        if seg not in node:
            raise ModelError(f"{join(path)}: no {seg!r} (has {sorted(node)})")
        return node[seg]
    if isinstance(node, list):
        i = _index(node, seg)
        if i is None:
            raise ModelError(f"{join(path)}: no {seg!r} among {[_name(e) or k for k, e in enumerate(node)]}")
        return node[i]
    raise ModelError(f"{join(path)}: a value, not a container")


def get(data, path, default=REQUIRED) -> Any:
    """The value at a path (default when given — MISSING too — and the path is not there)."""
    node: Any = data
    done: list = []
    try:
        for seg in split(path):
            node = _step(node, seg, done)
            done.append(seg)
    except ModelError:
        if default is not REQUIRED:
            return default
        raise
    return node


def exists(data, path) -> bool:
    """Whether the path leads somewhere."""
    return get(data, path, MISSING) is not MISSING


def put(data, path, value) -> Any:
    """Set the value at a path (a dict key may be new); the value before (MISSING when there was none)."""
    segs = split(path)
    if not segs:
        raise ModelError("a path names what to set")
    parent = get(data, segs[:-1])
    last = segs[-1]
    if isinstance(parent, dict):
        before = parent.get(last, MISSING)
        parent[last] = value
        return before
    if isinstance(parent, list):
        i = _index(parent, last)
        if i is None:
            raise ModelError(f"{join(segs)}: no element {last!r} (insert adds one)")
        before, parent[i] = parent[i], value
        return before
    raise ModelError(f"{join(segs[:-1])}: a value, not a container")


def remove(data, path) -> Any:
    """Remove what is at a path; what it was."""
    segs = split(path)
    parent = get(data, segs[:-1])
    last = segs[-1]
    if isinstance(parent, dict):
        if last not in parent:
            raise ModelError(f"{join(segs)}: not there")
        return parent.pop(last)
    if isinstance(parent, list):
        i = _index(parent, last)
        if i is None:
            raise ModelError(f"{join(segs)}: not there")
        return parent.pop(i)
    raise ModelError(f"{join(segs[:-1])}: a value, not a container")


def insert(data, path, value, at: int | None = None) -> int:
    """Insert into the list at a path (at an index, default the end); the index it took."""
    target = get(data, path)
    if not isinstance(target, list):
        raise ModelError(f"{join(path)}: not a list")
    i = len(target) if at is None else int(at)
    target.insert(i, value)
    return i


def location(data, path) -> list:
    """The path with every list segment as an index (stable while the list does not change)."""
    out: list = []
    node = data
    for seg in split(path):
        if isinstance(node, list):
            i = _index(node, seg)
            if i is None:
                raise ModelError(f"{join(out + [seg])}: not there")
            out.append(i)
            node = node[i]
        else:
            out.append(seg)
            node = _step(node, seg, out)
    return out


def snapshot(value):
    """A copy of a value (what a command keeps to undo; MISSING stays itself)."""
    return value if value is MISSING else copy.deepcopy(value)


def _names(items: list) -> list | None:
    """The names of a list's elements when every one has its own (else None: compared by index)."""
    names = [_name(e) for e in items]
    return names if all(n is not None for n in names) and len(set(names)) == len(names) else None


def diff(a, b, path=()) -> list[tuple[str, Any, Any]]:
    """Where two plain data differ, as small as it can be said: [(path, value in a, value in b)] — MISSING where
    a key is in one only; a list whose elements changed place or number is one difference (the whole list), else
    its elements are compared (by name when they have one)."""
    if type(a) is not type(b):
        return [(join(path), a, b)] if a != b or type(a) is not type(b) else []
    if isinstance(a, dict):
        out = []
        for k in list(a) + [k for k in b if k not in a]:
            out += diff(a.get(k, MISSING), b.get(k, MISSING), (*path, k)) if k in a and k in b else \
                [(join((*path, k)), a.get(k, MISSING), b.get(k, MISSING))]
        return out
    if isinstance(a, list):
        na, nb = _names(a), _names(b)
        if len(a) != len(b) or na != nb:
            return [] if a == b else [(join(path), a, b)]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, (*path, na[i] if na is not None else i))
        return out
    return [] if a == b else [(join(path), a, b)]
