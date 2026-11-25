"""
A simulation's workspace: its folder, its document (lp.Document — the one source of what the simulation is), every
change to it as a command (undo / redo), who wants to know about a change, the model or scene built from it, and the
results in its folder with what made them. The interface works through it; a script can too:

    ws = lp.Workspace.open("my_simulation")             a folder (its <name>.loudpy.json) or the document's path
    ws.set("studies/fr/params/freqs/to", 5000.0)        a command: checked on its node, refused (and undone) if not
    with ws.batch("crossover at 400 Hz"):               several changes, one undo step
        ws.set("circuits/xo/params/parts/hi/params/H/0/params/fc", 400.0)
        ws.set("circuits/xo/params/parts/lo/params/H/0/params/fc", 400.0)
    ws.undo(); ws.redo()
    with ws.tracking("material"):                       an edit made on the document's data (or its library file)
        …                                                directly: recorded as one undo step from what changed
    ws.on_change(lambda paths: …)                       the paths a change touched (undo / redo too)
    ws.target                                           the model or scene (built again only when it changed)
    ws.readiness()                                      what the simulation still needs (lp.readiness)
    ws.results()                                        its result files: study, current or out of date
    study, kwargs = ws.study("fr")                      its results stamped (provenance: Results.provenance)
    ws.save()                                           <name>.loudpy.json and the script generated from it

    folder/
      <name>.loudpy.json  meshes/  results/  exports/
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from loudpy import docpath, schema
from loudpy.document import SUFFIX, Document
from loudpy.errors import ModelError

SET, REMOVE, INSERT, FILE = "set", "remove", "insert", "file"   # a change's kinds: a value set (put), removed,
#                                                                inserted in a list; a file of the folder rewritten
FILE_SUFFIXES = (".json",)          # the files of its folder a document points at that edits write (a materials
FILE_LIMIT = 2 << 20                # library …): part of its state (undone); results, meshes, modes are not


@dataclass
class Change:
    """One change of the document: kind (set / remove / insert), its path, the value before and after, the place in
    a list (insert / remove) — enough to undo it exactly."""
    kind: str
    path: str
    before: Any = docpath.MISSING
    after: Any = docpath.MISSING
    index: int | None = None

    def inverse(self) -> "Change":
        """The change that undoes this one."""
        if self.kind in (SET, FILE):
            return Change(self.kind, self.path, self.after, self.before)
        if self.kind == INSERT:
            return Change(REMOVE, docpath.join(self.path, self.index), self.after, docpath.MISSING, self.index)
        if self.index is None:                                         # a dict's key removed: set again
            return Change(SET, self.path, docpath.MISSING, self.before)
        return Change(INSERT, docpath.join(docpath.split(self.path)[:-1]), docpath.MISSING, self.before, self.index)


@dataclass
class Step:
    """An undo step: its label and its changes, in order."""
    label: str
    changes: list = field(default_factory=list)


class Workspace:
    """A simulation's folder and document, edited by commands (module docstring)."""

    def __init__(self, document: Document, folder=None, path=None):
        self.document = document
        self.folder = Path(folder or document.folder or Path.cwd()).resolve()
        self.document.folder = self.folder
        self.path = Path(path) if path is not None else self.folder / f"{document.name}{SUFFIX}"
        self.undos: list[Step] = []
        self.redos: list[Step] = []
        self.limit = 100
        self._listeners: list[Callable] = []
        self._open: Step | None = None
        self._target: Any = None
        self._target_key: str | None = None
        self._version = 0                                              # changes so far (what caches follow)
        self._fingerprints: tuple = (-1, {})                          # (version, {what: fingerprint})
        self._catalog: dict = {}
        self._built: dict = {}                                         # parts built before (drivers): reused

    # ── opening ────────────────────────────────────────────────────────────────────────────────────────────────────
    @classmethod
    def open(cls, path) -> "Workspace":
        """A workspace from a simulation folder (its document) or a document file."""
        path = Path(path).resolve()
        if path.is_dir():
            docs = sorted(path.glob(f"*{SUFFIX}"))
            if not docs:
                raise ModelError(f"{path}: no simulation document (*{SUFFIX}) in it")
            path = docs[0]
        return cls(Document.load(path), path.parent, path)

    @classmethod
    def of(cls, target, folder, studies=None, circuits=None, name: str | None = None) -> "Workspace":
        """A workspace for a model or scene (and studies, circuits) in `folder`."""
        doc = Document.of(target, studies, name=name, folder=folder, circuits=circuits)
        return cls(doc, folder)

    # ── commands ───────────────────────────────────────────────────────────────────────────────────────────────────
    def set(self, path, value, check: bool = True):
        """Set the value at a path; checked on its node (lp.schema) unless check=False, refused — the document left as
        it was — when its node does not hold. The value before."""
        path = docpath.join(path)
        before = docpath.snapshot(self.document.get(path, docpath.MISSING))
        self.document.set(path, docpath.snapshot(value))
        change = Change(SET, path, before, docpath.snapshot(value))
        self._done(change, check)
        return before

    def remove(self, path):
        """Remove what is at a path; what it was."""
        path = docpath.join(path)
        segs = docpath.split(path)
        parent = self.document.get(segs[:-1])
        index = docpath.location(self.document._root(), path)[-1] if isinstance(parent, list) else None
        gone = self.document.remove(path)
        change = Change(REMOVE, path, gone, docpath.MISSING, index)
        self._done(change, check=False)
        return gone

    def insert(self, path, value, at: int | None = None, check: bool = True) -> int:
        """Insert a value into the list at a path; its index."""
        path = docpath.join(path)
        i = self.document.insert(path, docpath.snapshot(value), at)
        change = Change(INSERT, path, docpath.MISSING, docpath.snapshot(value), i)
        self._done(change, check, docpath.join(path, i))
        return i

    @contextlib.contextmanager
    def batch(self, label: str = "edit"):
        """Several changes as one undo step (a refused one undoes them all)."""
        if self._open is not None:                                     # (nested: the outer one keeps them)
            yield self
            return
        self._open = Step(label)
        try:
            yield self
        except Exception:
            step, self._open = self._open, None
            for c in reversed(step.changes):
                self._apply(c.inverse())
            self._notify([c.path for c in step.changes])
            raise
        step, self._open = self._open, None
        if step.changes:
            self._record(step)
            self._notify([c.path for c in step.changes])

    @contextlib.contextmanager
    def tracking(self, label: str = "edit"):
        """An edit made on the document's data (or its library files) directly, by code that knows its own checks:
        what changed is recorded as one undo step (none when nothing changed; an exception: put back as it was)."""
        before = self.state()
        try:
            yield self
        except Exception:
            self.put_back(before)
            raise
        self.record(label, before)

    def state(self) -> tuple:
        """The document's data and its files now (what record() compares with, put_back() restores)."""
        return docpath.snapshot(self.document._root()), self.files()

    def record(self, label: str, before: tuple) -> list[str]:
        """Record what changed since `before` (state() then) as one undo step; the paths that changed ([] and no
        step when nothing did)."""
        data, files = before
        changes = [Change(SET, path, docpath.snapshot(a), docpath.snapshot(b))
                   for path, a, b in docpath.diff(data, self.document._root())]
        now = self.files()
        for f in sorted(set(files) | set(now)):
            a, b = files.get(f, docpath.MISSING), now.get(f, docpath.MISSING)
            if a != b:
                changes.append(Change(FILE, f, a, b))
        if not changes:
            return []
        if self._open is not None:
            self._open.changes += changes
        else:
            self._record(Step(label, changes))
            self._notify([c.path for c in changes])
        return [c.path for c in changes]

    def files(self) -> dict[str, bytes]:
        """{path: content} of the files in the folder that the document points at and edits write (FILE_SUFFIXES,
        small): part of what an undo puts back."""
        out: dict[str, bytes] = {}

        def visit(node):
            if isinstance(node, dict):
                for v in node.values():
                    visit(v)
            elif isinstance(node, list):
                for v in node:
                    visit(v)
            elif isinstance(node, str) and node.endswith(FILE_SUFFIXES) and len(node) < 4096:
                p = Path(node) if Path(node).is_absolute() else self.folder / node
                try:
                    if p.is_file() and p.resolve().is_relative_to(self.folder) and p.stat().st_size < FILE_LIMIT:
                        out[str(p.resolve())] = p.read_bytes()
                except OSError:
                    pass
        visit(self.document._root())
        return out

    def put_back(self, before: tuple) -> None:
        """The document and its files as they were (state() then), no undo step."""
        data, files = before
        for k in self.document.PARTS:
            setattr(self.document, k, docpath.snapshot(data.get(k)))
        for f in set(self.files()) - set(files):
            Path(f).unlink(missing_ok=True)
        for f, content in files.items():
            Path(f).write_bytes(content)

    def take(self, document: Document) -> None:
        """The parts of another description of the same simulation (made again from what the interface edited)
        become this document's — no undo step: record() it."""
        for k in ("model", "scene", "studies", "circuits"):
            setattr(self.document, k, getattr(document, k))
        self._version += 1

    def undo(self) -> bool:
        """Undo the last step; False when there is none."""
        if not self.undos:
            return False
        step = self.undos.pop()
        for c in reversed(step.changes):
            self._apply(c.inverse())
        self.redos.append(step)
        self._notify([c.path for c in step.changes])
        return True

    def redo(self) -> bool:
        """Redo the last undone step; False when there is none."""
        if not self.redos:
            return False
        step = self.redos.pop()
        for c in step.changes:
            self._apply(c)
        self.undos.append(step)
        self._notify([c.path for c in step.changes])
        return True

    def on_change(self, callback: Callable) -> Callable:
        """Call callback(paths) after every change (a step, an undo, a redo); returns it (to remove it later)."""
        self._listeners.append(callback)
        return callback

    def off_change(self, callback: Callable) -> None:
        """Stop calling callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _done(self, change: Change, check: bool, node_path: str | None = None):
        """A change made: checked (else undone and refused), recorded, said."""
        if check:
            problems = self._check_around(node_path or change.path)
            if problems:
                self._apply(change.inverse())
                raise ModelError("; ".join(problems))
        if self._open is not None:
            self._open.changes.append(change)
            return
        self._record(Step(change.path, [change]))
        self._notify([change.path])

    def _check_around(self, path: str) -> list[str]:
        """The problems of the nearest node (a dict with a "type") holding a path."""
        segs = docpath.split(path)
        for k in range(len(segs), 0, -1):
            node = self.document.get(segs[:k], None)
            if isinstance(node, dict) and "type" in node and "params" in node:
                try:
                    return schema.check(node)
                except Exception as e:                                # noqa: BLE001  (a node the schema cannot read)
                    return [str(e)]
        return []

    def _apply(self, change: Change):
        root = self.document._root()
        if change.kind == FILE:
            if change.after is docpath.MISSING:
                Path(change.path).unlink(missing_ok=True)
            else:
                Path(change.path).write_bytes(change.after)
        elif change.kind == SET:
            if change.after is docpath.MISSING:
                segs = docpath.split(change.path)
                if len(segs) > 1 and docpath.get(root, segs, docpath.MISSING) is not docpath.MISSING:
                    docpath.remove(root, segs)
            else:
                self.document.set(change.path, docpath.snapshot(change.after))
        elif change.kind == REMOVE:
            self.document.remove(change.path)
        elif change.kind == INSERT:
            self.document.insert(change.path, docpath.snapshot(change.after), change.index)

    def _record(self, step: Step):
        self.undos.append(step)
        del self.undos[:-self.limit]
        self.redos.clear()

    def _notify(self, paths: list):
        self._version += 1
        for cb in list(self._listeners):
            cb(list(paths))

    # ── what is built from it ──────────────────────────────────────────────────────────────────────────────────────
    @property
    def target(self):
        """The model or the scene of the document, built again only when what it depends on changed (refresh(): its
        files changed outside)."""
        key = self.fingerprint("target")
        if self._target is None or key != self._target_key:
            self._target = self.build()
            self._target_key = key
        return self._target

    def fingerprint(self, what: str = "target") -> str:
        """Document.fingerprint, kept until the next change (refresh(): files changed outside)."""
        if self._fingerprints[0] != self._version:
            self._fingerprints = (self._version, {})
        found = self._fingerprints[1]
        if what not in found:
            found[what] = self.document.fingerprint(what)
        return found[what]

    def build(self):
        """A new build of the document (objects of its own: the target's are not shared); the drivers built
        before are used again."""
        return self.document.build(cache=self._built)

    def refresh(self) -> None:
        """Read the files again (a geometry or a modes file changed outside): the target is built again if they did."""
        self._fingerprints = (-1, {})
        self._catalog.clear()
        self._built.clear()

    def readiness(self) -> list:
        """What the simulation still needs (loudpy.readiness.Step), on its current model or scene."""
        try:
            target = self.target
        except Exception:                                            # noqa: BLE001  (the document says why)
            return self.document.readiness()
        return self.document.readiness(target)

    def study(self, name: str):
        """(study, solve kwargs) of a study on the current target; its results stamped with what made them."""
        return self.document.study(name, self.target)

    # ── results ────────────────────────────────────────────────────────────────────────────────────────────────────
    def results(self, folder: str = "results") -> list[dict]:
        """The result files of the simulation (results/, recursively): {"name", "path", "study", "fingerprint",
        "state"} — state "current" (made by a study as it is now), "out of date" (its study or what it depends on
        changed since), "unknown" (no provenance: made otherwise)."""
        base = self.folder / folder
        return [self.result(p) for p in (sorted(base.rglob("*.h5")) if base.exists() else [])]

    def result(self, path) -> dict:
        """One result file: {"name", "path", "study", "fingerprint", "state"} (results())."""
        path = Path(path)
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            stamp = -1
        cached = self._catalog.get(path)
        if cached is None or cached[0] != stamp:
            from loudpy.results.schema import info
            try:
                prov = info(path)["provenance"]                          # (a sweep: its first run's)
            except (OSError, ValueError, KeyError):
                prov = {}
            self._catalog[path] = cached = (stamp, prov)
        prov = cached[1]
        study = prov.get("study")
        if not prov:
            state = "unknown"
        else:
            try:
                state = "current" if self.fingerprint(f"study:{study}") == prov.get("fingerprint") \
                    else "out of date"
            except ModelError:
                state = "out of date"                                 # (its study no longer exists)
        return {"name": path.stem, "path": path, "study": study, "fingerprint": prov.get("fingerprint"),
                "state": state}

    # ── files ──────────────────────────────────────────────────────────────────────────────────────────────────────
    def save(self, script: bool = True) -> Path:
        """Write the document (and the script generated from it, <name>_ui.py); its path."""
        self.document.save(self.path)
        if script:
            text = self.document.to_script(self.folder)
            compile(text, str(self.path), "exec")
            (self.folder / f"{self.document.name}_ui.py").write_text(text)
        return self.path

    def __repr__(self):
        return f"Workspace({self.document.name!r} in {self.folder}, {len(self.undos)} undo step(s))"
