"""
Where results go, and what made them — one convention for scripts and the interface:

    <simulation folder>/results/<simulation>/<study>_<YYYYmmdd_HHMMSS>.h5

    path = lp.result_path(HERE, "voltage_sweep")        that path for a new result (not written); with the study
                                                        as third argument, the study is stamped (a scene's store=)
    study.save(HERE, name="voltage_sweep")              the study's results written there, stamped with what made them
    res.save(HERE, name="voltage_sweep")                (the same for results in hand)
    res = lp.load(HERE, study="voltage_sweep")          the newest result of that study in the folder
    lp.results.files.results_of(HERE)                   every result file of the folder, newest last

The stamp (Results.provenance) is {"study", "simulation", "fingerprint"}: the fingerprint is the simulation document's
(<folder>/<name>.loudpy.json, lp.Document.fingerprint("study:<name>")) when the folder has one holding that study — the
interface and lp.Workspace then know whether the result is current; a script without a document still records its
study and simulation. The simulation is the folder's document name, else the folder's name.
"""
from __future__ import annotations

import time
from pathlib import Path

from loudpy.errors import ResultsError

RESULTS = "results"
SUFFIX = ".h5"


def _document(folder: Path):
    """The simulation document of a folder (its <name>.loudpy.json), or None."""
    from loudpy.document import SUFFIX as DOC, Document
    found = sorted(Path(folder).glob(f"*{DOC}"))
    if not found:
        return None
    try:
        return Document.load(found[0])
    except Exception:                                                # noqa: BLE001  (a document being written)
        return None


def simulation_of(folder) -> str:
    """The simulation's name: its document's, else the folder's name."""
    doc = _document(Path(folder))
    return doc.name if doc is not None else Path(folder).resolve().name


def result_path(folder, name: str, study=None, simulation: str | None = None, stamp: str | None = None) -> Path:
    """The file a new result of the study `name` goes to: <folder>/results/<simulation>/<name>_<stamp>.h5 (a free
    name: a second result in the same second gets _2 …). Not written; the folder is made. study: the study object
    that will write it (a scene solve(store=…)): stamped with what made it (Results.provenance).

        res = lp.FrequencyResponse(scene, drive=…).solve(FREQS, store=lp.result_path(HERE, "fr", study))"""
    if not str(name).strip() or any(c in str(name) for c in "/\\"):
        raise ResultsError(f"a study name for the result file, got {name!r}")
    folder = Path(folder)
    sim = simulation or simulation_of(folder)
    if study is not None:
        study.provenance = provenance(folder, name, sim)
    base = folder / RESULTS / sim
    base.mkdir(parents=True, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    path = base / f"{name}_{stamp}{SUFFIX}"
    k = 2
    while path.exists():
        path = base / f"{name}_{stamp}_{k}{SUFFIX}"
        k += 1
    return path


def provenance(folder, study: str, simulation: str | None = None) -> dict:
    """What a result of `study` saved in `folder` is stamped with: {"study", "simulation"} and, when the folder's
    document holds that study, its "fingerprint"."""
    folder = Path(folder)
    doc = _document(folder)
    out: dict = {"study": study, "simulation": simulation or (doc.name if doc is not None else folder.resolve().name)}
    if doc is not None and any(s.get("name") == study for s in doc.studies):
        try:
            out["fingerprint"] = doc.fingerprint(f"study:{study}")
        except Exception:                                            # noqa: BLE001  (a study it cannot describe)
            pass
    return out


def stamp(results, made: dict) -> None:
    """Record what made results (Results.provenance) in their study settings (and those of each run of a sweep)."""
    for r in [results, *getattr(results, "results", [])]:
        st = getattr(r, "study", None)
        if isinstance(st, dict):
            st["provenance"] = dict(made)
        elif hasattr(r, "study"):
            r.study = {"provenance": dict(made)}


def save(results, folder, name: str, simulation: str | None = None) -> Path:
    """Write results where the convention puts them, stamped (the module docstring); returns the path."""
    from loudpy.results.schema import write
    path = result_path(folder, name, simulation=simulation)
    made = getattr(results, "provenance", None) or {}
    if made.get("study") != name or "fingerprint" not in made:
        stamp(results, provenance(folder, name, simulation))
    write(results, path)
    return path


def results_of(folder, study: str | None = None, simulation: str | None = None) -> list[Path]:
    """The result files of a simulation folder (results/<simulation>/, every simulation when it has several), those
    of one study when given (by their stamp, else their name), oldest first."""
    from loudpy.results.schema import info
    folder = Path(folder)
    base = folder / RESULTS
    dirs = [base / simulation] if simulation else ([p for p in base.iterdir() if p.is_dir()] + [base]
                                                   if base.is_dir() else [])
    out = []
    for d in dirs:
        for p in sorted(d.glob(f"*{SUFFIX}")) if d.is_dir() else []:
            if study is not None:
                try:
                    made = info(p).get("provenance") or {}
                except Exception:                                    # noqa: BLE001  (a file being written)
                    continue
                if made.get("study", p.stem.rsplit("_", 2)[0]) != study:
                    continue
            out.append(p)
    return sorted(out, key=lambda p: (p.stat().st_mtime, p.name))


def latest(folder, study: str, simulation: str | None = None) -> Path:
    """The newest result file of a study in a simulation folder."""
    found = results_of(folder, study, simulation)
    if not found:
        raise FileNotFoundError(f"no result of study {study!r} in {Path(folder) / RESULTS} (save one with "
                                f"study.save(folder, name={study!r}))")
    return found[-1]


__all__ = ["RESULTS", "result_path", "provenance", "stamp", "save", "results_of", "latest", "simulation_of"]
