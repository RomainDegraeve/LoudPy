"""
Test harness of loudpy: regression tests (numbers frozen from the reference code, matched by node
coordinates and MAC), verification against analytical solutions (test_verification.py), and tests of
the model, results and plotting layers.

    pytest                        the fast tests (~10 s)
    pytest --slow                 all of them, with the 3D coupling (~30 s)
    pytest -m slow                only the slow ones
    pytest --update-reference     (re)write the reference files from the current code
    pytest --slow --verification-report docs/guide/verification_report.rst
                                  write the verification report from the numbers the tests measured

References live in tests/reference/<test name>.npz and are committed. Only update them on
purpose (a deliberate change of the numerics), never to make a failing test pass.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "loudpy_numba_cache"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

from support.reference import ReferenceStore  # noqa: E402
from support.verification import Recorder  # noqa: E402

RECORDER = Recorder()


def pytest_addoption(parser):
    parser.addoption("--update-reference", action="store_true", default=False,
                     help="write the reference files from the current code instead of comparing")
    parser.addoption("--slow", action="store_true", default=False,
                     help="also run the slow tests (3D coupling)")
    parser.addoption("--verification-report", default=None, metavar="PATH",
                     help="write the verification report (reStructuredText) from the measured numbers")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: longer cases (3D coupling); run with --slow or -m slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow") or "slow" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason="slow test: run with --slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def _repo_root():
    """Run from the repository root: the example geometries use paths relative to it."""
    old = os.getcwd()
    os.chdir(ROOT)
    yield
    os.chdir(old)


MESH_DIR = ROOT / "tests" / "data" / "meshes"


def _mesh_digests() -> dict:
    import hashlib
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(MESH_DIR.glob("*.msh"))}


def pytest_sessionstart(session):
    session.config._mesh_digests = _mesh_digests() if MESH_DIR.exists() else {}


def pytest_sessionfinish(session, exitstatus):
    """A committed mesh rewritten by a test (its signature no longer matched) would silently change
    every regression number tied to it: fail the run instead."""
    before = getattr(session.config, "_mesh_digests", {})
    changed = sorted(n for n, d in _mesh_digests().items() if n in before and before[n] != d) if before else []
    if changed:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        print(f"\nERROR: committed meshes were rewritten during the tests: {changed}\n"
              f"(a model no longer matches the mesh signature). Restore them with "
              f"`git checkout tests/data/meshes` and fix the model or the signature.")


@pytest.fixture(scope="session")
def mesh_dir():
    """
    Committed meshes (tests/data/meshes): fixed inputs, loaded instead of remeshed.

    gmsh meshes with several threads are not reproducible (the same .geo gives slightly
    different triangulations from run to run, ~1e-6 on the frequencies), so the regression
    numbers are tied to these files. They are written by the first run if missing; a run that
    rewrites one fails (pytest_sessionfinish).
    """
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    return MESH_DIR


@pytest.fixture(scope="session")
def cache_dir():
    """Scratch space for intermediate results (git-ignored)."""
    d = ROOT / "tests" / ".cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def reference(request):
    """ReferenceStore bound to this test: reference.check(key=array, ...)."""
    store = ReferenceStore(ROOT / "tests" / "reference", request.node.name,
                           update=request.config.getoption("--update-reference"))
    yield store
    store.finish()


@pytest.fixture
def verify(request):
    """verify(area, case, quantity, value, bound=None, kind="max", reference="", unit="", note=""): record a
    measured number for the verification report (support/verification.py)."""
    def record(*args, **kwargs):
        RECORDER.add(request.node.nodeid, *args, **kwargs)
    return record


def pytest_sessionfinish(session, exitstatus):
    if not RECORDER.records:
        return
    RECORDER.save_json(ROOT / "tests" / ".cache" / "verification.json")
    out = session.config.getoption("--verification-report")
    if out:
        path = Path(out) if Path(out).is_absolute() else ROOT / out
        path.write_text(RECORDER.rst())


def pytest_runtest_logreport(report):
    """Keep each test's outcome for the verification report; on GitHub Actions, report each failing test as an
    annotation (readable without the job log)."""
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        RECORDER.set_outcome(report.nodeid, report.outcome)
    if os.environ.get("GITHUB_ACTIONS") != "true" or not report.failed:
        return
    path, line, _ = report.location
    text = "\n".join(str(report.longrepr).splitlines()[-25:])
    text = text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")
    sys.__stdout__.write(f"\n::error file={path},line={(line or 0) + 1},title={report.nodeid}::{text}\n")
    sys.__stdout__.flush()
