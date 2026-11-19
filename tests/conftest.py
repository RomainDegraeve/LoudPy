"""
Regression test harness for the loudpy refactor (docs/ARCHITECTURE.md, phase 0).

The tests freeze the numbers of the current code in small, fast cases (coarse meshes) and
check that every refactoring step reproduces them:

    pytest                        the fast tests (~10 s)
    pytest --slow                 all of them, with the 3D coupling (~30 s)
    pytest -m slow                only the slow ones
    pytest --update-reference     (re)write the reference files from the current code

References live in tests/reference/<test name>.npz and are committed. Only update them on
purpose (a deliberate change of the numerics), never to make a failing test pass.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "loudpy_numba_cache"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

from support.reference import ReferenceStore  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--update-reference", action="store_true", default=False,
                     help="write the reference files from the current code instead of comparing")
    parser.addoption("--slow", action="store_true", default=False,
                     help="also run the slow tests (3D coupling)")


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


@pytest.fixture(scope="session")
def mesh_dir():
    """
    Committed meshes (tests/data/meshes): fixed inputs, loaded instead of remeshed.

    gmsh meshes with several threads are not reproducible (the same .geo gives slightly
    different triangulations from run to run, ~1e-6 on the frequencies), so the regression
    numbers are tied to these files. They are written by the first run if missing.
    """
    d = ROOT / "tests" / "data" / "meshes"
    d.mkdir(parents=True, exist_ok=True)
    return d


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
