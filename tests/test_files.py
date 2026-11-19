"""
Result files: what a study saves must read back to the same numbers.

The first run (pytest --update-reference) also keeps one small file per study type in
tests/data/legacy/. Those files are the old format: after the refactor, the new loader must
still read them to the same numbers (test_legacy_*).
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from support import current_api as api

LEGACY = Path(__file__).parent / "data" / "legacy"


def _same_by_coords(ca, va, cb, vb, rtol=1e-10, digits=10):
    """Values at the same coordinates agree (any node ordering)."""
    ka = {tuple(np.round(c, digits)): i for i, c in enumerate(ca)}
    idx = [ka[tuple(np.round(c, digits))] for c in cb]
    a, b = np.asarray(va)[..., idx, :] if np.ndim(va) == 3 else np.asarray(va)[idx], np.asarray(vb)
    assert np.abs(a - b).max() <= rtol * np.abs(b).max()


def _keep_legacy(request, src, name):
    if request.config.getoption("--update-reference"):
        LEGACY.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, LEGACY / name)


def test_freq_file_roundtrip(mesh_dir, tmp_path, request):
    freqs = [100.0, 1000.0]
    res, study = api.speaker_freq_meca(mesh_dir, freqs)
    path = tmp_path / "freq.h5"
    study.save(path)
    back = api.read_freq_file(path)
    assert [f for f, _, _ in back] == freqs
    for (f, coords, u), u_mem in zip(back, res["u"]):
        _same_by_coords(coords, u, res["coords"], u_mem)
    _keep_legacy(request, path, "freq_meca.h5")


def test_eigen_file_roundtrip(mesh_dir, tmp_path, request):
    modes, study = api.speaker_modes(mesh_dir, f_max=800.0, orders=(0, 1))
    path = tmp_path / "eigen.h5"
    study.save(path)
    back = api.read_eigen_file(path)
    assert sorted(back) == [0, 1]
    for m, (freqs, coords, shapes) in back.items():
        np.testing.assert_allclose(np.sort(freqs), np.sort(modes[m]["freqs"]), rtol=1e-12)
        _same_by_coords(coords, shapes, modes[m]["coords"], modes[m]["shapes"])
    _keep_legacy(request, path, "eigen_fourier.h5")


def test_time_file_roundtrip(mesh_dir, tmp_path, request):
    run, study = api.speaker_transient(mesh_dir, n_steps=10)
    path = tmp_path / "time.h5"
    study.save(path)
    t, coords, U = api.read_time_file(path)
    np.testing.assert_allclose(t, run["time"], rtol=1e-14)
    _same_by_coords(coords, U, run["coords"], run["U"])
    _keep_legacy(request, path, "time.h5")


@pytest.mark.parametrize("name", ["freq_meca.h5", "eigen_fourier.h5", "time.h5"])
def test_legacy_files_readable(reference, name):
    """The kept old-format files still load to the same numbers (backward compatibility)."""
    path = LEGACY / name
    if not path.exists():
        pytest.skip("legacy files are written by pytest --update-reference")
    if name.startswith("freq"):
        for f, coords, u in api.read_freq_file(path):
            reference.check_nodal(f"u_{f:.0f}Hz", coords, u)
    elif name.startswith("eigen"):
        for m, (freqs, coords, shapes) in api.read_eigen_file(path).items():
            reference.check_modes(f"m{m}", freqs, shapes, coords, n_shapes=3)
    else:
        t, coords, U = api.read_time_file(path)
        reference.check(time=t)
        reference.check_nodal("U_last", coords, U[-1])
