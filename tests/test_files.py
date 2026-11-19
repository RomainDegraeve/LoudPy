"""
Result files: what a study saves must read back to the same numbers.

tests/data/legacy/ holds one small file per study type in result format 1, written by the former
Problem-based studies in phase 0. They are fixed data now: the loaders must keep reading them to
the same numbers (test_legacy_*, and tests/test_results.py for lp.load).
"""
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


def test_freq_file_roundtrip(mesh_dir, tmp_path):
    freqs = [100.0, 1000.0]
    res, study = api.speaker_freq_meca(mesh_dir, freqs)
    path = tmp_path / "freq.h5"
    study.save(path)
    back = api.load_freq_file(path)
    assert [f for f, _, _ in back] == freqs
    for (f, coords, u), u_mem in zip(back, res["u"]):
        _same_by_coords(coords, u, res["coords"], u_mem)


def test_eigen_file_roundtrip(mesh_dir, tmp_path):
    modes, study = api.speaker_modes(mesh_dir, f_max=800.0, orders=(0, 1))
    path = tmp_path / "eigen.h5"
    study.save(path)
    back = api.load_eigen_file(path)
    assert sorted(back) == [0, 1]
    for m, (freqs, coords, shapes) in back.items():
        np.testing.assert_allclose(np.sort(freqs), np.sort(modes[m]["freqs"]), rtol=1e-12)
        _same_by_coords(coords, shapes, modes[m]["coords"], modes[m]["shapes"])


def test_time_file_roundtrip(mesh_dir, tmp_path):
    run, study = api.speaker_transient(mesh_dir, n_steps=10)
    path = tmp_path / "time.h5"
    study.save(path)
    t, coords, U = api.load_time_file(path)
    np.testing.assert_allclose(t, run["time"], rtol=1e-14)
    _same_by_coords(coords, U, run["coords"], run["U"])


@pytest.mark.parametrize("name", ["freq_meca.h5", "eigen_fourier.h5", "time.h5"])
def test_legacy_files_readable(reference, name):
    """The kept old-format files still load to the same numbers (backward compatibility)."""
    path = LEGACY / name
    if not path.exists():
        pytest.skip("legacy files are written by pytest --update-reference")
    if name.startswith("freq"):
        for f, coords, u in api.load_freq_file(path):
            reference.check_nodal(f"u_{f:.0f}Hz", coords, u)
    elif name.startswith("eigen"):
        for m, (freqs, coords, shapes) in api.load_eigen_file(path).items():
            reference.check_modes(f"m{m}", freqs, shapes, coords, n_shapes=3)
    else:
        t, coords, U = api.load_time_file(path)
        reference.check(time=t)
        reference.check_nodal("U_last", coords, U[-1])
