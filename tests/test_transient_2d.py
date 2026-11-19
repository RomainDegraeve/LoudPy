"""Time study (2D axisymmetric): short nonlinear transient of the coarse loudspeaker."""
import numpy as np
import pytest

from support import current_api as api


@pytest.fixture(scope="module")
def run(mesh_dir):
    return api.speaker_transient(mesh_dir, n_steps=200)[0]


def test_transient_history(run, reference):
    reference.check(time=run["time"], rtol=1e-12)
    reference.check_nodal("U_last", run["coords"], run["U"][-1], rtol=1e-7)
    reference.check_nodal("U_mid", run["coords"], run["U"][len(run["U"]) // 2], rtol=1e-7)


def test_transient_starts_at_rest(run):
    assert np.abs(run["U"][0]).max() < 1e-3 * np.abs(run["U"]).max()
