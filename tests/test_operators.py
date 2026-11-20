"""
Assembled operators on the committed meshes (structure, air with PML, coupling, coil force,
Fourier order 1, nonlinear tangent). Phase 2 of the refactor moves the element kernels and
the assemblers: these numbers must not change (rtol 1e-10, the cross-platform floating-point spread).
"""
import pytest

from support import current_api as api

from support.reference import RTOL_OPERATOR as RTOL


@pytest.fixture(scope="module")
def ops(mesh_dir):
    return api.assembled_operators(mesh_dir)


@pytest.mark.parametrize("name", ["K", "M", "C", "F", "H", "Q", "Cam", "K_m1", "M_m1", "C_m1",
                                  "KT", "M_nl", "C_nl", "Fint", "rho", "free", "T_m1"])
def test_operator(ops, reference, name):
    d = ops[name]
    if "Ax" in d:
        reference.check_nodal("Ax", d["coords"], d["Ax"], rtol=RTOL)
    for k in ("forms", "nnz", "abs_sum", "value"):
        if k in d:
            reference.check(rtol=RTOL, **{k: d[k]})


def test_dtn_system(mesh_dir, reference):
    """Structure + air + exact DtN radiation: the symmetric (Ks, Ms, Cs) system of EigenStudy."""
    d = api.dtn_system(mesh_dir)
    reference.check(rtol=RTOL, **d)


def test_dtn_system_blocks(mesh_dir, reference):
    """The coupled DtN system block by block (structure, air, coupling, radiation, auxiliaries)."""
    for key, inv in api.dtn_system_blocks(mesh_dir).items():
        reference.check(rtol=RTOL, **{key: inv})
