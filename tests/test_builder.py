"""
Phase E: the generic system builder (fields, terms as functions of s, loads) and the contributions.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from loudpy.errors import ModelError
from loudpy.physics.builder import Field, Load, SystemBuilder, Term


def small_builder(n=0.7):
    rng = np.random.default_rng(0)
    K, C, M = (sp.csr_matrix(rng.standard_normal((3, 3))) for _ in range(3))
    b = SystemBuilder()
    b.add_field(Field("u", 3))
    b.add_field(Field("I", 1))
    col = sp.csr_matrix(rng.standard_normal((3, 1)))
    b.add(Term("u", "u", K), Term("u", "u", C, power=1), Term("u", "u", M, power=2),
          Term("u", "I", -col), Term("I", "u", col.T, power=1),
          Term("I", "I", sp.csr_matrix([[6.0]])), Term("I", "I", sp.csr_matrix([[3e-4]]), func=lambda z: z ** n))
    b.add_load(Load("I", np.array([2.0])))
    return b, (K, C, M, col)


def test_at_matches_hand_assembly():
    b, (K, C, M, col) = small_builder()
    z = 1j * 2 * np.pi * 300.0
    Ze = 6.0 + z ** 0.7 * 3e-4
    A = sp.bmat([[K + z * C + z ** 2 * M, -col], [z * col.T, sp.csr_matrix([[Ze]])]]).toarray()
    np.testing.assert_allclose(b.at(z).toarray(), A, rtol=1e-15, atol=1e-15)
    np.testing.assert_array_equal(b.rhs(z), [0, 0, 0, 2.0])
    parts = b.split(np.arange(4.0))
    np.testing.assert_array_equal(parts["u"], [0, 1, 2])
    np.testing.assert_array_equal(parts["I"], [3])


def test_polynomial_form():
    b, (K, C, M, col) = small_builder(n=1.0)
    with pytest.raises(ModelError, match="not polynomial"):
        b.polynomial()                                   # the inductance is a callable even for n = 1
    b.terms[-1] = Term("I", "I", sp.csr_matrix([[3e-4]]), power=1)
    K0, C0, M0 = b.polynomial()
    z = 0.3 + 2.0j
    np.testing.assert_allclose((K0 + z * C0 + z ** 2 * M0).toarray(), b.at(z).toarray(), rtol=1e-14)


def test_checks():
    b = SystemBuilder()
    b.add_field(Field("u", 2))
    with pytest.raises(ModelError, match="twice"):
        b.add_field(Field("u", 2))
    with pytest.raises(ModelError, match="unknown field"):
        b.add(Term("u", "p", sp.eye(2)))
    with pytest.raises(ModelError, match="shape"):
        b.add(Term("u", "u", sp.eye(3)))
    with pytest.raises(ModelError, match="unknown field"):
        b.add_load(Load("p", np.zeros(2)))
    b.add_field(Field("q", 1))                               # a field with no term keeps its block
    b.add(Term("u", "u", sp.eye(2)))
    assert b.at(1j).shape == (3, 3)


def test_registry_covers_the_roles():
    """Every contribution belongs to a declared role; the forms are checked; the DtN radiation exists in
    the symmetric form only."""
    from loudpy.physics.builder import CONTRIBUTIONS, contribute
    from loudpy.physics.registry import ROLES
    assert set(CONTRIBUTIONS) <= set(ROLES)
    b = SystemBuilder()
    with pytest.raises(ValueError, match="form"):
        contribute(b, None, [], None, ["solid"], form="time")
    with pytest.raises(ModelError, match="symmetric form"):
        contribute(b, None, [], None, ["radiation"], form="frequency")


def _blocks_problem(rank_core):
    """A small two-field problem: dense q (eliminated), sparse p with a low-rank boundary term."""
    from loudpy.physics.builder import LowRank
    rng = np.random.default_rng(1)
    nq, n_p, r = 4, 30, 3
    Kp = sp.diags(4.0 + rng.random(n_p)) + sp.diags(-np.ones(n_p - 1), 1) + sp.diags(-np.ones(n_p - 1), -1)
    Bqp = sp.csr_matrix(rng.standard_normal((nq, n_p)) * (rng.random((nq, n_p)) < 0.2))
    rows = np.arange(5, 12)
    U = rng.standard_normal((len(rows), r))
    b = SystemBuilder()
    b.add_field(Field("q", nq))
    b.add_field(Field("p", n_p))
    b.add(Term("q", "q", np.diag(1.0 + np.arange(nq))), Term("q", "q", np.eye(nq), power=2),
          Term("q", "q", lambda z: np.full((nq, nq), 0.01 * z.imag), power=0),
          Term("p", "p", Kp.tocsr()), Term("p", "p", sp.eye(n_p) * 0.1, power=2),
          Term("p", "p", lambda z: LowRank((n_p, n_p), rows, U, rank_core(z))),
          Term("q", "p", Bqp), Term("p", "q", -0.5 * Bqp.T.tocsr(), power=2))
    b.add_load(Load("q", rng.standard_normal(nq) + 0j))
    dense = (Kp.toarray(), Bqp.toarray(), rows, U)
    return b, dense


def test_block_solver_matches_direct_solve():
    """Schur complement + GMRES with a LowRank block = the dense direct solve of the same system."""
    import scipy.sparse.linalg as spla
    from loudpy.physics.builder import BlockSolver
    core = lambda z: np.array([0.3, -0.2j, 0.1]) * (1 + 0.001 * z.imag)               # noqa: E731
    b, (Kp, Bqp, rows, U) = _blocks_problem(core)
    z = 1j * 0.7
    nq, n_p = 4, 30
    App = Kp + 0.1 * z ** 2 * np.eye(n_p)
    App[np.ix_(rows, rows)] += U @ np.diag(core(z)) @ U.T
    Aqq = np.diag(1.0 + np.arange(nq)) + z ** 2 * np.eye(nq) + 0.01 * z.imag
    A = np.block([[Aqq, Bqp], [-0.5 * z ** 2 * Bqp.T, App]])
    x_ref = np.linalg.solve(A, b.rhs(z))
    lu = spla.splu(sp.csc_matrix(Kp + 0.1 * z ** 2 * np.eye(n_p)))
    M = spla.LinearOperator((n_p, n_p), matvec=lu.solve, dtype=complex)
    x, it, info = BlockSolver(b, small="q", big="p", rtol=1e-12).solve(z, preconditioner=M)
    assert info == 0 and it > 0
    np.testing.assert_allclose(np.concatenate([x["q"], x["p"]]), x_ref, rtol=1e-9, atol=1e-12)
    with pytest.raises(ModelError, match="LowRank"):
        b.at(z)


def test_block_solver_small_only():
    """Only eliminated fields: a direct solve, several right-hand sides at once."""
    from loudpy.physics.builder import BlockSolver
    b = SystemBuilder()
    b.add_field(Field("q", 3))
    b.add(Term("q", "q", np.diag([1.0, 2.0, 3.0])), Term("q", "q", np.eye(3), power=2))
    z = 0.5j
    F = np.eye(3, dtype=complex)
    x, _, _ = BlockSolver(b, small="q").solve(z, rhs=F)
    np.testing.assert_allclose(x["q"], np.linalg.inv(np.diag([1.0, 2.0, 3.0]) + z ** 2 * np.eye(3)), rtol=1e-14)
