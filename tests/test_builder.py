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
    with pytest.raises(ModelError, match="symmetric form only"):
        contribute(b, None, [], None, ["radiation"], form="frequency")
