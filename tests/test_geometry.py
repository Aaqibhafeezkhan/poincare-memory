import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from poincare_memory import PoincareBall
from poincare_memory.geometry import lorentz_distance, lorentz_inner, mips_scores, to_mips_vectors

from .conftest import ball_points

dims = st.integers(min_value=2, max_value=8)
seeds = st.integers(min_value=0, max_value=10_000)


@given(dim=dims, seed=seeds)
def test_distance_formulas_agree(ball, dim, seed):
    P = ball_points(ball, 20, dim, seed)
    Q = ball_points(ball, 20, dim, seed + 1)
    d1 = ball.distance(P, Q)
    d2 = ball.distance_arcosh(P, Q)
    np.testing.assert_allclose(d1, d2, rtol=1e-8, atol=1e-8)


@given(dim=dims, seed=seeds)
def test_metric_axioms(ball, dim, seed):
    P = ball_points(ball, 12, dim, seed)
    D = ball.pairwise_distance(P, P)
    np.testing.assert_allclose(D, D.T, atol=1e-9)
    assert np.all(np.diag(D) < 1e-9)
    assert np.all(D >= -1e-12)
    # triangle inequality
    for i in range(len(P)):
        for j in range(len(P)):
            assert np.all(D[i, j] <= D[i] + D[:, j] + 1e-8)


@given(dim=dims, seed=seeds)
def test_exp_log_origin_inverse(ball, dim, seed):
    P = ball_points(ball, 10, dim, seed)
    np.testing.assert_allclose(ball.expmap0(ball.logmap0(P)), P, atol=1e-9)
    V = ball.logmap0(P)
    # the Riemannian norm at the origin is lambda_0 * ||v|| = 2 ||v||
    np.testing.assert_allclose(
        ball.distance_to_origin(P), 2 * np.linalg.norm(V, axis=-1), atol=1e-9
    )


@given(dim=dims, seed=seeds)
def test_exp_log_at_point_inverse(ball, dim, seed):
    X = ball_points(ball, 10, dim, seed, max_radius=2.0)
    Y = ball_points(ball, 10, dim, seed + 7, max_radius=2.0)
    V = ball.logmap(X, Y)
    np.testing.assert_allclose(ball.expmap(X, V), Y, atol=1e-8)
    # the norm of the log map in the metric at X equals the distance
    lam = ball.conformal_factor(X)[..., 0]
    np.testing.assert_allclose(lam * np.linalg.norm(V, axis=-1), ball.distance(X, Y), atol=1e-8)


@given(dim=dims, seed=seeds)
def test_geodesic_endpoints_and_additivity(ball, dim, seed):
    X = ball_points(ball, 5, dim, seed)
    Y = ball_points(ball, 5, dim, seed + 3)
    np.testing.assert_allclose(ball.geodesic(X, Y, 0.0), X, atol=1e-9)
    np.testing.assert_allclose(ball.geodesic(X, Y, 1.0), Y, atol=1e-8)
    M = ball.geodesic(X, Y, 0.4)
    np.testing.assert_allclose(
        ball.distance(X, M) + ball.distance(M, Y), ball.distance(X, Y), atol=1e-7
    )


@given(dim=dims, seed=seeds)
def test_mobius_identities(ball, dim, seed):
    X = ball_points(ball, 6, dim, seed)
    zero = np.zeros_like(X)
    np.testing.assert_allclose(ball.mobius_add(zero, X), X, atol=1e-12)
    np.testing.assert_allclose(ball.mobius_add(X, zero), X, atol=1e-12)
    np.testing.assert_allclose(ball.mobius_add(-X, X), zero, atol=1e-12)
    # left translation is an isometry
    Y = ball_points(ball, 6, dim, seed + 1)
    A = ball_points(ball, 1, dim, seed + 2)
    np.testing.assert_allclose(
        ball.distance(ball.mobius_add(A, X), ball.mobius_add(A, Y)), ball.distance(X, Y), atol=1e-7
    )


def test_point_at_and_set_depth(ball):
    d = np.array([3.0, 4.0])
    for r in [0.1, 1.0, 5.0]:
        p = ball.point_at(d, r)
        assert np.isclose(ball.distance_to_origin(p), r)
        assert np.allclose(p / np.linalg.norm(p), d / 5.0)
    q = ball.set_depth(ball.point_at(d, 2.0), 0.7)
    assert np.isclose(ball.distance_to_origin(q), 0.7)


def test_project_keeps_points_inside(ball):
    far = np.array([[10.0, 0.0], [0.0, -3.0]])
    p = ball.project(far)
    assert np.all(np.linalg.norm(p, axis=1) < ball.radius)
    inside = np.array([[0.1, 0.1]])
    np.testing.assert_array_equal(ball.project(inside), inside)


def test_float32_support():
    ball = PoincareBall(1.0)
    P = ball.random_points(5, 3, seed=1).astype(np.float32)
    Q = ball.random_points(5, 3, seed=2).astype(np.float32)
    d = ball.distance(P, Q)
    assert d.dtype == np.float32
    np.testing.assert_allclose(
        d, ball.distance(P.astype(np.float64), Q.astype(np.float64)), rtol=1e-4
    )


@given(dim=dims, seed=seeds)
def test_lorentz_round_trip_and_distance(ball, dim, seed):
    P = ball_points(ball, 8, dim, seed)
    Q = ball_points(ball, 8, dim, seed + 5)
    L = ball.to_lorentz(P)
    np.testing.assert_allclose(lorentz_inner(L, L), -1.0 / ball.c, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(ball.from_lorentz(L), P, atol=1e-9)
    np.testing.assert_allclose(
        lorentz_distance(L, ball.to_lorentz(Q), ball.c), ball.distance(P, Q), atol=1e-7
    )


def test_mips_ordering_matches_hyperbolic_nn(ball):
    P = ball_points(ball, 200, 4, seed=11)
    q = ball_points(ball, 1, 4, seed=12)[0]
    stored = to_mips_vectors(ball.to_lorentz(P))
    scores = mips_scores(stored, ball.to_lorentz(q))
    d = ball.distance(q[None, :], P)
    assert list(np.argsort(-scores)[:10]) == list(np.argsort(d)[:10])


def test_invalid_curvature():
    with pytest.raises(ValueError):
        PoincareBall(0.0)
