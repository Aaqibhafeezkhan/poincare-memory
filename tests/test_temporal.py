from datetime import datetime, timedelta, timezone

import numpy as np

from poincare_memory import (
    Ebbinghaus,
    HalfLife,
    NoDecay,
    Product,
    ValidityWindow,
    generative_agents_recency,
)

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)


def call(policy, times, access=None, vf=None, vu=None):
    n = len(times)
    return policy.weight(NOW, times, access or [0] * n, vf or [None] * n, vu or [None] * n)


def test_half_life():
    hl = HalfLife(hours=10)
    w = call(
        hl, [NOW, NOW - timedelta(hours=10), NOW - timedelta(hours=20), NOW + timedelta(hours=5)]
    )
    np.testing.assert_allclose(w, [1.0, 0.5, 0.25, 1.0])


def test_generative_agents_recency_matches_0995_per_hour():
    hl = generative_agents_recency()
    w = call(hl, [NOW - timedelta(hours=100)])
    np.testing.assert_allclose(w, [0.995**100], rtol=1e-9)


def test_ebbinghaus_access_boost():
    e = Ebbinghaus(stability_hours=24, boost_per_access=1.0)
    w0 = call(e, [NOW - timedelta(hours=24)], access=[0])[0]
    w3 = call(e, [NOW - timedelta(hours=24)], access=[3])[0]
    assert np.isclose(w0, np.exp(-1))
    assert w3 > w0


def test_validity_and_product():
    v = ValidityWindow()
    w = call(v, [NOW, NOW], vf=[None, NOW + timedelta(days=1)], vu=[NOW - timedelta(days=1), None])
    assert w[0] < 1e-5 and w[1] < 1e-5
    p = Product(HalfLife(hours=10), ValidityWindow())
    w = call(p, [NOW - timedelta(hours=10)], vf=[None], vu=[None])
    np.testing.assert_allclose(w, [0.5])
    assert np.all(call(NoDecay(), [NOW] * 3) == 1.0)
