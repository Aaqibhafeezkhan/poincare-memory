import numpy as np
import pytest
from hypothesis import HealthCheck, settings

from poincare_memory import PoincareBall

settings.register_profile(
    "ci",
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.load_profile("ci")


@pytest.fixture(params=[1.0, 0.5, 2.0], ids=["c=1", "c=0.5", "c=2"])
def ball(request):
    return PoincareBall(request.param)


def ball_points(
    ball: PoincareBall, n: int, dim: int, seed: int, max_radius: float = 3.0
) -> np.ndarray:
    return ball.random_points(n, dim, max_hyperbolic_radius=max_radius, seed=seed)
