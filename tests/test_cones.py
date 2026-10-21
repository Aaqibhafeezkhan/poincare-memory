import numpy as np

from poincare_memory import PoincareBall, in_cone
from poincare_memory.retrieval import cone_energy, half_aperture


def test_point_on_same_ray_is_inside_cone():
    ball = PoincareBall()
    x = ball.point_at(np.array([1.0, 0.3]), 1.0)
    y = ball.point_at(np.array([1.0, 0.3]), 2.5)
    assert in_cone(x, y, ball)
    assert not in_cone(y, x, ball)  # cones open toward the boundary, so the reverse fails


def test_far_angle_outside_cone_and_central_region():
    ball = PoincareBall(0.5)
    x = ball.point_at(np.array([1.0, 0.0]), 1.0)
    y = ball.point_at(np.array([-1.0, 0.0]), 2.0)
    assert cone_energy(x, y, ball) > 0
    centre = np.array([0.01, 0.0])
    assert half_aperture(centre, ball) == np.pi  # undefined cone treated as everything
    assert in_cone(centre, y, ball)
