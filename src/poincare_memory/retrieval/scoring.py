"""Combining geometry and time into one retrieval score."""

from __future__ import annotations

import numpy as np

from ..geometry.numerics import FloatArray
from ..temporal.policies import MIN_WEIGHT


def combine(
    distances: FloatArray,
    time_weights: FloatArray | None = None,
    depths: FloatArray | None = None,
    time_coef: float = 1.0,
    depth_bias: float = 0.0,
) -> FloatArray:
    """``score = -distance + depth_bias * depth + time_coef * log(time_weight)``.

    Larger is better. ``depth_bias > 0`` prefers specific (deep) memories,
    ``< 0`` prefers general ones; ``time_coef`` scales how strongly the temporal
    policy counts (1.0 means a half-life halving of the weight costs ``ln 2``
    in score, the same as moving ``ln 2 ≈ 0.69`` further away in the ball).
    """
    score = -np.asarray(distances, dtype=np.float64)
    if depths is not None and depth_bias != 0.0:
        score = score + depth_bias * np.asarray(depths, dtype=np.float64)
    if time_weights is not None and time_coef != 0.0:
        score = score + time_coef * np.log(
            np.maximum(np.asarray(time_weights, dtype=np.float64), MIN_WEIGHT)
        )
    return score
