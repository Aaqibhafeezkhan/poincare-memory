"""Mapping Euclidean embeddings into the Poincaré ball.

The library separates *direction* from *depth*: the semantic embedding gives
the direction of a point, the hierarchy gives its hyperbolic radius (depth).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import PoincareBall
from ..geometry.numerics import FloatArray, norm, safe_div


@dataclass(frozen=True)
class BallProjector:
    """Place L2-normalised Euclidean vectors in the ball at a chosen hyperbolic depth."""

    ball: PoincareBall

    def direction(self, X: FloatArray) -> FloatArray:
        X = np.asarray(X, dtype=np.float64)
        return safe_div(X, norm(X))

    def at_depth(self, X: FloatArray, depth: float | FloatArray) -> FloatArray:
        """Points with the direction of ``X`` and hyperbolic distance ``depth`` from the origin."""
        return self.ball.point_at(self.direction(X), depth)

    def expmap0(self, X: FloatArray, scale: float = 1.0) -> FloatArray:
        """Alternative: exponential map of the (scaled) raw vectors; depth = 2 * scale * ||x|| (the conformal factor at the origin is 2)."""
        return self.ball.expmap0(scale * np.asarray(X, dtype=np.float64))
