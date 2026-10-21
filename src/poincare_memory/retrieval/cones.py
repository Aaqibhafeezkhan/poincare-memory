"""Hyperbolic entailment cones (Ganea, Bécigneul & Hofmann, ICML 2018).

Every point ``x`` in the ball (outside a small central region of radius ``K``)
owns a geodesically convex cone opening toward the boundary. "``y`` lies in the
cone of ``x``" is the geometric form of "``x`` entails / is more general than
``y``". With depth encoded by the norm, cones give *ancestor* and *descendant*
queries that plain nearest-neighbour search cannot express.

Formulas (unit ball; points are scaled by ``sqrt(c)`` first):
    half-aperture   psi(x)   = arcsin(K (1 - ||x||^2) / ||x||)
    angle at x      Xi(x, y) = arccos( (<x,y>(1 + ||x||^2) - ||x||^2 (1 + ||y||^2))
                                      / (||x|| ||x - y|| sqrt(1 + ||x||^2 ||y||^2 - 2<x,y>)) )
    energy          E(x, y)  = max(0, Xi(x, y) - psi(x))     (0 means y is inside the cone)
"""

from __future__ import annotations

import numpy as np

from ..geometry import PoincareBall
from ..geometry.numerics import FloatArray, norm, safe_div

DEFAULT_K = 0.1


def half_aperture(x: FloatArray, ball: PoincareBall, K: float = DEFAULT_K) -> FloatArray:
    xs = np.asarray(x) * ball.sqrt_c
    n = norm(xs, keepdims=False)
    arg = np.clip(safe_div(K * (1 - n * n), n), -1.0, 1.0)
    psi = np.arcsin(arg)
    # inside the central region the cone is undefined: treat it as the whole space
    return np.where(n < K, np.pi, psi)


def cone_angle(x: FloatArray, y: FloatArray, ball: PoincareBall) -> FloatArray:
    xs = np.asarray(x) * ball.sqrt_c
    ys = np.asarray(y) * ball.sqrt_c
    xy = np.sum(xs * ys, axis=-1)
    x2 = np.sum(xs * xs, axis=-1)
    y2 = np.sum(ys * ys, axis=-1)
    nx = np.sqrt(np.maximum(x2, 1e-30))
    nxy = norm(xs - ys, keepdims=False)
    denom = nx * np.maximum(nxy, 1e-30) * np.sqrt(np.maximum(1 + x2 * y2 - 2 * xy, 1e-30))
    cos_xi = np.clip((xy * (1 + x2) - x2 * (1 + y2)) / denom, -1.0, 1.0)
    return np.asarray(np.arccos(cos_xi))


def cone_energy(
    x: FloatArray, y: FloatArray, ball: PoincareBall, K: float = DEFAULT_K
) -> FloatArray:
    """``max(0, Xi(x, y) - psi(x))``; zero when ``y`` is inside the entailment cone of ``x``."""
    return np.maximum(0.0, cone_angle(x, y, ball) - half_aperture(x, ball, K))


def in_cone(
    x: FloatArray, y: FloatArray, ball: PoincareBall, K: float = DEFAULT_K, tol: float = 1e-9
) -> np.ndarray:
    return cone_energy(x, y, ball, K) <= tol
