"""Numerically careful elementary functions for hyperbolic geometry.

All clamps follow the conventions used by the hyperbolic deep-learning
literature: points are kept strictly inside the ball with a dtype-dependent
margin, ``artanh`` arguments are clamped away from ±1, and ``tanh`` arguments
are clamped at ±15 (beyond which the result is 1 in double precision).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]

# Margin that keeps points strictly inside the unit ball, per dtype.
BALL_EPS: dict[type, float] = {np.float16: 1e-2, np.float32: 4e-3, np.float64: 1e-5}
# Clamp for artanh arguments, per dtype.
ATANH_CLAMP: dict[type, float] = {np.float16: 1 - 1e-3, np.float32: 1 - 1e-7, np.float64: 1 - 1e-15}
TANH_CLAMP = 15.0
TINY = 1e-15


def dtype_eps(x: FloatArray) -> float:
    """Ball margin for the dtype of ``x`` (float64 margin for unknown dtypes)."""
    return BALL_EPS.get(x.dtype.type, BALL_EPS[np.float64])


def atanh_clamp(x: FloatArray) -> float:
    return ATANH_CLAMP.get(x.dtype.type, ATANH_CLAMP[np.float64])


def artanh(x: FloatArray) -> FloatArray:
    """Inverse hyperbolic tangent with the argument clamped away from ±1."""
    x = np.asarray(x)
    c = atanh_clamp(x)
    x = np.clip(x, -c, c)
    return 0.5 * (np.log1p(x) - np.log1p(-x))


def tanh(x: FloatArray) -> FloatArray:
    """Hyperbolic tangent with the argument clamped to ±15."""
    return np.tanh(np.clip(np.asarray(x), -TANH_CLAMP, TANH_CLAMP))


def arcosh(x: FloatArray) -> FloatArray:
    """Inverse hyperbolic cosine with the argument clamped to be ≥ 1."""
    out: FloatArray = np.arccosh(np.maximum(np.asarray(x), 1.0))
    return out


def norm(x: FloatArray, keepdims: bool = True) -> FloatArray:
    """Euclidean norm along the last axis."""
    out: FloatArray = np.linalg.norm(np.asarray(x), axis=-1, keepdims=keepdims)
    return out


def safe_div(a: FloatArray, b: FloatArray) -> FloatArray:
    """Elementwise ``a / b`` with ``b`` kept away from zero (sign preserved)."""
    b = np.asarray(b)
    tiny = np.asarray(TINY, dtype=b.dtype) if np.issubdtype(b.dtype, np.floating) else TINY
    b_safe = np.where(np.abs(b) < TINY, np.copysign(tiny, b), b)
    return a / b_safe
