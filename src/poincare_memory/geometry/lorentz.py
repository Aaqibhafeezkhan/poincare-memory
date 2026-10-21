"""Lorentz (hyperboloid) model helpers.

The hyperboloid of curvature ``-c`` is ``{X in R^{d+1} : <X, X>_L = -1/c, X_0 > 0}``
with the Lorentzian inner product ``<X, Y>_L = -X_0 Y_0 + sum_i X_i Y_i``.

Why it matters for retrieval: the geodesic distance is
``d(X, Y) = (1/sqrt(c)) arcosh(-c <X, Y>_L)``, a decreasing function of the
Lorentzian inner product. Negating the time coordinate of every stored vector
turns ``<X, Y>_L`` into an ordinary dot product, so any maximum-inner-product
(dot-product) ANN index returns exact hyperbolic nearest neighbours. This is
the trick used by hyperbolic RAG systems and documented by vector-database
vendors; no custom metric is needed.
"""

from __future__ import annotations

import numpy as np

from . import numerics as num
from .numerics import FloatArray


def lorentz_inner(X: FloatArray, Y: FloatArray) -> FloatArray:
    """Lorentzian inner product along the last axis."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    return -X[..., 0] * Y[..., 0] + np.sum(X[..., 1:] * Y[..., 1:], axis=-1)


def lorentz_distance(X: FloatArray, Y: FloatArray, curvature: float = 1.0) -> FloatArray:
    """Geodesic distance on the hyperboloid of curvature ``-curvature``."""
    inner = lorentz_inner(X, Y)
    out: FloatArray = num.arcosh(-curvature * inner) / np.sqrt(curvature)
    return out


def to_mips_vectors(X: FloatArray) -> FloatArray:
    """Negate the time coordinate so that ``dot(to_mips_vectors(X), Y) == <X, Y>_L``.

    Store ``to_mips_vectors(points)`` in a dot-product index and query with the
    raw hyperboloid coordinates of the query point; the largest dot products are
    the hyperbolic nearest neighbours.
    """
    X = np.array(X, copy=True)
    X[..., 0] = -X[..., 0]
    return X


def mips_scores(stored_mips: FloatArray, query: FloatArray) -> FloatArray:
    """Dot-product scores (larger = closer) between MIPS-prepared stored vectors and a query."""
    return np.asarray(np.asarray(stored_mips) @ np.asarray(query))
