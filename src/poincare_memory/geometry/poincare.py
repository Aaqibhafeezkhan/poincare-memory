"""The Poincaré ball model of hyperbolic space with curvature ``-c`` (``c > 0``).

The ball has radius ``1/sqrt(c)``. With ``c = 1`` the formulas reduce to the
unit ball used in the paper (Section III-A): the metric tensor
``g_x = (2 / (1 - ||x||^2))^2 g^E`` and the distance

    d(x, y) = arcosh(1 + 2 ||x - y||^2 / ((1 - ||x||^2)(1 - ||y||^2))).

Everything is vectorised over leading dimensions and works in float32 or
float64 (float64 is the default throughout the library).

References (implemented from the papers, not from any third-party code):
- Nickel & Kiela, "Poincaré Embeddings for Learning Hierarchical
  Representations", NeurIPS 2017.
- Ganea, Bécigneul & Hofmann, "Hyperbolic Neural Networks", NeurIPS 2018
  (Möbius addition, exponential and logarithmic maps, conformal factor).
- Ungar, "A Gyrovector Space Approach to Hyperbolic Geometry", 2008.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import numerics as num
from .numerics import FloatArray


@dataclass(frozen=True)
class PoincareBall:
    """Poincaré ball of curvature ``-curvature``.

    Parameters
    ----------
    curvature:
        Positive number ``c``; the space has constant sectional curvature
        ``-c`` and the ball has Euclidean radius ``1/sqrt(c)``.
    """

    curvature: float = 1.0

    def __post_init__(self) -> None:
        if not (self.curvature > 0):
            raise ValueError("curvature must be a positive number")

    # ------------------------------------------------------------------ basics
    @property
    def c(self) -> float:
        return float(self.curvature)

    @property
    def sqrt_c(self) -> float:
        return float(np.sqrt(self.curvature))

    @property
    def radius(self) -> float:
        """Euclidean radius of the ball."""
        return 1.0 / self.sqrt_c

    def project(self, x: FloatArray) -> FloatArray:
        """Clip points to lie strictly inside the ball (dtype-dependent margin)."""
        x = np.asarray(x)
        eps = num.dtype_eps(x)
        maxnorm = (1.0 - eps) / self.sqrt_c
        n = num.norm(x)
        scaled = num.safe_div(x, n) * maxnorm
        return np.where(n > maxnorm, scaled, x)

    def is_inside(self, x: FloatArray) -> np.ndarray:
        """Boolean mask: which points lie inside the open ball."""
        return num.norm(np.asarray(x), keepdims=False) < self.radius

    def conformal_factor(self, x: FloatArray) -> FloatArray:
        """``lambda_x = 2 / (1 - c ||x||^2)``, the metric's conformal factor."""
        x = np.asarray(x)
        return 2.0 / np.maximum(1.0 - self.c * np.sum(x * x, axis=-1, keepdims=True), num.TINY)

    # ------------------------------------------------------------ gyrovectors
    def mobius_add(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Möbius addition ``x ⊕_c y`` (non-commutative, non-associative)."""
        x = np.asarray(x)
        y = np.asarray(y)
        c = self.c
        xy = np.sum(x * y, axis=-1, keepdims=True)
        x2 = np.sum(x * x, axis=-1, keepdims=True)
        y2 = np.sum(y * y, axis=-1, keepdims=True)
        numer = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c * c * x2 * y2
        return self.project(num.safe_div(numer, np.maximum(denom, num.TINY)))

    def mobius_scalar_mul(self, r: float | FloatArray, x: FloatArray) -> FloatArray:
        """Möbius scalar multiplication ``r ⊗_c x`` (scales hyperbolic distance from 0 by r)."""
        x = np.asarray(x)
        n = num.norm(x)
        coef = num.tanh(np.asarray(r) * num.artanh(self.sqrt_c * n)) / self.sqrt_c
        return self.project(num.safe_div(x, n) * coef)

    # --------------------------------------------------------------- distances
    def distance(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Geodesic distance via the gyrovector formula.

        ``d_c(x, y) = (2 / sqrt(c)) artanh(sqrt(c) || (-x) ⊕_c y ||)``.
        """
        diff = self.mobius_add(-np.asarray(x), np.asarray(y))
        n = num.norm(diff, keepdims=False)
        return 2.0 / self.sqrt_c * num.artanh(self.sqrt_c * n)

    def distance_arcosh(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Geodesic distance via the arcosh formula (paper eq. 3, generalised to curvature c).

        ``d_c(x, y) = (1 / sqrt(c)) arcosh(1 + 2c ||x - y||^2 / ((1 - c||x||^2)(1 - c||y||^2)))``.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        c = self.c
        d2 = np.sum((x - y) ** 2, axis=-1)
        ax = np.maximum(1 - c * np.sum(x * x, axis=-1), num.TINY)
        ay = np.maximum(1 - c * np.sum(y * y, axis=-1), num.TINY)
        return num.arcosh(1 + 2 * c * d2 / (ax * ay)) / self.sqrt_c

    def distance_to_origin(self, x: FloatArray) -> FloatArray:
        """``d_c(0, x) = (2 / sqrt(c)) artanh(sqrt(c) ||x||)`` — the depth coordinate."""
        n = num.norm(np.asarray(x), keepdims=False)
        return 2.0 / self.sqrt_c * num.artanh(self.sqrt_c * n)

    def pairwise_distance(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """All-pairs distances: ``x`` of shape (n, d), ``y`` of shape (m, d) → (n, m)."""
        return self.distance(np.asarray(x)[:, None, :], np.asarray(y)[None, :, :])

    # ------------------------------------------------------- exp and log maps
    def expmap0(self, v: FloatArray) -> FloatArray:
        """Exponential map at the origin: tangent vector → point in the ball."""
        v = np.asarray(v)
        n = num.norm(v)
        coef = np.where(n > num.TINY, num.safe_div(num.tanh(self.sqrt_c * n), self.sqrt_c * n), 1.0)
        return self.project(coef * v)

    def logmap0(self, y: FloatArray) -> FloatArray:
        """Logarithmic map at the origin: point in the ball → tangent vector."""
        y = np.asarray(y)
        n = num.norm(y)
        coef = np.where(
            n > num.TINY, num.safe_div(num.artanh(self.sqrt_c * n), self.sqrt_c * n), 1.0
        )
        return coef * y

    def expmap(self, x: FloatArray, v: FloatArray) -> FloatArray:
        """Exponential map at ``x``: ``x ⊕_c (tanh(sqrt(c) λ_x ||v|| / 2) v / (sqrt(c) ||v||))``."""
        x = np.asarray(x)
        v = np.asarray(v)
        n = num.norm(v)
        lam = self.conformal_factor(x)
        coef = np.where(
            n > num.TINY,
            num.safe_div(num.tanh(self.sqrt_c * lam * n / 2.0), self.sqrt_c * n),
            lam / 2.0,
        )
        return self.mobius_add(x, coef * v)

    def logmap(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Logarithmic map at ``x`` (inverse of :meth:`expmap`)."""
        x = np.asarray(x)
        y = np.asarray(y)
        u = self.mobius_add(-x, y)
        n = num.norm(u)
        lam = self.conformal_factor(x)
        coef = np.where(
            n > num.TINY,
            num.safe_div(2.0 * num.artanh(self.sqrt_c * n), self.sqrt_c * lam * n),
            2.0 / lam,
        )
        return coef * u

    def geodesic(self, x: FloatArray, y: FloatArray, t: float | FloatArray) -> FloatArray:
        """Point at fraction ``t`` along the geodesic from ``x`` (t=0) to ``y`` (t=1)."""
        return self.expmap(x, np.asarray(t)[..., None] * self.logmap(x, y))

    def point_at(self, direction: FloatArray, hyperbolic_radius: float | FloatArray) -> FloatArray:
        """Point at hyperbolic distance ``hyperbolic_radius`` from the origin along ``direction``."""
        d = np.asarray(direction)
        unit = num.safe_div(d, num.norm(d))
        r = num.tanh(self.sqrt_c * np.asarray(hyperbolic_radius)[..., None] / 2.0) / self.sqrt_c
        return self.project(unit * r)

    def set_depth(self, x: FloatArray, hyperbolic_radius: float | FloatArray) -> FloatArray:
        """Move ``x`` radially so that its distance from the origin equals ``hyperbolic_radius``."""
        return self.point_at(x, hyperbolic_radius)

    # ----------------------------------------------------------- Lorentz model
    def to_lorentz(self, x: FloatArray) -> FloatArray:
        """Map ball points to the hyperboloid ``{X : <X,X>_L = -1/c, X_0 > 0}``.

        Returns an array with one extra leading coordinate (the time coordinate).
        """
        x = np.asarray(x)
        c = self.c
        x2 = np.sum(x * x, axis=-1, keepdims=True)
        denom = np.maximum(1 - c * x2, num.TINY)
        x0 = (1 + c * x2) / (self.sqrt_c * denom)
        xs = 2 * x / denom
        return np.concatenate([x0, xs], axis=-1)

    def from_lorentz(self, X: FloatArray) -> FloatArray:
        """Inverse of :meth:`to_lorentz`."""
        X = np.asarray(X)
        x0 = X[..., :1]
        xs = X[..., 1:]
        return self.project(xs / (1 + self.sqrt_c * x0))

    # ------------------------------------------------------- optimisation aids
    def egrad2rgrad(self, x: FloatArray, grad: FloatArray) -> FloatArray:
        """Rescale a Euclidean gradient to the Riemannian gradient: ``((1 - c||x||^2)^2 / 4) grad``."""
        x = np.asarray(x)
        lam = self.conformal_factor(x)
        return grad / (lam * lam)

    def retraction(self, x: FloatArray, v: FloatArray) -> FloatArray:
        """First-order retraction ``proj(x + v)`` used by Riemannian SGD (Nickel & Kiela 2017)."""
        return self.project(np.asarray(x) + np.asarray(v))

    def random_points(
        self, n: int, dim: int, max_hyperbolic_radius: float = 2.0, seed: int | None = None
    ) -> FloatArray:
        """Uniform-in-direction points with hyperbolic radius uniform in ``[0, max_hyperbolic_radius]``."""
        rng = np.random.default_rng(seed)
        directions = rng.normal(size=(n, dim))
        radii = rng.uniform(0.0, max_hyperbolic_radius, size=n)
        return self.point_at(directions, radii)
