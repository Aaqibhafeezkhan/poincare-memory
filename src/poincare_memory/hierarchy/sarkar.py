"""Training-free tree embedding by Sarkar's combinatorial construction.

Sarkar (2011) showed that any tree embeds in the hyperbolic plane with
arbitrarily low distortion: put the root at the origin, place its children on
a circle of hyperbolic radius ``tau`` at evenly spaced angles, then for every
child move it to the origin by an isometry, place its own children on a circle
around it while avoiding the direction of its parent, and move everything back.
Sala, De Sa, Gu & Ré (ICML 2018) analysed the construction and extended it to
higher dimensions by spreading child directions on a sphere.

This module implements both from the papers. Larger ``tau`` lowers distortion
but pushes deep nodes toward the boundary; :func:`max_safe_depth` tells you how
deep a tree can go before floating-point projection starts compressing it.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Mapping, Sequence
from typing import TypeVar

import numpy as np

from ..geometry import PoincareBall
from ..geometry.numerics import FloatArray, dtype_eps, norm, safe_div
from .tree import children_map

N = TypeVar("N", bound=Hashable)


def max_safe_depth(ball: PoincareBall, tau: float, dtype: type = np.float64) -> float:
    """Largest total hyperbolic radius representable before the ball margin clips points.

    Returns the number of ``tau``-length steps from the origin that stay strictly
    inside ``(1 - eps) / sqrt(c)``.
    """
    eps = dtype_eps(np.zeros(1, dtype=dtype))
    max_radius = 2.0 / ball.sqrt_c * np.arctanh(1.0 - eps)
    return float(max_radius / tau)


def spread_directions(
    k: int, dim: int, avoid: FloatArray | None = None, seed: int = 0, iterations: int = 200
) -> FloatArray:
    """``k`` unit vectors in ``R^dim`` spread apart (and away from ``avoid`` if given).

    In two dimensions the angles are exact: children fill the circle evenly, with
    the parent's direction occupying one slot. In higher dimensions a seeded
    repulsion on the sphere (a Thomson-style relaxation) spreads the directions.
    """
    if k <= 0:
        return np.zeros((0, dim))
    if dim < 2:
        raise ValueError("dim must be at least 2")
    if dim == 2:
        if avoid is None:
            angles = 2 * np.pi * np.arange(k) / k
        else:
            base = float(np.arctan2(avoid[1], avoid[0]))
            angles = base + 2 * np.pi * np.arange(1, k + 1) / (k + 1)
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)

    rng = np.random.default_rng(seed)
    pts: FloatArray = rng.normal(size=(k, dim))
    pts = safe_div(pts, norm(pts))
    fixed = None if avoid is None else safe_div(np.asarray(avoid, dtype=np.float64), norm(avoid))
    step = 0.1
    for _ in range(iterations):
        allpts = pts if fixed is None else np.vstack([pts, fixed[None, :]])
        diff = pts[:, None, :] - allpts[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist[:, :k], np.inf)
        force = np.sum(diff / (dist[..., None] ** 3 + 1e-9), axis=1)
        # project the force onto the tangent plane of the sphere and step
        force -= np.sum(force * pts, axis=1, keepdims=True) * pts
        pts = pts + step * force
        pts = safe_div(pts, norm(pts))
        step *= 0.99
    return pts


def sarkar_layout(
    edges: Sequence[tuple[N, N]],
    ball: PoincareBall | None = None,
    dim: int = 2,
    tau: float = 1.0,
    edge_length: Mapping[tuple[N, N], float] | None = None,
    root: N | None = None,
    seed: int = 0,
) -> dict[N, FloatArray]:
    """Embed a rooted tree given as ``(parent, child)`` edges.

    Every child is placed at hyperbolic distance ``tau * edge_length[(parent, child)]``
    from its parent (``edge_length`` defaults to 1). Returns ``node -> point``.
    A forest is handled by treating the listed ``root`` (or the single root) as
    the origin; pass a virtual root edge set if you have several roots.
    """
    ball = ball or PoincareBall()
    children, roots = children_map(edges)
    if root is None:
        if len(roots) != 1:
            raise ValueError(
                f"expected exactly one root, found {len(roots)}: pass root= explicitly"
            )
        root = roots[0]
    edge_length = edge_length or {}
    from .tree import depths as _depths

    deepest = max(_depths(children, root).values(), default=0)
    if deepest * tau > 0.95 * max_safe_depth(ball, 1.0):
        warnings.warn(
            f"tree depth {deepest} x tau {tau:g} exceeds the float64-safe radius "
            f"({max_safe_depth(ball, 1.0):.1f}); deep nodes will be clipped and distances compressed. "
            "Reduce tau, the depth, or use a smaller curvature.",
            RuntimeWarning,
            stacklevel=2,
        )
    pos: dict[N, FloatArray] = {root: np.zeros(dim)}
    stack: list[tuple[N, N | None]] = [(root, None)]
    while stack:
        node, parent = stack.pop()
        kids = children.get(node, [])
        if not kids:
            continue
        x = pos[node]
        avoid = None
        if parent is not None:
            p0 = ball.mobius_add(-x, pos[parent])  # parent in the frame centred on ``node``
            if float(norm(p0, keepdims=False)) > 1e-12:
                avoid = p0
        dirs = spread_directions(len(kids), dim, avoid=avoid, seed=seed)
        for kid, u in zip(kids, dirs, strict=True):
            r = tau * float(edge_length.get((node, kid), 1.0))
            q0 = ball.point_at(u, r)
            pos[kid] = ball.mobius_add(x, q0)  # translate back: the map y -> x ⊕ y sends 0 to x
            stack.append((kid, node))
    return pos
