"""Matplotlib rendering of a memory on the Poincaré disk."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from .export import disk_coordinates, structural_coordinates

if TYPE_CHECKING:  # pragma: no cover
    from ..store.memory import Memory


def _geodesic_points(a: np.ndarray, b: np.ndarray, n: int = 24) -> np.ndarray:
    """Points along the unit-disk geodesic between ``a`` and ``b`` (both inside the unit disk)."""
    from ..geometry import PoincareBall

    ball = PoincareBall(1.0)
    ts = np.linspace(0.0, 1.0, n)
    return ball.geodesic(a[None, :].repeat(n, 0), b[None, :].repeat(n, 0), ts)


def plot_memory(memory: Memory, ax=None, labels: bool = False, at_time: datetime | None = None):  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ids = memory.ids
    has_links = any(memory.get(i).parent_id is not None for i in ids)
    xy = (
        structural_coordinates(memory)
        if has_links
        else disk_coordinates(memory.points, memory.ball)
    )
    idx = {i: j for j, i in enumerate(ids)}
    w = memory.time_weights(ids, at_time) if ids else np.zeros(0)
    circle = Circle((0, 0), 1.0, fill=False, lw=1.2, color="#444")
    ax.add_patch(circle)
    for i in ids:
        p = memory.get(i).parent_id
        if p is not None:
            pts = _geodesic_points(xy[idx[p]], xy[idx[i]])
            ax.plot(pts[:, 0], pts[:, 1], color="#999", lw=0.8, alpha=0.8, zorder=1)
    depth = np.array([memory.level(i) for i in ids]) if ids else np.zeros(0)
    sizes: np.ndarray = 60 * (1 - np.sum(xy * xy, axis=1)) + 8 if len(ids) else np.zeros(0)
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=depth,
        s=sizes,
        cmap="viridis",
        alpha=np.clip(0.25 + 0.75 * w, 0, 1),
        zorder=2,
    )
    if labels:
        for j, i in enumerate(ids):
            ax.annotate(memory.get(i).text[:24], xy[j], fontsize=6, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    ax.set_title(f"poincare-memory · {len(ids)} memories · c={memory.ball.c:g}")
    if len(ids):
        plt.colorbar(sc, ax=ax, fraction=0.04, label="hierarchy level")
    return ax
