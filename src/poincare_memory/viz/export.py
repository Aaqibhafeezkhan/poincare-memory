"""Export a memory as JSON / NetworkX and project points to the 2-D disk for drawing."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from ..geometry import PoincareBall
from ..geometry.numerics import FloatArray, norm, safe_div

if TYPE_CHECKING:  # pragma: no cover
    from ..store.memory import Memory


def disk_coordinates(points: FloatArray, ball: PoincareBall) -> FloatArray:
    """2-D Poincaré-disk coordinates for drawing.

    In 2-D the points are returned unchanged (scaled to the unit disk). In
    higher dimensions the *directions* are reduced with an uncentred SVD of the
    tangent vectors at the origin (so the origin stays the origin) and every
    point keeps its exact hyperbolic depth; angles are approximate, depths are
    not.
    """
    P = np.asarray(points, dtype=np.float64)
    if P.size == 0:
        return np.zeros((0, 2))
    depth = ball.distance_to_origin(P)
    if P.shape[1] == 2:
        return P * ball.sqrt_c
    T = ball.logmap0(P)
    _, _, vt = np.linalg.svd(T, full_matrices=False)
    basis = vt[:2].T if vt.shape[0] >= 2 else np.pad(vt.T, ((0, 0), (0, 2 - vt.shape[0])))
    dirs2 = T @ basis
    unit = safe_div(dirs2, norm(dirs2))
    r = np.tanh(depth / 2.0)  # unit-disk radius for hyperbolic depth
    return unit * r[:, None]


def structural_coordinates(memory: Memory) -> FloatArray:
    """2-D disk coordinates for inspection: angles from the hierarchy, radii from the true depths.

    Every memory keeps its exact hyperbolic distance from the origin (the paper's
    depth coordinate); its angle comes from a radial tree layout in which each
    subtree owns an angular wedge proportional to its number of leaves, so
    siblings spread evenly and subtrees do not overlap. This is a drawing, not
    an isometric embedding: use the real points for distances.
    """
    ids = memory.ids
    if not ids:
        return np.zeros((0, 2))
    children = {i: memory.children(i) for i in ids}
    leaves: dict[str, int] = {}

    def count(n: str) -> int:
        if n not in leaves:
            leaves[n] = 1 if not children[n] else sum(count(c) for c in children[n])
        return leaves[n]

    roots = memory.roots()
    total = sum(count(r) for r in roots)
    angle: dict[str, float] = {}
    wedge: dict[str, tuple[float, float]] = {}
    cursor = 0.0
    for r in roots:
        span = np.pi if len(roots) == 1 else 2 * np.pi * count(r) / total
        centre = cursor + span / 2
        angle[r] = centre if len(roots) > 1 else 0.0
        wedge[r] = (angle[r] - span / 2, angle[r] + span / 2)
        cursor += span
    stack = list(roots)
    while stack:
        n = stack.pop()
        lo, hi = wedge[n]
        kids = children[n]
        if not kids:
            continue
        tot = sum(count(c) for c in kids)
        pos = lo
        for c in kids:
            span = (hi - lo) * count(c) / tot
            angle[c] = pos + span / 2
            wedge[c] = (pos, pos + span)
            pos += span
            stack.append(c)
    depth = memory.ball.distance_to_origin(memory.points)
    radius = np.tanh(depth / 2.0)
    theta = np.array([angle[i] for i in ids])
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)


def to_dict(
    memory: Memory, at_time: datetime | None = None, layout: str = "auto"
) -> dict[str, Any]:
    """Serialisable snapshot. ``layout``: ``"structural"`` (tree layout), ``"projection"`` (semantic
    directions), or ``"auto"`` (structural when any parent links exist)."""
    ids = memory.ids
    P = memory.points
    if layout not in ("auto", "structural", "projection"):
        raise ValueError("layout must be 'auto', 'structural' or 'projection'")
    has_links = any(memory.get(i).parent_id is not None for i in ids)
    use_structural = layout == "structural" or (layout == "auto" and has_links)
    xy = structural_coordinates(memory) if use_structural else disk_coordinates(P, memory.ball)
    w = memory.time_weights(ids, at_time) if ids else np.zeros(0)
    nodes = []
    for j, i in enumerate(ids):
        r = memory.get(i)
        nodes.append(
            {
                "id": i,
                "text": r.text,
                "parent": r.parent_id,
                "level": memory.level(i),
                "depth": float(memory.ball.distance_to_origin(P[j])),
                "x": float(xy[j, 0]),
                "y": float(xy[j, 1]),
                "t": r.t.isoformat(),
                "t_ms": r.t.timestamp() * 1000.0,
                "weight": float(w[j]),
                "accesses": r.access_count,
                "meta": r.meta,
            }
        )
    edges = [
        {"source": memory.get(i).parent_id, "target": i} for i in ids if memory.get(i).parent_id
    ]
    from ..store.memory import _decay_spec

    return {
        "config": {
            "curvature": memory.ball.c,
            "dim": memory.dim,
            "base_depth": memory.base_depth,
            "depth_step": memory.depth_step,
            "decay": _decay_spec(memory.decay),
            "now_ms": (at_time or memory.clock()).timestamp() * 1000.0,
            "count": len(ids),
            "layout": "structural" if use_structural else "projection",
        },
        "nodes": nodes,
        "edges": edges,
    }


def to_networkx(memory: Memory):  # type: ignore[no-untyped-def]
    """``networkx.DiGraph`` with node attributes (requires ``networkx``)."""
    import networkx as nx

    g = nx.DiGraph()
    data = to_dict(memory)
    for n in data["nodes"]:
        g.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
    for e in data["edges"]:
        g.add_edge(e["source"], e["target"])
    return g
