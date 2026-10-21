"""Standard hierarchy-embedding evaluation: reconstruction MAP and mean rank."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from ..geometry.numerics import FloatArray

N = TypeVar("N", bound=Hashable)


@dataclass
class ReconstructionResult:
    mean_average_precision: float
    mean_rank: float
    n_queries: int


def reconstruction(
    points: Mapping[N, FloatArray],
    closure: Sequence[tuple[N, N]],
    distance: Callable[[FloatArray, FloatArray], FloatArray],
) -> ReconstructionResult:
    """Reconstruction protocol of Nickel & Kiela (2017).

    For every node ``u`` with related nodes ``R(u)``: the *mean rank* of each
    related node counts only unrelated nodes that are closer to ``u`` (so hub
    nodes with many descendants are not penalised for their own descendants),
    and the *average precision* is computed over the ranking of all other nodes.
    """
    nodes = list(points.keys())
    index = {n: i for i, n in enumerate(nodes)}
    P = np.stack([points[n] for n in nodes])
    related: dict[int, set[int]] = {}
    for u, v in closure:
        related.setdefault(int(index[u]), set()).add(int(index[v]))
    aps: list[float] = []
    ranks: list[float] = []
    for qi, rel in related.items():
        d = np.asarray(distance(P[qi][None, :], P), dtype=np.float64)
        d[qi] = np.inf
        rel_idx = np.fromiter(rel, dtype=np.int64)
        is_rel = np.zeros(len(nodes), dtype=bool)
        is_rel[rel_idx] = True
        neg_d = np.sort(d[~is_rel])
        # mean rank: for each related node, 1 + the number of *unrelated* nodes that are closer
        # (the protocol of Nickel & Kiela 2017; related nodes do not count against each other)
        rel_ranks = 1 + np.searchsorted(neg_d, d[rel_idx], side="left")
        ranks.extend(rel_ranks.tolist())
        # average precision over the full ranking of all other nodes
        order = np.argsort(d, kind="stable")
        hits = is_rel[order][: len(nodes) - 1]
        cum = np.cumsum(hits)
        precision_at_hits = cum[hits] / (np.nonzero(hits)[0] + 1)
        aps.append(float(precision_at_hits.mean()) if len(precision_at_hits) else 0.0)
    return ReconstructionResult(
        mean_average_precision=float(np.mean(aps)) if aps else float("nan"),
        mean_rank=float(np.mean(ranks)) if ranks else float("nan"),
        n_queries=len(aps),
    )
