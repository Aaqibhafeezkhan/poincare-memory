"""Nearest-neighbour indexes for the reference-shell metric.

Semantic search in :class:`~poincare_memory.Memory` compares the query with every
memory on the root shell. For large stores that comparison is offloaded to a
maximum-inner-product index: shell points are mapped to the Lorentz model and
their time coordinate negated (:func:`~poincare_memory.geometry.to_mips_vectors`),
after which the largest dot products are exactly the hyperbolic nearest
neighbours. Two backends:

- ``numpy``: exact, vectorised, fine to a few hundred thousand points;
- ``faiss``: ``IndexHNSWFlat`` with inner-product metric (``pip install poincare-memory[ann]``),
  approximate, for millions of points.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .geometry import PoincareBall, to_mips_vectors
from .geometry.numerics import FloatArray

Backend = Literal["auto", "numpy", "faiss"]


class MIPSIndex:
    """Maximum-inner-product index over Lorentz-mapped points."""

    def __init__(
        self, ball: PoincareBall, backend: Backend = "auto", hnsw_m: int = 32, ef_search: int = 64
    ) -> None:
        self.ball = ball
        self.hnsw_m = hnsw_m
        self.ef_search = ef_search
        if backend == "auto":
            try:
                import faiss  # noqa: F401

                backend = "faiss"
            except ImportError:
                backend = "numpy"
        if backend not in ("numpy", "faiss"):
            raise ValueError("backend must be 'auto', 'numpy' or 'faiss'")
        self.backend: str = backend
        self._vectors: FloatArray | None = None
        self._faiss: Any = None
        self.size = 0

    def build(self, points: FloatArray) -> MIPSIndex:
        """Index ``points`` (ball coordinates, shape ``(n, dim)``)."""
        P = np.asarray(points, dtype=np.float64)
        self.size = len(P)
        if self.size == 0:
            self._vectors = np.zeros((0, P.shape[1] + 1 if P.ndim == 2 else 1))
            self._faiss = None
            return self
        vecs = to_mips_vectors(self.ball.to_lorentz(P))
        self._vectors = vecs
        if self.backend == "faiss":
            import faiss

            d = vecs.shape[1]
            index = faiss.IndexHNSWFlat(d, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = self.ef_search
            index.add(np.ascontiguousarray(vecs, dtype=np.float32))
            self._faiss = index
        return self

    def query(self, point: FloatArray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Indices and inner-product scores of the ``k`` nearest stored points to a ball point."""
        if self._vectors is None or self.size == 0 or k <= 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        k = min(k, self.size)
        q = self.ball.to_lorentz(np.asarray(point, dtype=np.float64))
        if self.backend == "faiss" and self._faiss is not None:
            scores, idx = self._faiss.search(np.ascontiguousarray(q[None, :], dtype=np.float32), k)
            keep = idx[0] >= 0
            return idx[0][keep].astype(np.int64), scores[0][keep].astype(np.float64)
        scores = self._vectors @ q
        idx = np.argpartition(-scores, k - 1)[:k]
        order = np.argsort(-scores[idx], kind="stable")
        idx = idx[order]
        return idx.astype(np.int64), scores[idx]
