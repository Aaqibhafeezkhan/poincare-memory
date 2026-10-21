"""The ``Memory`` class: a hierarchy-aware, time-aware store on the Poincaré ball."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..embed import BallProjector, Embedder, HashingEmbedder
from ..geometry import PoincareBall
from ..geometry.numerics import FloatArray
from ..index import Backend, MIPSIndex
from ..retrieval import DEFAULT_K, combine, cone_energy
from ..temporal import DecayPolicy, HalfLife, NoDecay, to_utc, utcnow
from .records import MemoryRecord, SearchResult

SearchMode = Literal["nearest", "subtree", "children", "ancestors", "siblings", "cone"]
Metric = Literal["shell", "hyperbolic"]
Filter = Mapping[str, Any] | Callable[[MemoryRecord], bool] | None

logger = logging.getLogger("poincare_memory")
FORMAT_VERSION = 1
_UNSET: Any = object()


class Memory:
    """Hierarchy-aware, time-aware memory for agents and RAG.

    Each memory gets a point in the Poincaré ball. Its *direction* comes from
    the text embedding (optionally pulled toward its parent's direction); its
    *depth* (hyperbolic distance from the origin) comes from its level in the
    hierarchy you declare with ``parent=``. Structural queries (``subtree``,
    ``ancestors``, ``siblings``, ``cone``) and the temporal policy are geometric
    and explicit, and every memory can be inspected visually.

    Scoring. ``search`` ranks memories by
    ``-distance + depth_bias * depth + time_coef * log(time_weight)``.
    With ``metric="shell"`` (default) ``distance`` is the hyperbolic distance
    between the query and the memory after both are moved to the reference
    shell at ``base_depth``: it measures semantic (angular) mismatch in
    hyperbolic units and is comparable across levels. With
    ``metric="hyperbolic"`` it is the raw distance to the memory's actual point,
    which strongly favours shallow, general memories (a root at depth 1 is close
    to everything) and is the right choice only when the embedder itself is
    hyperbolic (e.g. Hierarchy Transformers) or when generality is wanted. Use
    ``depth_bias`` to prefer specific (``> 0``) or general (``< 0``) memories
    explicitly.

    Thread safety. All mutating and searching methods take an internal
    re-entrant lock, so a ``Memory`` can be shared between threads; the
    embedder you pass must be safe to call from several threads if you do.

    Parameters
    ----------
    embedder:
        Anything with ``dim`` and ``embed(texts)``; defaults to the offline
        :class:`~poincare_memory.embed.HashingEmbedder`.
    curvature:
        Positive curvature magnitude ``c`` of the ball.
    decay:
        Temporal policy; defaults to :class:`~poincare_memory.temporal.NoDecay`.
    base_depth, depth_step:
        Hyperbolic radius of root memories and the extra radius per level.
    inherit:
        How much of the parent's direction a child inherits (0 = none, 1 = all).
    cone_constrained:
        Force children inside the parent's entailment cone. Makes ``cone``
        queries exact for the declared hierarchy but collapses semantic
        directions onto a few rays; off by default.
    cone_k:
        The ``K`` constant of entailment cones (Ganea et al. 2018).
    metric:
        Default distance for ``search``; see above.
    time_coef, depth_bias:
        Score weights; see :func:`poincare_memory.retrieval.combine`.
    clock:
        Callable returning "now"; override in tests or simulations.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        curvature: float = 1.0,
        decay: DecayPolicy | None = None,
        base_depth: float = 1.0,
        depth_step: float = 0.5,
        inherit: float = 0.25,
        cone_constrained: bool = False,
        cone_k: float = DEFAULT_K,
        metric: Metric = "shell",
        time_coef: float = 1.0,
        depth_bias: float = 0.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.embedder: Embedder = embedder or HashingEmbedder()
        if int(self.embedder.dim) < 2:
            raise ValueError("the embedder must produce vectors of dimension >= 2")
        self.ball = PoincareBall(curvature)
        self.projector = BallProjector(self.ball)
        self.decay: DecayPolicy = decay or NoDecay()
        if base_depth <= 0 or depth_step <= 0:
            raise ValueError("base_depth and depth_step must be positive")
        if not 0.0 <= inherit <= 1.0:
            raise ValueError("inherit must be in [0, 1]")
        if metric not in ("shell", "hyperbolic"):
            raise ValueError("metric must be 'shell' or 'hyperbolic'")
        self.base_depth = float(base_depth)
        self.depth_step = float(depth_step)
        self.inherit = float(inherit)
        self.cone_constrained = bool(cone_constrained)
        self.cone_k = float(cone_k)
        self.metric: Metric = metric
        self.time_coef = float(time_coef)
        self.depth_bias = float(depth_bias)
        self.clock = clock or utcnow
        self._records: dict[str, MemoryRecord] = {}
        self._order: list[str] = []
        self._points: dict[str, FloatArray] = {}
        self._embeddings: dict[str, FloatArray] = {}
        self._children: dict[str, list[str]] = {}
        self._cache: FloatArray | None = None
        self._lock = threading.RLock()
        self._index: MIPSIndex | None = None
        self._index_ids: list[str] = []
        self._index_dirty = True

    # ------------------------------------------------------------- basics
    def __repr__(self) -> str:
        return (
            f"Memory(n={len(self)}, dim={self.dim}, curvature={self.ball.c:g}, "
            f"decay={self.decay!r}, metric={self.metric!r})"
        )

    @property
    def dim(self) -> int:
        return int(self.embedder.dim)

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, id: object) -> bool:
        return id in self._records

    def __iter__(self):  # type: ignore[no-untyped-def]
        with self._lock:
            return iter([self._records[i] for i in self._order])

    @property
    def ids(self) -> list[str]:
        return list(self._order)

    @property
    def points(self) -> FloatArray:
        """All points as an ``(n, dim)`` array in insertion order."""
        with self._lock:
            if self._cache is None:
                self._cache = (
                    np.stack([self._points[i] for i in self._order])
                    if self._order
                    else np.zeros((0, self.dim))
                )
            return self._cache

    def get(self, id: str) -> MemoryRecord:
        return self._records[id]

    def point(self, id: str) -> FloatArray:
        return self._points[id]

    def depth(self, id: str) -> float:
        return float(self.ball.distance_to_origin(self._points[id]))

    def children(self, id: str) -> list[str]:
        return list(self._children.get(id, []))

    def parent(self, id: str) -> str | None:
        return self._records[id].parent_id

    def ancestors(self, id: str) -> list[str]:
        """Ancestors from the nearest parent up to the root."""
        out: list[str] = []
        cur = self._records[id].parent_id
        while cur is not None:
            out.append(cur)
            cur = self._records[cur].parent_id
        return out

    def descendants(self, id: str) -> list[str]:
        out: list[str] = []
        stack = list(self._children.get(id, []))
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(self._children.get(n, []))
        return out

    def siblings(self, id: str) -> list[str]:
        p = self._records[id].parent_id
        pool = (
            self._children.get(p, [])
            if p is not None
            else [i for i in self._order if self._records[i].parent_id is None]
        )
        return [i for i in pool if i != id]

    def roots(self) -> list[str]:
        return [i for i in self._order if self._records[i].parent_id is None]

    def level(self, id: str) -> int:
        return len(self.ancestors(id))

    def stats(self, at_time: datetime | None = None) -> dict[str, Any]:
        """Summary numbers: counts, levels, depths, timestamps, mean temporal weight."""
        with self._lock:
            ids = self._order
            levels = [self.level(i) for i in ids]
            hist: dict[int, int] = {}
            for lv in levels:
                hist[lv] = hist.get(lv, 0) + 1
            times = [self._records[i].t for i in ids]
            weights = self.time_weights(ids, at_time) if ids else np.zeros(0)
            depths = self.ball.distance_to_origin(self.points) if ids else np.zeros(0)
            return {
                "count": len(ids),
                "roots": len(self.roots()),
                "max_level": max(levels, default=0),
                "levels": dict(sorted(hist.items())),
                "dim": self.dim,
                "curvature": self.ball.c,
                "metric": self.metric,
                "decay": _decay_spec(self.decay),
                "depth_min": float(np.min(depths)) if len(ids) else None,
                "depth_max": float(np.max(depths)) if len(ids) else None,
                "oldest": min(times).isoformat() if times else None,
                "newest": max(times).isoformat() if times else None,
                "mean_time_weight": float(weights.mean()) if len(ids) else None,
                "accesses": int(sum(self._records[i].access_count for i in ids)),
            }

    # ---------------------------------------------------------------- add
    def _embed(self, texts: Sequence[str]) -> FloatArray:
        for t in texts:
            if not isinstance(t, str) or not t.strip():
                raise ValueError("memory text must be a non-empty string")
        E = np.asarray(self.embedder.embed(list(texts)), dtype=np.float64)
        if E.ndim != 2 or E.shape[0] != len(texts) or E.shape[1] != self.dim:
            raise ValueError(
                f"embedder returned shape {E.shape}, expected ({len(texts)}, {self.dim})"
            )
        if not np.all(np.isfinite(E)):
            raise ValueError("embedder returned non-finite values")
        return E

    def _insert(
        self,
        text: str,
        e: FloatArray,
        parent: str | None,
        t: datetime | None,
        id: str | None,
        meta: Mapping[str, Any] | None,
        valid_from: datetime | None,
        valid_until: datetime | None,
    ) -> str:
        if parent is not None and parent not in self._records:
            raise KeyError(f"unknown parent id {parent!r}")
        if id is None:
            id = uuid.uuid4().hex[:12]
        elif not isinstance(id, str) or not id.strip():
            raise ValueError("id must be a non-empty string")
        if id in self._records:
            raise ValueError(f"duplicate id {id!r}")
        rec = MemoryRecord(
            id=id,
            text=text,
            parent_id=parent,
            t=to_utc(t) if t is not None else self.clock(),
            meta=dict(meta or {}),
            valid_from=to_utc(valid_from) if valid_from else None,
            valid_until=to_utc(valid_until) if valid_until else None,
        )
        self._records[id] = rec
        self._order.append(id)
        self._embeddings[id] = e
        self._points[id] = self._place(e, parent)
        if parent is not None:
            self._children.setdefault(parent, []).append(id)
        self._touch_structure()
        logger.debug("add %s parent=%s level=%d", id, parent, self.level(id))
        return id

    def _touch_structure(self) -> None:
        self._cache = None
        self._index_dirty = True

    def add(
        self,
        text: str,
        parent: str | None = None,
        t: datetime | None = None,
        id: str | None = None,
        meta: Mapping[str, Any] | None = None,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
    ) -> str:
        """Store a memory; returns its id."""
        with self._lock:
            e = self._embed([text])[0]
            return self._insert(text, e, parent, t, id, meta, valid_from, valid_until)

    def add_many(self, items: Iterable[Mapping[str, Any]]) -> list[str]:
        """Store many memories with one embedding call.

        Each item is a dict of ``add`` keyword arguments; parents must precede
        their children in the iteration order. Returns the ids in order.
        """
        with self._lock:
            rows = [dict(item) for item in items]
            if not rows:
                return []
            E = self._embed([str(r.get("text", "")) for r in rows])
            ids: list[str] = []
            for r, e in zip(rows, E, strict=True):
                ids.append(
                    self._insert(
                        r["text"],
                        e,
                        r.get("parent"),
                        r.get("t"),
                        r.get("id"),
                        r.get("meta"),
                        r.get("valid_from"),
                        r.get("valid_until"),
                    )
                )
            return ids

    def update(
        self,
        id: str,
        text: str | None = None,
        meta: Mapping[str, Any] | None = None,
        t: datetime | None = None,
        valid_from: datetime | None = _UNSET,
        valid_until: datetime | None = _UNSET,
    ) -> None:
        """Change a memory in place. New ``text`` re-embeds it and re-places its subtree;
        ``meta`` replaces the metadata dict; ``valid_from`` / ``valid_until`` may be set to ``None``."""
        with self._lock:
            rec = self._records[id]
            if text is not None:
                rec.text = text
                self._embeddings[id] = self._embed([text])[0]
                self._replace_subtree(id)
            if meta is not None:
                rec.meta = dict(meta)
            if t is not None:
                rec.t = to_utc(t)
            if valid_from is not _UNSET:
                rec.valid_from = to_utc(valid_from) if valid_from else None
            if valid_until is not _UNSET:
                rec.valid_until = to_utc(valid_until) if valid_until else None
            self._touch_structure()
            logger.debug("update %s", id)

    def remove(self, id: str, cascade: bool = False) -> None:
        with self._lock:
            kids = self._children.get(id, [])
            if kids and not cascade:
                raise ValueError(f"{id!r} has children; pass cascade=True to remove the subtree")
            for k in list(kids):
                self.remove(k, cascade=True)
            rec = self._records.pop(id)
            self._order.remove(id)
            self._points.pop(id)
            self._embeddings.pop(id)
            self._children.pop(id, None)
            if rec.parent_id is not None:
                self._children[rec.parent_id].remove(id)
            self._touch_structure()
            logger.debug("remove %s", id)

    def link(self, child: str, parent: str | None) -> None:
        """Re-parent ``child`` (and re-place its subtree)."""
        with self._lock:
            if parent is not None and (parent == child or parent in self.descendants(child)):
                raise ValueError("cannot link a node under itself")
            if parent is not None and parent not in self._records:
                raise KeyError(f"unknown parent id {parent!r}")
            rec = self._records[child]
            if rec.parent_id is not None:
                self._children[rec.parent_id].remove(child)
            rec.parent_id = parent
            if parent is not None:
                self._children.setdefault(parent, []).append(child)
            self._replace_subtree(child)
            self._touch_structure()

    def _replace_subtree(self, id: str) -> None:
        rec = self._records[id]
        self._points[id] = self._place(self._embeddings[id], rec.parent_id)
        for k in self._children.get(id, []):
            self._replace_subtree(k)

    def _place(self, e: FloatArray, parent: str | None) -> FloatArray:
        if parent is None:
            return self.projector.at_depth(e, self.base_depth)
        p = self._points[parent]
        target = self.depth(parent) + self.depth_step
        raw = self.projector.at_depth(e, target)
        t_max = 1.0 - self.inherit
        point = self.ball.set_depth(self.ball.geodesic(p, raw, t_max), target)
        if self.cone_constrained and cone_energy(p, point, self.ball, self.cone_k) > 1e-9:
            lo, hi = 0.0, t_max  # t=0 lies on the parent's ray and is always inside its cone
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                cand = self.ball.set_depth(self.ball.geodesic(p, raw, mid), target)
                if cone_energy(p, cand, self.ball, self.cone_k) <= 1e-9:
                    lo = mid
                else:
                    hi = mid
            point = self.ball.set_depth(self.ball.geodesic(p, raw, lo), target)
        return point

    # ------------------------------------------------------------- search
    def _candidates(self, mode: SearchMode, root: str | None) -> list[str]:
        if mode == "nearest":
            return list(self._order)
        if root is None:
            raise ValueError(f"mode={mode!r} needs root=<memory id>")
        if root not in self._records:
            raise KeyError(f"unknown id {root!r}")
        if mode == "subtree":
            return [root, *self.descendants(root)]
        if mode == "children":
            return self.children(root)
        if mode == "ancestors":
            return self.ancestors(root)
        if mode == "siblings":
            return self.siblings(root)
        if mode == "cone":
            others = [i for i in self._order if i != root]
            if not others:
                return []
            P = np.stack([self._points[i] for i in others])
            energy = cone_energy(self._points[root][None, :], P, self.ball, self.cone_k)
            return [i for i, en in zip(others, energy, strict=True) if en <= 1e-9]
        raise ValueError(f"unknown mode {mode!r}")

    def _apply_filter(self, ids: list[str], filters: Filter) -> list[str]:
        if filters is None:
            return ids
        if callable(filters):
            return [i for i in ids if filters(self._records[i])]
        return [
            i for i in ids if all(self._records[i].meta.get(k) == v for k, v in filters.items())
        ]

    def time_weights(self, ids: Sequence[str], at_time: datetime | None = None) -> FloatArray:
        now = to_utc(at_time) if at_time else self.clock()
        recs = [self._records[i] for i in ids]
        return np.asarray(
            self.decay.weight(
                now,
                [r.t for r in recs],
                [r.access_count for r in recs],
                [r.valid_from for r in recs],
                [r.valid_until for r in recs],
            ),
            dtype=np.float64,
        )

    def embed_query(self, query: str, depth: float | None = None) -> FloatArray:
        """Point for a free-text query (direction from the embedder, depth given or ``base_depth``)."""
        e = self._embed([query])[0]
        return self.projector.at_depth(e, self.base_depth if depth is None else depth)

    # ---- index
    def build_index(self, backend: Backend = "auto") -> MIPSIndex:
        """Build (or rebuild) the shell-metric nearest-neighbour index over all memories."""
        with self._lock:
            shell = (
                self.ball.set_depth(self.points, self.base_depth)
                if self._order
                else np.zeros((0, self.dim))
            )
            self._index = MIPSIndex(self.ball, backend=backend).build(shell)
            self._index_ids = list(self._order)
            self._index_dirty = False
            logger.info("index built: backend=%s n=%d", self._index.backend, len(self._index_ids))
            return self._index

    def _index_candidates(self, q: FloatArray, k: int) -> list[str]:
        if self._index is None or self._index_dirty:
            self.build_index(self._index.backend if self._index else "auto")  # type: ignore[arg-type]
        assert self._index is not None
        idx, _ = self._index.query(q, k)
        return [self._index_ids[i] for i in idx]

    def search(
        self,
        query: str | None = None,
        k: int = 5,
        mode: SearchMode = "nearest",
        root: str | None = None,
        at_time: datetime | None = None,
        filters: Filter = None,
        metric: Metric | None = None,
        query_depth: float | None = None,
        touch: bool = True,
        use_index: bool | None = None,
        oversample: int = 8,
    ) -> list[SearchResult]:
        """Rank memories by ``-distance + depth_bias*depth + time_coef*log(time_weight)``.

        ``mode`` restricts the candidates: ``nearest`` (all), ``subtree``/``children``/
        ``ancestors``/``siblings`` of ``root``, or ``cone`` (points inside the
        entailment cone of ``root``). ``query`` may be omitted for structural modes,
        in which case candidates are ranked by time weight and depth only.
        ``metric`` overrides the memory's default (see the class docstring);
        ``query_depth`` only matters for ``metric="hyperbolic"`` (default ``base_depth``).

        ``use_index``: with ``True`` (or ``None`` once :meth:`build_index` has been
        called) a ``nearest`` + ``shell`` search first takes the ``k * oversample``
        closest memories from the index, then scores them exactly with the temporal
        policy and filters. The index is rebuilt automatically after changes.
        """
        with self._lock:
            metric = metric or self.metric
            if k < 1:
                return []
            indexed = (
                query is not None
                and mode == "nearest"
                and metric == "shell"
                and (use_index is True or (use_index is None and self._index is not None))
            )
            if indexed:
                assert query is not None
                q = self.embed_query(query, self.base_depth)
                ids = self._apply_filter(self._index_candidates(q, k * max(1, oversample)), filters)
            else:
                ids = self._apply_filter(self._candidates(mode, root), filters)
            if not ids:
                return []
            P = np.stack([self._points[i] for i in ids])
            depths = self.ball.distance_to_origin(P)
            if query is not None:
                if metric == "shell":
                    q = self.embed_query(query, self.base_depth)
                    shell = self.ball.set_depth(P, self.base_depth)
                    d = self.ball.distance(q[None, :], shell)
                else:
                    q = self.embed_query(
                        query, self.base_depth if query_depth is None else query_depth
                    )
                    d = self.ball.distance(q[None, :], P)
            else:
                d = np.zeros(len(ids))
            w = self.time_weights(ids, at_time)
            score = combine(d, w, depths, self.time_coef, self.depth_bias)
            energies = None
            if mode == "cone" and root is not None:
                energies = cone_energy(self._points[root][None, :], P, self.ball, self.cone_k)
            order = np.argsort(-score, kind="stable")[:k]
            now = self.clock()
            out: list[SearchResult] = []
            for j in order:
                rec = self._records[ids[j]]
                if touch:
                    rec.access_count += 1
                    rec.last_accessed = now
                out.append(
                    SearchResult(
                        record=rec,
                        score=float(score[j]),
                        distance=float(d[j]),
                        time_weight=float(w[j]),
                        depth=float(depths[j]),
                        cone_energy=None if energies is None else float(energies[j]),
                    )
                )
            logger.debug(
                "search mode=%s metric=%s k=%d candidates=%d indexed=%s",
                mode,
                metric,
                k,
                len(ids),
                indexed,
            )
            return out

    # ------------------------------------------------------- inspection
    def table(self, at_time: datetime | None = None) -> list[dict[str, Any]]:
        """Plain rows: id, text, parent, level, depth, time, weight, accesses, meta."""
        with self._lock:
            if not self._order:
                return []
            w = self.time_weights(self._order, at_time)
            rows = []
            for i, wi in zip(self._order, w, strict=True):
                r = self._records[i]
                rows.append(
                    {
                        "id": i,
                        "text": r.text,
                        "parent": r.parent_id,
                        "level": self.level(i),
                        "depth": round(self.depth(i), 4),
                        "t": r.t.isoformat(),
                        "weight": round(float(wi), 6),
                        "accesses": r.access_count,
                        "meta": dict(r.meta),
                    }
                )
            return rows

    def to_dataframe(self, at_time: datetime | None = None):  # type: ignore[no-untyped-def]
        """``pandas.DataFrame`` of :meth:`table` (``pip install poincare-memory[pandas]``)."""
        import pandas as pd

        return pd.DataFrame(self.table(at_time))

    def tree(self, root: str | None = None, max_text: int = 60) -> str:
        """ASCII tree of the hierarchy (all roots, or below ``root``)."""
        lines: list[str] = []

        def walk(i: str, prefix: str, last: bool, top: bool) -> None:
            r = self._records[i]
            txt = r.text if len(r.text) <= max_text else r.text[: max_text - 1] + "…"
            branch = "" if top else ("└── " if last else "├── ")
            lines.append(f"{prefix}{branch}{txt}  [{i} d={self.depth(i):.2f}]")
            kids = self._children.get(i, [])
            for n, k in enumerate(kids):
                ext = "" if top else ("    " if last else "│   ")
                walk(k, prefix + ext, n == len(kids) - 1, False)

        with self._lock:
            tops = [root] if root is not None else self.roots()
            for t in tops:
                walk(t, "", True, True)
        return "\n".join(lines)

    def inspect(
        self,
        path: str | Path | None = None,
        open_browser: bool = False,
        title: str | None = None,
        layout: str = "auto",
    ) -> str:
        """Self-contained interactive HTML (disk, table, tree). Returns the HTML; writes it if ``path``.

        ``layout="structural"`` draws the stored hierarchy as a radial tree with exact
        depths (default when parents exist); ``"projection"`` draws the semantic directions.
        """
        from ..viz.html import render_html

        with self._lock:
            html = render_html(self, title=title, layout=layout)
        if path is not None:
            Path(path).write_text(html, encoding="utf-8")
            if open_browser:
                import webbrowser

                webbrowser.open(Path(path).resolve().as_uri())
        return html

    def _repr_html_(self) -> str:
        import html as _html

        return (
            '<iframe style="width:100%;height:620px;border:1px solid #ddd;border-radius:8px" '
            f'srcdoc="{_html.escape(self.inspect(), quote=True)}"></iframe>'
        )

    def plot(self, ax=None, labels: bool = False, at_time: datetime | None = None):  # type: ignore[no-untyped-def]
        """Matplotlib Poincaré-disk plot (``pip install poincare-memory[viz]``)."""
        from ..viz.mpl import plot_memory

        return plot_memory(self, ax=ax, labels=labels, at_time=at_time)

    # -------------------------------------------------------- persistence
    def _config(self) -> dict[str, Any]:
        return {
            "format": FORMAT_VERSION,
            "curvature": self.ball.c,
            "dim": self.dim,
            "base_depth": self.base_depth,
            "depth_step": self.depth_step,
            "inherit": self.inherit,
            "cone_constrained": self.cone_constrained,
            "cone_k": self.cone_k,
            "metric": self.metric,
            "time_coef": self.time_coef,
            "depth_bias": self.depth_bias,
            "embedder": _embedder_spec(self.embedder),
            "decay": _decay_spec(self.decay),
        }

    @classmethod
    def _from_config(
        cls,
        cfg: Mapping[str, Any],
        embedder: Embedder | None,
        decay: DecayPolicy | None,
        clock: Callable[[], datetime] | None,
    ) -> Memory:
        if int(cfg.get("format", 0)) != FORMAT_VERSION:
            raise ValueError(
                f"unsupported file format {cfg.get('format')!r}; this version reads format {FORMAT_VERSION}"
            )
        embedder = embedder or _embedder_from_spec(cfg["embedder"])
        if embedder is None:
            raise ValueError(
                "this file was saved with a custom embedder: pass embedder= to load it"
            )
        if int(embedder.dim) != int(cfg["dim"]):
            raise ValueError(
                f"embedder dimension {embedder.dim} does not match the saved dimension {cfg['dim']}"
            )
        return cls(
            embedder=embedder,
            curvature=cfg["curvature"],
            decay=decay or _decay_from_spec(cfg["decay"]),
            base_depth=cfg["base_depth"],
            depth_step=cfg["depth_step"],
            inherit=cfg["inherit"],
            cone_constrained=cfg["cone_constrained"],
            cone_k=cfg["cone_k"],
            metric=cfg.get("metric", "shell"),
            time_coef=cfg["time_coef"],
            depth_bias=cfg["depth_bias"],
            clock=clock,
        )

    def _restore(
        self,
        id: str,
        text: str,
        parent_id: str | None,
        t: str,
        vf: str | None,
        vu: str | None,
        acc: int,
        la: str | None,
        meta: dict[str, Any],
        point: FloatArray,
        emb: FloatArray,
    ) -> None:
        rec = MemoryRecord(
            id=id,
            text=text,
            parent_id=parent_id,
            t=datetime.fromisoformat(t),
            meta=meta,
            valid_from=datetime.fromisoformat(vf) if vf else None,
            valid_until=datetime.fromisoformat(vu) if vu else None,
            access_count=int(acc),
            last_accessed=datetime.fromisoformat(la) if la else None,
        )
        self._records[id] = rec
        self._order.append(id)
        self._points[id] = np.asarray(point, dtype=np.float64).reshape(self.dim).copy()
        self._embeddings[id] = np.asarray(emb, dtype=np.float64).copy()
        if parent_id is not None:
            self._children.setdefault(parent_id, []).append(id)

    def save(self, path: str | Path) -> None:
        """Persist records, points and configuration to a SQLite file."""
        with self._lock:
            path = Path(path)
            if path.exists():
                path.unlink()
            con = sqlite3.connect(path)
            try:
                con.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
                con.execute(
                    "CREATE TABLE records (pos INTEGER PRIMARY KEY, id TEXT UNIQUE, text TEXT, parent_id TEXT, "
                    "t TEXT, valid_from TEXT, valid_until TEXT, access_count INTEGER, last_accessed TEXT, "
                    "meta TEXT, point BLOB, embedding BLOB)"
                )
                con.executemany(
                    "INSERT INTO config VALUES (?, ?)",
                    [(k, json.dumps(v)) for k, v in self._config().items()],
                )
                rows = []
                for pos, i in enumerate(self._order):
                    r = self._records[i]
                    rows.append(
                        (
                            pos,
                            i,
                            r.text,
                            r.parent_id,
                            r.t.isoformat(),
                            r.valid_from.isoformat() if r.valid_from else None,
                            r.valid_until.isoformat() if r.valid_until else None,
                            r.access_count,
                            r.last_accessed.isoformat() if r.last_accessed else None,
                            json.dumps(r.meta),
                            np.asarray(self._points[i], dtype=np.float64).tobytes(),
                            np.asarray(self._embeddings[i], dtype=np.float64).tobytes(),
                        )
                    )
                con.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                con.commit()
            finally:
                con.close()
            logger.info("saved %d memories to %s", len(self._order), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedder: Embedder | None = None,
        decay: DecayPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> Memory:
        """Load a file written by :meth:`save`.

        The hashing embedder and the built-in decay policies are restored
        automatically; for other embedders pass the same ``embedder`` again.
        """
        con = sqlite3.connect(Path(path))
        try:
            cfg = {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM config")}
            mem = cls._from_config(cfg, embedder, decay, clock)
            dim = int(cfg["dim"])
            for row in con.execute(
                "SELECT id, text, parent_id, t, valid_from, valid_until, access_count, last_accessed, meta, point, embedding "
                "FROM records ORDER BY pos"
            ):
                (i, text, parent_id, t, vf, vu, acc, la, meta, point, emb) = row
                mem._restore(
                    i,
                    text,
                    parent_id,
                    t,
                    vf,
                    vu,
                    acc,
                    la,
                    json.loads(meta),
                    np.frombuffer(point, dtype=np.float64).reshape(dim),
                    np.frombuffer(emb, dtype=np.float64),
                )
            mem._touch_structure()
            logger.info("loaded %d memories from %s", len(mem), path)
            return mem
        finally:
            con.close()

    def to_json(self, path: str | Path | None = None, indent: int | None = None) -> str:
        """Portable JSON (config, records, points, embeddings). Returns the text; writes it if ``path``."""
        with self._lock:
            payload = {
                "config": self._config(),
                "records": [
                    {
                        "id": i,
                        "text": r.text,
                        "parent": r.parent_id,
                        "t": r.t.isoformat(),
                        "valid_from": r.valid_from.isoformat() if r.valid_from else None,
                        "valid_until": r.valid_until.isoformat() if r.valid_until else None,
                        "access_count": r.access_count,
                        "last_accessed": r.last_accessed.isoformat() if r.last_accessed else None,
                        "meta": r.meta,
                        "point": [float(x) for x in self._points[i]],
                        "embedding": [float(x) for x in self._embeddings[i]],
                    }
                    for i in self._order
                    for r in [self._records[i]]
                ],
            }
        text = json.dumps(payload, indent=indent)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(
        cls,
        source: str | Path,
        embedder: Embedder | None = None,
        decay: DecayPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> Memory:
        """Inverse of :meth:`to_json`; ``source`` is a path or the JSON text itself."""
        text = (
            source
            if isinstance(source, str) and source.lstrip().startswith("{")
            else Path(source).read_text(encoding="utf-8")
        )
        payload = json.loads(text)
        mem = cls._from_config(payload["config"], embedder, decay, clock)
        for r in payload["records"]:
            mem._restore(
                r["id"],
                r["text"],
                r.get("parent"),
                r["t"],
                r.get("valid_from"),
                r.get("valid_until"),
                int(r.get("access_count", 0)),
                r.get("last_accessed"),
                dict(r.get("meta") or {}),
                np.asarray(r["point"], dtype=np.float64),
                np.asarray(r["embedding"], dtype=np.float64),
            )
        mem._touch_structure()
        return mem


def _embedder_spec(e: Embedder) -> dict[str, Any] | None:
    if isinstance(e, HashingEmbedder):
        return {"type": "hashing", "dim": e.dim, "ngram_range": list(e.ngram_range), "seed": e.seed}
    return {"type": type(e).__name__, "dim": int(e.dim)}


def _embedder_from_spec(spec: dict[str, Any] | None) -> Embedder | None:
    if spec and spec.get("type") == "hashing":
        return HashingEmbedder(
            dim=spec["dim"], ngram_range=tuple(spec["ngram_range"]), seed=spec["seed"]
        )
    return None


def _decay_spec(d: DecayPolicy) -> dict[str, Any]:
    from ..temporal import Ebbinghaus, Product, ValidityWindow

    if isinstance(d, NoDecay):
        return {"type": "none"}
    if isinstance(d, HalfLife):
        return {"type": "half_life", "hours": d.hours}
    if isinstance(d, Ebbinghaus):
        return {
            "type": "ebbinghaus",
            "stability_hours": d.stability_hours,
            "boost_per_access": d.boost_per_access,
        }
    if isinstance(d, ValidityWindow):
        return {"type": "validity", "outside_weight": d.outside_weight}
    if isinstance(d, Product):
        return {"type": "product", "policies": [_decay_spec(p) for p in d.policies]}
    return {"type": type(d).__name__}


def _decay_from_spec(spec: dict[str, Any]) -> DecayPolicy:
    from ..temporal import Ebbinghaus, Product, ValidityWindow

    t = spec.get("type")
    if t == "none":
        return NoDecay()
    if t == "half_life":
        return HalfLife(hours=spec["hours"])
    if t == "ebbinghaus":
        return Ebbinghaus(
            stability_hours=spec["stability_hours"], boost_per_access=spec["boost_per_access"]
        )
    if t == "validity":
        return ValidityWindow(outside_weight=spec["outside_weight"])
    if t == "product":
        return Product(*[_decay_from_spec(p) for p in spec["policies"]])
    return NoDecay()
