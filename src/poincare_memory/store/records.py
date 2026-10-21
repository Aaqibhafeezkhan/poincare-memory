"""Data records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..temporal.policies import utcnow


@dataclass
class MemoryRecord:
    """One stored memory. Positions live in the :class:`~poincare_memory.Memory` that owns it."""

    id: str
    text: str
    parent_id: str | None = None
    t: datetime = field(default_factory=utcnow)
    meta: dict[str, Any] = field(default_factory=dict)
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    access_count: int = 0
    last_accessed: datetime | None = None


@dataclass
class SearchResult:
    record: MemoryRecord
    score: float
    distance: float
    time_weight: float
    depth: float
    cone_energy: float | None = None

    @property
    def id(self) -> str:
        return self.record.id

    @property
    def text(self) -> str:
        return self.record.text

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.record.id,
            "text": self.record.text,
            "parent": self.record.parent_id,
            "score": self.score,
            "distance": self.distance,
            "time_weight": self.time_weight,
            "depth": self.depth,
            "cone_energy": self.cone_energy,
            "t": self.record.t.isoformat(),
            "meta": dict(self.record.meta),
        }
