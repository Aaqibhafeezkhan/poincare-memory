"""Named temporal weighting policies.

The paper describes "time-weighted embeddings" that favour recent information
while keeping established knowledge reachable. Here that idea is a pluggable,
testable policy that maps a memory's age (and optionally its access history or
validity window) to a weight in ``(0, 1]``. Weights enter the retrieval score
as ``time_coef * log(weight)``, so a weight of 1 is neutral.

Forms implemented from the literature:
- exponential half-life (the ``0.995^hours`` recency of Generative Agents,
  Park et al. 2023, is ``HalfLife(hours=138.3)``);
- Ebbinghaus forgetting ``exp(-t / S)`` with stability that grows with
  accesses (MemoryBank, Zhong et al. 2024);
- validity windows (bi-temporal facts as in Graphiti/Zep).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import numpy as np

from ..geometry.numerics import FloatArray

MIN_WEIGHT = 1e-6


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_utc(t: datetime) -> datetime:
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def ages_hours(now: datetime, times: list[datetime]) -> FloatArray:
    now = to_utc(now)
    return np.array([(now - to_utc(t)).total_seconds() / 3600.0 for t in times], dtype=np.float64)


@runtime_checkable
class DecayPolicy(Protocol):
    def weight(
        self,
        now: datetime,
        times: list[datetime],
        access_counts: list[int],
        valid_from: list[datetime | None],
        valid_until: list[datetime | None],
    ) -> FloatArray:
        """Weights in ``[MIN_WEIGHT, 1]`` for each memory."""
        ...


@dataclass(frozen=True)
class NoDecay:
    def weight(self, now, times, access_counts, valid_from, valid_until):  # type: ignore[no-untyped-def]
        return np.ones(len(times), dtype=np.float64)


@dataclass(frozen=True)
class HalfLife:
    """``w = 0.5 ** (age_hours / hours)``; memories older than the future (negative age) get weight 1."""

    hours: float = 24 * 30

    def __post_init__(self) -> None:
        if self.hours <= 0:
            raise ValueError("hours must be positive")

    def weight(self, now, times, access_counts, valid_from, valid_until):  # type: ignore[no-untyped-def]
        age = np.maximum(ages_hours(now, times), 0.0)
        return np.maximum(0.5 ** (age / self.hours), MIN_WEIGHT)


def generative_agents_recency() -> HalfLife:
    """The ``0.995 ** hours`` recency used by Generative Agents, as a half-life (≈138.3 h)."""
    return HalfLife(hours=float(np.log(0.5) / np.log(0.995)))


@dataclass(frozen=True)
class Ebbinghaus:
    """``w = exp(-age_hours / S)`` with ``S = stability_hours * (1 + boost_per_access * accesses)``."""

    stability_hours: float = 24 * 7
    boost_per_access: float = 0.5

    def __post_init__(self) -> None:
        if self.stability_hours <= 0:
            raise ValueError("stability_hours must be positive")

    def weight(self, now, times, access_counts, valid_from, valid_until):  # type: ignore[no-untyped-def]
        age = np.maximum(ages_hours(now, times), 0.0)
        s = self.stability_hours * (
            1.0 + self.boost_per_access * np.asarray(access_counts, dtype=np.float64)
        )
        return np.maximum(np.exp(-age / s), MIN_WEIGHT)


@dataclass(frozen=True)
class ValidityWindow:
    """Weight 1 while ``valid_from <= now <= valid_until``, ``outside_weight`` otherwise.

    Missing bounds are open. Combine with another policy via :class:`Product`.
    """

    outside_weight: float = MIN_WEIGHT

    def weight(self, now, times, access_counts, valid_from, valid_until):  # type: ignore[no-untyped-def]
        now_u = to_utc(now)
        out = np.ones(len(times), dtype=np.float64)
        for i, (vf, vu) in enumerate(zip(valid_from, valid_until, strict=True)):
            if (vf is not None and now_u < to_utc(vf)) or (vu is not None and now_u > to_utc(vu)):
                out[i] = max(self.outside_weight, MIN_WEIGHT)
        return out


@dataclass(frozen=True)
class Product:
    """Elementwise product of several policies (e.g. ``Product(HalfLife(72), ValidityWindow())``)."""

    policies: tuple[DecayPolicy, ...]

    def __init__(self, *policies: DecayPolicy) -> None:
        object.__setattr__(self, "policies", tuple(policies))

    def weight(self, now, times, access_counts, valid_from, valid_until):  # type: ignore[no-untyped-def]
        out = np.ones(len(times), dtype=np.float64)
        for p in self.policies:
            out *= p.weight(now, times, access_counts, valid_from, valid_until)
        return np.maximum(out, MIN_WEIGHT)
