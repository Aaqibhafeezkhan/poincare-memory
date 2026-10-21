"""Embedder protocol and dependency-free implementations."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..geometry.numerics import FloatArray

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns texts into fixed-size Euclidean vectors."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> FloatArray:
        """Return an array of shape ``(len(texts), dim)``."""
        ...


@dataclass
class HashingEmbedder:
    """Deterministic, dependency-free text embedder (word + character n-gram hashing).

    It is not a semantic model; it is a stable, fast stand-in that makes the
    library usable offline and in tests. Swap in :class:`SentenceTransformerEmbedder`
    or any callable for real semantics.

    Algorithm (stable across versions and ports): lower-case the text; tokens are
    ``[\w']+``; features are ``"w:" + token`` and ``"c:" + ngram`` for character
    n-grams of ``"#" + token + "#"`` with lengths in ``ngram_range``; each feature is
    hashed with keyed BLAKE2b (digest 8 bytes, key = 4-byte little-endian ``seed``);
    index = first 4 bytes as little-endian integer mod ``dim``; sign = +1 if the
    fifth byte is odd else -1; the row is L2-normalised.
    """

    dim: int = 256
    ngram_range: tuple[int, int] = (3, 5)
    seed: int = 0
    _salt: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dim < 8:
            raise ValueError("dim must be at least 8")
        self._salt = self.seed.to_bytes(4, "little", signed=False)  # used as the BLAKE2b key

    def _features(self, text: str):  # type: ignore[no-untyped-def]
        words = _WORD_RE.findall(text.lower())
        lo, hi = self.ngram_range
        for w in words:
            yield "w:" + w
            padded = f"#{w}#"
            for n in range(lo, hi + 1):
                for i in range(0, max(0, len(padded) - n + 1)):
                    yield "c:" + padded[i : i + n]

    def embed(self, texts: Sequence[str]) -> FloatArray:
        X = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8, key=self._salt).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0
                X[row, idx] += sign
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms


@dataclass
class CallableEmbedder:
    """Wrap any ``fn(texts) -> array`` as an :class:`Embedder`."""

    fn: Callable[[Sequence[str]], FloatArray]
    dim: int

    def embed(self, texts: Sequence[str]) -> FloatArray:
        X = np.asarray(self.fn(texts), dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(
                f"embedding function returned shape {X.shape}, expected (n, {self.dim})"
            )
        return X


class SentenceTransformerEmbedder:
    """Sentence-Transformers wrapper (``pip install poincare-memory[embed]``)."""

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs: object
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "install the 'embed' extra: pip install 'poincare-memory[embed]'"
            ) from exc
        self.model = SentenceTransformer(model_name, **kwargs)
        self._dim = int(self.model.get_sentence_embedding_dimension() or 0)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> FloatArray:
        return np.asarray(
            self.model.encode(list(texts), normalize_embeddings=True), dtype=np.float64
        )
