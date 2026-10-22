# Embedders

An embedder is anything with a `dim` and an `embed(texts) -> array of shape (n, dim)`.

## The offline hashing embedder (default)

`HashingEmbedder(dim=256)` hashes word and character n-gram features with keyed BLAKE2b into a fixed-size vector and L2-normalises it. It is deterministic, dependency-free and fast, which keeps the library usable offline and in tests, but it is not a semantic model: it matches on shared words and sub-words, not meaning. Its exact algorithm is documented in the class docstring and ported to TypeScript in the explainer, so vectors are identical across languages.

## Sentence-Transformers

```python
# docs: skip (downloads a model)
from poincare_memory import Memory, SentenceTransformerEmbedder

mem = Memory(embedder=SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2"))
```

`pip install "poincare-memory[embed]"`. Vectors are L2-normalised; the direction is all the memory uses.

## Any callable

```python
import numpy as np
from poincare_memory import CallableEmbedder, Memory


def my_encode(texts):  # stand-in for any model or API call
    rng = np.random.default_rng(abs(hash(tuple(texts))) % 2**32)
    return rng.normal(size=(len(texts), 1024))


emb = CallableEmbedder(fn=my_encode, dim=1024)
mem = Memory(embedder=emb)
mem.add("works with any vector source")
```

API embedders (OpenAI, Cohere, Voyage, …) fit this shape; batch them, because `add_many` sends all texts in one call.

## Hyperbolic encoders

Hierarchy Transformers (He et al. 2024, `pip install hierarchy-transformers`) produce points that already live in a Poincaré ball of radius \(\sqrt{d}\). To use them natively, construct the ball with the matching curvature (`Memory(curvature=1/d, ...)`), feed the raw outputs through a `CallableEmbedder`, and search with `metric="hyperbolic"`.

## Persistence note

`Memory.load` and `Memory.from_json` restore the hashing embedder automatically. For any other embedder, pass the same instance to `load`; the saved dimension is checked against it.
