# Scaling and vector databases

## The index

The default search compares the query with every memory on the reference shell, which is fine to tens of thousands of memories. Beyond that, build an index:

```python
from poincare_memory.demo import build_demo_memory

mem = build_demo_memory()
mem.build_index()                  # NumPy by default; FAISS when installed
hits = mem.search("calf strain", k=5)
```

Once built, `nearest` searches with the shell metric take the `k * oversample` closest candidates from the index and score only those exactly (temporal policy, filters, depth bias). The index is marked stale on every change and rebuilt automatically on the next search; call `build_index()` yourself after a bulk load to control when that happens.

| Backend | Install | Behaviour |
|---|---|---|
| `numpy` | none | exact maximum inner product, vectorised |
| `faiss` | `pip install "poincare-memory[ann]"` | HNSW with inner-product metric, approximate, millions of points |

## Why a dot product is enough

Points on the ball map to the Lorentz model (`PoincareBall.to_lorentz`), where the distance is \(d = \operatorname{arcosh}(-c\,\langle X, Y\rangle_L)\), a decreasing function of the Lorentzian inner product. Negating the time coordinate of every stored vector (`to_mips_vectors`) turns \(\langle X, Y\rangle_L\) into an ordinary dot product, so any maximum-inner-product index returns exact hyperbolic nearest neighbours. No custom metric is needed in Qdrant, Milvus, Weaviate, pgvector, LanceDB or FAISS.

```python
import numpy as np
from poincare_memory import PoincareBall, to_mips_vectors

ball = mem.ball
shell = ball.set_depth(mem.points, mem.base_depth)        # what the shell metric compares
stored = to_mips_vectors(ball.to_lorentz(shell))          # put these in your dot-product index
q = ball.to_lorentz(mem.embed_query("calf strain"))        # query with the raw Lorentz vector
scores = stored @ q                                        # larger = closer
top = np.argsort(-scores)[:5]
```

Store `stored` with the memory ids as payload; at query time take the top candidates from the database and let `Memory` score them with the temporal policy.

## Costs

Placement is O(level) per `add`; `add_many` embeds in one batch. Exact search is O(n · dim). Memory per point is `dim` float64 for the point plus `dim` for the embedding.
