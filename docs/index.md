# poincare-memory

**Hierarchy-aware, time-aware memory for LLM agents and RAG, on the Poincaré ball.**

Flat vector stores put every memory in Euclidean space, where a hierarchy of a few thousand concepts has nowhere to go. Hyperbolic space has exponentially more room near its boundary, which is exactly the shape of a hierarchy. `poincare-memory` places each memory in the Poincaré ball with its **direction from the text embedding** and its **depth from its place in the hierarchy**, weights it by **age** at query time, answers **structural questions** (subtree, ancestors, siblings, entailment cones) geometrically, and lets you **see** all of it in one self-contained HTML file.

It is the official open-source reference implementation of the geometry-aware memory component of

> D. Mojila, "LLM Orchestration with Geometry-Aware Memory," *2026 2nd International Conference on Big Data & Machine Learning (ICBDML)*, Bhopal, India, Feb. 2026, pp. 1–7. doi:[10.1109/ICBDML68582.2026.11544690](https://doi.org/10.1109/ICBDML68582.2026.11544690)

!!! note "Provenance"
    This package was written after the paper, from the paper's description and the primary literature. The system used for the paper's experiments is not released, and every number in these docs comes from code in this repository.

## In one minute

```python
from datetime import datetime, timedelta, timezone
from poincare_memory import Memory, HalfLife

mem = Memory(decay=HalfLife(hours=24 * 30))  # observations fade with a 30-day half-life

body = mem.add("Human musculoskeletal system")
lower = mem.add("Lower limb musculature", parent=body)
hams = mem.add("Hamstring group: knee flexion and hip extension", parent=lower)
bf = mem.add("Biceps femoris: lateral hamstring", parent=hams)
mem.add(
    "Athlete A reported tight hamstrings after 400 m repeats",
    parent=hams,
    t=datetime.now(timezone.utc) - timedelta(days=30),
)

for r in mem.search("hamstring tightness", k=3):
    print(f"{r.score:6.2f}  depth={r.depth:.1f}  w={r.time_weight:.2f}  {r.text}")

mem.search("knee", mode="subtree", root=lower)  # only the lower-limb branch
mem.search(mode="ancestors", root=bf)  # hams → lower → body
mem.inspect("memory.html")  # interactive disk, table and tree
mem.save("memory.db")  # one SQLite file
```

## What you get

| Need | How |
|---|---|
| Hierarchy encoded in geometry | depth = hyperbolic distance from the origin; a child sits `depth_step` beyond its parent |
| Structural retrieval | `mode="subtree" / "children" / "ancestors" / "siblings" / "cone"` |
| Temporal weighting | `HalfLife`, `Ebbinghaus`, `ValidityWindow`, `Product`; evaluate at any `at_time` |
| Visual inspection | `inspect()` HTML, `plot()`, `tree()`, `table()`, `to_dataframe()`, inline in Jupyter |
| Exact tree embedding | `sarkar_layout` (Sarkar's construction) with a float64 safety check |
| Learned embeddings | `fit_poincare_embeddings` (Riemannian SGD) plus a Euclidean baseline and `reconstruction()` |
| Scale | `build_index()` with a NumPy or FAISS backend; the Lorentz dot-product trick for any vector database |
| Persistence | `save()` / `load()` (SQLite), `to_json()` / `from_json()` |
| Batches and updates | `add_many()`, `update()`, `remove()`, `link()`, `stats()` |

## Where to go next

- [Getting started](getting-started.md): install, first memory, first search, first picture.
- [Concepts](concepts.md): the geometry, how depth and direction are chosen, how scoring works and why.
- [Guides](guides/hierarchy.md): hierarchy, time, structural queries, inspection, persistence, scaling, embedders, agents.
- [Paper mapping](paper-mapping.md) and [Benchmarks](benchmarks.md).
- The companion package [`poincare-orchestra`](https://github.com/dhruvmojila/poincare-orchestra) implements the paper's controller, ensemble and synthesis on top of this memory.
