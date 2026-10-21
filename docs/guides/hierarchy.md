# Building a hierarchy

## Parents first

A memory's depth comes from its parent, so parents must exist before children. Give stable ids to structural nodes so you can refer to them later:

```python
from poincare_memory import Memory

mem = Memory()
mem.add_many([
    {"text": "Human musculoskeletal system", "id": "body"},
    {"text": "Lower limb musculature", "parent": "body", "id": "lower"},
    {"text": "Hamstring group: knee flexion and hip extension", "parent": "lower", "id": "hams"},
    {"text": "Biceps femoris: lateral hamstring", "parent": "hams", "id": "bf"},
    {"text": "Semitendinosus: medial hamstring", "parent": "hams", "id": "st"},
])
print(mem.level("bf"), round(mem.depth("bf"), 2))   # 3 2.5
```

`add_many` embeds all texts in one call, which matters with real embedding models.

## Several roots

A memory can hold several independent hierarchies; each root sits at `base_depth` with its own direction. `mem.roots()` lists them, and `search` runs across all of them unless you restrict it with `mode="subtree"`.

## Depth settings

`base_depth` (default 1.0) and `depth_step` (default 0.5) are hyperbolic units. Points must stay inside the float64-safe radius, about 12 units for curvature 1 (`max_safe_depth`), so with the defaults you have room for roughly twenty levels. A larger `depth_step` spreads levels further apart, which strengthens the separation structural modes see and weakens nothing in the shell metric.

## Re-parenting, updating, removing

```python
mem.link("st", "lower")                     # move a subtree; depths and directions are recomputed
mem.update("hams", text="Hamstrings: knee flexion and hip extension", meta={"source": "textbook"})
mem.remove("st")                            # leaves only; pass cascade=True to drop a subtree
```

`update` with new text re-embeds the memory and re-places its descendants, because their directions inherit from it.

## Inspecting the structure

```python
print(mem.tree())
print(mem.ancestors("bf"), mem.descendants("lower"), mem.siblings("bf"))
print(mem.stats()["levels"])
```

## Embedding a whole tree at once

When you have a complete taxonomy rather than a stream of memories, `sarkar_layout` gives an exact, training-free embedding and `fit_poincare_embeddings` a learned one; see [Concepts](../concepts.md#training-free-versus-learned-placement) and [Benchmarks](../benchmarks.md).
