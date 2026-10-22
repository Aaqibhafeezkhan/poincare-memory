# Structural queries

Flat vector stores answer one question: what is similar to this text. A hyperbolic memory also answers where things sit.

| Mode | Candidates | Needs `root` | Needs `query` |
|---|---|---|---|
| `nearest` (default) | every memory | no | yes |
| `subtree` | `root` and all its descendants | yes | optional |
| `children` | direct children of `root` | yes | optional |
| `ancestors` | parent, grandparent, … of `root` | yes | optional |
| `siblings` | other children of `root`'s parent | yes | optional |
| `cone` | memories inside the entailment cone of `root` | yes | optional |

Without a query, candidates are ranked by temporal weight and `depth_bias` only.

```python
from poincare_memory import Memory
from poincare_memory.demo import build_demo_memory

mem = build_demo_memory()
print([r.id for r in mem.search("knee extension", mode="subtree", root="lower", k=3)])
print([r.id for r in mem.search(mode="ancestors", root="type2x")])
print([r.id for r in mem.search(mode="siblings", root="hams")])
```

## Combining with filters

`filters` is a dict of metadata equalities or a callable on the record, applied before ranking:

```python
events = mem.search("hamstring", k=5, filters={"kind": "event"})
recent = mem.search("hamstring", k=5, filters=lambda r: r.access_count == 0)
```

## Zooming with `depth_bias`

```python
general = mem.search("calf", k=3, touch=False)  # depth_bias = 0
mem.depth_bias = 0.5  # prefer deeper, more specific memories
specific = mem.search("calf", k=3, touch=False)
mem.depth_bias = 0.0
```

Positive `depth_bias` prefers specific memories, negative prefers general ones; it is a per-memory setting you can change between calls.

## Entailment cones

`mode="cone"` is the geometric version of "everything this memory implies". With the default placement, cones are approximate (children are pulled only partly toward their parents); with `Memory(cone_constrained=True)` every child is forced inside its parent's cone and cone queries become exact for the declared hierarchy, at the cost of collapsing semantic directions. `SearchResult.cone_energy` is 0 inside the cone and grows with the violation angle.
