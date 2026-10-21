# Getting started

## Install

```bash
pip install poincare-memory                 # NumPy only, runs offline
pip install "poincare-memory[embed]"        # + Sentence-Transformers embedder
pip install "poincare-memory[viz,pandas]"   # + matplotlib plot, DataFrame export
pip install "poincare-memory[ann]"          # + FAISS index for large stores
```

Python 3.10 or newer. No services, no GPU. For development, see [Contributing](https://github.com/dhruvmojila/poincare-memory/blob/main/CONTRIBUTING.md).

## A first memory

```python
from poincare_memory import Memory

mem = Memory()
support = mem.add("Support knowledge base", id="kb")
api = mem.add("Public API: authentication, rate limits, pagination", parent="kb", id="api")
mem.add("API rate limit is 250 requests per second per project", parent="api", id="rate")
mem.add("Tokens expire after 60 minutes", parent="api", id="tokens")
print(mem)
print(mem.tree())
```

```text
Memory(n=4, dim=256, curvature=1, decay=NoDecay(), metric='shell')
Support knowledge base  [kb d=1.00]
└── Public API: authentication, rate limits, pagination  [api d=1.50]
    ├── API rate limit is 250 requests per second per project  [rate d=2.00]
    └── Tokens expire after 60 minutes  [tokens d=2.00]
```

Three things happened on each `add`:

1. the text was embedded (by default with the offline [hashing embedder](guides/embedders.md); swap in a real model for real semantics);
2. the point's **direction** came from that embedding, pulled a quarter of the way toward its parent's direction;
3. the point's **depth**, its hyperbolic distance from the origin, was set to the parent's depth plus `depth_step` (roots sit at `base_depth`). The `d=` values in the tree are those depths.

## A first search

```python
for r in mem.search("how fast can I call the API", k=2):
    print(f"{r.score:+.3f}  distance={r.distance:.3f}  depth={r.depth:.2f}  {r.id}: {r.text}")
```

Scores are `-distance + depth_bias * depth + time_coef * log(time_weight)`. With the default settings only the distance counts, and it is measured on the **reference shell**: the query and the memory are both moved to the root depth so that the number reflects semantic mismatch only ([why](concepts.md#scoring)).

Structural questions do not need a query:

```python
mem.search(mode="children", root="api")      # rate, tokens
mem.search(mode="ancestors", root="rate")    # api, kb
mem.search("limits", mode="subtree", root="api", k=5)
```

## Time

```python
from datetime import datetime, timedelta, timezone
from poincare_memory import HalfLife

now = datetime.now(timezone.utc)
mem = Memory(decay=HalfLife(hours=24 * 7))
mem.add("Release 2.3: rate limit 100 rps", id="old", t=now - timedelta(days=60))
mem.add("Release 2.4: rate limit 250 rps", id="new", t=now - timedelta(days=2))
print([r.id for r in mem.search("rate limit", k=2)])          # ['new', 'old']
print([r.id for r in mem.search("rate limit", k=2, at_time=now - timedelta(days=50))])  # the past
```

Facts that stop being true on a date get a `valid_until` and a `ValidityWindow` policy; see [Time and forgetting](guides/temporal.md).

## A first picture

```python
mem.inspect("memory.html")           # add open_browser=True to open it
```

One self-contained file: an interactive Poincaré disk (hover, click to light up ancestors and descendants, drag to pan hyperbolically, slide time), a sortable table and a collapsible tree. In Jupyter, a `Memory` renders the same inspector inline. See [Inspecting a memory](guides/inspection.md).

## Persist

```python
mem.save("memory.db")
again = Memory.load("memory.db")
mem.to_json("memory.json")
```

## The command line

```bash
poincare-memory demo -o demo --open       # build the anatomy demo, save demo.db, open demo.html
poincare-memory inspect memory.db         # write memory.html
poincare-memory tree memory.db
poincare-memory search memory.db "rate limit" -k 5 --mode subtree --root api
poincare-memory stats memory.db
poincare-memory export memory.db memory.json
```
