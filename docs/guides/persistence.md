# Persistence and export

## SQLite

```python
from poincare_memory import Memory
from poincare_memory.demo import build_demo_memory

mem = build_demo_memory()
mem.save("memory.db")
mem2 = Memory.load("memory.db")
assert mem2.ids == mem.ids
```

`save` writes one file with the configuration, every record, its point and its embedding. `load` restores the hashing embedder and the built-in temporal policies automatically; if you used another embedder, pass the same one again (`Memory.load(path, embedder=my_embedder)`), and the dimension is checked.

## JSON

```python
text = mem.to_json()  # portable, human-readable
mem.to_json("memory.json", indent=1)
mem3 = Memory.from_json("memory.json")
```

JSON is the format to use for sharing a memory, versioning it in git, or moving it between machines; SQLite is faster for large stores.

## From the command line

```bash
poincare-memory export memory.db memory.json
poincare-memory import memory.json memory.db
poincare-memory stats memory.db
```

## Format version

Files carry a `format` number. A newer library refuses files it cannot read with a clear error rather than guessing.
