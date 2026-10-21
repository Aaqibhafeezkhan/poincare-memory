# Inspecting a memory

Seeing is the point of putting memories in a disk.

## The HTML inspector

```python
from poincare_memory.demo import build_demo_memory

mem = build_demo_memory()
html = mem.inspect("memory.html", title="my memory")   # open_browser=True to open it
```

One self-contained file (no CDN, no server) with three tabs:

- **Disk**: the Poincaré disk; hover a point for its text, id, level, depth, time, weight and parent; click to highlight ancestors and descendants; drag to pan hyperbolically (a Möbius translation); the **time slider** re-weights every memory under the configured policy and sets opacity accordingly; a text filter dims non-matches; "download JSON" exports the drawing data.
- **Table**: sortable columns; clicking a row selects the memory in the disk.
- **Tree**: the hierarchy as collapsible nodes with depth and weight.

### Layouts

`layout="structural"` (default when parents exist) draws a radial tree: angles come from the hierarchy (each subtree owns a wedge proportional to its leaves) and radii from each memory's **true** hyperbolic depth. It is a drawing for reading, not an isometric embedding. `layout="projection"` draws the actual semantic directions reduced to 2-D (uncentred SVD of the tangent vectors at the origin, depths preserved); in high dimensions it tends to bunch memories, which is why it is not the default.

## Jupyter

A `Memory` object renders the inspector inline (`_repr_html_`).

## Matplotlib

```python
ax = mem.plot(labels=True)        # pip install "poincare-memory[viz]"
ax.figure.savefig("memory.png", dpi=160)
```

## Text views

```python
print(mem.tree())                 # ASCII tree with depths
rows = mem.table()                # list of dicts
df = mem.to_dataframe()           # pip install "poincare-memory[pandas]"
print(mem.stats())                # counts, levels, depths, timestamps, mean weight
```

## Exporting the drawing

`poincare_memory.viz.to_dict(mem)` returns the nodes (with 2-D coordinates), edges and configuration the inspector uses; `to_networkx(mem)` builds a `networkx.DiGraph` with the same attributes.
