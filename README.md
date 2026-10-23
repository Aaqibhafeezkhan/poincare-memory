# poincare-memory

**Hierarchy-aware, time-aware memory for LLM agents and RAG, on the Poincaré ball.**

[![CI](https://github.com/dhruvmojila/poincare-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/dhruvmojila/poincare-memory/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/poincare-memory.svg)](https://pypi.org/project/poincare-memory/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/IEEE%20ICBDML%202026-10.1109%2FICBDML68582.2026.11544690-00629B)](https://doi.org/10.1109/ICBDML68582.2026.11544690)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23551721.svg)](https://doi.org/10.5281/zenodo.23551721)

`poincare-memory` is the official open-source reference implementation of the **geometry-aware memory** component of

> D. Mojila, "LLM Orchestration with Geometry-Aware Memory," *2026 2nd International Conference on Big Data & Machine Learning (ICBDML)*, Bhopal, India, Feb. 2026, pp. 1–7. doi:[10.1109/ICBDML68582.2026.11544690](https://doi.org/10.1109/ICBDML68582.2026.11544690)

Vector stores treat every memory as a point in flat space, so a taxonomy of a few thousand concepts ends up crowded and shapeless. Hyperbolic space has exponentially more room near its boundary, which is exactly the shape of a hierarchy: this library puts each memory in the Poincaré ball with its **direction from the text embedding** and its **depth from its place in the hierarchy**, weights it by **age** at query time, and lets you **see** all of it.

> **Provenance.** This package was written after the paper, from the paper's description and the primary literature. The system used for the paper's experiments is not released, and the numbers in this repository's benchmarks come only from the code here.

![Interactive inspector: Poincaré disk, time slider, table and tree views](https://raw.githubusercontent.com/dhruvmojila/poincare-memory/main/docs/assets/inspector-demo.gif)

## Install

```bash
pip install poincare-memory                 # NumPy only
pip install "poincare-memory[embed]"        # + sentence-transformers embedder
pip install "poincare-memory[viz,pandas]"   # + matplotlib plot, DataFrame export
```

Python ≥ 3.10. No GPU, no services.

## Thirty seconds

```python
from datetime import datetime, timedelta, timezone
from poincare_memory import Memory, HalfLife

mem = Memory(decay=HalfLife(hours=24 * 30))  # memories lose weight with a 30-day half-life

body = mem.add("Human musculoskeletal system")
lower = mem.add("Lower limb musculature", parent=body)
hams = mem.add("Hamstring group: knee flexion and hip extension", parent=lower)
bf = mem.add("Biceps femoris: lateral hamstring", parent=hams)
mem.add(
    "Athlete A reported tight hamstrings after 400 m repeats",
    parent=hams,
    t=datetime.now(timezone.utc) - timedelta(days=30),
)
mem.add(
    "Athlete A: flexibility improved after two weeks of eccentric work",
    parent=hams,
    t=datetime.now(timezone.utc) - timedelta(days=3),
)

for r in mem.search("hamstring tightness", k=3):
    print(f"{r.score:6.2f}  depth={r.depth:.1f}  w={r.time_weight:.2f}  {r.text}")

mem.search("knee", mode="subtree", root=lower)  # only the lower-limb branch
mem.search(mode="ancestors", root=bf)  # hams → lower → body
mem.inspect("memory.html", open_browser=True)  # disk + table + tree, offline, self-contained
mem.save("memory.db")  # SQLite; Memory.load("memory.db")
```

The default embedder is a deterministic hashing embedder so everything runs offline; pass `embedder=SentenceTransformerEmbedder()` or any callable for real semantics.

## What you get

| Need | How |
|---|---|
| Hierarchy encoded in geometry | depth = hyperbolic distance from the origin; children sit at `depth_step` beyond their parent, pulled toward its direction by `inherit` |
| Structural retrieval | `mode="subtree" / "children" / "ancestors" / "siblings"` and `mode="cone"` (entailment cones, Ganea et al. 2018) |
| Temporal weighting | named policies: `HalfLife`, `Ebbinghaus` (access-strengthened), `ValidityWindow`, `Product`; evaluate at any `at_time` |
| Visual inspection | `mem.inspect()` → one HTML file with an interactive Poincaré disk, a sortable table and a collapsible tree; `mem.plot()` for matplotlib; `mem.tree()` / `mem.table()` / `mem.to_dataframe()` in the terminal; Jupyter renders `mem` inline |
| Exact tree embedding | `sarkar_layout(edges)`: Sarkar's training-free construction (2-D exact, n-D spread) with a float64 safety check |
| Learned embeddings | `fit_poincare_embeddings(closure)`: Riemannian SGD (Nickel & Kiela 2017) in NumPy, plus a Euclidean twin for fair baselines, and `reconstruction()` (MAP, mean rank) |
| Any vector database | `to_lorentz` + `to_mips_vectors` turn hyperbolic nearest-neighbour search into a plain dot-product query |
| Persistence | `save()` / `load()` to a single SQLite file; `to_json()` / `from_json()` for portability |
| Scale | `build_index()` with NumPy or FAISS; `add_many()` embeds in one batch; `update()`, `stats()`; thread-safe |

## How scoring works (and why not raw hyperbolic distance)

`search` ranks by `-distance + depth_bias * depth + time_coef * log(time_weight)`.

By default `distance` is measured **on a reference shell**: the query and the memory are both moved to the root depth and the hyperbolic distance between them measures semantic mismatch in hyperbolic units, comparable across levels. The raw distance to a memory's real point (`metric="hyperbolic"`) is also available, but be aware of the geometry: the centre is close to *everything*. From a query on the root shell (depth 1), a root memory 90° away is at distance 1.51, while a memory at depth 3 is at least 2.0 away even when it points exactly along the query, so it can never win; a memory at depth 2 wins only within about 36° of the query's direction. With off-the-shelf text embeddings that almost never happens, so raw hyperbolic distance degenerates into "return the most general memory". The shell metric keeps semantics in charge and leaves generality versus specificity to `depth_bias` and to the structural modes, which is where hyperbolic geometry earns its keep. Use `metric="hyperbolic"` with embedders that are themselves hyperbolic (for example Hierarchy Transformers).

## Mapping to the paper

| Paper (Section III–IV) | Here |
|---|---|
| Poincaré ball, metric tensor, distance (eq. 1–3) | `PoincareBall.distance`, `distance_arcosh`, `conformal_factor` |
| Structured knowledge representation with hierarchy | `Memory.add(parent=…)`, `sarkar_layout`, `fit_poincare_embeddings` |
| Temporal context management, time-weighted embeddings | `temporal.HalfLife / Ebbinghaus / ValidityWindow`, `search(at_time=…)` |
| Efficient retrieval over hierarchical structure | `search(mode=…)`, entailment cones, Lorentz dot-product trick |
| Cross-domain integration in one geometric space | one ball, many roots: `Memory.roots()` |

The metacognitive controller, provider-agnostic ensemble and response synthesis of the paper live in the companion package [`poincare-orchestra`](https://github.com/dhruvmojila/poincare-orchestra).

## Benchmarks

`benchmarks/wordnet_reconstruction.py` reproduces the classic WordNet reconstruction protocol (transitive closure, MAP and mean rank) for Poincaré versus Euclidean embeddings at equal dimension, with this library's NumPy trainer. On the `mammal.n.01` subtree (1,170 nodes, 200 epochs, 2026-10-10):

| dim | Poincaré MAP | Euclidean MAP |
|---:|---:|---:|
| 2 | 0.779 | 0.533 |
| 5 | 0.975 | 0.802 |
| 10 | 0.957 | 0.837 |

![WordNet mammals reconstruction MAP, Poincaré vs Euclidean](https://raw.githubusercontent.com/dhruvmojila/poincare-memory/main/docs/assets/wordnet-mammals-map.png)

Full table, environment and raw JSON in `benchmarks/README.md`; run it yourself with `pip install "poincare-memory[bench]"`.

## Documentation

Full docs at https://dhruvmojila.github.io/poincare-memory (getting started, concepts, guides, API, FAQ). Every code example in the docs is executed in CI.

## Command line

```bash
poincare-memory demo -o demo --open    # build the anatomy demo, save demo.db, open demo.html
poincare-memory inspect memory.db      # write memory.html
poincare-memory tree memory.db
poincare-memory search memory.db "calf strain" -k 5 --mode subtree --root lower
poincare-memory stats memory.db
poincare-memory export memory.db memory.json
```

## Citing

If you use this library, please cite the paper. The software itself is archived on Zenodo under the concept DOI [10.5281/zenodo.23551721](https://doi.org/10.5281/zenodo.23551721), which always resolves to the latest release; version DOIs are listed in `CITATION.cff`.

```bibtex
@inproceedings{mojila2026llmorchestration,
  author    = {Mojila, Dhruv},
  title     = {{LLM} Orchestration with Geometry-Aware Memory},
  booktitle = {2026 2nd International Conference on Big Data \& Machine Learning (ICBDML)},
  year      = {2026},
  pages     = {1--7},
  publisher = {IEEE},
  address   = {Bhopal, India},
  doi       = {10.1109/ICBDML68582.2026.11544690}
}
```

## Development

```bash
git clone https://github.com/dhruvmojila/poincare-memory && cd poincare-memory
uv venv && uv pip install -e ".[dev,viz,pandas]"
pytest            # property-based geometry tests + store tests
ruff check . && mypy
mkdocs serve      # docs
```

Clean-room note for contributors: implement from the papers cited in the docstrings. Several canonical hyperbolic-embedding repositories are unlicensed or non-commercial; do not copy from them.

## Licence

Apache-2.0. © 2026 Dhruv Mojila.
