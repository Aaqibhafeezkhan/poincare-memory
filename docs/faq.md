# FAQ

**Is this the system from the paper?**
No. It is the author's open-source reference implementation of the memory component, written after the paper from its description and the primary literature. The system used for the paper's experiments is not released, and every number here comes from this repository's code.

**Do I need a GPU or a service?**
No. The core is NumPy; the default embedder runs offline; persistence is one SQLite or JSON file.

**Why not just rank by hyperbolic distance?**
Because a point near the centre is close to everything. See [Concepts → Scoring](concepts.md#scoring); the raw metric is available as `metric="hyperbolic"`.

**How deep can a hierarchy go?**
About twenty levels with the defaults (`base_depth=1`, `depth_step=0.5`) before float64 runs out of room near the boundary; `max_safe_depth` tells you the limit for your settings, and `sarkar_layout` warns when a tree would exceed it.

**Can I use my own vector database?**
Yes; see [Scaling](guides/scaling.md): a dot product on Lorentz-mapped vectors is an exact hyperbolic nearest-neighbour search.

**Is it thread-safe?**
Yes for one process: every mutating and searching call takes an internal re-entrant lock. Your embedder must be safe to call concurrently if you share a `Memory` across threads.

**What about the orchestration part of the paper?**
[`poincare-orchestra`](https://github.com/dhruvmojila/poincare-orchestra) implements the controller, ensemble and synthesis on top of this memory.

**How do I cite it?**
Cite the paper (BibTeX in the README) and, for the software, the Zenodo release listed in `CITATION.cff`.
