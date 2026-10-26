# Changelog

All notable changes to this project are documented here. The format follows Keep a Changelog; versions follow SemVer.

## [Unreleased]
### Added
- Add `--root` to the `tree` CLI command to print a selected memory subtree.

## [0.1.4] — 2026-10-22
### Fixed
- Documentation: the scoring explanation (README "How scoring works" and Concepts) gave an angular crossover figure derived from the large-depth approximation of the Poincaré distance, which is inaccurate at the default depths; replaced with exact figures. The default shell metric and all behaviour are unchanged.

## [0.1.3] — 2026-10-21
### Changed
- The README is the package description again, with the inspector demo and benchmark chart included.

## [0.1.2] — 2026-10-21
### Changed
- Temporarily used a description without repository-hosted images.

## [0.1.1] — 2026-10-21
### Fixed
- A docstring in the hashing embedder contained an unescaped regular-expression sequence, which raised a `SyntaxWarning` on import under Python 3.12 and later.

## [0.1.0] — 2026-10-21
### Added
- Poincaré ball geometry with curvature (`distance`, `distance_arcosh`, exp/log maps, Möbius operations, Lorentz conversions, dot-product trick for ANN indexes), float32/float64.
- `Memory`: hierarchy-aware, time-aware store with structural search modes (`subtree`, `children`, `ancestors`, `siblings`, `cone`), named temporal policies, SQLite persistence.
- Visual inspection: self-contained interactive HTML (disk, table, tree), matplotlib plot, Jupyter rendering, ASCII tree, DataFrame export.
- Sarkar's training-free tree layout (2-D exact, n-D spread) with a float64 depth safety check; Riemannian SGD Poincaré embeddings and a Euclidean baseline; reconstruction MAP / mean rank.
- Offline hashing embedder, Sentence-Transformers wrapper, callable adapter.
- Inspector layouts: `structural` (radial tree at true hyperbolic depths, default when parents exist) and `projection` (semantic directions); `DecayPolicy` exported at package level.
- WordNet mammals benchmark results recorded with raw JSON and a chart; README demo GIF.
- Production hardening: `add_many` (one embedding call), `update`, `to_json` / `from_json`, `stats`, `__repr__`, input validation, module logging, a re-entrant lock around all mutating and searching calls, and `build_index` with NumPy or FAISS (`[ann]` extra) backends for indexed shell-metric search; `SearchResult.to_dict`; CLI `stats`, `export`, `import`.
- Documentation site (mkdocs-material): getting started, concepts, eight guides, API reference, FAQ; every Python code block in the docs is executed in CI by `scripts/check_docs_examples.py`.
- The hashing embedder now keys BLAKE2b with the seed (instead of salting), so the TypeScript port in the explainer produces identical vectors.
- CLI (`demo`, `inspect`, `tree`, `search`), WordNet reconstruction benchmark, documentation.
