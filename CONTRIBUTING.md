# Contributing

Thanks for helping. A few rules keep the project trustworthy:

1. **Clean room.** Implement from the papers cited in the docstrings (Nickel & Kiela 2017; Ganea et al. 2018; Sala et al. 2018; He et al. 2024). Do not copy code from repositories that are unlicensed or licensed for non-commercial use only.
2. **Numbers must be reproducible.** Any benchmark figure added to the docs must come from a script in `benchmarks/` that a reader can run.
3. **Tests.** Geometry changes need property-based tests (`hypothesis`); store changes need behavioural tests. `pytest`, `ruff check .`, `ruff format --check .` and `mypy` must pass.
4. **Provenance wording.** The README's provenance paragraph stays as it is: this is a reference implementation written after the paper; the paper's experimental system is not part of this repository.

Development setup:

```bash
uv venv && uv pip install -e ".[dev,viz,pandas]"
pre-commit install
pytest
```

Open an issue before large changes so the design can be discussed first.
