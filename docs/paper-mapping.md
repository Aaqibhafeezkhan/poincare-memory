# Mapping to the paper

| Paper | Code |
|---|---|
| §III-A Poincaré ball, metric tensor, distance (eq. 1–3) | `geometry/poincare.py` |
| §IV-C structured knowledge representation | `Memory.add(parent=…)`, `hierarchy/sarkar.py`, `hierarchy/rsgd.py` |
| §IV-C temporal context management | `temporal/policies.py`, `Memory.search(at_time=…)` |
| §IV-C efficient retrieval mechanisms | `Memory.search(mode=…)`, `retrieval/cones.py`, `geometry/lorentz.py` |
| §IV-C cross-domain knowledge integration | several roots in one `Memory` |
| §IV-B controller, §IV-D ensemble, §IV-E synthesis | companion package `poincare-orchestra` |

This is a reference implementation written from the paper's description; the evaluation in the paper used a system that is not part of this repository.
