# Benchmarks

Results are recorded here only when produced by the scripts in this directory. Each entry states the command, the environment and the numbers; the raw JSON sits next to this file.

## WordNet mammals reconstruction — 2026-10-10

Command:

```bash
python benchmarks/wordnet_reconstruction.py --subtree mammal.n.01 --dims 2 5 10 --epochs 200 --lr 0.3
```

Environment: Python 3.12.7, NumPy 2.5.3, single CPU core, Linux. Subtree `mammal.n.01`: 1,170 nodes, 6,448 transitive-closure pairs. Same loss, sampling (10 negatives), epochs and seed for both geometries; Euclidean learning rate = 0.3 / 6 (the Riemannian step is rescaled by the conformal factor, so a smaller Euclidean rate is the fair comparison).

| dim | Poincaré MAP | Poincaré mean rank | Euclidean MAP | Euclidean mean rank |
|---:|---:|---:|---:|---:|
| 2 | 0.779 | 1.86 | 0.533 | 2.25 |
| 5 | 0.975 | 1.49 | 0.802 | 1.46 |
| 10 | 0.957 | 1.38 | 0.837 | 1.39 |

Mean rank counts only unrelated nodes closer than a related one (Nickel & Kiela 2017); MAP is average precision over the full ranking. The pattern matches the literature: the advantage of hyperbolic space is largest at low dimension, where the Euclidean embedding has no room for the hierarchy. Raw output: `results-wordnet-mammals-2026-10-10.json`.

A shorter run (50 epochs, lr 0.1) gives Poincaré MAP 0.58 / 0.80 / 0.82 versus Euclidean 0.50 / 0.70 / 0.74 at dims 2 / 5 / 10; training length matters, so compare like with like.
