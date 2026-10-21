# Benchmarks

## WordNet reconstruction
Protocol (Nickel & Kiela 2017): take the transitive closure of a WordNet noun sub-hierarchy, embed its nodes, and for every node rank all other nodes by distance; report mean average precision and mean rank of the related nodes. Run:

```bash
pip install "poincare-memory[bench]"
python benchmarks/wordnet_reconstruction.py --subtree mammal.n.01 --dims 2 5 10 --epochs 200 --lr 0.3
```

The script trains the Poincaré and Euclidean embeddings with the same loss, sampling and epochs, and prints a table. Latest run (2026-10-10, 200 epochs, lr 0.3, mammal.n.01 with 1,170 nodes):

| dim | Poincaré MAP | Euclidean MAP |
|---:|---:|---:|
| 2 | 0.779 | 0.533 |
| 5 | 0.975 | 0.802 |
| 10 | 0.957 | 0.837 |

Numbers are only published once produced by this script; `benchmarks/README.md` has the full table with mean ranks, the environment and the raw JSON.
