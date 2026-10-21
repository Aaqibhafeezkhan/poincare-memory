"""WordNet noun hierarchy reconstruction: Poincaré vs Euclidean at equal dimension.

Usage:
    pip install "poincare-memory[bench]"
    python benchmarks/wordnet_reconstruction.py --subtree mammal.n.01 --dims 2 5 10 --epochs 100

Protocol: Nickel & Kiela (2017). The transitive closure of the hypernym relation
below ``--subtree`` is embedded; for each node all other nodes are ranked by
distance and the related nodes' mean rank and MAP are reported.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from poincare_memory import (
    PoincareBall,
    fit_euclidean_embeddings,
    fit_poincare_embeddings,
    reconstruction,
)


def closure_below(root_name: str) -> list[tuple[str, str]]:
    import nltk

    try:
        from nltk.corpus import wordnet as wn

        wn.synsets("dog")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn
    root = wn.synset(root_name)
    nodes = {root}
    stack = [root]
    while stack:
        s = stack.pop()
        for h in s.hyponyms():
            if h not in nodes:
                nodes.add(h)
                stack.append(h)
    pairs: set[tuple[str, str]] = set()
    for s in nodes:
        for path in s.hypernym_paths():
            for anc in path:
                if anc in nodes and anc != s:
                    pairs.add((anc.name(), s.name()))
    return sorted(pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtree", default="mammal.n.01")
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--negatives", type=int, default=10)
    ap.add_argument(
        "--lr",
        type=float,
        default=0.3,
        help="Riemannian SGD learning rate (0.3 follows Nickel & Kiela)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write results as JSON")
    a = ap.parse_args()
    closure = closure_below(a.subtree)
    nodes = {n for e in closure for n in e}
    print(f"{a.subtree}: {len(nodes)} nodes, {len(closure)} closure pairs")
    ball = PoincareBall()
    results = []
    for dim in a.dims:
        t0 = time.time()
        pts, _rep = fit_poincare_embeddings(
            closure, dim=dim, epochs=a.epochs, negatives=a.negatives, seed=a.seed, ball=ball
        )
        rh = reconstruction(pts, closure, lambda x, y: ball.distance(x, y))
        th = time.time() - t0
        t0 = time.time()
        pe, _ = fit_euclidean_embeddings(
            closure, dim=dim, epochs=a.epochs, negatives=a.negatives, seed=a.seed
        )
        re_ = reconstruction(pe, closure, lambda x, y: np.linalg.norm(x - y, axis=-1))
        te = time.time() - t0
        row = {
            "dim": dim,
            "poincare_map": rh.mean_average_precision,
            "poincare_rank": rh.mean_rank,
            "euclid_map": re_.mean_average_precision,
            "euclid_rank": re_.mean_rank,
            "poincare_s": th,
            "euclid_s": te,
        }
        results.append(row)
        print(
            f"dim={dim:3d}  Poincaré MAP={rh.mean_average_precision:.3f} rank={rh.mean_rank:7.2f} ({th:5.1f}s)   Euclidean MAP={re_.mean_average_precision:.3f} rank={re_.mean_rank:7.2f} ({te:5.1f}s)"
        )
    if a.out:
        with open(a.out, "w") as f:
            json.dump(
                {
                    "subtree": a.subtree,
                    "nodes": len(nodes),
                    "pairs": len(closure),
                    "epochs": a.epochs,
                    "results": results,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
