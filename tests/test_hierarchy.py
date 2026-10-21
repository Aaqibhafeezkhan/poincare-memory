import numpy as np
import pytest

from poincare_memory import (
    PoincareBall,
    fit_euclidean_embeddings,
    fit_poincare_embeddings,
    reconstruction,
    sarkar_layout,
    transitive_closure,
)
from poincare_memory.hierarchy import children_map, depths, max_safe_depth, spread_directions


def balanced_tree(branching: int, depth: int):
    edges = []
    frontier = ["r"]
    for _lvl in range(depth):
        nxt = []
        for p in frontier:
            for k in range(branching):
                c = f"{p}.{k}"
                edges.append((p, c))
                nxt.append(c)
        frontier = nxt
    return edges


@pytest.mark.parametrize("dim", [2, 3, 5])
@pytest.mark.parametrize("c", [1.0, 0.25])
def test_sarkar_child_distance_equals_tau(dim, c):
    ball = PoincareBall(c)
    edges = balanced_tree(3, 4)
    tau = 0.8
    pos = sarkar_layout(edges, ball=ball, dim=dim, tau=tau)
    for p, ch in edges:
        assert np.isclose(ball.distance(pos[p], pos[ch]), tau, atol=1e-6)
    assert np.allclose(pos["r"], 0.0)


def test_sarkar_depth_monotone_and_low_distortion_2d():
    ball = PoincareBall()
    edges = balanced_tree(2, 4)
    pos = sarkar_layout(edges, ball=ball, dim=2, tau=2.0)
    children, roots = children_map(edges)
    dep = depths(children, roots[0])
    for n, _lvl in dep.items():
        if n != "r":
            parent = n.rsplit(".", 1)[0]
            assert ball.distance_to_origin(pos[n]) > ball.distance_to_origin(pos[parent])
    # tree distance vs hyperbolic distance: distortion should be small for a large tau
    nodes = list(pos)
    rng = np.random.default_rng(0)
    for _ in range(100):
        a, b = rng.choice(nodes, 2, replace=False)
        pa = ["r", *a.split(".")[1:]]
        pb = ["r", *b.split(".")[1:]]
        common = 0
        for x, y in zip(pa, pb, strict=False):
            if x == y:
                common += 1
            else:
                break
        tree_dist = (len(pa) - common) + (len(pb) - common)
        hd = ball.distance(pos[a], pos[b]) / 2.0  # tau = 2.0, so this is in edge units
        assert hd <= tree_dist + 1e-6
        assert hd >= 0.75 * tree_dist


def test_spread_directions():
    d2 = spread_directions(4, 2, avoid=np.array([1.0, 0.0]))
    assert d2.shape == (4, 2)
    assert np.allclose(np.linalg.norm(d2, axis=1), 1.0)
    assert np.all(d2 @ np.array([1.0, 0.0]) < 0.999)
    d5 = spread_directions(6, 5, seed=3)
    assert np.allclose(np.linalg.norm(d5, axis=1), 1.0)
    G = d5 @ d5.T
    np.fill_diagonal(G, -1)
    assert G.max() < 0.9  # no two directions nearly coincide


def test_children_map_validation():
    with pytest.raises(ValueError):
        children_map([("a", "b"), ("b", "a")])
    with pytest.raises(ValueError):
        children_map([("a", "c"), ("b", "c")])
    ch, roots = children_map([("a", "b"), ("a", "c")])
    assert roots == ["a"] and ch["a"] == ["b", "c"]


def test_sarkar_warns_when_too_deep():
    import warnings

    ball = PoincareBall()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sarkar_layout(balanced_tree(2, 7), ball=ball, dim=2, tau=2.0)
    assert any(issubclass(x.category, RuntimeWarning) for x in w)


def test_max_safe_depth():
    ball = PoincareBall()
    assert 20 < max_safe_depth(ball, 0.5) < 30
    assert max_safe_depth(ball, 1.0) == pytest.approx(max_safe_depth(ball, 0.5) / 2)


def test_rsgd_reconstructs_small_tree_better_than_euclidean():
    edges = balanced_tree(3, 3)
    children, roots = children_map(edges)
    closure = transitive_closure(children, roots[0])
    ball = PoincareBall()
    pts, report = fit_poincare_embeddings(
        closure, dim=5, epochs=60, lr=0.1, negatives=10, burn_in=5, seed=0, ball=ball
    )
    assert report.losses[-1] < report.losses[0]
    res_h = reconstruction(pts, closure, lambda a, b: ball.distance(a, b))
    pts_e, _ = fit_euclidean_embeddings(closure, dim=5, epochs=60, lr=0.05, negatives=10, seed=0)
    res_e = reconstruction(pts_e, closure, lambda a, b: np.linalg.norm(a - b, axis=-1))
    assert res_h.mean_average_precision > 0.6
    assert res_h.mean_average_precision >= res_e.mean_average_precision - 0.05
