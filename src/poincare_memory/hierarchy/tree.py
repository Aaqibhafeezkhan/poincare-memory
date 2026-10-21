"""Small rooted-tree utilities shared by the layout and training code."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TypeVar

N = TypeVar("N", bound=Hashable)


def children_map(edges: Iterable[tuple[N, N]]) -> tuple[dict[N, list[N]], list[N]]:
    """Build ``parent -> [children]`` from ``(parent, child)`` edges and return the roots.

    Raises ``ValueError`` on cycles or on a node with two parents.
    """
    children: dict[N, list[N]] = defaultdict(list)
    parent_of: dict[N, N] = {}
    nodes: set[N] = set()
    for p, c in edges:
        if c in parent_of and parent_of[c] != p:
            raise ValueError(f"node {c!r} has two parents: {parent_of[c]!r} and {p!r}")
        parent_of[c] = p
        children[p].append(c)
        nodes.add(p)
        nodes.add(c)
    roots = [n for n in nodes if n not in parent_of]
    if nodes and not roots:
        raise ValueError("the edge set contains a cycle (no root found)")
    # cycle check by walking up from every node
    for n in nodes:
        seen = {n}
        cur = n
        while cur in parent_of:
            cur = parent_of[cur]
            if cur in seen:
                raise ValueError("the edge set contains a cycle")
            seen.add(cur)
    return dict(children), roots


def depths(children: Mapping[N, Sequence[N]], root: N) -> dict[N, int]:
    """Depth of every node below ``root`` (root has depth 0)."""
    out = {root: 0}
    stack = [root]
    while stack:
        n = stack.pop()
        for c in children.get(n, ()):
            out[c] = out[n] + 1
            stack.append(c)
    return out


def transitive_closure(children: Mapping[N, Sequence[N]], root: N) -> list[tuple[N, N]]:
    """All ``(ancestor, descendant)`` pairs below ``root``."""
    pairs: list[tuple[N, N]] = []

    def walk(n: N, ancestors: list[N]) -> None:
        for a in ancestors:
            pairs.append((a, n))
        for c in children.get(n, ()):
            walk(c, [*ancestors, n])

    walk(root, [])
    return pairs
