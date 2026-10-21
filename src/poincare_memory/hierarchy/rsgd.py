"""Poincaré embeddings trained with Riemannian SGD (Nickel & Kiela, NeurIPS 2017), in NumPy.

Use this when the hierarchy is noisy, partial, or a DAG rather than a tree; for a
clean tree :func:`~poincare_memory.hierarchy.sarkar.sarkar_layout` is exact and
instantaneous. A Euclidean twin with the same loss is provided as a baseline for
benchmarks. Implemented from the paper; no third-party code was consulted.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from ..geometry import PoincareBall
from ..geometry.numerics import FloatArray

N = TypeVar("N", bound=Hashable)


@dataclass
class TrainingReport:
    epochs: int
    final_loss: float
    losses: list[float]


def _poincare_distance_and_grad(theta: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """d(theta, x) in the unit ball and its Euclidean gradient w.r.t. ``theta`` (closed form)."""
    alpha = np.maximum(1 - np.sum(theta * theta, axis=-1, keepdims=True), 1e-12)
    beta = np.maximum(1 - np.sum(x * x, axis=-1, keepdims=True), 1e-12)
    diff2 = np.sum((theta - x) ** 2, axis=-1, keepdims=True)
    gamma = 1 + 2 * diff2 / (alpha * beta)
    d = np.arccosh(np.maximum(gamma, 1.0 + 1e-12))
    tx = np.sum(theta * x, axis=-1, keepdims=True)
    x2 = np.sum(x * x, axis=-1, keepdims=True)
    coef = 4.0 / (beta * np.sqrt(np.maximum(gamma * gamma - 1, 1e-12)))
    grad = coef * ((x2 - 2 * tx + 1) / (alpha * alpha) * theta - x / alpha)
    return d[..., 0], grad


def _softmax_neg_log_prob(
    d_pos: FloatArray, d_neg: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Loss -log softmax over {pos} ∪ negs with logits -d, plus d(loss)/d(d_pos), d(loss)/d(d_neg)."""
    logits = np.concatenate([-d_pos[:, None], -d_neg], axis=1)
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=1, keepdims=True)
    loss = -np.log(np.maximum(p[:, 0], 1e-300))
    # dloss/dlogit = p - onehot(0); logit = -d  ->  dloss/dd = -(p - onehot)
    dpos = -(p[:, 0] - 1.0)
    dneg = -p[:, 1:]
    return loss, dpos, dneg


def fit_poincare_embeddings(
    edges: Sequence[tuple[N, N]],
    dim: int = 10,
    epochs: int = 100,
    lr: float = 0.1,
    negatives: int = 10,
    burn_in: int = 10,
    burn_in_lr_factor: float = 0.1,
    batch_size: int = 64,
    seed: int = 0,
    ball: PoincareBall | None = None,
) -> tuple[dict[N, FloatArray], TrainingReport]:
    """Learn points for every node so related pairs are close in the Poincaré ball.

    ``edges`` are ``(u, v)`` related pairs (typically the transitive closure of a
    hierarchy). Returns ``node -> point`` (scaled to the curvature of ``ball``)
    and a training report.
    """
    ball = ball or PoincareBall()
    rng = np.random.default_rng(seed)
    nodes = sorted({n for e in edges for n in e}, key=repr)
    index = {n: i for i, n in enumerate(nodes)}
    pairs = np.array([[index[u], index[v]] for u, v in edges], dtype=np.int64)
    n = len(nodes)
    theta = rng.uniform(-1e-3, 1e-3, size=(n, dim))
    eps = 1e-5
    losses: list[float] = []
    for epoch in range(epochs):
        step = lr * (burn_in_lr_factor if epoch < burn_in else 1.0)
        rng.shuffle(pairs)
        epoch_loss = 0.0
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            u = batch[:, 0]
            v = batch[:, 1]
            neg = rng.integers(0, n, size=(len(batch), negatives))
            tu = theta[u]
            tv = theta[v]
            tn = theta[neg]
            d_pos, g_u_pos = _poincare_distance_and_grad(tu, tv)
            _, g_v_pos = _poincare_distance_and_grad(tv, tu)
            d_neg, g_u_neg = _poincare_distance_and_grad(tu[:, None, :], tn)
            _, g_n = _poincare_distance_and_grad(tn, tu[:, None, :])
            loss, dpos, dneg = _softmax_neg_log_prob(d_pos, d_neg)
            epoch_loss += float(loss.sum())
            grad = np.zeros_like(theta)
            np.add.at(grad, u, dpos[:, None] * g_u_pos + np.sum(dneg[..., None] * g_u_neg, axis=1))
            np.add.at(grad, v, dpos[:, None] * g_v_pos)
            np.add.at(grad, neg.ravel(), (dneg[..., None] * g_n).reshape(-1, dim))
            touched = np.unique(np.concatenate([u, v, neg.ravel()]))
            scale = ((1 - np.sum(theta[touched] ** 2, axis=1, keepdims=True)) ** 2) / 4.0
            theta[touched] -= step * scale * grad[touched]
            nrm = np.linalg.norm(theta[touched], axis=1, keepdims=True)
            over = nrm >= 1 - eps
            theta[touched] = np.where(over, theta[touched] / nrm * (1 - eps), theta[touched])
        losses.append(epoch_loss / max(1, len(pairs)))
    points = {node: theta[i] / ball.sqrt_c for node, i in index.items()}
    return points, TrainingReport(
        epochs=epochs, final_loss=losses[-1] if losses else float("nan"), losses=losses
    )


def fit_euclidean_embeddings(
    edges: Sequence[tuple[N, N]],
    dim: int = 10,
    epochs: int = 100,
    lr: float = 0.05,
    negatives: int = 10,
    batch_size: int = 64,
    seed: int = 0,
) -> tuple[dict[N, FloatArray], TrainingReport]:
    """Same loss and sampling as :func:`fit_poincare_embeddings`, Euclidean distance (baseline)."""
    rng = np.random.default_rng(seed)
    nodes = sorted({n for e in edges for n in e}, key=repr)
    index = {n: i for i, n in enumerate(nodes)}
    pairs = np.array([[index[u], index[v]] for u, v in edges], dtype=np.int64)
    n = len(nodes)
    theta = rng.uniform(-1e-3, 1e-3, size=(n, dim))
    losses: list[float] = []
    for _ in range(epochs):
        rng.shuffle(pairs)
        epoch_loss = 0.0
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            u, v = batch[:, 0], batch[:, 1]
            neg = rng.integers(0, n, size=(len(batch), negatives))
            tu, tv, tn = theta[u], theta[v], theta[neg]
            diff_pos = tu - tv
            d_pos = np.linalg.norm(diff_pos, axis=1) + 1e-12
            diff_neg = tu[:, None, :] - tn
            d_neg = np.linalg.norm(diff_neg, axis=2) + 1e-12
            loss, dpos, dneg = _softmax_neg_log_prob(d_pos, d_neg)
            epoch_loss += float(loss.sum())
            g_u = dpos[:, None] * diff_pos / d_pos[:, None] + np.sum(
                dneg[..., None] * diff_neg / d_neg[..., None], axis=1
            )
            g_v = -dpos[:, None] * diff_pos / d_pos[:, None]
            g_n = -(dneg[..., None] * diff_neg / d_neg[..., None])
            grad = np.zeros_like(theta)
            np.add.at(grad, u, g_u)
            np.add.at(grad, v, g_v)
            np.add.at(grad, neg.ravel(), g_n.reshape(-1, dim))
            theta -= lr * grad
        losses.append(epoch_loss / max(1, len(pairs)))
    return {node: theta[i] for node, i in index.items()}, TrainingReport(
        epochs=epochs, final_loss=losses[-1] if losses else float("nan"), losses=losses
    )
