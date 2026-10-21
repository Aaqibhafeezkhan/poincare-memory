"""Hierarchy placement: training-free Sarkar layout, Riemannian SGD training, evaluation."""

from .evaluation import ReconstructionResult, reconstruction
from .rsgd import TrainingReport, fit_euclidean_embeddings, fit_poincare_embeddings
from .sarkar import max_safe_depth, sarkar_layout, spread_directions
from .tree import children_map, depths, transitive_closure

__all__ = [
    "ReconstructionResult",
    "TrainingReport",
    "children_map",
    "depths",
    "fit_euclidean_embeddings",
    "fit_poincare_embeddings",
    "max_safe_depth",
    "reconstruction",
    "sarkar_layout",
    "spread_directions",
    "transitive_closure",
]
