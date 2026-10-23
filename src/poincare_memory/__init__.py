"""poincare-memory: hierarchy-aware, time-aware memory for LLM agents on the Poincaré ball.

Official open-source reference implementation of the geometry-aware memory
component described in

    D. Mojila, "LLM Orchestration with Geometry-Aware Memory," 2026 2nd Int.
    Conf. on Big Data & Machine Learning (ICBDML), Bhopal, India, 2026, pp. 1-7,
    doi:10.1109/ICBDML68582.2026.11544690.

The system used for the paper's experiments is not released; this package was
written afterwards from the paper's description and the primary literature.
"""

from .embed import (
    BallProjector,
    CallableEmbedder,
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
)
from .geometry import PoincareBall, lorentz_distance, to_mips_vectors
from .hierarchy import (
    fit_euclidean_embeddings,
    fit_poincare_embeddings,
    max_safe_depth,
    reconstruction,
    sarkar_layout,
    transitive_closure,
)
from .index import MIPSIndex
from .retrieval import cone_energy, in_cone
from .store import Memory, MemoryRecord, SearchResult
from .temporal import (
    DecayPolicy,
    Ebbinghaus,
    HalfLife,
    NoDecay,
    Product,
    ValidityWindow,
    generative_agents_recency,
)

__version__ = "0.1.4"
__all__ = [
    "BallProjector",
    "CallableEmbedder",
    "DecayPolicy",
    "Ebbinghaus",
    "Embedder",
    "HalfLife",
    "HashingEmbedder",
    "MIPSIndex",
    "Memory",
    "MemoryRecord",
    "NoDecay",
    "PoincareBall",
    "Product",
    "SearchResult",
    "SentenceTransformerEmbedder",
    "ValidityWindow",
    "__version__",
    "cone_energy",
    "fit_euclidean_embeddings",
    "fit_poincare_embeddings",
    "generative_agents_recency",
    "in_cone",
    "lorentz_distance",
    "max_safe_depth",
    "reconstruction",
    "sarkar_layout",
    "to_mips_vectors",
    "transitive_closure",
]
