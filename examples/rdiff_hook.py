"""Project hook for the offline example.

``build_context`` returns a :class:`~retrieval_diff.config.ProjectContext` wiring
the in-memory hybrid retriever, the query set, the declared golden chunks, and an
attribution factory. The ``RDIFF_DEMO_MODEL`` environment variable swaps the
embedding model to simulate a regressing change for the demo.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from retrieval_diff.config import ProjectContext
from retrieval_diff.retrievers.factory import MemoryRetrieverFactory
from retrieval_diff.retrievers.memory import Corpus, HashingEmbedder, HybridRetriever

#: A tiny documentation corpus (id -> text).
CORPUS = Corpus.from_mapping(
    {
        "install": "install the package with pip install retrieval-diff",
        "snapshot": "snapshot captures top-k chunk ids scores and ranks per query",
        "diff": "diff compares two snapshots and reports added removed reranked chunks",
        "attribute": "attribute explains which config axis caused a retrieval change",
        "budget": "the regression budget fails ci when a golden chunk drops",
        "vectors": "dense vector embeddings power semantic search over the corpus",
    }
)

#: Declared golden chunks per query (the signal CI cares about).
GOLDENS: dict[str, set[str]] = {
    "how do I snapshot retrieval": {"snapshot"},
    "what does diff report": {"diff"},
    "semantic search with vectors": {"vectors"},
}

#: The two embedding models the factory can reconstruct.
_EMBEDDERS = {
    "demo-embed-v1": HashingEmbedder(dim=48, model_id="demo-embed-v1"),
    "demo-embed-v2": HashingEmbedder(dim=96, model_id="demo-embed-v2"),
}


def build_context(queries: Iterable[str] | None = None) -> ProjectContext:
    """Build the demo project context.

    Args:
        queries: Ignored; the hook supplies its own committed query set so the
            demo is self-contained.

    Returns:
        A :class:`ProjectContext` for the in-memory retriever.
    """
    model_id = os.environ.get("RDIFF_DEMO_MODEL", "demo-embed-v1")
    embedder = _EMBEDDERS[model_id]
    retriever = HybridRetriever(CORPUS, embedder, alpha=0.5)
    factory = MemoryRetrieverFactory(CORPUS, embedders=_EMBEDDERS)
    return ProjectContext(
        retriever=retriever,
        queries=tuple(GOLDENS),
        goldens=GOLDENS,
        factory=factory,
        corpus=CORPUS,
    )
