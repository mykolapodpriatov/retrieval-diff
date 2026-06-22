"""Qdrant adapter stub (behind the ``[qdrant]`` extra).

Wraps a ``qdrant_client`` collection as a
:class:`~retrieval_diff.retrievers.Retriever`. The ``qdrant_client`` import is
guarded; construction raises
:class:`~retrieval_diff.retrievers.adapters.MissingDependencyError` until the
backend is installed and the adapter is implemented.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.retrievers.adapters import require
from retrieval_diff.types import ScoredHit

#: pip extra that provides this adapter's backend.
EXTRA = "qdrant"
#: Backend module name, import-guarded.
BACKEND_MODULE = "qdrant_client"


class QdrantRetriever:
    """Adapter exposing a Qdrant collection as a retriever (M4 stub).

    Args:
        client: A ``qdrant_client.QdrantClient`` instance.
        collection_name: The collection to query.
        embed: Callable turning a query string into a vector.
        fingerprint: The configuration fingerprint to report.
    """

    def __init__(
        self,
        client: object,
        collection_name: str,
        embed: Callable[[str], Sequence[float]],
        fingerprint: ConfigFingerprint,
    ) -> None:
        require(BACKEND_MODULE, backend="qdrant", extra=EXTRA)
        self._client = client
        self._collection_name = collection_name
        self._embed = embed
        self._fingerprint = fingerprint

    def search(self, query: str, k: int) -> list[ScoredHit]:
        """Return the top-``k`` hits for ``query`` (not yet implemented)."""
        raise NotImplementedError(
            "QdrantRetriever.search is an M4 stub; the Qdrant query path is "
            "implemented in the next milestone"
        )

    def fingerprint(self) -> ConfigFingerprint:
        """Return the configured fingerprint."""
        return self._fingerprint


__all__ = ["BACKEND_MODULE", "EXTRA", "QdrantRetriever"]
