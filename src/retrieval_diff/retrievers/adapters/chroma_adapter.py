"""Chroma adapter stub (behind the ``[chroma]`` extra).

Wraps a ``chromadb`` collection as a
:class:`~retrieval_diff.retrievers.Retriever`. The ``chromadb`` import is guarded;
construction raises
:class:`~retrieval_diff.retrievers.adapters.MissingDependencyError` until the
backend is installed and the adapter is implemented.
"""

from __future__ import annotations

from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.retrievers.adapters import require
from retrieval_diff.types import ScoredHit

#: pip extra that provides this adapter's backend.
EXTRA = "chroma"
#: Backend module name, import-guarded.
BACKEND_MODULE = "chromadb"


class ChromaRetriever:
    """Adapter exposing a Chroma collection as a retriever (M4 stub).

    Args:
        collection: A ``chromadb`` collection handle.
        fingerprint: The configuration fingerprint to report.
    """

    def __init__(self, collection: object, fingerprint: ConfigFingerprint) -> None:
        require(BACKEND_MODULE, backend="chroma", extra=EXTRA)
        self._collection = collection
        self._fingerprint = fingerprint

    def search(self, query: str, k: int) -> list[ScoredHit]:
        """Return the top-``k`` hits for ``query`` (not yet implemented)."""
        raise NotImplementedError(
            "ChromaRetriever.search is an M4 stub; the Chroma query path is "
            "implemented in the next milestone"
        )

    def fingerprint(self) -> ConfigFingerprint:
        """Return the configured fingerprint."""
        return self._fingerprint


__all__ = ["BACKEND_MODULE", "EXTRA", "ChromaRetriever"]
