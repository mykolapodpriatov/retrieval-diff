"""FAISS adapter stub (behind the ``[faiss]`` extra).

Wraps a FAISS index plus an id list and embedder as a
:class:`~retrieval_diff.retrievers.Retriever`. The ``faiss`` import is guarded so
this module loads without the dependency; construction raises
:class:`~retrieval_diff.retrievers.adapters.MissingDependencyError` until the
backend is installed and the adapter is implemented.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.retrievers.adapters import require
from retrieval_diff.types import ScoredHit

#: pip extra that provides this adapter's backend.
EXTRA = "faiss"
#: Backend module name, import-guarded.
BACKEND_MODULE = "faiss"


class FaissRetriever:
    """Adapter exposing a FAISS index as a retriever (M4 stub).

    Args:
        index: A FAISS index object (``faiss.Index``).
        ids: Row-aligned chunk ids for the vectors in ``index``.
        embed: Callable turning a query string into a vector.
        fingerprint: The configuration fingerprint to report.
    """

    def __init__(
        self,
        index: object,
        ids: Sequence[str],
        embed: Callable[[str], Sequence[float]],
        fingerprint: ConfigFingerprint,
    ) -> None:
        require(BACKEND_MODULE, backend="faiss", extra=EXTRA)
        self._index = index
        self._ids = list(ids)
        self._embed = embed
        self._fingerprint = fingerprint

    def search(self, query: str, k: int) -> list[ScoredHit]:
        """Return the top-``k`` hits for ``query`` (not yet implemented)."""
        raise NotImplementedError(
            "FaissRetriever.search is an M4 stub; the FAISS query path is "
            "implemented in the next milestone"
        )

    def fingerprint(self) -> ConfigFingerprint:
        """Return the configured fingerprint."""
        return self._fingerprint


__all__ = ["BACKEND_MODULE", "EXTRA", "FaissRetriever"]
