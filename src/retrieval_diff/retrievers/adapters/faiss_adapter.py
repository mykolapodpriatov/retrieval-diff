"""FAISS adapter (behind the ``[faiss]`` extra).

Wraps a FAISS index plus an id list and embedder as a
:class:`~retrieval_diff.retrievers.Retriever`. The ``faiss`` import is guarded so
this module loads without the dependency; construction raises
:class:`~retrieval_diff.retrievers.adapters.MissingDependencyError` until the
backend is installed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

import numpy as np

from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.retrievers.adapters import require
from retrieval_diff.types import ScoredHit

#: pip extra that provides this adapter's backend.
EXTRA = "faiss"
#: Backend module name, import-guarded.
BACKEND_MODULE = "faiss"


class _FaissIndex(Protocol):
    """Structural type for the subset of ``faiss.Index`` this adapter calls.

    Kept local (rather than importing ``faiss`` for typing) so this module never
    needs the optional dependency to type-check.
    """

    def search(self, x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, row_indices)``, each shaped ``(len(x), k)``."""
        ...


def _rank_hits(scores: Mapping[str, float], k: int) -> list[ScoredHit]:
    """Convert a ``{id: score}`` map into the rank-ordered top-``k`` hits.

    Hits are sorted by descending score, ties broken by ascending chunk id, with
    0-based contiguous ranks — the same contract ``retrievers/memory.py`` uses.
    """
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        ScoredHit(id=cid, score=float(score), rank=rank)
        for rank, (cid, score) in enumerate(ordered[:k])
    ]


class FaissRetriever:
    """Adapter exposing a FAISS index as a retriever.

    The wrapped index is assumed to score matches higher-is-better (e.g. an
    inner-product / cosine-similarity index over normalized vectors), matching
    the convention every other retriever in this package uses.

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
        """Return the rank-ordered top-``k`` hits for ``query``.

        Embeds ``query``, runs the wrapped FAISS index's ``search`` for the
        top-``k`` neighbors, and maps the returned row indices back to chunk ids
        via ``self._ids``. Handles ``k`` larger than the index size and an empty
        index by returning fewer than ``k`` (or zero) hits rather than raising.
        """
        if k <= 0 or not self._ids:
            return []
        index = cast(_FaissIndex, self._index)
        query_vector = np.asarray([self._embed(query)], dtype=np.float32)
        search_k = min(k, len(self._ids))
        raw_scores, row_indices = index.search(query_vector, search_k)
        scores: dict[str, float] = {}
        for row_index, score in zip(row_indices[0], raw_scores[0], strict=True):
            if row_index < 0:
                continue
            scores[self._ids[int(row_index)]] = float(score)
        return _rank_hits(scores, k)

    def fingerprint(self) -> ConfigFingerprint:
        """Return the configured fingerprint."""
        return self._fingerprint


__all__ = ["BACKEND_MODULE", "EXTRA", "FaissRetriever"]
