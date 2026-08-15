"""Chroma adapter (behind the ``[chroma]`` extra).

Wraps a ``chromadb`` collection as a
:class:`~retrieval_diff.retrievers.Retriever`. The ``chromadb`` import is guarded
so this module loads without the dependency; construction raises
:class:`~retrieval_diff.retrievers.adapters.MissingDependencyError` until the
backend is installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.retrievers.adapters import require
from retrieval_diff.types import ScoredHit

#: pip extra that provides this adapter's backend.
EXTRA = "chroma"
#: Backend module name, import-guarded.
BACKEND_MODULE = "chromadb"


class _ChromaCollection(Protocol):
    """Structural type for the subset of a Chroma collection this adapter calls.

    Kept local (rather than importing ``chromadb`` for typing) so this module
    never needs the optional dependency to type-check.
    """

    def count(self) -> int:
        """Return the number of embeddings stored in the collection."""
        ...

    def query(
        self,
        *,
        query_texts: Sequence[str],
        n_results: int,
        include: Sequence[str],
    ) -> Mapping[str, object]:
        """Return a query-result map whose ``ids`` / ``distances`` rows are lists."""
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


def _first_row(value: object) -> Sequence[object]:
    """Return the first row of a Chroma ``ids`` / ``distances`` payload."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return []
    first = value[0]
    if not isinstance(first, Sequence) or isinstance(first, (str, bytes)):
        return []
    return first


class ChromaRetriever:
    """Adapter exposing a Chroma collection as a retriever.

    Queries go through ``collection.query(query_texts=...)`` so the wrapped
    collection's own embedding function embeds the query. This keeps the
    constructor to ``(collection, fingerprint)`` — no separate ``embed``
    callable — and matches how Chroma collections are typically used as a
    local/prototype store. Collections that only hold precomputed embeddings
    (no embedding function) are not supported; attach an embedding function
    or query them through a different adapter.

    Chroma returns distances (lower-is-better). Those are converted to this
    package's higher-is-better scores by negation (``score = -distance``),
    which preserves neighbor order for every Chroma space (``l2``, ``ip``,
    ``cosine``) without depending on the space.

    Args:
        collection: A ``chromadb`` collection handle.
        fingerprint: The configuration fingerprint to report.
    """

    def __init__(self, collection: object, fingerprint: ConfigFingerprint) -> None:
        require(BACKEND_MODULE, backend="chroma", extra=EXTRA)
        self._collection = collection
        self._fingerprint = fingerprint

    def search(self, query: str, k: int) -> list[ScoredHit]:
        """Return the rank-ordered top-``k`` hits for ``query``.

        Embeds ``query`` via the wrapped collection's embedding function, asks
        Chroma for the nearest neighbors, and maps distances onto scores. Handles
        ``k`` larger than the collection size and an empty collection by
        returning fewer than ``k`` (or zero) hits rather than raising.
        """
        if k <= 0:
            return []
        collection = cast(_ChromaCollection, self._collection)
        count = int(collection.count())
        if count <= 0:
            return []
        result = collection.query(
            query_texts=[query],
            n_results=min(k, count),
            include=["distances"],
        )
        raw_ids = _first_row(result.get("ids"))
        raw_distances = _first_row(result.get("distances"))
        if not raw_ids or not raw_distances:
            return []
        scores: dict[str, float] = {}
        for chunk_id, distance in zip(raw_ids, raw_distances, strict=True):
            scores[str(chunk_id)] = -float(cast(float, distance))
        return _rank_hits(scores, k)

    def fingerprint(self) -> ConfigFingerprint:
        """Return the configured fingerprint."""
        return self._fingerprint


__all__ = ["BACKEND_MODULE", "EXTRA", "ChromaRetriever"]
