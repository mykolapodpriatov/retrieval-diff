"""Build :class:`~retrieval_diff.types.Snapshot` objects from retrievers.

Snapshotting runs each query through a retriever, captures the top-K hits, and
records the retriever's fingerprint. ``created_label`` is supplied by the caller
(never read from the clock), keeping snapshots reproducible.
"""

from __future__ import annotations

from collections.abc import Iterable

from retrieval_diff.retrievers import Retriever
from retrieval_diff.types import QueryResult, ScoredHit, Snapshot

#: Lockfile schema version produced by this library.
SNAPSHOT_VERSION = 1


def _normalize_hits(hits: list[ScoredHit], k: int) -> tuple[ScoredHit, ...]:
    """Validate and renumber hits into contiguous 0-based ranks, truncated to ``k``.

    A retriever is expected to already return rank-ordered hits, but ranks are
    re-derived here to defend against off-by-one or duplicate-rank bugs in
    third-party adapters. Duplicate ids are rejected.
    """
    seen: set[str] = set()
    normalized: list[ScoredHit] = []
    for rank, hit in enumerate(hits[:k]):
        if hit.id in seen:
            raise ValueError(f"duplicate chunk id {hit.id!r} in retriever output")
        seen.add(hit.id)
        normalized.append(ScoredHit(id=hit.id, score=hit.score, rank=rank))
    return tuple(normalized)


def snapshot(
    retriever: Retriever,
    queries: Iterable[str],
    k: int,
    *,
    label: str,
) -> Snapshot:
    """Capture a retriever's top-``k`` output for ``queries`` into a snapshot.

    Args:
        retriever: The retriever to snapshot.
        queries: The query set; order is irrelevant (results are keyed by query).
        k: The top-K depth to capture; must be positive.
        label: Caller-supplied label (e.g. a git SHA). Required and never read
            from the clock to keep snapshots deterministic.

    Returns:
        A :class:`Snapshot` keyed by query.

    Raises:
        ValueError: If ``k`` is not positive, ``label`` is empty, or duplicate
            queries are supplied.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not label:
        raise ValueError("label is required (supply a git SHA or other stable id)")

    results: dict[str, QueryResult] = {}
    for query in queries:
        if query in results:
            raise ValueError(f"duplicate query in query set: {query!r}")
        hits = _normalize_hits(retriever.search(query, k), k)
        results[query] = QueryResult(query=query, hits=hits)

    return Snapshot(
        version=SNAPSHOT_VERSION,
        created_label=label,
        fingerprint=retriever.fingerprint(),
        k=k,
        results=results,
    )
