"""Uniquely-named shared builders for the test suite.

This module is intentionally *not* called ``helpers``/``conftest`` so it never
collides with an unrelated ``tests`` package that may be on ``sys.path``. pytest
prepends each test file's directory to ``sys.path`` (default import mode), so
``import rdiff_testkit`` resolves here.

Everything is offline and deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.types import QueryResult, ScoredHit, Snapshot

#: A small, fixed corpus used across retriever-backed tests.
CORPUS_TEXT: dict[str, str] = {
    "doc_alpha": "the quick brown fox jumps over the lazy dog",
    "doc_beta": "a lazy dog sleeps in the warm sun all day",
    "doc_gamma": "foxes and wolves hunt together in the cold forest",
    "doc_delta": "machine learning models embed text into dense vectors",
    "doc_epsilon": "vector search retrieves the nearest dense embeddings",
    "doc_zeta": "the brown fox and the grey wolf are distant cousins",
}

#: The query set used by retriever-backed tests.
QUERIES: tuple[str, ...] = ("brown fox", "lazy dog", "dense vector search")


def make_hits(pairs: Sequence[tuple[str, float]]) -> tuple[ScoredHit, ...]:
    """Build a rank-ordered hit tuple from ``(id, score)`` pairs."""
    return tuple(
        ScoredHit(id=cid, score=score, rank=rank) for rank, (cid, score) in enumerate(pairs)
    )


def make_snapshot(
    results: Mapping[str, Sequence[tuple[str, float]]],
    *,
    k: int,
    label: str = "test",
    fingerprint: ConfigFingerprint | None = None,
) -> Snapshot:
    """Construct a :class:`Snapshot` from ``{query: [(id, score), ...]}`` data."""
    query_results = {
        query: QueryResult(query=query, hits=make_hits(pairs)) for query, pairs in results.items()
    }
    return Snapshot(
        version=1,
        created_label=label,
        fingerprint=fingerprint or ConfigFingerprint(),
        k=k,
        results=query_results,
    )
