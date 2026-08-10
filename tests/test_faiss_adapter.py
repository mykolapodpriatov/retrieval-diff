"""Tests for the FAISS adapter's real query path.

Gated behind the ``faiss`` extra: skipped cleanly via ``pytest.importorskip``
when ``faiss-cpu`` is not installed (the default CI test matrix does not install
optional adapter extras).
"""

from __future__ import annotations

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from retrieval_diff.fingerprint import ConfigFingerprint  # noqa: E402
from retrieval_diff.retrievers.adapters.faiss_adapter import FaissRetriever  # noqa: E402


def _build_index(vectors: list[list[float]]) -> object:
    """Build a small inner-product FAISS index over ``vectors``."""
    dim = len(vectors[0])
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(vectors, dtype=np.float32))
    return index


def test_basic_topk_ordering_and_ranks() -> None:
    """Hits come back sorted by descending score with 0-based contiguous ranks."""
    ids = ["a", "b", "c", "d"]
    vectors = [
        [1.0, 0.0],  # exact match, highest score
        [0.9, 0.1],  # close second
        [0.0, 1.0],  # orthogonal, lowest of the top-3
        [-1.0, 0.0],  # opposite, excluded by k=3
    ]
    index = _build_index(vectors)
    retriever = FaissRetriever(index, ids, lambda query: [1.0, 0.0], ConfigFingerprint())

    hits = retriever.search("q", k=3)

    assert [hit.id for hit in hits] == ["a", "b", "c"]
    assert [hit.rank for hit in hits] == [0, 1, 2]
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_tie_break_ascending_chunk_id() -> None:
    """Equal-score hits are ordered by ascending chunk id, not FAISS row order."""
    ids = ["z", "a", "m"]
    vectors = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    index = _build_index(vectors)
    retriever = FaissRetriever(index, ids, lambda query: [1.0, 0.0], ConfigFingerprint())

    hits = retriever.search("q", k=3)

    assert [hit.id for hit in hits] == ["a", "m", "z"]
    assert [hit.rank for hit in hits] == [0, 1, 2]
    assert {round(hit.score, 6) for hit in hits} == {1.0}


def test_k_exceeds_index_size_returns_all_hits_without_crashing() -> None:
    """Requesting more hits than the index holds returns what's available."""
    ids = ["a", "b"]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    index = _build_index(vectors)
    retriever = FaissRetriever(index, ids, lambda query: [1.0, 0.0], ConfigFingerprint())

    hits = retriever.search("q", k=10)

    assert [hit.id for hit in hits] == ["a", "b"]
    assert [hit.rank for hit in hits] == [0, 1]


def test_empty_index_returns_no_hits_without_crashing() -> None:
    """An empty index (no vectors added) yields an empty hit list, not an error."""
    index = faiss.IndexFlatIP(2)
    retriever = FaissRetriever(index, [], lambda query: [1.0, 0.0], ConfigFingerprint())

    assert retriever.search("q", k=5) == []


def test_non_positive_k_returns_no_hits() -> None:
    """A non-positive ``k`` returns an empty list rather than reaching FAISS."""
    index = _build_index([[1.0, 0.0]])
    retriever = FaissRetriever(index, ["a"], lambda query: [1.0, 0.0], ConfigFingerprint())

    assert retriever.search("q", k=0) == []


def test_fingerprint_passthrough() -> None:
    """``fingerprint()`` returns exactly what the retriever was constructed with."""
    index = faiss.IndexFlatIP(2)
    fp = ConfigFingerprint(embedding_model="faiss-test")
    retriever = FaissRetriever(index, [], lambda query: [1.0, 0.0], fp)

    assert retriever.fingerprint() == fp
