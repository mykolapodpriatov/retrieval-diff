"""Tests for the Chroma adapter's real query path.

Gated behind the ``chroma`` extra: skipped cleanly via ``pytest.importorskip``
when ``chromadb`` is not installed (the default CI test matrix does not install
optional adapter extras).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import pytest

chromadb = pytest.importorskip("chromadb")

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings  # noqa: E402
from chromadb.config import Settings  # noqa: E402

from retrieval_diff.fingerprint import ConfigFingerprint  # noqa: E402
from retrieval_diff.retrievers.adapters.chroma_adapter import ChromaRetriever  # noqa: E402


class _ConstantEmbeddingFunction(EmbeddingFunction[Documents]):
    """Return a fixed query vector so tests never download a real model."""

    def __init__(self, vector: Sequence[float]) -> None:
        self._vector = [float(value) for value in vector]

    @staticmethod
    def name() -> str:
        return "constant-ef"

    def __call__(self, input: Documents) -> Embeddings:
        return [list(self._vector) for _ in input]

    def get_config(self) -> dict[str, object]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, object]) -> _ConstantEmbeddingFunction:
        return _ConstantEmbeddingFunction([1.0, 0.0])


def _client() -> object:
    return chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))


def _build_collection(ids: list[str], vectors: list[list[float]]) -> object:
    """Build an in-memory Chroma collection over ``ids`` / ``vectors``.

    Documents are stored with the given embeddings; queries are embedded by a
    constant function that always returns ``[1.0, 0.0]``, matching the FAISS
    adapter tests' query vector.
    """
    client = _client()
    collection = client.create_collection(
        name=f"c-{uuid.uuid4().hex}",
        embedding_function=_ConstantEmbeddingFunction([1.0, 0.0]),
        metadata={"hnsw:space": "ip"},
    )
    collection.add(ids=ids, embeddings=vectors, documents=ids)
    return collection


def test_basic_topk_ordering_and_ranks() -> None:
    """Hits come back sorted by descending score with 0-based contiguous ranks."""
    ids = ["a", "b", "c", "d"]
    vectors = [
        [1.0, 0.0],  # exact match, highest score
        [0.9, 0.1],  # close second
        [0.0, 1.0],  # orthogonal, lowest of the top-3
        [-1.0, 0.0],  # opposite, excluded by k=3
    ]
    collection = _build_collection(ids, vectors)
    retriever = ChromaRetriever(collection, ConfigFingerprint())

    hits = retriever.search("q", k=3)

    assert [hit.id for hit in hits] == ["a", "b", "c"]
    assert [hit.rank for hit in hits] == [0, 1, 2]
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_tie_break_ascending_chunk_id() -> None:
    """Equal-score hits are ordered by ascending chunk id, not Chroma row order."""
    ids = ["z", "a", "m"]
    vectors = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    collection = _build_collection(ids, vectors)
    retriever = ChromaRetriever(collection, ConfigFingerprint())

    hits = retriever.search("q", k=3)

    assert [hit.id for hit in hits] == ["a", "m", "z"]
    assert [hit.rank for hit in hits] == [0, 1, 2]
    assert {round(hit.score, 6) for hit in hits} == {0.0}


def test_k_exceeds_index_size_returns_all_hits_without_crashing() -> None:
    """Requesting more hits than the collection holds returns what's available."""
    ids = ["a", "b"]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    collection = _build_collection(ids, vectors)
    retriever = ChromaRetriever(collection, ConfigFingerprint())

    hits = retriever.search("q", k=10)

    assert [hit.id for hit in hits] == ["a", "b"]
    assert [hit.rank for hit in hits] == [0, 1]


def test_empty_index_returns_no_hits_without_crashing() -> None:
    """An empty collection yields an empty hit list, not an error."""
    client = _client()
    collection = client.create_collection(
        name=f"e-{uuid.uuid4().hex}",
        embedding_function=_ConstantEmbeddingFunction([1.0, 0.0]),
    )
    retriever = ChromaRetriever(collection, ConfigFingerprint())

    assert retriever.search("q", k=5) == []


def test_non_positive_k_returns_no_hits() -> None:
    """A non-positive ``k`` returns an empty list rather than reaching Chroma."""
    collection = _build_collection(["a"], [[1.0, 0.0]])
    retriever = ChromaRetriever(collection, ConfigFingerprint())

    assert retriever.search("q", k=0) == []


def test_fingerprint_passthrough() -> None:
    """``fingerprint()`` returns exactly what the retriever was constructed with."""
    client = _client()
    collection = client.create_collection(
        name=f"f-{uuid.uuid4().hex}",
        embedding_function=_ConstantEmbeddingFunction([1.0, 0.0]),
    )
    fp = ConfigFingerprint(embedding_model="chroma-test")
    retriever = ChromaRetriever(collection, fp)

    assert retriever.fingerprint() == fp
