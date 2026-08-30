"""Tests for the Qdrant adapter's real query path.

Gated behind the ``qdrant`` extra: skipped cleanly via ``pytest.importorskip``
when ``qdrant_client`` is not installed (the default CI test matrix does not
install optional adapter extras). Everything runs against ``QdrantClient
(":memory:")`` — local mode, no server, no network.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient, models

from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.retrievers.adapters.qdrant_adapter import QdrantRetriever

QUERY_VECTOR = [1.0, 0.0]


def _embed(_query: str) -> Sequence[float]:
    """Return a fixed query vector so tests never load a real model."""
    return QUERY_VECTOR


def _collection(
    ids: Sequence[str],
    vectors: Sequence[Sequence[float]],
    *,
    distance: models.Distance = models.Distance.DOT,
    id_field: str | None = "chunk_id",
) -> tuple[QdrantClient, str]:
    """Build an in-memory collection whose payloads carry the chunk ids."""
    client = QdrantClient(":memory:")
    name = f"c-{uuid.uuid4().hex}"
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=2, distance=distance),
    )
    client.upsert(
        collection_name=name,
        points=[
            models.PointStruct(
                id=index + 1,
                vector=list(vector),
                payload={id_field: chunk_id} if id_field else None,
            )
            for index, (chunk_id, vector) in enumerate(zip(ids, vectors, strict=True))
        ],
    )
    return client, name


def test_basic_topk_ordering_and_ranks() -> None:
    """Hits come back sorted by descending score with 0-based contiguous ranks."""
    client, name = _collection(
        ["a", "b", "c", "d"],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]],
    )
    retriever = QdrantRetriever(client, name, _embed, ConfigFingerprint(), id_field="chunk_id")

    hits = retriever.search("q", k=3)

    assert [hit.id for hit in hits] == ["a", "b", "c"]
    assert [hit.rank for hit in hits] == [0, 1, 2]
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_tie_break_ascending_chunk_id() -> None:
    """Equal-score hits are ordered by ascending chunk id, not Qdrant point order."""
    client, name = _collection(["z", "a", "m"], [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    retriever = QdrantRetriever(client, name, _embed, ConfigFingerprint(), id_field="chunk_id")

    hits = retriever.search("q", k=3)

    assert [hit.id for hit in hits] == ["a", "m", "z"]
    assert [hit.rank for hit in hits] == [0, 1, 2]
    assert {round(hit.score, 6) for hit in hits} == {1.0}


def test_euclid_distances_are_negated_into_scores() -> None:
    """A distance metric still yields higher-is-better, correctly ordered scores."""
    client, name = _collection(
        ["near", "mid", "far"],
        [[1.0, 0.0], [0.9, 0.0], [0.0, 1.0]],
        distance=models.Distance.EUCLID,
    )
    retriever = QdrantRetriever(client, name, _embed, ConfigFingerprint(), id_field="chunk_id")

    hits = retriever.search("q", k=3)

    assert [hit.id for hit in hits] == ["near", "mid", "far"]
    assert [hit.rank for hit in hits] == [0, 1, 2]
    # Euclid reports 0.0 for the exact match; negation keeps every score <= 0.
    assert hits[0].score == pytest.approx(0.0)
    assert all(hit.score <= 0.0 for hit in hits)


def test_point_ids_used_when_no_id_field_configured() -> None:
    """Without ``id_field`` the Qdrant point id becomes the chunk id."""
    client, name = _collection(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], id_field=None)
    retriever = QdrantRetriever(client, name, _embed, ConfigFingerprint())

    hits = retriever.search("q", k=2)

    assert [hit.id for hit in hits] == ["1", "2"]


def test_point_missing_the_payload_key_falls_back_to_its_point_id() -> None:
    """One unlabelled point does not collapse the rest of the result set."""
    client, name = _collection(["a", "b"], [[1.0, 0.0], [0.9, 0.0]])
    client.set_payload(collection_name=name, payload={"chunk_id": None}, points=[2])
    retriever = QdrantRetriever(client, name, _embed, ConfigFingerprint(), id_field="chunk_id")

    hits = retriever.search("q", k=2)

    assert [hit.id for hit in hits] == ["a", "2"]


def test_named_vector_collection_is_queried_by_name() -> None:
    """A collection with named vectors is queried through ``vector_name``."""
    client = QdrantClient(":memory:")
    name = f"n-{uuid.uuid4().hex}"
    client.create_collection(
        collection_name=name,
        vectors_config={"text": models.VectorParams(size=2, distance=models.Distance.DOT)},
    )
    client.upsert(
        collection_name=name,
        points=[
            models.PointStruct(id=1, vector={"text": [1.0, 0.0]}, payload={"chunk_id": "a"}),
            models.PointStruct(id=2, vector={"text": [0.0, 1.0]}, payload={"chunk_id": "b"}),
        ],
    )
    retriever = QdrantRetriever(
        client, name, _embed, ConfigFingerprint(), id_field="chunk_id", vector_name="text"
    )

    hits = retriever.search("q", k=2)

    assert [hit.id for hit in hits] == ["a", "b"]
    assert [hit.rank for hit in hits] == [0, 1]


def test_k_exceeds_collection_size_returns_all_hits_without_crashing() -> None:
    """Requesting more hits than the collection holds returns what's available."""
    client, name = _collection(["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
    retriever = QdrantRetriever(client, name, _embed, ConfigFingerprint(), id_field="chunk_id")

    hits = retriever.search("q", k=10)

    assert [hit.id for hit in hits] == ["a", "b"]
    assert [hit.rank for hit in hits] == [0, 1]


def test_empty_collection_returns_no_hits_without_crashing() -> None:
    """An empty collection yields an empty hit list, not an error."""
    client, name = _collection([], [])
    retriever = QdrantRetriever(client, name, _embed, ConfigFingerprint(), id_field="chunk_id")

    assert retriever.search("q", k=5) == []


def test_non_positive_k_returns_no_hits() -> None:
    """A non-positive ``k`` returns an empty list rather than reaching Qdrant."""
    client, name = _collection(["a"], [[1.0, 0.0]])
    retriever = QdrantRetriever(client, name, _embed, ConfigFingerprint(), id_field="chunk_id")

    assert retriever.search("q", k=0) == []


def test_repeated_search_is_deterministic() -> None:
    """Identical ``(query, k)`` inputs yield identical hits, as snapshots require."""
    client, name = _collection(
        ["a", "b", "c"],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )
    retriever = QdrantRetriever(client, name, _embed, ConfigFingerprint(), id_field="chunk_id")

    assert retriever.search("q", k=3) == retriever.search("q", k=3)


def test_fingerprint_passthrough() -> None:
    """``fingerprint()`` returns exactly what the retriever was constructed with."""
    client, name = _collection(["a"], [[1.0, 0.0]])
    fp = ConfigFingerprint(embedding_model="qdrant-test")
    retriever = QdrantRetriever(client, name, _embed, fp, id_field="chunk_id")

    assert retriever.fingerprint() == fp
