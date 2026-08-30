"""Qdrant adapter (behind the ``[qdrant]`` extra).

Wraps a ``qdrant_client`` collection as a
:class:`~retrieval_diff.retrievers.Retriever`. The ``qdrant_client`` import is
guarded so this module loads without the dependency; construction raises
:class:`~retrieval_diff.retrievers.adapters.MissingDependencyError` until the
backend is installed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.retrievers.adapters import require
from retrieval_diff.types import ScoredHit

#: pip extra that provides this adapter's backend.
EXTRA = "qdrant"
#: Backend module name, import-guarded.
BACKEND_MODULE = "qdrant_client"

#: Qdrant distance metrics whose ``score`` is a distance (lower is better).
#: Cosine and Dot report similarities and are already higher-is-better.
_DISTANCE_METRICS = frozenset({"euclid", "manhattan"})


class _CountResult(Protocol):
    """Structural type for ``qdrant_client``'s count response."""

    count: int


class _ScoredPoint(Protocol):
    """Structural type for a single Qdrant search hit."""

    id: object
    score: float
    payload: Mapping[str, object] | None


class _QueryResponse(Protocol):
    """Structural type for ``query_points``' response envelope."""

    points: Sequence[_ScoredPoint]


class _QdrantClient(Protocol):
    """Structural type for the subset of ``QdrantClient`` this adapter calls.

    Kept local (rather than importing ``qdrant_client`` for typing) so this
    module never needs the optional dependency to type-check.
    """

    def count(self, collection_name: str, exact: bool = ...) -> _CountResult:
        """Return the number of points stored in the collection."""
        ...

    def get_collection(self, collection_name: str) -> object:
        """Return the collection's info record, including its vector params."""
        ...

    def query_points(
        self,
        collection_name: str,
        query: Sequence[float],
        using: str | None = ...,
        limit: int = ...,
        with_payload: bool = ...,
    ) -> _QueryResponse:
        """Return the nearest points to ``query``."""
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


def _distance_name(params: object, vector_name: str | None) -> str:
    """Return the lowercased distance metric name for ``params``, or ``""``.

    ``params.vectors`` is either a single ``VectorParams`` (unnamed vector) or a
    ``{name: VectorParams}`` mapping (named vectors). Anything unrecognized
    yields ``""`` so the caller falls back to the higher-is-better default.
    """
    vectors = getattr(params, "vectors", None)
    if isinstance(vectors, Mapping):
        vectors = vectors.get(vector_name) if vector_name is not None else None
    distance = getattr(vectors, "distance", None)
    name = getattr(distance, "value", distance)
    return name.lower() if isinstance(name, str) else ""


class QdrantRetriever:
    """Adapter exposing a Qdrant collection as a retriever.

    Qdrant's ``score`` means different things per collection: Cosine and Dot
    report similarities (higher is better), while Euclid and Manhattan report
    distances (lower is better). The adapter reads the collection's configured
    metric once and negates distances, so hits always follow this package's
    higher-is-better convention. A metric it cannot read is treated as a
    similarity.

    Qdrant point ids are integers or UUIDs, so a corpus keyed by string chunk
    ids normally carries the real id in the payload. Pass ``id_field`` to read
    chunk ids from that payload key; without it, ``str(point.id)`` is used. A
    point missing the payload key falls back to ``str(point.id)`` too, so one
    unlabelled point cannot collapse a whole result set.

    Args:
        client: A ``qdrant_client.QdrantClient`` instance.
        collection_name: The collection to query.
        embed: Callable turning a query string into a vector.
        fingerprint: The configuration fingerprint to report.
        id_field: Payload key holding the chunk id. Defaults to the point id.
        vector_name: Named vector to query, for collections that define several.
    """

    def __init__(
        self,
        client: object,
        collection_name: str,
        embed: Callable[[str], Sequence[float]],
        fingerprint: ConfigFingerprint,
        *,
        id_field: str | None = None,
        vector_name: str | None = None,
    ) -> None:
        require(BACKEND_MODULE, backend="qdrant", extra=EXTRA)
        self._client = client
        self._collection_name = collection_name
        self._embed = embed
        self._fingerprint = fingerprint
        self._id_field = id_field
        self._vector_name = vector_name
        self._lower_is_better: bool | None = None

    def _scores_are_distances(self, client: _QdrantClient) -> bool:
        """Return whether this collection's ``score`` is a distance, cached."""
        if self._lower_is_better is None:
            info = client.get_collection(self._collection_name)
            params = getattr(getattr(info, "config", None), "params", None)
            metric = _distance_name(params, self._vector_name)
            self._lower_is_better = metric in _DISTANCE_METRICS
        return self._lower_is_better

    def _chunk_id(self, point: _ScoredPoint) -> str:
        """Return the chunk id for ``point``, preferring the payload key."""
        if self._id_field is not None and point.payload is not None:
            value = point.payload.get(self._id_field)
            if value is not None:
                return str(value)
        return str(point.id)

    def search(self, query: str, k: int) -> list[ScoredHit]:
        """Return the rank-ordered top-``k`` hits for ``query``.

        Embeds ``query``, asks Qdrant for the nearest points, and maps them onto
        chunk ids and higher-is-better scores. Handles ``k`` larger than the
        collection size and an empty collection by returning fewer than ``k``
        (or zero) hits rather than raising.
        """
        if k <= 0:
            return []
        client = cast(_QdrantClient, self._client)
        count = int(client.count(self._collection_name, exact=True).count)
        if count <= 0:
            return []
        negate = self._scores_are_distances(client)
        response = client.query_points(
            collection_name=self._collection_name,
            query=list(self._embed(query)),
            using=self._vector_name,
            limit=min(k, count),
            with_payload=self._id_field is not None,
        )
        scores: dict[str, float] = {}
        for point in response.points:
            score = float(point.score)
            scores[self._chunk_id(point)] = -score if negate else score
        return _rank_hits(scores, k)

    def fingerprint(self) -> ConfigFingerprint:
        """Return the configured fingerprint."""
        return self._fingerprint


__all__ = ["BACKEND_MODULE", "EXTRA", "QdrantRetriever"]
