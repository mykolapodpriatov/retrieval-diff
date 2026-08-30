"""Tests for the pgvector adapter's real query path.

Gated behind the ``pg`` extra: skipped cleanly via ``pytest.importorskip`` when
``psycopg`` is not installed (the default CI test matrix does not install
optional adapter extras).

Two layers. The first drives the adapter through a fake connection, so the SQL
it builds, its parameter binding, its score mapping and its connection
ownership are all checked without a database. The second runs the same adapter
against a real PostgreSQL + pgvector, and is skipped unless
``RETRIEVAL_DIFF_PG_DSN`` points at one -- the ``pgvector`` CI job sets it.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import pytest

pytest.importorskip("psycopg")

from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.retrievers.adapters.pgvector_adapter import (
    PgVectorRetriever,
    _quote_identifier,
    _vector_literal,
)

QUERY_VECTOR = [1.0, 0.0]
PG_DSN = os.environ.get("RETRIEVAL_DIFF_PG_DSN")


def _embed(_query: str) -> Sequence[float]:
    """Return a fixed query vector so tests never load a real model."""
    return QUERY_VECTOR


class _FakeCursor:
    """Records the statement it was given and replays canned rows."""

    def __init__(self, owner: _FakeConnection) -> None:
        self._owner = owner

    def execute(self, query: str, params: Sequence[object]) -> None:
        self._owner.statements.append((query, tuple(params)))

    def fetchall(self) -> Sequence[Sequence[object]]:
        return self._owner.rows

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


class _FakeConnection:
    """Minimal stand-in for a psycopg connection."""

    def __init__(self, rows: Sequence[Sequence[object]] = ()) -> None:
        self.rows = rows
        self.statements: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def close(self) -> None:
        self.closed = True


def _retriever(connection: _FakeConnection, **kwargs: object) -> PgVectorRetriever:
    return PgVectorRetriever(
        "postgresql:///unused",
        "chunks",
        _embed,
        ConfigFingerprint(),
        connection=connection,
        **kwargs,  # type: ignore[arg-type]
    )


# --- identifier and literal helpers ----------------------------------------


@pytest.mark.parametrize("name", ["id", "chunk_id", "public.chunks", "_x$1"])
def test_valid_identifiers_are_quoted(name: str) -> None:
    """Each dotted part is quoted separately."""
    assert _quote_identifier(name) == ".".join(f'"{part}"' for part in name.split("."))


@pytest.mark.parametrize(
    "name",
    ['chunks"; DROP TABLE chunks; --', "chunks chunks", "1chunks", "", "a.", "a-b"],
)
def test_invalid_identifiers_are_rejected(name: str) -> None:
    """Identifiers are rejected outright rather than escaped."""
    with pytest.raises(ValueError, match="not a valid SQL identifier"):
        _quote_identifier(name)


def test_vector_literal_round_trips_floats() -> None:
    """The literal is pgvector's bracketed text form, full precision."""
    assert _vector_literal([1, 0.5]) == "[1.0,0.5]"


# --- query construction and mapping, without a database ---------------------


def test_search_binds_the_vector_and_limit_as_parameters() -> None:
    """The vector and k are bound, never interpolated into the statement."""
    connection = _FakeConnection([("a", 0.0)])
    _retriever(connection).search("q", k=3)

    statement, params = connection.statements[0]
    assert params == ("[1.0,0.0]", 3)
    assert "%s::vector" in statement
    assert "[1.0,0.0]" not in statement


@pytest.mark.parametrize(
    ("metric", "operator"),
    [("cosine", "<=>"), ("l2", "<->"), ("ip", "<#>")],
)
def test_metric_selects_its_operator(metric: str, operator: str) -> None:
    """Each supported metric maps onto its pgvector operator."""
    connection = _FakeConnection([("a", 0.0)])
    _retriever(connection, metric=metric).search("q", k=1)

    statement, _params = connection.statements[0]
    assert f'"embedding" {operator} %s::vector' in statement
    assert "ORDER BY distance ASC LIMIT %s" in statement


def test_unsupported_metric_is_rejected_at_construction() -> None:
    """A bad metric fails fast, not on the first query."""
    with pytest.raises(ValueError, match="unsupported metric"):
        _retriever(_FakeConnection(), metric="jaccard")


def test_custom_table_and_columns_are_quoted_into_the_statement() -> None:
    """Table and column names reach the statement quoted."""
    connection = _FakeConnection([("a", 0.0)])
    PgVectorRetriever(
        "postgresql:///unused",
        "public.docs",
        _embed,
        ConfigFingerprint(),
        id_column="chunk_id",
        embedding_column="vec",
        connection=connection,
    ).search("q", k=1)

    statement, _params = connection.statements[0]
    assert 'SELECT "chunk_id", "vec" <=>' in statement
    assert 'FROM "public"."docs"' in statement


def test_invalid_table_is_rejected_at_construction() -> None:
    """An injection attempt in the table name never reaches a query."""
    with pytest.raises(ValueError, match="not a valid SQL identifier"):
        PgVectorRetriever(
            "postgresql:///unused",
            "chunks; DROP TABLE chunks",
            _embed,
            ConfigFingerprint(),
            connection=_FakeConnection(),
        )


def test_distances_are_negated_into_ranked_scores() -> None:
    """Rows come back as descending scores with 0-based contiguous ranks."""
    connection = _FakeConnection([("a", 0.0), ("b", 0.25), ("c", 1.0)])
    hits = _retriever(connection).search("q", k=3)

    assert [hit.id for hit in hits] == ["a", "b", "c"]
    assert [hit.rank for hit in hits] == [0, 1, 2]
    assert [hit.score for hit in hits] == [0.0, -0.25, -1.0]


def test_tie_break_ascending_chunk_id() -> None:
    """Equal-distance rows are ordered by chunk id, not by row order."""
    connection = _FakeConnection([("z", 0.5), ("a", 0.5), ("m", 0.5)])
    hits = _retriever(connection).search("q", k=3)

    assert [hit.id for hit in hits] == ["a", "m", "z"]
    assert [hit.rank for hit in hits] == [0, 1, 2]


def test_empty_table_returns_no_hits() -> None:
    """No rows means no hits, not an error."""
    assert _retriever(_FakeConnection([])).search("q", k=5) == []


def test_non_positive_k_never_reaches_the_database() -> None:
    """A non-positive ``k`` short-circuits before any statement is executed."""
    connection = _FakeConnection([("a", 0.0)])

    assert _retriever(connection).search("q", k=0) == []
    assert connection.statements == []


def test_close_leaves_an_injected_connection_open() -> None:
    """A caller-owned connection is the caller's to close."""
    connection = _FakeConnection([("a", 0.0)])
    retriever = _retriever(connection)
    retriever.search("q", k=1)
    retriever.close()

    assert connection.closed is False


def test_close_closes_a_connection_the_adapter_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection opened from the DSN is closed, and closing twice is safe."""
    import psycopg

    opened = _FakeConnection([("a", 0.0)])
    monkeypatch.setattr(psycopg, "connect", lambda _dsn: opened)
    retriever = PgVectorRetriever("postgresql:///unused", "chunks", _embed, ConfigFingerprint())

    retriever.search("q", k=1)
    retriever.close()
    retriever.close()

    assert opened.closed is True


def test_fingerprint_passthrough() -> None:
    """``fingerprint()`` returns exactly what the retriever was constructed with."""
    fp = ConfigFingerprint(embedding_model="pg-test")
    retriever = PgVectorRetriever(
        "postgresql:///unused", "chunks", _embed, fp, connection=_FakeConnection()
    )

    assert retriever.fingerprint() == fp


# --- against a real PostgreSQL + pgvector -----------------------------------

pg = pytest.mark.skipif(PG_DSN is None, reason="RETRIEVAL_DIFF_PG_DSN is not set")


@pytest.fixture
def pg_table() -> object:
    """Create a throwaway pgvector table and drop it afterwards."""
    import psycopg

    assert PG_DSN is not None
    connection = psycopg.connect(PG_DSN)
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute("DROP TABLE IF EXISTS rd_chunks")
        cursor.execute("CREATE TABLE rd_chunks (id text PRIMARY KEY, embedding vector(2))")
        cursor.execute(
            "INSERT INTO rd_chunks (id, embedding) VALUES "
            "('a', '[1,0]'), ('b', '[0.9,0.1]'), ('c', '[0,1]'), ('d', '[-1,0]')"
        )
    connection.commit()
    try:
        yield connection
    finally:
        with connection.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS rd_chunks")
        connection.commit()
        connection.close()


@pg
@pytest.mark.parametrize("metric", ["cosine", "l2", "ip"])
def test_real_query_orders_by_similarity(pg_table: object, metric: str) -> None:
    """Every metric ranks the exact match first and stays higher-is-better."""
    assert PG_DSN is not None
    retriever = PgVectorRetriever(
        PG_DSN, "rd_chunks", _embed, ConfigFingerprint(), metric=metric, connection=pg_table
    )

    hits = retriever.search("q", k=3)

    assert [hit.id for hit in hits] == ["a", "b", "c"]
    assert [hit.rank for hit in hits] == [0, 1, 2]
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


@pg
def test_real_query_is_deterministic(pg_table: object) -> None:
    """Identical ``(query, k)`` inputs yield identical hits, as snapshots require."""
    assert PG_DSN is not None
    retriever = PgVectorRetriever(
        PG_DSN, "rd_chunks", _embed, ConfigFingerprint(), connection=pg_table
    )

    assert retriever.search("q", k=4) == retriever.search("q", k=4)


@pg
def test_real_query_with_k_larger_than_the_table(pg_table: object) -> None:
    """Asking for more rows than exist returns what's there."""
    assert PG_DSN is not None
    retriever = PgVectorRetriever(
        PG_DSN, "rd_chunks", _embed, ConfigFingerprint(), connection=pg_table
    )

    hits = retriever.search("q", k=99)

    assert [hit.id for hit in hits] == ["a", "b", "c", "d"]
    assert [hit.rank for hit in hits] == [0, 1, 2, 3]


@pg
def test_real_connection_is_opened_from_the_dsn_when_not_injected(pg_table: object) -> None:
    """Without an injected connection the adapter opens (and closes) its own."""
    assert PG_DSN is not None
    retriever = PgVectorRetriever(PG_DSN, "rd_chunks", _embed, ConfigFingerprint())
    try:
        hits = retriever.search("q", k=2)
    finally:
        retriever.close()

    assert [hit.id for hit in hits] == ["a", "b"]
