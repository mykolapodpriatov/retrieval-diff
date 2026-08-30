"""pgvector adapter (behind the ``[pg]`` extra).

Wraps a PostgreSQL + ``pgvector`` table as a
:class:`~retrieval_diff.retrievers.Retriever`. The ``psycopg`` import is guarded
so this module loads without the dependency; construction raises
:class:`~retrieval_diff.retrievers.adapters.MissingDependencyError` until the
backend is installed.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.retrievers.adapters import require
from retrieval_diff.types import ScoredHit

#: pip extra that provides this adapter's backend.
EXTRA = "pg"
#: Backend module name, import-guarded.
BACKEND_MODULE = "psycopg"

#: Supported metrics mapped to their pgvector operator. Every one of these
#: operators is lower-is-better -- ``<#>`` returns the *negative* inner product
#: -- so a single negation converts any of them into this package's scores.
_OPERATORS: Mapping[str, str] = {
    "cosine": "<=>",
    "l2": "<->",
    "ip": "<#>",
}

#: Unquoted SQL identifier: a letter or underscore, then letters/digits/_/$.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


class _Cursor(Protocol):
    """Structural type for the subset of a psycopg cursor this adapter calls."""

    def execute(self, query: str, params: Sequence[object]) -> object:
        """Execute ``query`` with the given bound parameters."""
        ...

    def fetchall(self) -> Sequence[Sequence[object]]:
        """Return every remaining result row."""
        ...

    def __enter__(self) -> _Cursor: ...

    def __exit__(self, *exc: object) -> None: ...


class _Connection(Protocol):
    """Structural type for the subset of a psycopg connection this adapter calls."""

    def cursor(self) -> _Cursor:
        """Return a new cursor."""
        ...

    def close(self) -> None:
        """Close the connection."""
        ...


class _Psycopg(Protocol):
    """Structural type for the ``psycopg`` module surface this adapter calls."""

    def connect(self, conninfo: str) -> _Connection:
        """Open a connection to ``conninfo``."""
        ...


def _quote_identifier(name: str) -> str:
    """Return ``name`` as a quoted SQL identifier, or raise ``ValueError``.

    Table and column names cannot be bound as query parameters, so they are
    validated against :data:`_IDENTIFIER` and then double-quoted. A dotted name
    is treated as ``schema.table`` and each part is validated separately.
    Anything else -- whitespace, quotes, semicolons -- is rejected outright
    rather than escaped, so no caller-supplied string can reach the query
    unvalidated.
    """
    parts = name.split(".")
    if not all(_IDENTIFIER.match(part) for part in parts):
        raise ValueError(f"not a valid SQL identifier: {name!r}")
    return ".".join(f'"{part}"' for part in parts)


def _vector_literal(vector: Sequence[float]) -> str:
    """Render ``vector`` as a pgvector text literal such as ``[1.0,0.0]``.

    Passed as a bound parameter and cast with ``::vector`` in the query, so the
    adapter works without ``pgvector.psycopg.register_vector``.
    """
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


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


class PgVectorRetriever:
    """Adapter exposing a pgvector table as a retriever.

    Queries are a single ``ORDER BY <distance> LIMIT k`` over ``table``, so
    pgvector's index (if any) does the work. All three supported operators are
    lower-is-better — ``<#>`` returns the *negative* inner product — so scores
    are the negated operator result, matching this package's higher-is-better
    convention.

    The connection is opened lazily on the first search and reused. Pass
    ``connection`` to reuse one you already own (from a pool, say); an injected
    connection is never closed by :meth:`close`.

    Args:
        dsn: A PostgreSQL connection string.
        table: The table holding ``(id, embedding)`` rows. May be
            ``schema.table``.
        embed: Callable turning a query string into a vector.
        fingerprint: The configuration fingerprint to report.
        id_column: Column holding the chunk id.
        embedding_column: Column holding the ``vector`` value.
        metric: One of ``cosine``, ``l2`` or ``ip``. Must match the operator
            class of the index on ``embedding_column``, or Postgres will fall
            back to a sequential scan.
        connection: An open psycopg connection to reuse instead of ``dsn``.

    Raises:
        ValueError: If ``metric`` is unsupported, or if ``table``,
            ``id_column`` or ``embedding_column`` is not a valid SQL identifier.
    """

    def __init__(
        self,
        dsn: str,
        table: str,
        embed: Callable[[str], Sequence[float]],
        fingerprint: ConfigFingerprint,
        *,
        id_column: str = "id",
        embedding_column: str = "embedding",
        metric: str = "cosine",
        connection: object | None = None,
    ) -> None:
        self._psycopg = require(BACKEND_MODULE, backend="pgvector", extra=EXTRA)
        if metric not in _OPERATORS:
            supported = ", ".join(sorted(_OPERATORS))
            raise ValueError(f"unsupported metric {metric!r}; expected one of: {supported}")
        self._dsn = dsn
        self._table = table
        self._embed = embed
        self._fingerprint = fingerprint
        self._metric = metric
        self._connection = connection
        self._owns_connection = connection is None
        self._sql = (
            f"SELECT {_quote_identifier(id_column)}, "
            f"{_quote_identifier(embedding_column)} {_OPERATORS[metric]} %s::vector AS distance "
            f"FROM {_quote_identifier(table)} "
            f"ORDER BY distance ASC LIMIT %s"
        )

    def _connect(self) -> _Connection:
        """Return the connection, opening one from ``dsn`` on first use."""
        if self._connection is None:
            self._connection = cast(_Psycopg, self._psycopg).connect(self._dsn)
        return cast(_Connection, self._connection)

    def search(self, query: str, k: int) -> list[ScoredHit]:
        """Return the rank-ordered top-``k`` hits for ``query``.

        Embeds ``query``, asks Postgres for the ``k`` nearest rows, and negates
        the distances into scores. An empty table simply yields no hits.
        """
        if k <= 0:
            return []
        vector = _vector_literal(self._embed(query))
        with self._connect().cursor() as cursor:
            cursor.execute(self._sql, (vector, k))
            rows = cursor.fetchall()
        scores: dict[str, float] = {}
        for chunk_id, distance in rows:
            scores[str(chunk_id)] = -float(cast(float, distance))
        return _rank_hits(scores, k)

    def close(self) -> None:
        """Close the connection, if this retriever opened it.

        A connection passed in as ``connection`` belongs to the caller and is
        left open. Calling this more than once is harmless.
        """
        if self._owns_connection and self._connection is not None:
            cast(_Connection, self._connection).close()
            self._connection = None

    def fingerprint(self) -> ConfigFingerprint:
        """Return the configured fingerprint."""
        return self._fingerprint


__all__ = ["BACKEND_MODULE", "EXTRA", "PgVectorRetriever"]
