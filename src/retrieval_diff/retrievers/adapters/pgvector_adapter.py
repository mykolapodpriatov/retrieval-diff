"""pgvector adapter stub (behind the ``[pg]`` extra).

Wraps a PostgreSQL + ``pgvector`` table as a
:class:`~retrieval_diff.retrievers.Retriever`. The ``psycopg`` import is guarded;
construction raises
:class:`~retrieval_diff.retrievers.adapters.MissingDependencyError` until the
backend is installed and the adapter is implemented.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.retrievers.adapters import require
from retrieval_diff.types import ScoredHit

#: pip extra that provides this adapter's backend.
EXTRA = "pg"
#: Backend module name, import-guarded.
BACKEND_MODULE = "psycopg"


class PgVectorRetriever:
    """Adapter exposing a pgvector table as a retriever (M4 stub).

    Args:
        dsn: A PostgreSQL connection string.
        table: The table holding ``(id, embedding)`` rows.
        embed: Callable turning a query string into a vector.
        fingerprint: The configuration fingerprint to report.
    """

    def __init__(
        self,
        dsn: str,
        table: str,
        embed: Callable[[str], Sequence[float]],
        fingerprint: ConfigFingerprint,
    ) -> None:
        require(BACKEND_MODULE, backend="pgvector", extra=EXTRA)
        self._dsn = dsn
        self._table = table
        self._embed = embed
        self._fingerprint = fingerprint

    def search(self, query: str, k: int) -> list[ScoredHit]:
        """Return the top-``k`` hits for ``query`` (not yet implemented)."""
        raise NotImplementedError(
            "PgVectorRetriever.search is an M4 stub; the pgvector query path is "
            "implemented in the next milestone"
        )

    def fingerprint(self) -> ConfigFingerprint:
        """Return the configured fingerprint."""
        return self._fingerprint


__all__ = ["BACKEND_MODULE", "EXTRA", "PgVectorRetriever"]
