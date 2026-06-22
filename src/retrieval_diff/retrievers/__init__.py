"""Retriever protocol and built-in implementations.

A :class:`Retriever` is anything that can answer ``search(query, k)`` with a
rank-ordered list of :class:`~retrieval_diff.types.ScoredHit` and report its
:class:`~retrieval_diff.fingerprint.ConfigFingerprint`. The package ships
deterministic, offline in-memory retrievers (:mod:`retrieval_diff.retrievers.memory`)
and a :class:`~retrieval_diff.retrievers.factory.RetrieverFactory` for held-fixed
replay during attribution.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.types import ScoredHit


@runtime_checkable
class Retriever(Protocol):
    """Structural type for any retriever the tool can snapshot.

    Implementations must be deterministic: identical ``(query, k)`` inputs must
    yield identical hits (including tie-breaking) so snapshots are reproducible.
    """

    def search(self, query: str, k: int) -> list[ScoredHit]:
        """Return the rank-ordered top-``k`` hits for ``query``.

        Hits must be sorted by descending score with ties broken by ascending
        chunk id, and ranks must be 0-based and contiguous.
        """
        ...

    def fingerprint(self) -> ConfigFingerprint:
        """Return the configuration fingerprint of this retriever."""
        ...


__all__ = ["Retriever"]
