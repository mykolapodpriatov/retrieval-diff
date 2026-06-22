"""Core data types for snapshots, diffs, and attribution.

These are the value objects exchanged between the snapshot, diff, attribution,
budget, and report layers. They are intentionally plain pydantic models so they
serialize cleanly into the versioned lockfile and so equality is structural.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from retrieval_diff.fingerprint import ConfigFingerprint


class ChangeKind(StrEnum):
    """The kind of change a chunk underwent between two snapshots.

    A common chunk may carry **both** ``REORDERED`` and ``SCORE_CHANGED``;
    ``ADDED``, ``REMOVED`` and ``UNCHANGED`` are mutually exclusive singletons.
    """

    ADDED = "added"
    REMOVED = "removed"
    REORDERED = "reordered"
    SCORE_CHANGED = "score_changed"
    UNCHANGED = "unchanged"


class ScoredHit(BaseModel):
    """A single retrieved chunk with its score and 0-based rank."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    score: float
    rank: int = Field(ge=0)


class QueryResult(BaseModel):
    """The rank-ordered top-K hits for one query (length ``<= k``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    hits: tuple[ScoredHit, ...]

    def ids(self) -> list[str]:
        """Return hit ids in rank order."""
        return [hit.id for hit in self.hits]

    def rank_of(self) -> dict[str, int]:
        """Return a mapping of chunk id -> rank."""
        return {hit.id: hit.rank for hit in self.hits}

    def score_of(self) -> dict[str, float]:
        """Return a mapping of chunk id -> score."""
        return {hit.id: hit.score for hit in self.hits}


class Snapshot(BaseModel):
    """A versioned capture of a retriever's top-K output over a query set.

    ``created_label`` is supplied by the caller (e.g. a git SHA); the library
    never reads the wall clock, which keeps snapshots reproducible.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = 1
    created_label: str
    fingerprint: ConfigFingerprint
    k: int = Field(gt=0)
    results: dict[str, QueryResult]

    def queries(self) -> set[str]:
        """Return the set of queries captured in this snapshot."""
        return set(self.results)


class QueryDiff(BaseModel):
    """The per-chunk diff for a single query between two snapshots."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    kinds: dict[str, set[ChangeKind]]
    rank_delta: dict[str, int]
    score_delta: dict[str, float]
    churn: float = Field(ge=0.0, le=1.0)

    def ids_with(self, kind: ChangeKind) -> list[str]:
        """Return sorted chunk ids whose change set contains ``kind``."""
        return sorted(cid for cid, kinds in self.kinds.items() if kind in kinds)


class QuerySetDelta(BaseModel):
    """Queries that appeared in or disappeared from the new snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    added_queries: list[str] = Field(default_factory=list)
    removed_queries: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        """Return whether the query set is unchanged."""
        return not self.added_queries and not self.removed_queries


class GoldenMovement(BaseModel):
    """The movement of a declared golden chunk for a query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    golden_id: str
    old_rank: int | None
    new_rank: int | None
    rank_delta: int | None
    removed: bool


class DiffSummary(BaseModel):
    """Aggregate statistics over a whole-snapshot diff."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind_counts: dict[ChangeKind, int]
    mean_churn: float
    max_churn: float
    golden_movements: list[GoldenMovement] = Field(default_factory=list)


class SnapshotDiff(BaseModel):
    """The full diff between two snapshots over the query-set intersection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    per_query: dict[str, QueryDiff]
    query_set_delta: QuerySetDelta
    summary: DiffSummary
    fingerprint_delta: list[str]
    k: int = Field(gt=0)


#: Attribution confidence verdict.
Confidence = Literal["confirmed", "ambiguous", "not_attributable"]


class ChangeRef(BaseModel):
    """A reference to a single per-chunk change being attributed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    chunk_id: str
    kind: ChangeKind


class AxisAttribution(BaseModel):
    """The attribution verdict for a single change.

    ``axis`` is empty when ``confidence`` is ``not_attributable`` and no single
    axis explains the change.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    change_ref: ChangeRef
    axis: str
    confidence: Confidence
    evidence: dict[str, object] = Field(default_factory=dict)
