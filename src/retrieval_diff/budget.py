"""The regression budget and CI gate evaluation.

A :class:`RegressionBudget` declares the thresholds a retrieval change must stay
within: removed goldens fail, golden rank drops beyond a cap fail, a never-retrieved
declared golden fails, the worst single query's churn beyond a cap fails, new
chunks may be allowed or blocked, and a non-empty query-set delta fails by default
(a newly added query has no baseline in the lock and could hide a bad result).

:func:`evaluate_budget` turns a :class:`~retrieval_diff.types.SnapshotDiff` (plus
optional ``audit_k`` golden re-ranking) into a list of
:class:`BudgetViolation`\\ s and a pass/fail :class:`BudgetReport`. Each violation
names the affected golden/query so the failure is actionable.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from retrieval_diff.retrievers import Retriever
from retrieval_diff.types import ChangeKind, Snapshot, SnapshotDiff


class ViolationKind(StrEnum):
    """The category of a budget violation."""

    REMOVED_GOLDEN = "removed_golden"
    GOLDEN_RANK_DROP = "golden_rank_drop"
    GOLDEN_DISPLACED_PAST_K = "golden_displaced_past_k"
    GOLDEN_NEVER_RETRIEVED = "golden_never_retrieved"
    CHURN_EXCEEDED = "churn_exceeded"
    NEW_CHUNK_BLOCKED = "new_chunk_blocked"
    QUERY_SET_CHANGED = "query_set_changed"


class BudgetViolation(BaseModel):
    """A single budget violation with a human-readable, actionable message."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ViolationKind
    query: str | None
    chunk_id: str | None
    message: str
    detail: dict[str, object] = Field(default_factory=dict)


class BudgetReport(BaseModel):
    """The result of evaluating a diff against a budget."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    violations: list[BudgetViolation] = Field(default_factory=list)

    def summary(self) -> str:
        """Return a one-line human summary of the report."""
        if self.passed:
            return "retrieval budget OK: no regressions"
        lines = [f"retrieval budget FAILED with {len(self.violations)} violation(s):"]
        lines.extend(f"  - {v.message}" for v in self.violations)
        return "\n".join(lines)


class RegressionBudget(BaseModel):
    """Thresholds the CI gate enforces on a retrieval diff."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    removed_golden_fails: bool = True
    max_golden_rank_drop: int = Field(default=2, ge=0)
    #: Caps the **maximum** per-query churn: no single query may regress past
    #: this. A mean would let one fully-flipped query hide behind many calm ones.
    max_churn: float = Field(default=0.5, ge=0.0, le=1.0)
    allow_new_chunks: bool = True
    query_set_change_fails: bool = True
    #: When set, declared goldens are re-ranked at this larger depth to detect a
    #: golden that was pushed just past K (invisible to a pure top-K diff).
    audit_k: int | None = Field(default=None, gt=0)


def _check_query_set(diff: SnapshotDiff, budget: RegressionBudget) -> list[BudgetViolation]:
    """Fail on a non-empty query-set delta unless explicitly allowed."""
    if not budget.query_set_change_fails or diff.query_set_delta.is_empty():
        return []
    added = diff.query_set_delta.added_queries
    removed = diff.query_set_delta.removed_queries
    return [
        BudgetViolation(
            kind=ViolationKind.QUERY_SET_CHANGED,
            query=None,
            chunk_id=None,
            message=(
                "query set changed vs lock "
                f"(added={added}, removed={removed}); "
                "pass --allow-query-set-change to migrate the lock consciously"
            ),
            detail={"added_queries": added, "removed_queries": removed},
        )
    ]


def _check_goldens(diff: SnapshotDiff, budget: RegressionBudget) -> list[BudgetViolation]:
    """Check removed goldens and golden rank drops from the diff summary."""
    violations: list[BudgetViolation] = []
    for move in diff.summary.golden_movements:
        if move.old_rank is None and move.new_rank is None:
            # Declared but absent from both snapshots' top-K: a misdeclared id
            # or a golden the retriever never surfaces. Silently passing here
            # would give false CI coverage, so flag it explicitly.
            violations.append(
                BudgetViolation(
                    kind=ViolationKind.GOLDEN_NEVER_RETRIEVED,
                    query=move.query,
                    chunk_id=move.golden_id,
                    message=(
                        f"golden chunk {move.golden_id!r} for query {move.query!r} "
                        "was never retrieved in either snapshot's top-K; "
                        "check the golden id or widen K (audit_k)"
                    ),
                    detail={"old_rank": None, "new_rank": None},
                )
            )
            continue
        if move.removed and budget.removed_golden_fails:
            violations.append(
                BudgetViolation(
                    kind=ViolationKind.REMOVED_GOLDEN,
                    query=move.query,
                    chunk_id=move.golden_id,
                    message=(
                        f"golden chunk {move.golden_id!r} for query {move.query!r} "
                        f"was removed from top-K (was rank {move.old_rank})"
                    ),
                    detail={"old_rank": move.old_rank},
                )
            )
            continue
        if move.rank_delta is not None and move.rank_delta > budget.max_golden_rank_drop:
            violations.append(
                BudgetViolation(
                    kind=ViolationKind.GOLDEN_RANK_DROP,
                    query=move.query,
                    chunk_id=move.golden_id,
                    message=(
                        f"golden chunk {move.golden_id!r} for query {move.query!r} "
                        f"dropped {move.rank_delta} ranks "
                        f"({move.old_rank} -> {move.new_rank}), "
                        f"exceeding max_golden_rank_drop={budget.max_golden_rank_drop}"
                    ),
                    detail={
                        "old_rank": move.old_rank,
                        "new_rank": move.new_rank,
                        "rank_delta": move.rank_delta,
                    },
                )
            )
    return violations


def _check_churn(diff: SnapshotDiff, budget: RegressionBudget) -> list[BudgetViolation]:
    """Fail when the worst single query's churn exceeds the cap.

    The cap bounds the **maximum** per-query churn rather than the mean: a mean
    dilutes one fully-regressed query across the whole set, so a single query
    that flipped entirely (churn ``1.0``) would slip past a small ``max_churn``
    once enough unchanged queries surround it. Capping the maximum guarantees no
    individual query may regress past the threshold.
    """
    worst_query, worst_churn = _worst_query_churn(diff)
    if worst_churn <= budget.max_churn:
        return []
    return [
        BudgetViolation(
            kind=ViolationKind.CHURN_EXCEEDED,
            query=worst_query,
            chunk_id=None,
            message=(
                f"max per-query churn {worst_churn:.4f} exceeds max_churn={budget.max_churn}"
                + (f" (query {worst_query!r})" if worst_query is not None else "")
            ),
            detail={
                "max_churn": worst_churn,
                "mean_churn": diff.summary.mean_churn,
                "budget_max_churn": budget.max_churn,
                "query": worst_query,
            },
        )
    ]


def _worst_query_churn(diff: SnapshotDiff) -> tuple[str | None, float]:
    """Return the ``(query, churn)`` of the single worst-churning query.

    Falls back to ``diff.summary.max_churn`` (with no attributable query) when
    there are no per-query diffs. Ties are broken by query name for determinism.
    """
    if not diff.per_query:
        return None, diff.summary.max_churn
    worst_query = max(diff.per_query, key=lambda q: (diff.per_query[q].churn, q))
    return worst_query, diff.per_query[worst_query].churn


def _check_new_chunks(diff: SnapshotDiff, budget: RegressionBudget) -> list[BudgetViolation]:
    """Fail on newly added chunks when ``allow_new_chunks`` is off."""
    if budget.allow_new_chunks:
        return []
    violations: list[BudgetViolation] = []
    for query in sorted(diff.per_query):
        for cid in diff.per_query[query].ids_with(ChangeKind.ADDED):
            violations.append(
                BudgetViolation(
                    kind=ViolationKind.NEW_CHUNK_BLOCKED,
                    query=query,
                    chunk_id=cid,
                    message=(
                        f"new chunk {cid!r} appeared for query {query!r} but allow_new_chunks=False"
                    ),
                    detail={},
                )
            )
    return violations


def audit_goldens_past_k(
    new_snapshot: Snapshot,
    retriever: Retriever,
    goldens: Mapping[str, set[str]],
    budget: RegressionBudget,
) -> list[BudgetViolation]:
    """Detect goldens pushed just past K via a larger ``audit_k`` re-rank.

    A golden that was at rank ``K-1`` and is now displaced to ``K+1`` by a new
    chunk vanishes from the top-K diff entirely. This re-ranks declared goldens
    at ``audit_k > K`` and flags any whose audited rank lands at or beyond K
    while still appearing within the audit window.

    Args:
        new_snapshot: The candidate snapshot (provides K and per-query top-K).
        retriever: The live retriever to re-query at ``audit_k``.
        goldens: ``{query: {golden_id, ...}}`` declarations.
        budget: The budget; no-op unless ``audit_k`` is set and ``> K``.

    Returns:
        A list of :class:`BudgetViolation` for displaced goldens.
    """
    if budget.audit_k is None or budget.audit_k <= new_snapshot.k:
        return []
    k = new_snapshot.k
    violations: list[BudgetViolation] = []
    for query in sorted(goldens):
        declared = goldens[query]
        if not declared or query not in new_snapshot.results:
            continue
        top_k_ids = set(new_snapshot.results[query].ids())
        audited = retriever.search(query, budget.audit_k)
        audit_rank = {hit.id: hit.rank for hit in audited}
        for golden_id in sorted(declared):
            if golden_id in top_k_ids:
                continue  # safely inside top-K, handled by the normal diff
            rank = audit_rank.get(golden_id)
            if rank is not None and rank >= k:
                violations.append(
                    BudgetViolation(
                        kind=ViolationKind.GOLDEN_DISPLACED_PAST_K,
                        query=query,
                        chunk_id=golden_id,
                        message=(
                            f"golden chunk {golden_id!r} for query {query!r} "
                            f"was displaced past K={k} to audited rank {rank} "
                            f"(audit_k={budget.audit_k}); it is no longer retrieved"
                        ),
                        detail={"audited_rank": rank, "k": k, "audit_k": budget.audit_k},
                    )
                )
    return violations


def evaluate_budget(
    diff: SnapshotDiff,
    budget: RegressionBudget,
    *,
    audit_violations: list[BudgetViolation] | None = None,
) -> BudgetReport:
    """Evaluate a diff against a budget and return a pass/fail report.

    Args:
        diff: The old-vs-new snapshot diff.
        budget: The thresholds to enforce.
        audit_violations: Optional pre-computed ``audit_k`` golden-displacement
            violations from :func:`audit_goldens_past_k` (kept separate because
            auditing requires a live retriever, which the pure diff lacks).

    Returns:
        A :class:`BudgetReport` whose ``violations`` are ordered deterministically
        by kind then query then chunk id.
    """
    violations: list[BudgetViolation] = []
    violations.extend(_check_query_set(diff, budget))
    violations.extend(_check_goldens(diff, budget))
    violations.extend(_check_churn(diff, budget))
    violations.extend(_check_new_chunks(diff, budget))
    if audit_violations:
        violations.extend(audit_violations)

    violations.sort(key=lambda v: (v.kind.value, v.query or "", v.chunk_id or ""))
    return BudgetReport(passed=not violations, violations=violations)


__all__ = [
    "BudgetReport",
    "BudgetViolation",
    "RegressionBudget",
    "ViolationKind",
    "audit_goldens_past_k",
    "evaluate_budget",
]
