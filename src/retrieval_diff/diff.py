"""The per-query and whole-snapshot diff engine.

Computes, for each query in the **intersection** of two snapshots' query sets,
the chunk-level changes (added / removed / reordered / score_changed /
unchanged), per-id rank and score deltas, and the normalized churn metric. The
query-set difference itself is reported (not errored) so the lock can evolve; an
empty intersection is the only hard error.

See the plan's §3.4 for the exact churn formula, reproduced in
:func:`compute_churn`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from retrieval_diff.types import (
    ChangeKind,
    DiffSummary,
    GoldenMovement,
    QueryDiff,
    QueryResult,
    QuerySetDelta,
    Snapshot,
    SnapshotDiff,
)

#: Default score-equality epsilon. Score deltas with ``abs(d) <= score_eps`` are
#: treated as unchanged, which keeps re-snapshots of the same retriever clean.
DEFAULT_SCORE_EPS: Final[float] = 1e-6


class KMismatchError(ValueError):
    """Raised when two snapshots were captured at different K.

    The churn metric and ``max_churn`` budget are only comparable at a fixed K,
    so a mismatch is a hard error rather than a silently rescaled value.
    """


class EmptyIntersectionError(ValueError):
    """Raised when two snapshots share no queries, so nothing can be diffed."""


def compute_churn(rank_delta: Mapping[str, int], common_count: int, k: int) -> float:
    """Compute the normalized rank-displacement churn in ``[0, 1]``.

    Args:
        rank_delta: ``{id: new_rank - old_rank}`` for ids present in both lists.
        common_count: The number of ids common to both top-K lists.
        k: The shared snapshot K.

    Returns:
        ``0.0`` when ``k <= 1`` or there are no common ids (the equal/replaced
        decision is made by the caller). Otherwise
        ``min(1.0, sum(|delta|) / (common_count * (k - 1)))``. The denominator is
        the maximum achievable displacement for ``common_count`` items over ``k``
        ranks, which bounds the result to ``[0, 1]``.
    """
    if k <= 1 or common_count == 0:
        return 0.0
    total = sum(abs(d) for d in rank_delta.values())
    denom = common_count * (k - 1)
    return min(1.0, total / denom)


def diff_query(
    old: QueryResult,
    new: QueryResult,
    k: int,
    *,
    score_eps: float = DEFAULT_SCORE_EPS,
) -> QueryDiff:
    """Diff one query's old vs new top-K results.

    Args:
        old: The previously recorded result.
        new: The newly captured result.
        k: The shared snapshot K (used by the churn denominator).
        score_eps: Score-equality tolerance.

    Returns:
        The :class:`QueryDiff` for this query.
    """
    if old.query != new.query:
        raise ValueError(f"diff_query received mismatched queries: {old.query!r} vs {new.query!r}")

    old_ranks = old.rank_of()
    new_ranks = new.rank_of()
    old_scores = old.score_of()
    new_scores = new.score_of()

    old_ids = set(old_ranks)
    new_ids = set(new_ranks)
    common = old_ids & new_ids

    kinds: dict[str, set[ChangeKind]] = {}
    rank_delta: dict[str, int] = {}
    score_delta: dict[str, float] = {}

    for cid in sorted(new_ids - old_ids):
        kinds[cid] = {ChangeKind.ADDED}
    for cid in sorted(old_ids - new_ids):
        kinds[cid] = {ChangeKind.REMOVED}

    for cid in sorted(common):
        delta_rank = new_ranks[cid] - old_ranks[cid]
        delta_score = new_scores[cid] - old_scores[cid]
        rank_delta[cid] = delta_rank
        score_delta[cid] = delta_score
        change: set[ChangeKind] = set()
        if delta_rank != 0:
            change.add(ChangeKind.REORDERED)
        if abs(delta_score) > score_eps:
            change.add(ChangeKind.SCORE_CHANGED)
        if not change:
            change.add(ChangeKind.UNCHANGED)
        kinds[cid] = change

    if not common:
        churn = 0.0 if old_ids == new_ids else 1.0
    else:
        churn = compute_churn(rank_delta, len(common), k)

    return QueryDiff(
        query=old.query,
        kinds=kinds,
        rank_delta=rank_delta,
        score_delta=score_delta,
        churn=churn,
    )


def _golden_movements(
    per_query: Mapping[str, QueryDiff],
    old: Snapshot,
    new: Snapshot,
    goldens: Mapping[str, set[str]],
) -> list[GoldenMovement]:
    """Compute golden-chunk movements for the queries in the intersection.

    Args:
        per_query: The computed per-query diffs.
        old: The old snapshot.
        new: The new snapshot.
        goldens: ``{query: {golden_id, ...}}`` declarations.

    Returns:
        A deterministically ordered list of :class:`GoldenMovement`.
    """
    movements: list[GoldenMovement] = []
    for query in sorted(per_query):
        declared = goldens.get(query)
        if not declared:
            continue
        old_ranks = old.results[query].rank_of()
        new_ranks = new.results[query].rank_of()
        for golden_id in sorted(declared):
            old_rank = old_ranks.get(golden_id)
            new_rank = new_ranks.get(golden_id)
            removed = old_rank is not None and new_rank is None
            delta = None if old_rank is None or new_rank is None else new_rank - old_rank
            movements.append(
                GoldenMovement(
                    query=query,
                    golden_id=golden_id,
                    old_rank=old_rank,
                    new_rank=new_rank,
                    rank_delta=delta,
                    removed=removed,
                )
            )
    return movements


def _summarize(
    per_query: Mapping[str, QueryDiff],
    movements: list[GoldenMovement],
) -> DiffSummary:
    """Aggregate per-query diffs into a :class:`DiffSummary`."""
    counts: dict[ChangeKind, int] = dict.fromkeys(ChangeKind, 0)
    churns: list[float] = []
    for query in sorted(per_query):
        qd = per_query[query]
        churns.append(qd.churn)
        for kinds in qd.kinds.values():
            for kind in kinds:
                counts[kind] += 1
    mean_churn = sum(churns) / len(churns) if churns else 0.0
    max_churn = max(churns) if churns else 0.0
    return DiffSummary(
        kind_counts=counts,
        mean_churn=mean_churn,
        max_churn=max_churn,
        golden_movements=movements,
    )


def diff_snapshots(
    old: Snapshot,
    new: Snapshot,
    *,
    score_eps: float = DEFAULT_SCORE_EPS,
    goldens: Mapping[str, set[str]] | None = None,
) -> SnapshotDiff:
    """Diff two snapshots over their query-set intersection.

    Args:
        old: The baseline snapshot.
        new: The candidate snapshot.
        score_eps: Score-equality tolerance.
        goldens: Optional ``{query: {golden_id, ...}}`` declarations for the
            golden-movement summary.

    Returns:
        A :class:`SnapshotDiff` with per-query diffs, the query-set delta, an
        aggregate summary, and the list of differing fingerprint axes.

    Raises:
        KMismatchError: If the snapshots were captured at different K.
        EmptyIntersectionError: If the snapshots share no queries.
    """
    if old.k != new.k:
        raise KMismatchError(
            f"snapshot K mismatch: old K={old.k}, new K={new.k}; "
            "churn and budgets are only comparable at a fixed K"
        )

    old_queries = old.queries()
    new_queries = new.queries()
    intersection = old_queries & new_queries

    delta = QuerySetDelta(
        added_queries=sorted(new_queries - old_queries),
        removed_queries=sorted(old_queries - new_queries),
    )

    if not intersection:
        raise EmptyIntersectionError(
            "snapshots share no queries; cannot diff "
            f"(old has {len(old_queries)}, new has {len(new_queries)})"
        )

    per_query: dict[str, QueryDiff] = {}
    for query in sorted(intersection):
        per_query[query] = diff_query(
            old.results[query], new.results[query], old.k, score_eps=score_eps
        )

    movements = _golden_movements(per_query, old, new, goldens or {})
    summary = _summarize(per_query, movements)
    fingerprint_delta = old.fingerprint.differing_axes(new.fingerprint)

    return SnapshotDiff(
        per_query=per_query,
        query_set_delta=delta,
        summary=summary,
        fingerprint_delta=fingerprint_delta,
        k=old.k,
    )
