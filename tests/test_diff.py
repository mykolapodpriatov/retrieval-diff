"""Tests for the diff engine: change kinds, deltas, churn, and edge cases."""

from __future__ import annotations

import pytest

from rdiff_testkit import QUERIES, make_snapshot
from retrieval_diff.diff import (
    DEFAULT_SCORE_EPS,
    EmptyIntersectionError,
    KMismatchError,
    compute_churn,
    diff_snapshots,
)
from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.snapshot import snapshot
from retrieval_diff.types import ChangeKind

# --- change kinds -----------------------------------------------------------


def test_diff_query_rejects_mismatched_queries() -> None:
    """diff_query guards against being handed two different queries."""
    from retrieval_diff.diff import diff_query
    from retrieval_diff.types import QueryResult, ScoredHit

    a = QueryResult(query="q1", hits=(ScoredHit(id="a", score=0.5, rank=0),))
    b = QueryResult(query="q2", hits=(ScoredHit(id="a", score=0.5, rank=0),))
    with pytest.raises(ValueError, match="mismatched queries"):
        diff_query(a, b, k=1)


def test_added_and_removed_chunks() -> None:
    """A chunk only in new is ADDED; only in old is REMOVED."""
    old = make_snapshot({"q": [("a", 0.9), ("b", 0.5)]}, k=2)
    new = make_snapshot({"q": [("a", 0.9), ("c", 0.4)]}, k=2)
    qd = diff_snapshots(old, new).per_query["q"]
    assert qd.kinds["c"] == {ChangeKind.ADDED}
    assert qd.kinds["b"] == {ChangeKind.REMOVED}
    assert qd.kinds["a"] == {ChangeKind.UNCHANGED}


def test_reordered_only() -> None:
    """A rank change with an equal score is REORDERED, not SCORE_CHANGED."""
    old = make_snapshot({"q": [("a", 0.9), ("b", 0.8)]}, k=2)
    new = make_snapshot({"q": [("b", 0.8), ("a", 0.9)]}, k=2)
    qd = diff_snapshots(old, new).per_query["q"]
    assert qd.kinds["a"] == {ChangeKind.REORDERED}
    assert qd.kinds["b"] == {ChangeKind.REORDERED}
    assert qd.rank_delta["a"] == 1
    assert qd.rank_delta["b"] == -1
    assert qd.score_delta["a"] == pytest.approx(0.0)


def test_score_changed_only() -> None:
    """A score change at the same rank is SCORE_CHANGED, not REORDERED."""
    old = make_snapshot({"q": [("a", 0.9), ("b", 0.5)]}, k=2)
    new = make_snapshot({"q": [("a", 0.7), ("b", 0.5)]}, k=2)
    qd = diff_snapshots(old, new).per_query["q"]
    assert qd.kinds["a"] == {ChangeKind.SCORE_CHANGED}
    assert qd.kinds["b"] == {ChangeKind.UNCHANGED}
    assert qd.score_delta["a"] == pytest.approx(-0.2)
    assert qd.rank_delta["a"] == 0


def test_reordered_and_score_changed_together() -> None:
    """One id can be BOTH reordered and score_changed simultaneously."""
    old = make_snapshot({"q": [("a", 0.9), ("b", 0.8)]}, k=2)
    new = make_snapshot({"q": [("b", 0.95), ("a", 0.85)]}, k=2)
    qd = diff_snapshots(old, new).per_query["q"]
    assert qd.kinds["a"] == {ChangeKind.REORDERED, ChangeKind.SCORE_CHANGED}
    assert qd.kinds["b"] == {ChangeKind.REORDERED, ChangeKind.SCORE_CHANGED}


def test_unchanged() -> None:
    """Identical results yield UNCHANGED for every id and zero churn."""
    snap = make_snapshot({"q": [("a", 0.9), ("b", 0.5)]}, k=2)
    diff = diff_snapshots(snap, snap)
    qd = diff.per_query["q"]
    assert all(kinds == {ChangeKind.UNCHANGED} for kinds in qd.kinds.values())
    assert diff.summary.mean_churn == 0.0


# --- churn formula ----------------------------------------------------------


def test_compute_churn_basic_formula() -> None:
    """sum(|delta|) / (common * (K-1)), bounded to [0, 1]."""
    # two common ids each displaced by 1 over K=3 -> 2 / (2 * 2) = 0.5
    assert compute_churn({"a": 1, "b": -1}, common_count=2, k=3) == pytest.approx(0.5)


def test_compute_churn_full_displacement_caps_at_one() -> None:
    """The denominator is the max displacement, so churn never exceeds 1.0."""
    # K=2, one common id displaced by 1 -> 1 / (1 * 1) = 1.0
    assert compute_churn({"a": 1}, common_count=1, k=2) == pytest.approx(1.0)


def test_compute_churn_k_le_1_is_zero() -> None:
    """K<=1 short-circuits to 0.0 (no displacement is possible)."""
    assert compute_churn({"a": 0}, common_count=1, k=1) == 0.0


def test_compute_churn_empty_common_is_zero() -> None:
    """No common ids short-circuits to 0.0 in the helper."""
    assert compute_churn({}, common_count=0, k=5) == 0.0


def test_churn_empty_common_equal_sets_is_zero() -> None:
    """Disjoint-but-equal? Impossible; equal sets with no overlap means empty top-K."""
    old = make_snapshot({"q": []}, k=3)
    new = make_snapshot({"q": []}, k=3)
    assert diff_snapshots(old, new).per_query["q"].churn == 0.0


def test_churn_empty_common_full_replacement_is_one() -> None:
    """When top-K is fully replaced (no overlap), churn is 1.0."""
    old = make_snapshot({"q": [("a", 0.9), ("b", 0.8)]}, k=2)
    new = make_snapshot({"q": [("c", 0.9), ("d", 0.8)]}, k=2)
    assert diff_snapshots(old, new).per_query["q"].churn == pytest.approx(1.0)


def test_churn_k1_equal_sets_is_zero() -> None:
    """K=1 with the same single id is churn 0.0 even though ranks can't move."""
    old = make_snapshot({"q": [("a", 0.9)]}, k=1)
    new = make_snapshot({"q": [("a", 0.5)]}, k=1)
    assert diff_snapshots(old, new).per_query["q"].churn == 0.0


def test_churn_k1_different_id_is_one() -> None:
    """K=1 with a different id is a full replacement (churn 1.0)."""
    old = make_snapshot({"q": [("a", 0.9)]}, k=1)
    new = make_snapshot({"q": [("b", 0.9)]}, k=1)
    assert diff_snapshots(old, new).per_query["q"].churn == pytest.approx(1.0)


def test_churn_is_in_unit_interval_for_random_like_case() -> None:
    """Churn stays within [0, 1] for a mixed add/remove/reorder case."""
    old = make_snapshot({"q": [("a", 0.9), ("b", 0.8), ("c", 0.7)]}, k=3)
    new = make_snapshot({"q": [("c", 0.95), ("a", 0.85), ("d", 0.6)]}, k=3)
    churn = diff_snapshots(old, new).per_query["q"].churn
    assert 0.0 <= churn <= 1.0


# --- K mismatch & empty intersection ----------------------------------------


def test_k_mismatch_raises() -> None:
    """Diffing snapshots captured at different K is a hard error."""
    old = make_snapshot({"q": [("a", 0.9)]}, k=2)
    new = make_snapshot({"q": [("a", 0.9)]}, k=3)
    with pytest.raises(KMismatchError, match="K mismatch"):
        diff_snapshots(old, new)


def test_empty_intersection_raises() -> None:
    """No shared queries is the only query-set condition that errors."""
    old = make_snapshot({"q1": [("a", 0.9)]}, k=1)
    new = make_snapshot({"q2": [("a", 0.9)]}, k=1)
    with pytest.raises(EmptyIntersectionError, match="share no queries"):
        diff_snapshots(old, new)


# --- query-set delta --------------------------------------------------------


def test_query_set_delta_reported_and_intersection_diffed() -> None:
    """Added/removed queries are reported; only the intersection is diffed."""
    old = make_snapshot({"shared": [("a", 0.9)], "gone": [("x", 0.5)]}, k=1)
    new = make_snapshot({"shared": [("a", 0.9)], "fresh": [("y", 0.5)]}, k=1)
    diff = diff_snapshots(old, new)
    assert diff.query_set_delta.added_queries == ["fresh"]
    assert diff.query_set_delta.removed_queries == ["gone"]
    assert set(diff.per_query) == {"shared"}


# --- baseline reproducibility & epsilon -------------------------------------


def test_baseline_reproducibility_with_real_retriever(hybrid) -> None:  # type: ignore[no-untyped-def]
    """Snapshotting the same retriever twice yields an empty diff."""
    a = snapshot(hybrid, QUERIES, 4, label="run-a")
    b = snapshot(hybrid, QUERIES, 4, label="run-b")
    diff = diff_snapshots(a, b)
    assert diff.summary.mean_churn == 0.0
    for qd in diff.per_query.values():
        assert all(kinds == {ChangeKind.UNCHANGED} for kinds in qd.kinds.values())


def test_score_drift_just_under_eps_is_unchanged() -> None:
    """A score delta below score_eps does not register as SCORE_CHANGED."""
    old = make_snapshot({"q": [("a", 0.5)]}, k=1)
    new = make_snapshot({"q": [("a", 0.5 + DEFAULT_SCORE_EPS / 2)]}, k=1)
    qd = diff_snapshots(old, new).per_query["q"]
    assert qd.kinds["a"] == {ChangeKind.UNCHANGED}


def test_score_drift_just_over_eps_is_score_changed() -> None:
    """A score delta above score_eps registers as SCORE_CHANGED."""
    old = make_snapshot({"q": [("a", 0.5)]}, k=1)
    new = make_snapshot({"q": [("a", 0.5 + DEFAULT_SCORE_EPS * 2)]}, k=1)
    qd = diff_snapshots(old, new).per_query["q"]
    assert qd.kinds["a"] == {ChangeKind.SCORE_CHANGED}


def test_deterministic_ties_ordered_by_id() -> None:
    """Equal-score hits are ordered by ascending id, so ties are reproducible."""
    from retrieval_diff.retrievers.memory import _rank_hits

    # Equal scores: id order decides ranks deterministically regardless of the
    # input iteration order -> stable, reproducible ranking with no RNG/clock.
    assert [h.id for h in _rank_hits({"b": 0.5, "a": 0.5, "c": 0.5}, 3)] == [
        "a",
        "b",
        "c",
    ]
    assert [h.id for h in _rank_hits({"c": 0.5, "b": 0.5, "a": 0.5}, 3)] == [
        "a",
        "b",
        "c",
    ]


def test_deterministic_ties_end_to_end(corpus, embedder) -> None:  # type: ignore[no-untyped-def]
    """A real retriever snapshotted twice agrees on tie order."""
    from retrieval_diff.retrievers.memory import HybridRetriever

    retriever = HybridRetriever(corpus, embedder, alpha=0.0)
    first = snapshot(retriever, ["the"], 4, label="t1")
    second = snapshot(retriever, ["the"], 4, label="t2")
    assert first.results["the"].ids() == second.results["the"].ids()


# --- fingerprint delta & golden movements -----------------------------------


def test_fingerprint_delta_lists_changed_axes() -> None:
    """The diff lists exactly the fingerprint axes that differ."""
    old = make_snapshot(
        {"q": [("a", 0.9)]},
        k=1,
        fingerprint=ConfigFingerprint(embedding_model="m1", alpha=0.5),
    )
    new = make_snapshot(
        {"q": [("a", 0.9)]},
        k=1,
        fingerprint=ConfigFingerprint(embedding_model="m2", alpha=0.5),
    )
    diff = diff_snapshots(old, new)
    assert diff.fingerprint_delta == ["embedding_model"]


def test_golden_movement_summary() -> None:
    """Declared goldens produce a movement entry with the rank delta."""
    old = make_snapshot({"q": [("g", 0.9), ("a", 0.8), ("b", 0.7)]}, k=3)
    new = make_snapshot({"q": [("a", 0.9), ("b", 0.8), ("g", 0.7)]}, k=3)
    diff = diff_snapshots(old, new, goldens={"q": {"g"}})
    movements = diff.summary.golden_movements
    assert len(movements) == 1
    assert movements[0].golden_id == "g"
    assert movements[0].old_rank == 0
    assert movements[0].new_rank == 2
    assert movements[0].rank_delta == 2
    assert movements[0].removed is False


def test_golden_removed_movement() -> None:
    """A golden that drops out of top-K is flagged removed with no new rank."""
    old = make_snapshot({"q": [("g", 0.9), ("a", 0.8)]}, k=2)
    new = make_snapshot({"q": [("a", 0.9), ("b", 0.8)]}, k=2)
    diff = diff_snapshots(old, new, goldens={"q": {"g"}})
    move = diff.summary.golden_movements[0]
    assert move.removed is True
    assert move.new_rank is None
    assert move.rank_delta is None
