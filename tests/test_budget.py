"""Tests for the regression budget and the audit_k golden-displacement check."""

from __future__ import annotations

from rdiff_testkit import make_snapshot
from retrieval_diff.budget import (
    RegressionBudget,
    ViolationKind,
    audit_goldens_past_k,
    evaluate_budget,
)
from retrieval_diff.diff import diff_snapshots
from retrieval_diff.types import ScoredHit


def _diff(old_data, new_data, *, k, goldens=None):  # type: ignore[no-untyped-def]
    old = make_snapshot(old_data, k=k)
    new = make_snapshot(new_data, k=k)
    return diff_snapshots(old, new, goldens=goldens or {})


# --- removed golden ---------------------------------------------------------


def test_removed_golden_fails() -> None:
    """A golden dropping out of top-K fails the budget."""
    diff = _diff(
        {"q": [("g", 0.9), ("a", 0.8)]},
        {"q": [("a", 0.9), ("b", 0.8)]},
        k=2,
        goldens={"q": {"g"}},
    )
    # max_churn=1.0 isolates the removed-golden rule from incidental churn.
    report = evaluate_budget(diff, RegressionBudget(max_churn=1.0))
    assert not report.passed
    assert report.violations[0].kind == ViolationKind.REMOVED_GOLDEN
    assert report.violations[0].chunk_id == "g"
    assert report.violations[0].query == "q"


def test_removed_golden_allowed_when_flag_off() -> None:
    """removed_golden_fails=False tolerates a dropped golden."""
    diff = _diff(
        {"q": [("g", 0.9), ("a", 0.8)]},
        {"q": [("a", 0.9), ("b", 0.8)]},
        k=2,
        goldens={"q": {"g"}},
    )
    report = evaluate_budget(diff, RegressionBudget(removed_golden_fails=False, max_churn=1.0))
    assert report.passed


# --- rank drop --------------------------------------------------------------


def test_golden_rank_drop_beyond_cap_fails() -> None:
    """A golden dropping more ranks than the cap fails."""
    diff = _diff(
        {"q": [("g", 0.9), ("a", 0.8), ("b", 0.7), ("c", 0.6)]},
        {"q": [("a", 0.9), ("b", 0.8), ("c", 0.7), ("g", 0.6)]},
        k=4,
        goldens={"q": {"g"}},
    )
    report = evaluate_budget(diff, RegressionBudget(max_golden_rank_drop=2))
    assert not report.passed
    assert report.violations[0].kind == ViolationKind.GOLDEN_RANK_DROP
    assert report.violations[0].detail["rank_delta"] == 3


def test_golden_rank_drop_within_cap_passes() -> None:
    """A golden dropping within the cap passes."""
    diff = _diff(
        {"q": [("g", 0.9), ("a", 0.8), ("b", 0.7)]},
        {"q": [("a", 0.9), ("g", 0.8), ("b", 0.7)]},
        k=3,
        goldens={"q": {"g"}},
    )
    report = evaluate_budget(diff, RegressionBudget(max_golden_rank_drop=2))
    assert report.passed


def test_golden_rank_improvement_passes() -> None:
    """A golden moving up (negative delta) never fails the rank-drop rule."""
    diff = _diff(
        {"q": [("a", 0.9), ("g", 0.8)]},
        {"q": [("g", 0.9), ("a", 0.8)]},
        k=2,
        goldens={"q": {"g"}},
    )
    report = evaluate_budget(diff, RegressionBudget(max_golden_rank_drop=0, max_churn=1.0))
    assert report.passed


# --- churn ------------------------------------------------------------------


def test_churn_beyond_cap_fails() -> None:
    """Mean churn above the cap fails."""
    diff = _diff(
        {"q": [("a", 0.9), ("b", 0.8)]},
        {"q": [("c", 0.9), ("d", 0.8)]},  # full replacement -> churn 1.0
        k=2,
    )
    report = evaluate_budget(diff, RegressionBudget(max_churn=0.5))
    assert not report.passed
    assert any(v.kind == ViolationKind.CHURN_EXCEEDED for v in report.violations)


def test_churn_within_cap_passes() -> None:
    """Mean churn at or below the cap passes."""
    diff = _diff(
        {"q": [("a", 0.9), ("b", 0.8)]},
        {"q": [("a", 0.9), ("b", 0.8)]},
        k=2,
    )
    report = evaluate_budget(diff, RegressionBudget(max_churn=0.5))
    assert report.passed


def test_max_churn_caps_worst_query_not_mean() -> None:
    """One fully-regressed query among many calm ones must fail the churn cap.

    Gating on the *mean* would dilute a single churn=1.0 query across the set and
    let it slip past max_churn; the cap bounds the worst per-query churn instead.
    """
    # 9 unchanged queries (churn 0.0) + 1 fully replaced query (churn 1.0).
    old_data = {f"q{i}": [("a", 0.9), ("b", 0.8)] for i in range(9)}
    new_data = {f"q{i}": [("a", 0.9), ("b", 0.8)] for i in range(9)}
    old_data["bad"] = [("a", 0.9), ("b", 0.8)]
    new_data["bad"] = [("c", 0.9), ("d", 0.8)]  # full replacement -> churn 1.0
    diff = _diff(old_data, new_data, k=2)
    # Mean churn is 1.0/10 = 0.1, well under the 0.5 cap, but the worst query is
    # 1.0: the old mean-based gate passed here; the max-based gate must fail.
    assert diff.summary.mean_churn < 0.5
    assert diff.summary.max_churn == 1.0
    report = evaluate_budget(diff, RegressionBudget(max_churn=0.5))
    assert not report.passed
    churn_violations = [v for v in report.violations if v.kind == ViolationKind.CHURN_EXCEEDED]
    assert len(churn_violations) == 1
    assert churn_violations[0].query == "bad"
    assert churn_violations[0].detail["max_churn"] == 1.0


# --- golden never retrieved -------------------------------------------------


def test_golden_never_retrieved_in_either_snapshot_fails() -> None:
    """A declared golden absent from both snapshots' top-K surfaces a violation.

    Previously this produced zero violations (old_rank=new_rank=None fell through
    every check), giving false CI coverage for a misdeclared/never-retrieved id.
    """
    diff = _diff(
        {"q": [("a", 0.9), ("b", 0.8)]},
        {"q": [("a", 0.9), ("b", 0.8)]},
        k=2,
        goldens={"q": {"never_seen"}},
    )
    report = evaluate_budget(diff, RegressionBudget(max_churn=1.0))
    assert not report.passed
    nr = [v for v in report.violations if v.kind == ViolationKind.GOLDEN_NEVER_RETRIEVED]
    assert len(nr) == 1
    assert nr[0].chunk_id == "never_seen"
    assert nr[0].query == "q"


def test_golden_present_in_both_is_not_never_retrieved() -> None:
    """A golden that is present is never flagged GOLDEN_NEVER_RETRIEVED."""
    diff = _diff(
        {"q": [("g", 0.9), ("a", 0.8)]},
        {"q": [("g", 0.9), ("a", 0.8)]},
        k=2,
        goldens={"q": {"g"}},
    )
    report = evaluate_budget(diff, RegressionBudget())
    assert report.passed
    assert not any(v.kind == ViolationKind.GOLDEN_NEVER_RETRIEVED for v in report.violations)


# --- new chunks -------------------------------------------------------------


def test_new_chunk_blocked_when_disallowed() -> None:
    """A new chunk fails when allow_new_chunks=False."""
    diff = _diff(
        {"q": [("a", 0.9)]},
        {"q": [("a", 0.9), ("b", 0.8)]},
        k=2,
    )
    report = evaluate_budget(diff, RegressionBudget(allow_new_chunks=False))
    assert not report.passed
    assert any(v.kind == ViolationKind.NEW_CHUNK_BLOCKED for v in report.violations)
    assert any(v.chunk_id == "b" for v in report.violations)


def test_new_chunk_allowed_by_default() -> None:
    """A new chunk passes by default (allow_new_chunks=True)."""
    diff = _diff(
        {"q": [("a", 0.9)]},
        {"q": [("a", 0.9), ("b", 0.8)]},
        k=2,
    )
    report = evaluate_budget(diff, RegressionBudget())
    assert report.passed


# --- query-set change -------------------------------------------------------


def test_query_set_change_fails_by_default() -> None:
    """A current query absent from the lock fails check by default."""
    old = make_snapshot({"q1": [("a", 0.9)]}, k=1)
    new = make_snapshot({"q1": [("a", 0.9)], "q2": [("b", 0.9)]}, k=1)
    diff = diff_snapshots(old, new)
    report = evaluate_budget(diff, RegressionBudget())
    assert not report.passed
    qsc = [v for v in report.violations if v.kind == ViolationKind.QUERY_SET_CHANGED]
    assert qsc
    assert "q2" in qsc[0].detail["added_queries"]


def test_query_set_change_allowed_when_flag_off() -> None:
    """query_set_change_fails=False migrates the lock consciously."""
    old = make_snapshot({"q1": [("a", 0.9)]}, k=1)
    new = make_snapshot({"q1": [("a", 0.9)], "q2": [("b", 0.9)]}, k=1)
    diff = diff_snapshots(old, new)
    report = evaluate_budget(diff, RegressionBudget(query_set_change_fails=False))
    assert report.passed


# --- audit_k golden displaced past K ----------------------------------------


class _AuditRetriever:
    """A retriever returning a fixed deeper ranking for audit_k re-ranking."""

    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def search(self, query: str, k: int) -> list[ScoredHit]:
        return [
            ScoredHit(id=cid, score=1.0 - 0.01 * rank, rank=rank)
            for rank, cid in enumerate(self._ids[:k])
        ]

    def fingerprint(self):  # type: ignore[no-untyped-def]
        from retrieval_diff.fingerprint import ConfigFingerprint

        return ConfigFingerprint()


def test_golden_displaced_past_k_caught_via_audit_k() -> None:
    """A golden pushed from rank K-1 to K+1 is caught by audit_k, not silently passed."""
    # K=2 snapshot: golden 'g' is no longer in the top-2 (a new chunk pushed it).
    new = make_snapshot({"q": [("x", 0.99), ("a", 0.9)]}, k=2)
    goldens = {"q": {"g"}}
    budget = RegressionBudget(audit_k=5)
    # At audit depth 5, 'g' surfaces at rank 3 (>= K=2) -> displaced.
    retriever = _AuditRetriever(["x", "a", "b", "g", "c"])
    violations = audit_goldens_past_k(new, retriever, goldens, budget)
    assert len(violations) == 1
    assert violations[0].kind == ViolationKind.GOLDEN_DISPLACED_PAST_K
    assert violations[0].chunk_id == "g"
    assert violations[0].detail["audited_rank"] == 3


def test_audit_k_noop_without_setting() -> None:
    """Without audit_k configured, the audit is a no-op."""
    new = make_snapshot({"q": [("x", 0.99), ("a", 0.9)]}, k=2)
    retriever = _AuditRetriever(["x", "a", "b", "g"])
    violations = audit_goldens_past_k(new, retriever, {"q": {"g"}}, RegressionBudget())
    assert violations == []


def test_audit_k_not_greater_than_k_is_noop() -> None:
    """audit_k that does not exceed K is a no-op (nothing deeper to inspect)."""
    new = make_snapshot({"q": [("x", 0.99), ("a", 0.9)]}, k=2)
    retriever = _AuditRetriever(["x", "a", "g"])
    violations = audit_goldens_past_k(new, retriever, {"q": {"g"}}, RegressionBudget(audit_k=2))
    assert violations == []


def test_audit_k_skips_queries_without_goldens() -> None:
    """Queries with no declared goldens are skipped by the audit."""
    new = make_snapshot({"q": [("x", 0.99), ("a", 0.9)]}, k=2)
    retriever = _AuditRetriever(["x", "a", "b", "g"])
    # Empty golden set for the query -> no work, no violations.
    violations = audit_goldens_past_k(new, retriever, {"q": set()}, RegressionBudget(audit_k=5))
    assert violations == []


def test_audit_k_golden_inside_top_k_not_flagged() -> None:
    """A golden safely inside top-K is not flagged by the audit."""
    new = make_snapshot({"q": [("g", 0.99), ("a", 0.9)]}, k=2)
    retriever = _AuditRetriever(["g", "a", "b", "c", "d"])
    violations = audit_goldens_past_k(new, retriever, {"q": {"g"}}, RegressionBudget(audit_k=5))
    assert violations == []


def test_audit_violations_merged_into_report() -> None:
    """audit_k violations surface in the combined budget report."""
    new = make_snapshot({"q": [("x", 0.99), ("a", 0.9)]}, k=2)
    old = make_snapshot({"q": [("g", 0.99), ("a", 0.9)]}, k=2)
    diff = diff_snapshots(old, new, goldens={"q": {"g"}})
    budget = RegressionBudget(audit_k=5, removed_golden_fails=False)
    retriever = _AuditRetriever(["x", "a", "b", "g", "c"])
    audit = audit_goldens_past_k(new, retriever, {"q": {"g"}}, budget)
    report = evaluate_budget(diff, budget, audit_violations=audit)
    assert not report.passed
    assert any(v.kind == ViolationKind.GOLDEN_DISPLACED_PAST_K for v in report.violations)


# --- report summary ---------------------------------------------------------


def test_report_summary_lists_violations() -> None:
    """The report summary enumerates each violation message."""
    diff = _diff(
        {"q": [("g", 0.9)]},
        {"q": [("a", 0.9)]},
        k=1,
        goldens={"q": {"g"}},
    )
    report = evaluate_budget(diff, RegressionBudget())
    summary = report.summary()
    assert "FAILED" in summary
    assert "g" in summary


def test_passing_report_summary() -> None:
    """A clean report has an OK summary."""
    diff = _diff({"q": [("a", 0.9)]}, {"q": [("a", 0.9)]}, k=1)
    report = evaluate_budget(diff, RegressionBudget())
    assert report.passed
    assert "OK" in report.summary()
