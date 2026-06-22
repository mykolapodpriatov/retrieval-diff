"""Tests for terminal, Markdown, and PR-comment rendering shapes."""

from __future__ import annotations

from rdiff_testkit import make_snapshot
from retrieval_diff.budget import RegressionBudget, evaluate_budget
from retrieval_diff.diff import diff_snapshots
from retrieval_diff.report import render_markdown, render_pr_comment, render_terminal


def _sample_diff():  # type: ignore[no-untyped-def]
    old = make_snapshot(
        {
            "alpha query": [("a", 0.9), ("b", 0.8), ("g", 0.7)],
            "beta query": [("c", 0.6), ("d", 0.5)],
        },
        k=3,
    )
    new = make_snapshot(
        {
            "alpha query": [("b", 0.95), ("a", 0.85), ("e", 0.5)],  # g removed, e added
            "beta query": [("c", 0.6), ("d", 0.5)],
        },
        k=3,
    )
    return diff_snapshots(old, new, goldens={"alpha query": {"g"}})


def test_terminal_render_contains_header_and_glyphs() -> None:
    """The terminal render includes the summary header and per-query tables."""
    text = render_terminal(_sample_diff())
    assert "retrieval-diff:" in text
    assert "alpha query" in text
    assert "golden movements:" in text


def test_terminal_render_shows_query_set_changes() -> None:
    """Query-set deltas appear in the terminal render."""
    old = make_snapshot({"shared": [("a", 0.9)], "old_q": [("x", 0.5)]}, k=1)
    new = make_snapshot({"shared": [("a", 0.9)], "new_q": [("y", 0.5)]}, k=1)
    text = render_terminal(diff_snapshots(old, new))
    assert "query-set changed" in text
    assert "new_q" in text
    assert "old_q" in text


def test_markdown_render_shape() -> None:
    """The Markdown render has the expected sections and tables."""
    md = render_markdown(_sample_diff())
    assert md.startswith("# retrieval-diff report")
    assert "## Summary" in md
    assert "## Golden movements" in md
    assert "## Per-query diff" in md
    assert "| chunk | change | Δrank | Δscore |" in md
    # Added/removed chunks are represented.
    assert "added" in md
    assert "removed" in md
    assert md.endswith("\n")


def test_markdown_includes_query_set_section_when_changed() -> None:
    """The Markdown render adds a query-set section only when it changed."""
    old = make_snapshot({"shared": [("a", 0.9)], "old_q": [("x", 0.5)]}, k=1)
    new = make_snapshot({"shared": [("a", 0.9)], "new_q": [("y", 0.5)]}, k=1)
    md = render_markdown(diff_snapshots(old, new))
    assert "## Query-set changes" in md


def test_pr_comment_compact_with_budget_status() -> None:
    """The PR comment leads with the budget verdict and a collapsible table."""
    diff = _sample_diff()
    report = evaluate_budget(diff, RegressionBudget(max_churn=1.0))
    comment = render_pr_comment(diff, budget_report=report)
    assert comment.startswith("### retrieval-diff")
    assert "retrieval budget" in comment
    assert "<details>" in comment
    assert "per-query changes" in comment
    # A removed golden is a failure -> the verdict is the FAILED form.
    assert "FAILED" in comment


def test_pr_comment_without_budget_report() -> None:
    """The PR comment renders without a budget report (diff-only)."""
    comment = render_pr_comment(_sample_diff())
    assert "### retrieval-diff" in comment
    assert "mean churn" in comment


def test_pr_comment_passing_budget() -> None:
    """A clean diff yields the OK verdict in the PR comment."""
    snap = make_snapshot({"q": [("a", 0.9), ("b", 0.8)]}, k=2)
    diff = diff_snapshots(snap, snap)
    report = evaluate_budget(diff, RegressionBudget())
    comment = render_pr_comment(diff, budget_report=report)
    assert "OK" in comment
    assert "(no changes)" in comment
