"""Tests for terminal, Markdown, PR-comment, and budget rendering shapes."""

from __future__ import annotations

import json

from rdiff_testkit import make_snapshot
from retrieval_diff.budget import RegressionBudget, evaluate_budget
from retrieval_diff.diff import diff_snapshots
from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.report import (
    render_attributions_markdown,
    render_attributions_terminal,
    render_budget_json,
    render_budget_markdown,
    render_markdown,
    render_pr_comment,
    render_snapshot_markdown,
    render_snapshot_terminal,
    render_terminal,
)
from retrieval_diff.types import AxisAttribution, ChangeKind, ChangeRef


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


def _sample_attributions() -> list[AxisAttribution]:
    """A confirmed and an ambiguous attribution for renderer assertions."""
    confirmed = AxisAttribution(
        change_ref=ChangeRef(query="alpha query", chunk_id="g", kind=ChangeKind.REMOVED),
        axis="alpha",
        confidence="confirmed",
        evidence={
            "differing_axes": ["alpha", "reranker"],
            "explaining_axes": ["alpha"],
            "non_replayable_axes": [],
        },
    )
    ambiguous = AxisAttribution(
        change_ref=ChangeRef(query="beta query", chunk_id="e", kind=ChangeKind.ADDED),
        axis="",
        confidence="ambiguous",
        evidence={
            "differing_axes": ["alpha", "reranker"],
            "explaining_axes": ["alpha", "reranker"],
            "non_replayable_axes": [],
        },
    )
    # A not_attributable verdict carries a scalar ``reason`` (exercises the
    # non-list evidence rendering path).
    not_attributable = AxisAttribution(
        change_ref=ChangeRef(query="beta query", chunk_id="f", kind=ChangeKind.REORDERED),
        axis="",
        confidence="not_attributable",
        evidence={"reason": "interaction effect: no single axis reproduces this change"},
    )
    return [confirmed, ambiguous, not_attributable]


def test_attributions_terminal_contains_axis_and_confidence() -> None:
    """The terminal attribution table shows both verdicts and the responsible axis."""
    text = render_attributions_terminal(_sample_attributions())
    assert "confirmed" in text
    assert "ambiguous" in text
    # The confirmed change's responsible axis is surfaced.
    assert "alpha" in text


def test_attributions_markdown_contains_axis_and_confidence() -> None:
    """The Markdown attribution table shows both verdicts and the responsible axis."""
    md = render_attributions_markdown(_sample_attributions())
    assert md.startswith("# retrieval-diff attribution")
    assert "| query | chunk | kind | axis | confidence | evidence |" in md
    assert "confirmed" in md
    assert "ambiguous" in md
    assert "alpha" in md
    assert md.endswith("\n")


def test_attributions_markdown_handles_empty() -> None:
    """With no attributions the Markdown render is a short, valid note."""
    md = render_attributions_markdown([])
    assert md.startswith("# retrieval-diff attribution")
    assert "no attributable changes" in md
    assert md.endswith("\n")


def test_render_budget_json_serializes_verdict_and_violations() -> None:
    """The budget JSON carries ``passed`` plus the documented per-violation fields."""
    diff = _sample_diff()  # a removed golden -> a failing budget
    report = evaluate_budget(diff, RegressionBudget(max_churn=1.0))
    payload = json.loads(render_budget_json(report))
    assert payload["passed"] is False
    assert payload["violations"]
    for violation in payload["violations"]:
        assert set(violation) == {"kind", "query", "chunk_id", "message"}
    # The removed golden 'g' for 'alpha query' is named in a violation.
    assert any(v["query"] == "alpha query" and v["chunk_id"] == "g" for v in payload["violations"])


def test_render_budget_json_passing_has_empty_violations() -> None:
    """A clean diff yields ``passed`` true and no violations."""
    snap = make_snapshot({"q": [("a", 0.9), ("b", 0.8)]}, k=2)
    report = evaluate_budget(diff_snapshots(snap, snap), RegressionBudget())
    payload = json.loads(render_budget_json(report))
    assert payload == {"passed": True, "violations": []}


def test_render_budget_markdown_failing_and_passing() -> None:
    """The budget Markdown leads with the verdict and tabulates violations."""
    diff = _sample_diff()
    failing = render_budget_markdown(evaluate_budget(diff, RegressionBudget(max_churn=1.0)))
    assert failing.startswith("## Retrieval budget")
    assert "FAILED" in failing
    assert "| kind | query | chunk | message |" in failing
    assert failing.endswith("\n")

    snap = make_snapshot({"q": [("a", 0.9), ("b", 0.8)]}, k=2)
    clean = evaluate_budget(diff_snapshots(snap, snap), RegressionBudget())
    passing = render_budget_markdown(clean)
    assert "PASSED" in passing
    assert "_no violations_" in passing


def test_snapshot_renderers_show_header_and_hits() -> None:
    """The snapshot renderers include the header (label/K/digest) and top-K hits."""
    fp = ConfigFingerprint(embedding_model="e1", alpha=0.5)
    snap = make_snapshot(
        {"q1": [("a", 0.9), ("b", 0.8)], "q2": [("c", 0.7)]},
        k=2,
        label="sha-1",
        fingerprint=fp,
    )
    text = render_snapshot_terminal(snap)
    assert "sha-1" in text
    assert "K=2" in text
    assert fp.digest() in text
    assert "q1" in text and "q2" in text

    md = render_snapshot_markdown(snap)
    assert md.startswith("# retrieval-lock")
    assert "| label | sha-1 |" in md
    assert f"| digest | {fp.digest()} |" in md
    assert md.endswith("\n")


def test_snapshot_renderers_filter_to_single_query() -> None:
    """A query filter renders only that query's table."""
    snap = make_snapshot({"keep": [("a", 0.9)], "drop": [("b", 0.8)]}, k=1)
    text = render_snapshot_terminal(snap, query="keep")
    assert "keep" in text
    assert "drop" not in text
    md = render_snapshot_markdown(snap, query="keep")
    assert "### `keep`" in md
    assert "drop" not in md
