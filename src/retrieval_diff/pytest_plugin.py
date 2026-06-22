"""pytest plugin exposing :func:`assert_no_regression`.

Drop this assertion into a test to gate a retriever against a committed
``retrieval.lock``: it re-snapshots the live retriever, diffs against the lock,
evaluates the regression budget (including optional ``audit_k`` golden
re-ranking), and **fails with a message that names the specific budget
violation and the affected golden/query** so a real regression is never
mistaken for a framework error.

The plugin is registered via the ``pytest11`` entry point in ``pyproject.toml``;
no ``conftest`` wiring is required.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

from retrieval_diff.budget import (
    BudgetReport,
    RegressionBudget,
    audit_goldens_past_k,
    evaluate_budget,
)
from retrieval_diff.diff import DEFAULT_SCORE_EPS, diff_snapshots
from retrieval_diff.lockfile import load
from retrieval_diff.report import render_terminal
from retrieval_diff.retrievers import Retriever
from retrieval_diff.snapshot import snapshot


class RetrievalRegressionError(AssertionError):
    """Raised by :func:`assert_no_regression` when the budget is exceeded.

    Subclasses :class:`AssertionError` so pytest reports it as a normal test
    failure while remaining distinguishable from unrelated assertions.
    """

    def __init__(self, report: BudgetReport, *, detail: str | None = None) -> None:
        self.report = report
        super().__init__(detail if detail is not None else report.summary())


def evaluate_regression(
    lock_path: str | Path,
    retriever: Retriever,
    queries: Iterable[str],
    budget: RegressionBudget,
    *,
    label: str = "pytest",
    goldens: Mapping[str, set[str]] | None = None,
    score_eps: float = DEFAULT_SCORE_EPS,
) -> tuple[BudgetReport, str]:
    """Re-snapshot, diff against the lock, and evaluate the budget.

    Args:
        lock_path: Path to the committed ``retrieval.lock``.
        retriever: The live retriever under test.
        queries: The query set (must overlap the lock's).
        budget: The regression budget to enforce.
        label: Label for the freshly captured snapshot.
        goldens: ``{query: {golden_id, ...}}`` declarations for golden checks
            and ``audit_k`` displacement detection.
        score_eps: Score-equality tolerance for the diff.

    Returns:
        A ``(report, rendered_diff)`` tuple. ``rendered_diff`` is the terminal
        table, included in failure output for context.
    """
    old = load(lock_path)
    new = snapshot(retriever, queries, old.k, label=label)
    diff = diff_snapshots(old, new, score_eps=score_eps, goldens=goldens or {})

    audit_violations = audit_goldens_past_k(new, retriever, goldens or {}, budget)
    report = evaluate_budget(diff, budget, audit_violations=audit_violations)
    return report, render_terminal(diff)


def assert_no_regression(
    lock_path: str | Path,
    retriever: Retriever,
    queries: Iterable[str],
    budget: RegressionBudget,
    *,
    label: str = "pytest",
    goldens: Mapping[str, set[str]] | None = None,
    score_eps: float = DEFAULT_SCORE_EPS,
) -> None:
    """Fail the current test if the retriever regresses beyond the budget.

    On failure the error message names every violated budget rule and the
    affected golden/query, followed by the rendered diff table.

    Raises:
        RetrievalRegressionError: If the budget is exceeded.
    """
    report, rendered = evaluate_regression(
        lock_path,
        retriever,
        queries,
        budget,
        label=label,
        goldens=goldens,
        score_eps=score_eps,
    )
    if not report.passed:
        message = report.summary() + "\n\n" + rendered
        raise RetrievalRegressionError(report, detail=message)


@pytest.fixture
def assert_no_regression_fixture() -> object:
    """Expose :func:`assert_no_regression` as a pytest fixture for convenience."""
    return assert_no_regression


__all__ = [
    "RetrievalRegressionError",
    "assert_no_regression",
    "assert_no_regression_fixture",
    "evaluate_regression",
]
