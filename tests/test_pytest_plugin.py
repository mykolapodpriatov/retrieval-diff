"""Tests for the pytest plugin's ``assert_no_regression`` via ``pytester``.

The regressed case must fail with a message that *names the specific budget
violation and the affected golden/query* -- not merely "an assertion failed" --
so a framework error can never masquerade as a caught regression.
"""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]


# Shared preamble written into the generated test files: builds a tiny in-memory
# retriever and a committed lock on disk, fully offline.
_PREAMBLE = """
from pathlib import Path

from retrieval_diff.budget import RegressionBudget
from retrieval_diff.lockfile import save
from retrieval_diff.pytest_plugin import assert_no_regression
from retrieval_diff.retrievers.memory import StaticRetriever
from retrieval_diff.snapshot import snapshot
from retrieval_diff.types import ScoredHit


def _retriever(order):
    return StaticRetriever(
        hits_by_query={
            "q": tuple(
                ScoredHit(id=cid, score=1.0 - 0.01 * i, rank=i)
                for i, cid in enumerate(order)
            )
        }
    )


def _write_lock(tmp_path, order):
    snap = snapshot(_retriever(order), ["q"], 3, label="base")
    lock = tmp_path / "retrieval.lock"
    save(snap, lock)
    return lock
"""


def test_plugin_passes_when_no_regression(pytester: pytest.Pytester) -> None:
    """A retriever identical to the lock passes assert_no_regression."""
    pytester.makepyfile(
        _PREAMBLE
        + """
def test_no_regression(tmp_path):
    lock = _write_lock(tmp_path, ["g", "a", "b"])
    assert_no_regression(
        lock,
        _retriever(["g", "a", "b"]),
        ["q"],
        RegressionBudget(),
        goldens={"q": {"g"}},
    )
"""
    )
    result = pytester.runpytest("-q")
    result.assert_outcomes(passed=1)


def test_plugin_fails_and_names_violation_and_golden(
    pytester: pytest.Pytester,
) -> None:
    """A dropped golden fails AND the message names the violation + golden/query."""
    pytester.makepyfile(
        _PREAMBLE
        + """
def test_regression(tmp_path):
    lock = _write_lock(tmp_path, ["g", "a", "b"])
    # New retriever drops the golden 'g' entirely.
    assert_no_regression(
        lock,
        _retriever(["a", "b", "c"]),
        ["q"],
        RegressionBudget(max_churn=1.0),
        goldens={"q": {"g"}},
    )
"""
    )
    result = pytester.runpytest("-q")
    result.assert_outcomes(failed=1)
    # The failure must name the specific budget rule and the affected golden+query,
    # proving a real regression was caught (not a generic framework error).
    result.stdout.fnmatch_lines(["*retrieval budget FAILED*"])
    result.stdout.fnmatch_lines(["*golden*'g'*'q'*"])
    # And it must be the typed regression error, not an unrelated exception.
    result.stdout.fnmatch_lines(["*RetrievalRegressionError*"])


def test_plugin_reports_specific_kind_for_rank_drop(
    pytester: pytest.Pytester,
) -> None:
    """A golden rank-drop violation names the rank-drop rule specifically."""
    pytester.makepyfile(
        _PREAMBLE
        + """
def test_rank_drop(tmp_path):
    lock = _write_lock(tmp_path, ["g", "a", "b"])
    # 'g' falls from rank 0 to rank 2 (drop of 2) with cap 0.
    assert_no_regression(
        lock,
        _retriever(["a", "b", "g"]),
        ["q"],
        RegressionBudget(max_golden_rank_drop=0, max_churn=1.0),
        goldens={"q": {"g"}},
    )
"""
    )
    result = pytester.runpytest("-q")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*dropped*ranks*"])
    result.stdout.fnmatch_lines(["*'g'*'q'*"])
