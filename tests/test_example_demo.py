"""Smoke test guaranteeing the offline ``examples/run_demo.py`` keeps working."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_demo_runs_offline_and_reports_regression() -> None:
    """The example demo runs to completion and demonstrates a caught regression."""
    result = subprocess.run(
        [sys.executable, "run_demo.py"],
        cwd=EXAMPLES,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "retrieval budget OK: no regressions" in out  # baseline passes
    assert "retrieval budget FAILED" in out  # regressed run fails
    assert "confirmed -> embedding_model" in out  # attribution works
    # The generated lockfile is a real artifact of the run.
    assert (EXAMPLES / "retrieval.lock").exists()


def test_demo_lockfile_is_valid() -> None:
    """The lockfile the demo writes loads cleanly through the public API."""
    from retrieval_diff.lockfile import load

    lock = EXAMPLES / "retrieval.lock"
    if not lock.exists():  # ensure the demo has run in this session
        subprocess.run(
            [sys.executable, "run_demo.py"],
            cwd=EXAMPLES,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
    snap = load(lock)
    assert snap.k == 4
    assert snap.created_label == "baseline"
    assert set(snap.results) == {
        "how do I snapshot retrieval",
        "what does diff report",
        "semantic search with vectors",
    }
