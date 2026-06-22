"""Offline end-to-end demo: snapshot -> diff -> check, plus a simulated regression.

Run from this directory with ``python run_demo.py``. Everything is deterministic
and network-free. The script:

1. snapshots the baseline retriever into ``retrieval.lock``;
2. diffs the baseline against itself (no changes);
3. checks the baseline against the lock (passes);
4. swaps the embedding model to simulate a regression, then checks again and
   shows the gate failing with an attributed, budget-named report.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the local hook importable and ensure the package is on the path when run
# straight from a checkout.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import rdiff_hook  # noqa: E402  (local example hook)

from retrieval_diff.attribution import attribute  # noqa: E402
from retrieval_diff.budget import (  # noqa: E402
    audit_goldens_past_k,
    evaluate_budget,
)
from retrieval_diff.config import load_project_config  # noqa: E402
from retrieval_diff.diff import diff_snapshots  # noqa: E402
from retrieval_diff.lockfile import load, save  # noqa: E402
from retrieval_diff.report import render_terminal  # noqa: E402
from retrieval_diff.snapshot import snapshot  # noqa: E402


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    """Run the demo and return a process exit code (0 = demo completed)."""
    config_path = HERE / "pyproject.toml"
    cfg = load_project_config(config_path)
    lock_path = HERE / "retrieval.lock"

    # 1) Baseline snapshot -> lockfile.
    _print_header("1) snapshot baseline -> retrieval.lock")
    os.environ.pop("RDIFF_DEMO_MODEL", None)
    baseline_ctx = rdiff_hook.build_context(None)
    baseline = snapshot(baseline_ctx.retriever, baseline_ctx.queries, cfg.k, label="baseline")
    save(baseline, lock_path)
    print(f"wrote {lock_path.name}: {len(baseline.results)} queries at K={cfg.k}")

    # 2) Diff baseline against itself -> no changes.
    _print_header("2) diff baseline vs baseline (expect: no changes)")
    committed = load(lock_path)
    self_diff = diff_snapshots(committed, baseline, goldens=baseline_ctx.goldens)
    print(render_terminal(self_diff).rstrip())

    # 3) check baseline vs lock -> passes.
    _print_header("3) check baseline vs lock (expect: PASS)")
    report = evaluate_budget(self_diff, cfg.budget)
    print(report.summary())

    # 4) Simulate a regression: swap the embedding model.
    _print_header("4) check after embedding swap (expect: FAIL + attribution)")
    os.environ["RDIFF_DEMO_MODEL"] = "demo-embed-v2"
    regressed_ctx = rdiff_hook.build_context(None)
    regressed = snapshot(regressed_ctx.retriever, regressed_ctx.queries, cfg.k, label="swap-v2")
    diff = diff_snapshots(committed, regressed, goldens=regressed_ctx.goldens)
    print(render_terminal(diff).rstrip())

    # Tighten the churn cap for this illustration so the reranking the embedding
    # swap caused trips the gate (the committed budget in pyproject.toml is more
    # lenient). This shows how the budget knob turns drift into a CI failure.
    strict_budget = cfg.budget.model_copy(update={"max_churn": 0.1})
    audit = audit_goldens_past_k(
        regressed, regressed_ctx.retriever, regressed_ctx.goldens, strict_budget
    )
    regressed_report = evaluate_budget(diff, strict_budget, audit_violations=audit)
    print("\n" + regressed_report.summary())

    if diff.fingerprint_delta:
        print("\nattribution of observed changes:")
        attributions = attribute(
            committed.fingerprint,
            regressed.fingerprint,
            diff,
            factory=regressed_ctx.factory,
            queries=list(regressed_ctx.queries),
            k=cfg.k,
            corpus=regressed_ctx.corpus,
        )
        for a in attributions:
            print(
                f"  - {a.change_ref.query!r} / {a.change_ref.chunk_id!r} "
                f"({a.change_ref.kind.value}): {a.confidence}" + (f" -> {a.axis}" if a.axis else "")
            )

    os.environ.pop("RDIFF_DEMO_MODEL", None)
    _print_header("demo complete")
    status = "FAILED (as expected)" if not regressed_report.passed else "passed"
    print(f"Baseline check passed; the regressed check {status}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
