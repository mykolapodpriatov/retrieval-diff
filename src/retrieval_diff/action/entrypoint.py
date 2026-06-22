"""Token-gated entrypoint for the retrieval-diff GitHub Action.

This runs inside the composite action. It loads the project context, re-snapshots
the wired retriever, diffs against the committed lock, evaluates the budget, and
writes the verdict to ``$GITHUB_OUTPUT``. If a token and PR event are present it
posts the compact Markdown diff as a PR comment via the GitHub REST API;
otherwise it cleanly skips the comment so the action works on forks.

Network access is only ever used to post the comment, behind an explicit token
check -- there is no hidden network use, keeping the rest of the tool offline.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from retrieval_diff.budget import (
    BudgetReport,
    RegressionBudget,
    audit_goldens_past_k,
    evaluate_budget,
)
from retrieval_diff.config import (
    ConfigError,
    ProjectContext,
    load_context,
    load_project_config,
    load_queries,
)
from retrieval_diff.diff import diff_snapshots
from retrieval_diff.lockfile import LockfileError, load
from retrieval_diff.report import render_pr_comment
from retrieval_diff.snapshot import snapshot
from retrieval_diff.types import SnapshotDiff

#: Exit code returned when a regression is detected.
EXIT_REGRESSION = 1
#: Exit code returned on a configuration/usage error.
EXIT_ERROR = 2


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable (``true``/``1``/``yes``)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _write_output(key: str, value: str) -> None:
    """Append ``key=value`` to ``$GITHUB_OUTPUT`` if it is set."""
    out_path = os.environ.get("GITHUB_OUTPUT")
    if not out_path:
        return
    with Path(out_path).open("a", encoding="utf-8") as handle:
        handle.write(f"{key}={value}\n")


def _build_context() -> tuple[ProjectContext, RegressionBudget, str]:
    """Load config + context from the action's environment."""
    config_path = Path(os.environ.get("RDIFF_CONFIG", "pyproject.toml"))
    cfg = load_project_config(config_path)
    if not cfg.hook:
        raise ConfigError("[tool.retrieval_diff].hook is not set")
    queries = None
    if cfg.queries_path is not None:
        queries = load_queries(cfg.queries_path)
    context = load_context(cfg.hook, queries=queries)
    budget = cfg.budget
    if _env_bool("RDIFF_ALLOW_QUERY_SET_CHANGE"):
        budget = budget.model_copy(update={"query_set_change_fails": False})
    lock_path = os.environ.get("RDIFF_LOCK", "retrieval.lock")
    return context, budget, lock_path


def _run_check() -> tuple[BudgetReport, SnapshotDiff]:
    """Execute the check and return the budget report and the diff."""
    context, budget, lock_path = _build_context()
    old = load(lock_path)
    new = snapshot(context.retriever, context.queries, old.k, label="action")
    diff = diff_snapshots(old, new, goldens=context.goldens)
    audit_violations = audit_goldens_past_k(new, context.retriever, context.goldens, budget)
    report = evaluate_budget(diff, budget, audit_violations=audit_violations)
    return report, diff


def _post_comment(body: str) -> bool:
    """Post ``body`` as a PR comment if a token and PR event are present.

    Returns:
        ``True`` if a comment was posted, ``False`` if skipped (no token, not a
        PR, or the event payload was unavailable).
    """
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token or not _env_bool("RDIFF_COMMENT", default=True):
        return False
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not event_path or not repo or not Path(event_path).exists():
        return False
    try:
        event: dict[str, Any] = json.loads(Path(event_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    pr = event.get("pull_request")
    if not isinstance(pr, dict):
        return False
    number = pr.get("number")
    if number is None:
        return False

    api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    url = f"{api}/repos/{repo}/issues/{number}/comments"
    return _http_post_comment(url, token, body)


def _http_post_comment(url: str, token: str, body: str) -> bool:
    """POST a comment via urllib; import-guarded so the module loads offline."""
    import urllib.error
    import urllib.request

    payload = json.dumps({"body": body}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "retrieval-diff-action",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = int(response.status)
            return 200 <= status < 300
    except (urllib.error.URLError, TimeoutError):  # pragma: no cover - network
        return False


def main() -> int:
    """Run the action: check, report, optionally comment, set the exit code."""
    try:
        report, diff = _run_check()
    except (ConfigError, LockfileError, ValueError) as exc:
        sys.stderr.write(f"retrieval-diff action error: {exc}\n")
        return EXIT_ERROR

    comment_body = render_pr_comment(diff, budget_report=report)
    sys.stdout.write(comment_body)
    _write_output("passed", "true" if report.passed else "false")

    posted = _post_comment(comment_body)
    if not posted:
        sys.stderr.write("retrieval-diff: PR comment skipped (no token/PR context)\n")

    if not report.passed:
        sys.stderr.write(report.summary() + "\n")
        return EXIT_REGRESSION
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI shim
    raise SystemExit(main())
