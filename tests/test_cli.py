"""Tests for the typer CLI: snapshot -> diff -> check happy path and exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from retrieval_diff.cli import EXIT_OK, EXIT_REGRESSION, EXIT_USER_ERROR, app

runner = CliRunner()

# A self-contained project hook written to disk; the CLI imports it by path. It
# builds a deterministic in-memory retriever whose ranking can be perturbed via
# an environment variable so tests can simulate a regression.
_HOOK = """
import os

from retrieval_diff.config import ProjectContext
from retrieval_diff.retrievers.factory import MemoryRetrieverFactory
from retrieval_diff.retrievers.memory import Corpus, HashingEmbedder, HybridRetriever

CORPUS = Corpus.from_mapping(
    {
        "d1": "dense vector search retrieval system",
        "d2": "the lazy dog sleeps under the brown fox",
        "d3": "vector embeddings power semantic search",
        "d4": "a grey wolf howls at the moon",
    }
)


def build_context(queries):
    embedder = HashingEmbedder(dim=32, model_id="e1")
    # RDIFF_TEST_ALPHA lets a test perturb ranking to force a regression.
    alpha = float(os.environ.get("RDIFF_TEST_ALPHA", "0.5"))
    retriever = HybridRetriever(CORPUS, embedder, alpha=alpha)
    factory = MemoryRetrieverFactory(CORPUS, embedders={"e1": embedder})
    return ProjectContext(
        retriever=retriever,
        queries=("the fox", "vector search"),
        goldens={},
        factory=factory,
        corpus=CORPUS,
    )
"""

_PYPROJECT = """
[tool.retrieval_diff]
hook = "{hook}"
k = 3
"""


def _project(tmp_path: Path) -> tuple[Path, Path]:
    """Write the hook + a pyproject pointing at it; return their paths."""
    hook = tmp_path / "rdiff_hook.py"
    hook.write_text(_HOOK, encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_PYPROJECT.format(hook=hook.as_posix()), encoding="utf-8")
    return hook, pyproject


def test_snapshot_diff_check_happy_path(tmp_path: Path) -> None:
    """snapshot -> diff -> check round trips and check passes with no change."""
    _hook, pyproject = _project(tmp_path)
    lock = tmp_path / "retrieval.lock"

    res = runner.invoke(
        app,
        ["snapshot", "--out", str(lock), "--label", "sha-1", "--config", str(pyproject)],
    )
    assert res.exit_code == EXIT_OK, res.output
    assert lock.exists()

    # Diff the lock against itself (identical) -> exits 0, empty changes.
    res_diff = runner.invoke(app, ["diff", str(lock), str(lock), "--format", "json"])
    assert res_diff.exit_code == EXIT_OK
    payload = json.loads(res_diff.stdout)
    assert payload["summary"]["mean_churn"] == 0.0
    assert payload["query_set_delta"] == {"added_queries": [], "removed_queries": []}

    # check against the freshly written lock with no change -> pass.
    res_check = runner.invoke(app, ["check", "--lock", str(lock), "--config", str(pyproject)])
    assert res_check.exit_code == EXIT_OK, res_check.output


def test_snapshot_requires_label(tmp_path: Path) -> None:
    """Omitting --label is a usage error from typer (missing required option)."""
    _hook, pyproject = _project(tmp_path)
    lock = tmp_path / "retrieval.lock"
    res = runner.invoke(app, ["snapshot", "--out", str(lock), "--config", str(pyproject)])
    assert res.exit_code != EXIT_OK


def test_check_fails_on_regression(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A ranking change beyond the churn cap makes check exit non-zero."""
    _hook, pyproject = _project(tmp_path)
    lock = tmp_path / "retrieval.lock"
    runner.invoke(
        app,
        ["snapshot", "--out", str(lock), "--label", "sha-1", "--config", str(pyproject)],
    )
    # Perturb alpha so the live retriever ranks differently from the lock, then
    # gate on a strict churn cap.
    monkeypatch.setenv("RDIFF_TEST_ALPHA", "0.0")
    res = runner.invoke(
        app,
        [
            "check",
            "--lock",
            str(lock),
            "--config",
            str(pyproject),
            "--max-churn",
            "0.0",
        ],
    )
    assert res.exit_code == EXIT_REGRESSION


def test_check_query_set_change_fails_without_flag(tmp_path: Path) -> None:
    """A lock with extra queries vs the live set fails check by default."""
    _hook, pyproject = _project(tmp_path)
    lock = tmp_path / "retrieval.lock"
    runner.invoke(
        app,
        ["snapshot", "--out", str(lock), "--label", "sha-1", "--config", str(pyproject)],
    )
    # Rewrite the lock to add a query not produced by the live retriever.
    data = json.loads(lock.read_text(encoding="utf-8"))
    data["results"]["a third query"] = {
        "query": "a third query",
        "hits": [{"id": "d1", "rank": 0, "score": 0.5}],
    }
    lock.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    res = runner.invoke(app, ["check", "--lock", str(lock), "--config", str(pyproject)])
    assert res.exit_code == EXIT_REGRESSION

    res_allow = runner.invoke(
        app,
        [
            "check",
            "--lock",
            str(lock),
            "--config",
            str(pyproject),
            "--allow-query-set-change",
        ],
    )
    assert res_allow.exit_code == EXIT_OK, res_allow.output


def test_diff_markdown_and_term_formats(tmp_path: Path) -> None:
    """diff supports md and term renderers."""
    _hook, pyproject = _project(tmp_path)
    lock = tmp_path / "retrieval.lock"
    runner.invoke(
        app,
        ["snapshot", "--out", str(lock), "--label", "sha-1", "--config", str(pyproject)],
    )
    res_md = runner.invoke(app, ["diff", str(lock), str(lock), "--format", "md"])
    assert res_md.exit_code == EXIT_OK
    assert "# retrieval-diff report" in res_md.stdout

    res_term = runner.invoke(app, ["diff", str(lock), str(lock), "--format", "term"])
    assert res_term.exit_code == EXIT_OK
    assert "retrieval-diff:" in res_term.stdout


def test_diff_unknown_format_errors(tmp_path: Path) -> None:
    """An unknown --format is a clean usage error."""
    _hook, pyproject = _project(tmp_path)
    lock = tmp_path / "retrieval.lock"
    runner.invoke(
        app,
        ["snapshot", "--out", str(lock), "--label", "sha-1", "--config", str(pyproject)],
    )
    res = runner.invoke(app, ["diff", str(lock), str(lock), "--format", "xml"])
    assert res.exit_code == EXIT_USER_ERROR


def test_attribute_cli_emits_json(tmp_path: Path) -> None:
    """attribute produces a JSON array of per-change verdicts."""
    _hook, pyproject = _project(tmp_path)
    old = tmp_path / "old.lock"
    new = tmp_path / "new.lock"
    runner.invoke(
        app,
        ["snapshot", "--out", str(old), "--label", "old", "--config", str(pyproject)],
    )
    # Build a different snapshot by perturbing alpha via a second config/hook env.
    import os

    os.environ["RDIFF_TEST_ALPHA"] = "0.0"
    try:
        runner.invoke(
            app,
            [
                "snapshot",
                "--out",
                str(new),
                "--label",
                "new",
                "--config",
                str(pyproject),
            ],
        )
    finally:
        del os.environ["RDIFF_TEST_ALPHA"]

    res = runner.invoke(app, ["attribute", str(old), str(new), "--config", str(pyproject)])
    assert res.exit_code == EXIT_OK, res.output
    payload = json.loads(res.stdout)
    assert isinstance(payload, list)
    for entry in payload:
        assert entry["confidence"] in {"confirmed", "ambiguous", "not_attributable"}


def test_report_cli_writes_markdown(tmp_path: Path) -> None:
    """report writes a Markdown file when --out is given."""
    _hook, pyproject = _project(tmp_path)
    lock = tmp_path / "retrieval.lock"
    runner.invoke(
        app,
        ["snapshot", "--out", str(lock), "--label", "sha-1", "--config", str(pyproject)],
    )
    out = tmp_path / "report.md"
    res = runner.invoke(app, ["report", str(lock), str(lock), "--out", str(out)])
    assert res.exit_code == EXIT_OK
    assert out.exists()
    assert "# retrieval-diff report" in out.read_text(encoding="utf-8")


def _attribute_lock_pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Snapshot a baseline and a perturbed candidate; return (old, new, pyproject)."""
    _hook, pyproject = _project(tmp_path)
    old = tmp_path / "old.lock"
    new = tmp_path / "new.lock"
    runner.invoke(
        app,
        ["snapshot", "--out", str(old), "--label", "old", "--config", str(pyproject)],
    )
    import os

    os.environ["RDIFF_TEST_ALPHA"] = "0.0"
    try:
        runner.invoke(
            app,
            ["snapshot", "--out", str(new), "--label", "new", "--config", str(pyproject)],
        )
    finally:
        del os.environ["RDIFF_TEST_ALPHA"]
    return old, new, pyproject


def test_attribute_unknown_format_errors(tmp_path: Path) -> None:
    """attribute with an unknown --format is a clean usage error, not a traceback."""
    old, new, pyproject = _attribute_lock_pair(tmp_path)
    res = runner.invoke(
        app,
        ["attribute", str(old), str(new), "--config", str(pyproject), "--format", "bogus"],
    )
    assert res.exit_code == EXIT_USER_ERROR
    assert "unknown --format" in res.output


def test_attribute_json_format_matches_default(tmp_path: Path) -> None:
    """--format json is byte-identical to the (json) default for back-compat."""
    old, new, pyproject = _attribute_lock_pair(tmp_path)
    res_default = runner.invoke(app, ["attribute", str(old), str(new), "--config", str(pyproject)])
    res_json = runner.invoke(
        app,
        ["attribute", str(old), str(new), "--config", str(pyproject), "--format", "json"],
    )
    assert res_default.exit_code == EXIT_OK, res_default.output
    assert res_json.exit_code == EXIT_OK, res_json.output
    assert res_json.stdout == res_default.stdout
    # And it is still valid JSON of the documented shape.
    payload = json.loads(res_json.stdout)
    assert isinstance(payload, list)


def test_attribute_term_and_markdown_formats(tmp_path: Path) -> None:
    """attribute renders term and md tables in addition to json."""
    old, new, pyproject = _attribute_lock_pair(tmp_path)
    res_term = runner.invoke(
        app,
        ["attribute", str(old), str(new), "--config", str(pyproject), "--format", "term"],
    )
    assert res_term.exit_code == EXIT_OK, res_term.output
    assert "retrieval-diff attribution" in res_term.stdout

    res_md = runner.invoke(
        app,
        ["attribute", str(old), str(new), "--config", str(pyproject), "-f", "md"],
    )
    assert res_md.exit_code == EXIT_OK, res_md.output
    assert res_md.stdout.startswith("# retrieval-diff attribution")


#: Queries and K produced by the shared test hook (see ``_HOOK`` / ``_PYPROJECT``).
_SHOW_QUERIES = ("the fox", "vector search")
_SHOW_K = 3


def _snapshot_lock(tmp_path: Path, *, label: str = "sha-1") -> Path:
    """Snapshot the wired retriever into a fresh lockfile and return its path.

    Self-contained: the lock is generated on the fly in ``tmp_path`` (never read
    from the gitignored ``examples/retrieval.lock``), mirroring how the other
    tests in this file produce a lock.
    """
    _hook, pyproject = _project(tmp_path)
    lock = tmp_path / "retrieval.lock"
    res = runner.invoke(
        app,
        ["snapshot", "--out", str(lock), "--label", label, "--config", str(pyproject)],
    )
    assert res.exit_code == EXIT_OK, res.output
    return lock


def test_show_term_format(tmp_path: Path) -> None:
    """show --format term prints the header and per-query hit tables."""
    lock = _snapshot_lock(tmp_path)
    res = runner.invoke(app, ["show", str(lock), "--format", "term"])
    assert res.exit_code == EXIT_OK, res.output
    assert "retrieval-lock:" in res.stdout
    assert f"K={_SHOW_K}" in res.stdout
    # Every captured query is rendered as its own table.
    for query in _SHOW_QUERIES:
        assert query in res.stdout


def test_show_markdown_format(tmp_path: Path) -> None:
    """show --format md prints a Markdown header table and per-query tables."""
    lock = _snapshot_lock(tmp_path)
    res = runner.invoke(app, ["show", str(lock), "-f", "md"])
    assert res.exit_code == EXIT_OK, res.output
    assert res.stdout.startswith("# retrieval-lock")
    assert f"| K | {_SHOW_K} |" in res.stdout
    assert "| rank | id | score |" in res.stdout


def test_show_json_format_and_query_filter(tmp_path: Path) -> None:
    """show --format json emits a structured view, filterable to one query."""
    lock = _snapshot_lock(tmp_path)
    res = runner.invoke(app, ["show", str(lock), "--format", "json"])
    assert res.exit_code == EXIT_OK, res.output
    payload = json.loads(res.stdout)
    assert payload["k"] == _SHOW_K
    assert payload["query_count"] == len(_SHOW_QUERIES)
    assert len(payload["fingerprint"]["digest"]) == 64
    assert set(payload["fingerprint"]["axes"]) == {
        "alpha",
        "chunk_params",
        "embedding_model",
        "index_content_hash",
        "reranker",
    }
    assert set(payload["results"]) == set(_SHOW_QUERIES)

    one = _SHOW_QUERIES[0]
    res_one = runner.invoke(app, ["show", str(lock), "--format", "json", "--query", one])
    assert res_one.exit_code == EXIT_OK, res_one.output
    filtered = json.loads(res_one.stdout)
    assert list(filtered["results"]) == [one]
    # The header still reflects the full lockfile.
    assert filtered["query_count"] == len(_SHOW_QUERIES)


def test_show_unknown_query_errors(tmp_path: Path) -> None:
    """A --query not present in the lockfile fails cleanly with a clear message."""
    lock = _snapshot_lock(tmp_path)
    res = runner.invoke(app, ["show", str(lock), "--query", "no such query"])
    assert res.exit_code == EXIT_USER_ERROR
    assert "not in lockfile" in res.output


def test_show_unknown_format_errors(tmp_path: Path) -> None:
    """An unknown --format is a clean usage error."""
    lock = _snapshot_lock(tmp_path)
    res = runner.invoke(app, ["show", str(lock), "--format", "xml"])
    assert res.exit_code == EXIT_USER_ERROR
    assert "unknown --format" in res.output


def test_show_malformed_lock_uses_lockfile_error(tmp_path: Path) -> None:
    """A corrupt lock surfaces the LockfileError message, not a traceback."""
    bad = tmp_path / "bad.lock"
    bad.write_text("{ not valid json", encoding="utf-8")
    res = runner.invoke(app, ["show", str(bad)])
    assert res.exit_code == EXIT_USER_ERROR
    assert "not valid JSON" in res.output


def test_show_unknown_version_lock_uses_lockfile_error(tmp_path: Path) -> None:
    """An unknown (future) lock version surfaces the LockfileError message."""
    lock = _snapshot_lock(tmp_path)
    future = tmp_path / "future.lock"
    data = json.loads(lock.read_text(encoding="utf-8"))
    data["version"] = 999
    future.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    res = runner.invoke(app, ["show", str(future)])
    assert res.exit_code == EXIT_USER_ERROR
    assert "newer than supported version" in res.output


# --- diff --format pr-comment (issue #7) -----------------------------------


def test_diff_pr_comment_format_emits_collapsible_summary(tmp_path: Path) -> None:
    """diff --format pr-comment emits the collapsible details block and glyph table."""
    old, new, _pyproject = _attribute_lock_pair(tmp_path)
    res = runner.invoke(app, ["diff", str(old), str(new), "--format", "pr-comment"])
    assert res.exit_code == EXIT_OK, res.output
    assert res.stdout.startswith("### retrieval-diff")
    assert "<details>" in res.stdout
    assert "</details>" in res.stdout
    assert "per-query changes" in res.stdout
    # The per-query glyph summary is a table keyed by query.
    assert "| query | churn | changes |" in res.stdout
    # The wired hook's queries appear as rows in the summary.
    for query in _SHOW_QUERIES:
        assert query in res.stdout


def test_diff_pr_comment_no_change_reports_no_changes(tmp_path: Path) -> None:
    """Diffing a lock against itself still renders the block with '(no changes)'."""
    lock = _snapshot_lock(tmp_path)
    res = runner.invoke(app, ["diff", str(lock), str(lock), "-f", "pr-comment"])
    assert res.exit_code == EXIT_OK, res.output
    assert "<details>" in res.stdout
    assert "(no changes)" in res.stdout


# --- check --format {term,md,json} (issue #6) ------------------------------


def test_check_json_format_serializes_budget_and_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """check --format json emits a machine-readable verdict alongside the diff."""
    _hook, pyproject = _project(tmp_path)
    lock = tmp_path / "retrieval.lock"
    runner.invoke(
        app,
        ["snapshot", "--out", str(lock), "--label", "sha-1", "--config", str(pyproject)],
    )
    monkeypatch.setenv("RDIFF_TEST_ALPHA", "0.0")
    res = runner.invoke(
        app,
        [
            "check",
            "--lock",
            str(lock),
            "--config",
            str(pyproject),
            "--max-churn",
            "0.0",
            "--format",
            "json",
        ],
    )
    assert res.exit_code == EXIT_REGRESSION
    payload = json.loads(res.stdout)
    assert payload["passed"] is False
    assert payload["violations"]
    for violation in payload["violations"]:
        assert set(violation) == {"kind", "query", "chunk_id", "message"}
    # The diff is serialized alongside the verdict.
    assert "diff" in payload
    assert set(payload["diff"]["query_set_delta"]) == {"added_queries", "removed_queries"}


def test_check_json_format_passes_cleanly(tmp_path: Path) -> None:
    """A clean check emits ``passed`` true, empty violations, and exit 0."""
    _hook, pyproject = _project(tmp_path)
    lock = tmp_path / "retrieval.lock"
    runner.invoke(
        app,
        ["snapshot", "--out", str(lock), "--label", "sha-1", "--config", str(pyproject)],
    )
    res = runner.invoke(
        app,
        ["check", "--lock", str(lock), "--config", str(pyproject), "--format", "json"],
    )
    assert res.exit_code == EXIT_OK, res.output
    payload = json.loads(res.stdout)
    assert payload["passed"] is True
    assert payload["violations"] == []


def test_check_markdown_format_includes_budget_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """check --format md renders the diff report plus the budget verdict section."""
    _hook, pyproject = _project(tmp_path)
    lock = tmp_path / "retrieval.lock"
    runner.invoke(
        app,
        ["snapshot", "--out", str(lock), "--label", "sha-1", "--config", str(pyproject)],
    )
    monkeypatch.setenv("RDIFF_TEST_ALPHA", "0.0")
    res = runner.invoke(
        app,
        [
            "check",
            "--lock",
            str(lock),
            "--config",
            str(pyproject),
            "--max-churn",
            "0.0",
            "-f",
            "md",
        ],
    )
    assert res.exit_code == EXIT_REGRESSION
    assert "# retrieval-diff report" in res.stdout
    assert "## Retrieval budget" in res.stdout
    assert "FAILED" in res.stdout


def test_check_unknown_format_errors(tmp_path: Path) -> None:
    """An unknown check --format is a clean usage error, not a traceback."""
    lock = _snapshot_lock(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    res = runner.invoke(
        app,
        ["check", "--lock", str(lock), "--config", str(pyproject), "--format", "xml"],
    )
    assert res.exit_code == EXIT_USER_ERROR
    assert "unknown --format" in res.output
