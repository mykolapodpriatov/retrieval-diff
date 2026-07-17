"""Tests for the typer CLI: snapshot -> diff -> check happy path and exit codes."""

from __future__ import annotations

import json
from pathlib import Path

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
