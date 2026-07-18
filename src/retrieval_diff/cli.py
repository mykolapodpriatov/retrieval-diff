"""The ``retrieval-diff`` command-line interface (typer).

Subcommands:

* ``snapshot`` -- capture a retriever's top-K output into a lockfile (``--label``
  is **required**; the library never reads the clock).
* ``diff`` -- diff two lockfiles (``--format term|md|json|pr-comment``).
* ``check`` -- re-snapshot the wired retriever, diff vs the committed lock, and
  exit non-zero on a budget regression. A non-empty query-set delta fails unless
  ``--allow-query-set-change``. Emits ``--format term|md|json`` and can also write
  a JUnit report via ``--junit-xml PATH``.
* ``attribute`` -- attribute the changes between two lockfiles to single config
  axes via held-fixed replay (requires a factory wired through the project hook),
  in ``--format term|md|json`` (json is the default for back-compat).
* ``show`` -- inspect a single committed lockfile (header + top-K hits),
  optionally filtered to one ``--query``, in ``--format term|md|json``.
* ``report`` -- render a diff to Markdown.

The retriever/query-set/factory are wired through a project hook (see
:mod:`retrieval_diff.config`), so the CLI never hardcodes a backend.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from retrieval_diff import lockfile
from retrieval_diff.attribution import attribute as attribute_changes
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
from retrieval_diff.diff import (
    DEFAULT_SCORE_EPS,
    EmptyIntersectionError,
    KMismatchError,
    diff_snapshots,
)
from retrieval_diff.report import (
    render_attributions_markdown,
    render_attributions_terminal,
    render_budget_markdown,
    render_junit,
    render_markdown,
    render_pr_comment,
    render_snapshot_markdown,
    render_snapshot_terminal,
    render_terminal,
)
from retrieval_diff.snapshot import snapshot
from retrieval_diff.types import Snapshot, SnapshotDiff

app = typer.Typer(
    name="retrieval-diff",
    help="A git-style diff for RAG retrievers: snapshot, diff, attribute, gate CI.",
    no_args_is_help=True,
    add_completion=False,
)

_err = Console(stderr=True)
_out = Console()

# Exit codes (documented for CI consumers).
EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_USER_ERROR = 2


def _fail(message: str, code: int = EXIT_USER_ERROR) -> NoReturn:
    """Print an error and exit with ``code`` (never returns)."""
    _err.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code)


def _resolve_context(
    config_path: Path, queries_override: list[str] | None
) -> tuple[ProjectContext, RegressionBudget, int]:
    """Load the project config and instantiate the wired :class:`ProjectContext`."""
    try:
        cfg = load_project_config(config_path)
    except ConfigError as exc:
        _fail(str(exc))
    if not cfg.hook:
        _fail(
            "[tool.retrieval_diff].hook is not set; declare a hook module that "
            "exposes build_context() -> ProjectContext"
        )
    queries = queries_override
    if queries is None and cfg.queries_path is not None:
        try:
            queries = load_queries(cfg.queries_path)
        except ConfigError as exc:
            _fail(str(exc))
    try:
        context = load_context(cfg.hook, queries=queries)
    except ConfigError as exc:
        _fail(str(exc))
    return context, cfg.budget, cfg.k


@app.command(name="snapshot")
def snapshot_cmd(
    out: Annotated[Path, typer.Option("--out", "-o", help="Lockfile path to write.")],
    label: Annotated[str, typer.Option("--label", help="Required stable label, e.g. a git SHA.")],
    queries: Annotated[
        Path | None,
        typer.Option("--queries", help="JSONL query file (overrides config)."),
    ] = None,
    config: Annotated[
        Path, typer.Option("--config", help="pyproject.toml with [tool.retrieval_diff].")
    ] = Path("pyproject.toml"),
    k: Annotated[int | None, typer.Option("--k", help="Top-K depth (overrides config).")] = None,
) -> None:
    """Snapshot the wired retriever's top-K output into a lockfile."""
    if not label:
        _fail("--label is required and must be non-empty")
    query_override = load_queries(queries) if queries is not None else None
    context, _budget, cfg_k = _resolve_context(config, query_override)
    depth = k if k is not None else cfg_k
    snap = snapshot(context.retriever, context.queries, depth, label=label)
    lockfile.save(snap, out)
    _out.print(f"[green]wrote[/green] {out} ({len(snap.results)} queries, K={depth})")


def _load_two(old_path: Path, new_path: Path) -> tuple[Snapshot, Snapshot]:
    """Load two lockfiles, failing cleanly on errors."""
    try:
        old = lockfile.load(old_path)
        new = lockfile.load(new_path)
    except lockfile.LockfileError as exc:
        _fail(str(exc))
    return old, new


def _diff_or_fail(old: Snapshot, new: Snapshot) -> SnapshotDiff:
    """Diff two snapshots, mapping diff errors to clean CLI failures."""
    try:
        return diff_snapshots(old, new)
    except (KMismatchError, EmptyIntersectionError) as exc:
        _fail(str(exc))


def _diff_payload(diff: SnapshotDiff) -> dict[str, object]:
    """Build a stable, JSON-ready mapping for a diff (shared by json renderers)."""
    return {
        "k": diff.k,
        "fingerprint_delta": diff.fingerprint_delta,
        "query_set_delta": {
            "added_queries": diff.query_set_delta.added_queries,
            "removed_queries": diff.query_set_delta.removed_queries,
        },
        "summary": {
            "kind_counts": {kind.value: count for kind, count in diff.summary.kind_counts.items()},
            "mean_churn": diff.summary.mean_churn,
            "max_churn": diff.summary.max_churn,
            "golden_movements": [
                {
                    "query": m.query,
                    "golden_id": m.golden_id,
                    "old_rank": m.old_rank,
                    "new_rank": m.new_rank,
                    "rank_delta": m.rank_delta,
                    "removed": m.removed,
                }
                for m in diff.summary.golden_movements
            ],
        },
        "per_query": {
            query: {
                "churn": qd.churn,
                "kinds": {
                    cid: sorted(k.value for k in kinds) for cid, kinds in sorted(qd.kinds.items())
                },
                "rank_delta": dict(sorted(qd.rank_delta.items())),
                "score_delta": dict(sorted(qd.score_delta.items())),
            }
            for query, qd in sorted(diff.per_query.items())
        },
    }


def _diff_to_json(diff: SnapshotDiff) -> str:
    """Serialize a diff to a stable JSON string for ``--format json``."""
    return json.dumps(_diff_payload(diff), sort_keys=True, indent=2, ensure_ascii=False)


def _budget_payload(report: BudgetReport) -> dict[str, object]:
    """Build a JSON-ready mapping of a budget verdict for the ``check`` gate."""
    return {
        "passed": report.passed,
        "violations": [
            {
                "kind": violation.kind.value,
                "query": violation.query,
                "chunk_id": violation.chunk_id,
                "message": violation.message,
            }
            for violation in report.violations
        ],
    }


@app.command(name="diff")
def diff_cmd(
    old: Annotated[Path, typer.Argument(help="Baseline lockfile.")],
    new: Annotated[Path, typer.Argument(help="Candidate lockfile.")],
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output: term | md | json | pr-comment."),
    ] = "term",
) -> None:
    """Diff two lockfiles and print the result in the requested format."""
    if fmt not in {"term", "md", "json", "pr-comment"}:
        _fail(f"unknown --format {fmt!r}; expected term|md|json|pr-comment")
    old_snap, new_snap = _load_two(old, new)
    diff = _diff_or_fail(old_snap, new_snap)
    if fmt == "term":
        sys.stdout.write(render_terminal(diff))
    elif fmt == "md":
        sys.stdout.write(render_markdown(diff))
    elif fmt == "pr-comment":
        sys.stdout.write(render_pr_comment(diff))
    else:
        sys.stdout.write(_diff_to_json(diff) + "\n")


@app.command(name="report")
def report_cmd(
    old: Annotated[Path, typer.Argument(help="Baseline lockfile.")],
    new: Annotated[Path, typer.Argument(help="Candidate lockfile.")],
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Write Markdown here (default: stdout)."),
    ] = None,
) -> None:
    """Render a Markdown report of the diff between two lockfiles."""
    old_snap, new_snap = _load_two(old, new)
    diff = _diff_or_fail(old_snap, new_snap)
    markdown = render_markdown(diff)
    if out is None:
        sys.stdout.write(markdown)
    else:
        Path(out).write_text(markdown, encoding="utf-8")
        _out.print(f"[green]wrote[/green] {out}")


@app.command(name="check")
def check_cmd(
    lock: Annotated[Path, typer.Option("--lock", help="Committed lockfile to gate on.")],
    config: Annotated[
        Path, typer.Option("--config", help="pyproject.toml with [tool.retrieval_diff].")
    ] = Path("pyproject.toml"),
    label: Annotated[
        str, typer.Option("--label", help="Label for the freshly captured snapshot.")
    ] = "check",
    allow_query_set_change: Annotated[
        bool,
        typer.Option(
            "--allow-query-set-change",
            help="Permit a non-empty query-set delta (otherwise fails).",
        ),
    ] = False,
    max_churn: Annotated[
        float | None, typer.Option("--max-churn", help="Override budget max_churn.")
    ] = None,
    max_golden_rank_drop: Annotated[
        int | None,
        typer.Option("--max-golden-rank-drop", help="Override golden rank-drop cap."),
    ] = None,
    fmt: Annotated[str, typer.Option("--format", "-f", help="Output: term | md | json.")] = "term",
    junit_xml: Annotated[
        Path | None,
        typer.Option("--junit-xml", help="Also write a JUnit XML report to this path."),
    ] = None,
) -> None:
    """Re-snapshot the wired retriever and fail CI on a budget regression."""
    if fmt not in {"term", "md", "json"}:
        _fail(f"unknown --format {fmt!r}; expected term|md|json")
    context, budget, _cfg_k = _resolve_context(config, None)

    if allow_query_set_change:
        budget = budget.model_copy(update={"query_set_change_fails": False})
    if max_churn is not None:
        budget = budget.model_copy(update={"max_churn": max_churn})
    if max_golden_rank_drop is not None:
        budget = budget.model_copy(update={"max_golden_rank_drop": max_golden_rank_drop})

    try:
        old = lockfile.load(lock)
    except lockfile.LockfileError as exc:
        _fail(str(exc))

    new = snapshot(context.retriever, context.queries, old.k, label=label)
    try:
        diff = diff_snapshots(old, new, score_eps=DEFAULT_SCORE_EPS, goldens=context.goldens)
    except (KMismatchError, EmptyIntersectionError) as exc:
        _fail(str(exc))

    audit_violations = audit_goldens_past_k(new, context.retriever, context.goldens, budget)
    report = evaluate_budget(diff, budget, audit_violations=audit_violations)

    if junit_xml is not None:
        junit_xml.write_text(render_junit(diff, report), encoding="utf-8")

    if fmt == "md":
        sys.stdout.write(render_markdown(diff))
        sys.stdout.write("\n")
        sys.stdout.write(render_budget_markdown(report))
    elif fmt == "json":
        payload = {**_budget_payload(report), "diff": _diff_payload(diff)}
        sys.stdout.write(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    else:  # term: keep the human-facing rich table and verdict lines
        sys.stdout.write(render_terminal(diff))
        if report.passed:
            _out.print("[green]check passed[/green]: no retrieval regression")
        else:
            _err.print(f"[red]{report.summary()}[/red]")

    raise typer.Exit(EXIT_OK if report.passed else EXIT_REGRESSION)


@app.command(name="attribute")
def attribute_cmd(
    old: Annotated[Path, typer.Argument(help="Baseline lockfile.")],
    new: Annotated[Path, typer.Argument(help="Candidate lockfile.")],
    config: Annotated[
        Path, typer.Option("--config", help="pyproject.toml with [tool.retrieval_diff].")
    ] = Path("pyproject.toml"),
    fmt: Annotated[str, typer.Option("--format", "-f", help="Output: term | md | json.")] = "json",
) -> None:
    """Attribute the changes between two lockfiles to single config axes."""
    if fmt not in {"term", "md", "json"}:
        _fail(f"unknown --format {fmt!r}; expected term|md|json")
    old_snap, new_snap = _load_two(old, new)
    diff = _diff_or_fail(old_snap, new_snap)

    context, _budget, _cfg_k = _resolve_context(config, None)

    attributions = attribute_changes(
        old_snap.fingerprint,
        new_snap.fingerprint,
        diff,
        factory=context.factory,
        queries=list(context.queries),
        k=old_snap.k,
        corpus=context.corpus,
    )

    if fmt == "term":
        sys.stdout.write(render_attributions_terminal(attributions))
    elif fmt == "md":
        sys.stdout.write(render_attributions_markdown(attributions))
    else:
        payload = [
            {
                "query": a.change_ref.query,
                "chunk_id": a.change_ref.chunk_id,
                "kind": a.change_ref.kind.value,
                "axis": a.axis,
                "confidence": a.confidence,
                "evidence": a.evidence,
            }
            for a in attributions
        ]
        sys.stdout.write(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False))
        sys.stdout.write("\n")


def _snapshot_to_json(snap: Snapshot, query: str | None) -> str:
    """Serialize a lockfile view to a stable JSON string for ``show --format json``.

    When ``query`` is given only that query's hits are included; the header
    (label, K, fingerprint digest and axes) and the total ``query_count`` always
    reflect the full lockfile.
    """
    fp = snap.fingerprint
    queries = [query] if query is not None else sorted(snap.results)
    payload = {
        "created_label": snap.created_label,
        "k": snap.k,
        "query_count": len(snap.results),
        "version": snap.version,
        "fingerprint": {
            "digest": fp.digest(),
            "axes": {
                "alpha": fp.alpha,
                "chunk_params": dict(sorted(fp.chunk_params.items())),
                "embedding_model": fp.embedding_model,
                "index_content_hash": fp.index_content_hash,
                "reranker": fp.reranker,
            },
        },
        "results": {
            q: {
                "hits": [
                    {"id": hit.id, "rank": hit.rank, "score": hit.score}
                    for hit in sorted(snap.results[q].hits, key=lambda h: h.rank)
                ]
            }
            for q in queries
        },
    }
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


@app.command(name="show")
def show_cmd(
    lock: Annotated[Path, typer.Argument(help="Lockfile to inspect.")],
    query: Annotated[
        str | None,
        typer.Option("--query", help="Show only this query (must exist in the lock)."),
    ] = None,
    fmt: Annotated[str, typer.Option("--format", "-f", help="Output: term | md | json.")] = "term",
) -> None:
    """Inspect a single lockfile: its header and top-K ids/scores/ranks."""
    if fmt not in {"term", "md", "json"}:
        _fail(f"unknown --format {fmt!r}; expected term|md|json")
    try:
        snap = lockfile.load(lock)
    except lockfile.LockfileError as exc:
        _fail(str(exc))
    if query is not None and query not in snap.results:
        _fail(f"query {query!r} not in lockfile; known queries: {sorted(snap.results)}")
    if fmt == "term":
        sys.stdout.write(render_snapshot_terminal(snap, query=query))
    elif fmt == "md":
        sys.stdout.write(render_snapshot_markdown(snap, query=query))
    else:
        sys.stdout.write(_snapshot_to_json(snap, query) + "\n")


def main() -> None:  # pragma: no cover - thin console-script wrapper
    """Console-script entry point."""
    app()


__all__ = ["app", "main"]
