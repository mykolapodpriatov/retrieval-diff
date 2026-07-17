"""Rendering of diffs and budget reports for terminals, Markdown, and PRs.

Three output shapes:

* :func:`render_terminal` -- a colored :mod:`rich` table for interactive use.
* :func:`render_markdown` -- a full Markdown report for artifacts/PRs.
* :func:`render_pr_comment` -- a compact, collapsible Markdown summary suited to
  a PR comment body.

All renderers are deterministic (stable id/query ordering) and read-only over
the diff, so the same diff always produces the same text.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence

from rich.console import Console
from rich.table import Table
from rich.text import Text

from retrieval_diff.budget import BudgetReport
from retrieval_diff.types import (
    AxisAttribution,
    ChangeKind,
    Confidence,
    QueryDiff,
    SnapshotDiff,
)

#: Glyphs used to summarize a chunk's change set in compact views.
_KIND_GLYPH = {
    ChangeKind.ADDED: "+",
    ChangeKind.REMOVED: "-",
    ChangeKind.REORDERED: "~",
    ChangeKind.SCORE_CHANGED: "$",
    ChangeKind.UNCHANGED: "=",
}

_KIND_STYLE = {
    ChangeKind.ADDED: "green",
    ChangeKind.REMOVED: "red",
    ChangeKind.REORDERED: "yellow",
    ChangeKind.SCORE_CHANGED: "cyan",
    ChangeKind.UNCHANGED: "dim",
}

#: Terminal styles keyed by attribution confidence verdict.
_CONFIDENCE_STYLE: dict[Confidence, str] = {
    "confirmed": "green",
    "ambiguous": "yellow",
    "not_attributable": "dim",
}

#: Placeholder shown where an attribution has no single responsible axis.
_NO_AXIS = "·"


def _kinds_label(kinds: set[ChangeKind]) -> str:
    """Return a stable comma-joined label for a chunk's change set."""
    return ",".join(sorted(k.value for k in kinds))


def _kinds_glyphs(kinds: set[ChangeKind]) -> str:
    """Return a stable glyph string for a chunk's change set."""
    ordered = sorted(kinds, key=lambda k: k.value)
    return "".join(_KIND_GLYPH[k] for k in ordered)


def _primary_kind(kinds: set[ChangeKind]) -> ChangeKind:
    """Return the most salient kind for styling purposes."""
    for kind in (
        ChangeKind.REMOVED,
        ChangeKind.ADDED,
        ChangeKind.REORDERED,
        ChangeKind.SCORE_CHANGED,
        ChangeKind.UNCHANGED,
    ):
        if kind in kinds:
            return kind
    return ChangeKind.UNCHANGED  # pragma: no cover - kinds is never empty


def _fmt_delta_int(value: int | None) -> str:
    """Format a signed integer rank delta (``+2``/``-1``/``·``)."""
    if value is None:
        return "·"
    return f"{value:+d}"


def _fmt_delta_float(value: float | None) -> str:
    """Format a signed float score delta with fixed precision."""
    if value is None:
        return "·"
    return f"{value:+.4f}"


def _query_table(query: str, qd: QueryDiff) -> Table:
    """Build a rich table for a single query's diff."""
    table = Table(title=f"query: {query!r}  (churn={qd.churn:.3f})", show_lines=False)
    table.add_column("chunk", overflow="fold")
    table.add_column("change")
    table.add_column("Δrank", justify="right")
    table.add_column("Δscore", justify="right")
    for cid in sorted(qd.kinds):
        kinds = qd.kinds[cid]
        style = _KIND_STYLE[_primary_kind(kinds)]
        table.add_row(
            cid,
            f"[{style}]{_kinds_label(kinds)}[/{style}]",
            _fmt_delta_int(qd.rank_delta.get(cid)),
            _fmt_delta_float(qd.score_delta.get(cid)),
        )
    return table


def render_terminal(diff: SnapshotDiff, *, console: Console | None = None) -> str:
    """Render the diff as colored tables and return the captured text.

    Args:
        diff: The diff to render.
        console: Optional console; a string-capturing console is used by default
            so callers can both print and test the output.

    Returns:
        The rendered text (ANSI stripped when captured to a string buffer).
    """
    buffer = io.StringIO()
    con = console or Console(file=buffer, force_terminal=False, width=100)
    con.print(_header_text(diff))
    if not diff.query_set_delta.is_empty():
        con.print(_query_set_text(diff))
    for query in sorted(diff.per_query):
        con.print(_query_table(query, diff.per_query[query]))
    if diff.summary.golden_movements:
        con.print(_golden_text(diff))
    return buffer.getvalue()


def _header_text(diff: SnapshotDiff) -> str:
    """Return the one-line diff header (counts + churn + axes)."""
    counts = diff.summary.kind_counts
    parts = [f"{kind.value}={counts.get(kind, 0)}" for kind in ChangeKind]
    axes = ", ".join(diff.fingerprint_delta) if diff.fingerprint_delta else "none"
    return (
        f"retrieval-diff: {', '.join(parts)} | "
        f"mean_churn={diff.summary.mean_churn:.3f} "
        f"max_churn={diff.summary.max_churn:.3f} | changed axes: {axes}"
    )


def _query_set_text(diff: SnapshotDiff) -> str:
    """Return a textual note on query-set changes."""
    delta = diff.query_set_delta
    return (
        "query-set changed -> added: "
        f"{delta.added_queries or '[]'}, removed: {delta.removed_queries or '[]'}"
    )


def _golden_text(diff: SnapshotDiff) -> str:
    """Return a textual summary of golden movements."""
    lines = ["golden movements:"]
    for move in diff.summary.golden_movements:
        if move.removed:
            lines.append(f"  - {move.query!r}/{move.golden_id!r}: REMOVED (was {move.old_rank})")
        else:
            lines.append(
                f"  - {move.query!r}/{move.golden_id!r}: "
                f"{move.old_rank} -> {move.new_rank} ({_fmt_delta_int(move.rank_delta)})"
            )
    return "\n".join(lines)


def render_markdown(diff: SnapshotDiff) -> str:
    """Render the diff as a full Markdown report.

    Returns:
        A Markdown document with a summary, optional query-set/golden sections,
        and one table per query in the intersection.
    """
    lines: list[str] = ["# retrieval-diff report", ""]
    counts = diff.summary.kind_counts
    lines.append("## Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for kind in ChangeKind:
        lines.append(f"| {kind.value} | {counts.get(kind, 0)} |")
    lines.append(f"| mean churn | {diff.summary.mean_churn:.4f} |")
    lines.append(f"| max churn | {diff.summary.max_churn:.4f} |")
    axes = ", ".join(diff.fingerprint_delta) if diff.fingerprint_delta else "none"
    lines.append(f"| changed axes | {axes} |")
    lines.append("")

    if not diff.query_set_delta.is_empty():
        lines.append("## Query-set changes")
        lines.append("")
        lines.append(f"- added queries: {diff.query_set_delta.added_queries or '[]'}")
        lines.append(f"- removed queries: {diff.query_set_delta.removed_queries or '[]'}")
        lines.append("")

    if diff.summary.golden_movements:
        lines.append("## Golden movements")
        lines.append("")
        lines.append("| query | golden | old rank | new rank | Δrank | removed |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for move in diff.summary.golden_movements:
            lines.append(
                f"| {move.query} | {move.golden_id} | {move.old_rank} | "
                f"{move.new_rank} | {_fmt_delta_int(move.rank_delta)} | "
                f"{'yes' if move.removed else 'no'} |"
            )
        lines.append("")

    lines.append("## Per-query diff")
    lines.append("")
    for query in sorted(diff.per_query):
        qd = diff.per_query[query]
        lines.append(f"### `{query}` (churn = {qd.churn:.4f})")
        lines.append("")
        lines.append("| chunk | change | Δrank | Δscore |")
        lines.append("| --- | --- | --- | --- |")
        for cid in sorted(qd.kinds):
            lines.append(
                f"| {cid} | {_kinds_label(qd.kinds[cid])} | "
                f"{_fmt_delta_int(qd.rank_delta.get(cid))} | "
                f"{_fmt_delta_float(qd.score_delta.get(cid))} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_pr_comment(diff: SnapshotDiff, *, budget_report: BudgetReport | None = None) -> str:
    """Render a compact, collapsible Markdown summary for a PR comment.

    Args:
        diff: The diff to summarize.
        budget_report: Optional budget result; its pass/fail status leads the
            comment so reviewers see the verdict immediately.

    Returns:
        A short Markdown string with a per-query glyph summary in a collapsible
        ``<details>`` block.
    """
    status = ""
    if budget_report is not None:
        status = (
            "✅ retrieval budget OK"
            if budget_report.passed
            else f"❌ retrieval budget FAILED ({len(budget_report.violations)} violation(s))"
        )
    header = "### retrieval-diff"
    summary = (
        f"mean churn `{diff.summary.mean_churn:.3f}`, "
        f"max churn `{diff.summary.max_churn:.3f}`, "
        f"changed axes: {', '.join(diff.fingerprint_delta) or 'none'}"
    )

    lines: list[str] = [header, ""]
    if status:
        lines.append(f"**{status}**")
        lines.append("")
    lines.append(summary)
    lines.append("")

    if budget_report is not None and not budget_report.passed:
        lines.append("**Violations:**")
        for violation in budget_report.violations:
            lines.append(f"- {violation.message}")
        lines.append("")

    lines.append("<details><summary>per-query changes</summary>")
    lines.append("")
    lines.append("| query | churn | changes |")
    lines.append("| --- | --- | --- |")
    for query in sorted(diff.per_query):
        qd = diff.per_query[query]
        glyphs = " ".join(
            f"{cid}{_kinds_glyphs(qd.kinds[cid])}"
            for cid in sorted(qd.kinds)
            if qd.kinds[cid] != {ChangeKind.UNCHANGED}
        )
        lines.append(f"| `{query}` | {qd.churn:.3f} | {glyphs or '(no changes)'} |")
    lines.append("")
    lines.append("</details>")
    return "\n".join(lines).rstrip() + "\n"


def _evidence_text(evidence: Mapping[str, object]) -> str:
    """Return a stable, single-string rendering of an attribution's evidence.

    Keys are sorted for determinism; list/tuple values render as ``[a, b]`` and
    scalars via ``str``. The result is plain text (no ``rich`` markup), so it is
    safe to place verbatim in a :class:`~rich.text.Text` cell even though axis
    lists contain the ``[`` and ``]`` characters that markup would try to parse.
    """
    parts: list[str] = []
    for key in sorted(evidence):
        value = evidence[key]
        if isinstance(value, (list, tuple)):
            rendered = "[" + ", ".join(str(item) for item in value) + "]"
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return "; ".join(parts)


def render_attributions_terminal(
    attributions: Sequence[AxisAttribution], *, console: Console | None = None
) -> str:
    """Render axis attributions as a colored :mod:`rich` table.

    Args:
        attributions: The per-change verdicts to render.
        console: Optional console; a string-capturing console is used by default
            so callers can both print and test the output.

    Returns:
        The rendered table text (ANSI stripped when captured to a string buffer).
    """
    buffer = io.StringIO()
    con = console or Console(file=buffer, force_terminal=False, width=120)
    table = Table(title="retrieval-diff attribution", show_lines=False)
    table.add_column("query", overflow="fold")
    table.add_column("chunk", overflow="fold")
    table.add_column("kind")
    table.add_column("axis")
    table.add_column("confidence")
    table.add_column("evidence", overflow="fold")
    for attribution in attributions:
        ref = attribution.change_ref
        # User-controlled fields are added as Text (never markup-parsed) so that
        # a query, chunk id, or bracketed evidence value can never be mistaken
        # for a rich style tag.
        table.add_row(
            Text(ref.query),
            Text(ref.chunk_id),
            Text(ref.kind.value),
            Text(attribution.axis or _NO_AXIS),
            Text(attribution.confidence, style=_CONFIDENCE_STYLE[attribution.confidence]),
            Text(_evidence_text(attribution.evidence)),
        )
    con.print(table)
    return buffer.getvalue()


def render_attributions_markdown(attributions: Sequence[AxisAttribution]) -> str:
    """Render axis attributions as a Markdown table.

    Returns:
        A Markdown document with one row per change (query, chunk, kind, axis,
        confidence, evidence), or a short note when there are no attributions.
    """
    lines: list[str] = ["# retrieval-diff attribution", ""]
    if not attributions:
        lines.append("_no attributable changes_")
        return "\n".join(lines).rstrip() + "\n"
    lines.append("| query | chunk | kind | axis | confidence | evidence |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for attribution in attributions:
        ref = attribution.change_ref
        lines.append(
            f"| {ref.query} | {ref.chunk_id} | {ref.kind.value} | "
            f"{attribution.axis or _NO_AXIS} | {attribution.confidence} | "
            f"{_evidence_text(attribution.evidence)} |"
        )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "render_attributions_markdown",
    "render_attributions_terminal",
    "render_markdown",
    "render_pr_comment",
    "render_terminal",
]
