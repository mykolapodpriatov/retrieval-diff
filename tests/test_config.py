"""Tests for project config loading, query parsing, and the hook mechanism."""

from __future__ import annotations

from pathlib import Path

import pytest

from retrieval_diff.config import (
    ConfigError,
    ProjectContext,
    load_context,
    load_project_config,
    load_queries,
)

# --- query loading ----------------------------------------------------------


def test_load_queries_object_and_string_lines(tmp_path: Path) -> None:
    """Both ``{"query": ...}`` objects and bare strings parse, blanks skipped."""
    qfile = tmp_path / "q.jsonl"
    qfile.write_text(
        '{"query": "first"}\n\n"second"\n   \n{"query": "third"}\n',
        encoding="utf-8",
    )
    assert load_queries(qfile) == ["first", "second", "third"]


def test_load_queries_rejects_duplicates(tmp_path: Path) -> None:
    """Duplicate queries are rejected with the offending line number."""
    qfile = tmp_path / "q.jsonl"
    qfile.write_text('"dup"\n"dup"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate query"):
        load_queries(qfile)


def test_load_queries_rejects_bad_json(tmp_path: Path) -> None:
    """A malformed JSON line is reported with its line number."""
    qfile = tmp_path / "q.jsonl"
    qfile.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_queries(qfile)


def test_load_queries_rejects_wrong_shape(tmp_path: Path) -> None:
    """A JSON value that is neither a string nor a {query} object is rejected."""
    qfile = tmp_path / "q.jsonl"
    qfile.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="string 'query' field"):
        load_queries(qfile)


def test_load_queries_rejects_empty_file(tmp_path: Path) -> None:
    """An all-blank query file is rejected."""
    qfile = tmp_path / "q.jsonl"
    qfile.write_text("\n   \n", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_queries(qfile)


def test_load_queries_missing_file() -> None:
    """A missing query file raises a clear error."""
    with pytest.raises(ConfigError, match="not found"):
        load_queries("/nonexistent/q.jsonl")


# --- project config ---------------------------------------------------------


def test_load_project_config_defaults(tmp_path: Path) -> None:
    """An empty section yields sensible defaults."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.retrieval_diff]\n", encoding="utf-8")
    cfg = load_project_config(pyproject)
    assert cfg.k == 10
    assert cfg.lock_path == Path("retrieval.lock")
    assert cfg.budget.max_churn == 0.5
    assert cfg.budget.query_set_change_fails is True


def test_load_project_config_full(tmp_path: Path) -> None:
    """All keys parse, including the budget table and audit_k."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.retrieval_diff]
hook = "pkg.hook"
queries = "queries.jsonl"
k = 7
audit_k = 20

[tool.retrieval_diff.budget]
removed_golden_fails = false
max_golden_rank_drop = 3
max_churn = 0.25
allow_new_chunks = false
query_set_change_fails = false
""",
        encoding="utf-8",
    )
    cfg = load_project_config(pyproject)
    assert cfg.hook == "pkg.hook"
    assert cfg.k == 7
    assert cfg.audit_k == 20
    assert cfg.queries_path == Path("queries.jsonl")
    assert cfg.budget.removed_golden_fails is False
    assert cfg.budget.max_golden_rank_drop == 3
    assert cfg.budget.max_churn == 0.25
    assert cfg.budget.allow_new_chunks is False
    assert cfg.budget.query_set_change_fails is False
    assert cfg.budget.audit_k == 20


def test_load_project_config_missing_file() -> None:
    """A missing pyproject.toml raises a clear error."""
    with pytest.raises(ConfigError, match="not found"):
        load_project_config("/nonexistent/pyproject.toml")


# --- hook loading -----------------------------------------------------------


_GOOD_HOOK = """
from retrieval_diff.config import ProjectContext
from retrieval_diff.retrievers.memory import Corpus, HashingEmbedder, HybridRetriever


def build_context(queries):
    corpus = Corpus.from_mapping({"a": "alpha text", "b": "beta text"})
    retriever = HybridRetriever(corpus, HashingEmbedder(dim=8, model_id="e1"))
    qs = tuple(queries) if queries else ("a query",)
    return ProjectContext(retriever=retriever, queries=qs, goldens={})
"""


def test_load_context_from_file_path(tmp_path: Path) -> None:
    """A hook given as a .py path loads and produces a ProjectContext."""
    hook = tmp_path / "myhook.py"
    hook.write_text(_GOOD_HOOK, encoding="utf-8")
    ctx = load_context(str(hook), queries=["x", "y"])
    assert isinstance(ctx, ProjectContext)
    assert ctx.queries == ("x", "y")


def test_load_context_missing_factory(tmp_path: Path) -> None:
    """A hook without build_context is rejected."""
    hook = tmp_path / "bad.py"
    hook.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must define a callable"):
        load_context(str(hook))


def test_load_context_wrong_return_type(tmp_path: Path) -> None:
    """A hook returning a non-ProjectContext is rejected."""
    hook = tmp_path / "wrong.py"
    hook.write_text("def build_context(queries):\n    return 42\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must return a ProjectContext"):
        load_context(str(hook))


def test_load_context_missing_file() -> None:
    """A hook path that does not exist is rejected."""
    with pytest.raises(ConfigError, match="hook file not found"):
        load_context("/nonexistent/hook.py")


def test_load_context_bad_dotted_module() -> None:
    """An unimportable dotted hook module is rejected clearly."""
    with pytest.raises(ConfigError, match="cannot import hook module"):
        load_context("definitely.not.a.real.module")
