"""Tests for the GitHub Action entrypoint (token-gated) and adapter stubs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from retrieval_diff.action import entrypoint
from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.retrievers.adapters import MissingDependencyError
from retrieval_diff.retrievers.adapters.chroma_adapter import ChromaRetriever
from retrieval_diff.retrievers.adapters.faiss_adapter import FaissRetriever
from retrieval_diff.retrievers.adapters.pgvector_adapter import PgVectorRetriever
from retrieval_diff.retrievers.adapters.qdrant_adapter import QdrantRetriever

# A project hook reused by the action entrypoint test.
_HOOK = """
from retrieval_diff.config import ProjectContext
from retrieval_diff.retrievers.memory import Corpus, HashingEmbedder, HybridRetriever

CORPUS = Corpus.from_mapping(
    {"d1": "vector search", "d2": "lazy dog", "d3": "brown fox"}
)


def build_context(queries):
    embedder = HashingEmbedder(dim=16, model_id="e1")
    retriever = HybridRetriever(CORPUS, embedder, alpha=0.5)
    return ProjectContext(retriever=retriever, queries=("the fox",), goldens={})
"""

_PYPROJECT = '[tool.retrieval_diff]\nhook = "{hook}"\nk = 3\n'


def _setup_project(tmp_path: Path) -> tuple[Path, Path]:
    hook = tmp_path / "rdiff_hook.py"
    hook.write_text(_HOOK, encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_PYPROJECT.format(hook=hook.as_posix()), encoding="utf-8")
    embedder_lock = tmp_path / "retrieval.lock"
    return pyproject, embedder_lock


def _write_lock(pyproject: Path, lock: Path) -> None:
    from retrieval_diff.config import load_context, load_project_config
    from retrieval_diff.lockfile import save
    from retrieval_diff.snapshot import snapshot

    cfg = load_project_config(pyproject)
    assert cfg.hook is not None
    ctx = load_context(cfg.hook)
    snap = snapshot(ctx.retriever, ctx.queries, cfg.k, label="base")
    save(snap, lock)


def test_action_skips_comment_without_token(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    """With no token the action runs the check and skips the PR comment cleanly."""
    pyproject, lock = _setup_project(tmp_path)
    _write_lock(pyproject, lock)

    monkeypatch.setenv("RDIFF_CONFIG", str(pyproject))
    monkeypatch.setenv("RDIFF_LOCK", str(lock))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    code = entrypoint.main()
    captured = capsys.readouterr()
    assert code == 0
    assert "### retrieval-diff" in captured.out  # comment body printed to stdout
    assert "PR comment skipped" in captured.err


def test_action_writes_github_output(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The action records passed=true/false in $GITHUB_OUTPUT."""
    pyproject, lock = _setup_project(tmp_path)
    _write_lock(pyproject, lock)
    out_file = tmp_path / "gh_output"

    monkeypatch.setenv("RDIFF_CONFIG", str(pyproject))
    monkeypatch.setenv("RDIFF_LOCK", str(lock))
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    assert entrypoint.main() == 0
    assert "passed=true" in out_file.read_text(encoding="utf-8")


def test_action_reports_error_on_missing_config(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    """A missing config yields a non-zero error exit, not a crash."""
    monkeypatch.setenv("RDIFF_CONFIG", str(tmp_path / "nope.toml"))
    monkeypatch.setenv("RDIFF_LOCK", str(tmp_path / "nope.lock"))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    code = entrypoint.main()
    assert code == entrypoint.EXIT_ERROR
    assert "action error" in capsys.readouterr().err


def test_post_comment_skipped_without_token(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """_post_comment returns False (skipped) when no token is present."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert entrypoint._post_comment("body") is False


def test_action_yaml_exists() -> None:
    """The composite action manifest ships inside the package."""
    action_yml = Path(entrypoint.__file__).resolve().parent / "action.yml"
    assert action_yml.exists()
    body = action_yml.read_text(encoding="utf-8")
    assert 'using: "composite"' in body
    assert "github-token" in body


@pytest.mark.parametrize(
    ("ctor", "args", "extra", "backend"),
    [
        (FaissRetriever, (object(), [], lambda q: []), "faiss", "faiss"),
        (ChromaRetriever, (object(),), "chroma", "chromadb"),
        (QdrantRetriever, (object(), "c", lambda q: []), "qdrant", "qdrant_client"),
        (PgVectorRetriever, ("dsn", "t", lambda q: []), "pg", "psycopg"),
    ],
)
def test_adapter_import_guard(ctor, args, extra, backend) -> None:  # type: ignore[no-untyped-def]
    """Each adapter raises MissingDependencyError naming its extra when uninstalled.

    Skipped per-backend when that backend *is* installed — the guard only fires
    on a missing import, so there is nothing to assert. The `adapters` CI job
    installs the extras; the `test` matrix does not.
    """
    if importlib.util.find_spec(backend) is not None:
        pytest.skip(f"{backend} is installed; the import guard cannot fire")
    with pytest.raises(MissingDependencyError) as excinfo:
        ctor(*args, ConfigFingerprint())
    assert excinfo.value.extra == extra
    assert f"[{extra}]" in str(excinfo.value)
