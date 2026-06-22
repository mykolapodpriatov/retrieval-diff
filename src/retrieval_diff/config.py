"""Project configuration loading and the retriever-wiring hook.

The CLI must never hardcode a backend, so the retriever, query set, and (optional)
attribution factory are wired through a project-supplied Python hook -- mirroring
how a user already constructs their retriever. The hook lives in a module
declared under ``[tool.retrieval_diff]`` in ``pyproject.toml`` (or pointed at via
``--config``) and exposes a :class:`ProjectContext`.

Golden chunks and budget thresholds are also declared here. Query sets load from
a JSONL file (one ``{"query": ...}`` object per line) so they are easy to commit
and review.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from retrieval_diff.budget import RegressionBudget
from retrieval_diff.retrievers import Retriever
from retrieval_diff.retrievers.factory import RetrieverFactory

#: Name of the factory function a project hook module must expose.
HOOK_FACTORY_NAME = "build_context"


class ConfigError(Exception):
    """Raised when project configuration is missing or malformed."""


@dataclass(frozen=True)
class ProjectContext:
    """Everything the CLI needs to snapshot/diff/check/attribute a project.

    Attributes:
        retriever: The live retriever to snapshot.
        queries: The query set.
        goldens: ``{query: {golden_id, ...}}`` declarations (may be empty).
        factory: Optional factory enabling attribution and ``audit_k`` re-ranks.
        corpus: Optional raw corpus passed to the factory for build-time axes.
    """

    retriever: Retriever
    queries: tuple[str, ...]
    goldens: Mapping[str, set[str]] = field(default_factory=dict)
    factory: RetrieverFactory | None = None
    corpus: object | None = None


@dataclass(frozen=True)
class ProjectConfig:
    """Parsed ``[tool.retrieval_diff]`` settings."""

    hook: str | None
    k: int
    lock_path: Path
    queries_path: Path | None
    budget: RegressionBudget
    audit_k: int | None


def load_queries(path: str | Path) -> list[str]:
    """Load a query set from a JSONL file.

    Each non-blank line must be a JSON object with a ``"query"`` string field, or
    a bare JSON string. Duplicate queries are rejected.

    Args:
        path: Path to the ``.jsonl`` query file.

    Returns:
        The ordered list of queries.

    Raises:
        ConfigError: On a missing file or a malformed line.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"query file not found: {p}")
    queries: list[str] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
        if isinstance(obj, str):
            query = obj
        elif isinstance(obj, dict) and isinstance(obj.get("query"), str):
            query = obj["query"]
        else:
            raise ConfigError(
                f"{p}:{lineno}: expected a JSON string or object with a string 'query' field"
            )
        if query in seen:
            raise ConfigError(f"{p}:{lineno}: duplicate query {query!r}")
        seen.add(query)
        queries.append(query)
    if not queries:
        raise ConfigError(f"{p}: query file is empty")
    return queries


def _parse_goldens(value: Any) -> dict[str, set[str]]:
    """Parse a goldens mapping from raw TOML/JSON data."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError("'goldens' must be a table mapping query -> list of ids")
    out: dict[str, set[str]] = {}
    for query, ids in value.items():
        if not isinstance(ids, (list, tuple)):
            raise ConfigError(f"goldens[{query!r}] must be a list of chunk ids")
        out[str(query)] = {str(i) for i in ids}
    return out


def load_project_config(pyproject: str | Path) -> ProjectConfig:
    """Load ``[tool.retrieval_diff]`` from a ``pyproject.toml``.

    Args:
        pyproject: Path to the project's ``pyproject.toml``.

    Returns:
        A :class:`ProjectConfig`. Missing keys fall back to sensible defaults
        (``k=10``, ``lock_path=retrieval.lock``).

    Raises:
        ConfigError: If the file is missing or the section is malformed.
    """
    p = Path(pyproject)
    if not p.exists():
        raise ConfigError(f"pyproject.toml not found: {p}")
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    section = data.get("tool", {}).get("retrieval_diff", {})
    if not isinstance(section, Mapping):
        raise ConfigError("[tool.retrieval_diff] must be a table")

    budget_section = section.get("budget", {})
    if not isinstance(budget_section, Mapping):
        raise ConfigError("[tool.retrieval_diff.budget] must be a table")
    audit_k = section.get("audit_k")
    budget = RegressionBudget(
        removed_golden_fails=bool(budget_section.get("removed_golden_fails", True)),
        max_golden_rank_drop=int(budget_section.get("max_golden_rank_drop", 2)),
        max_churn=float(budget_section.get("max_churn", 0.5)),
        allow_new_chunks=bool(budget_section.get("allow_new_chunks", True)),
        query_set_change_fails=bool(budget_section.get("query_set_change_fails", True)),
        audit_k=(int(audit_k) if audit_k is not None else None),
    )

    queries_path = section.get("queries")
    return ProjectConfig(
        hook=section.get("hook"),
        k=int(section.get("k", 10)),
        lock_path=Path(section.get("lock_path", "retrieval.lock")),
        queries_path=(Path(queries_path) if queries_path else None),
        budget=budget,
        audit_k=(int(audit_k) if audit_k is not None else None),
    )


def _import_hook_module(hook: str) -> Any:
    """Import a hook by dotted module path or filesystem path."""
    if hook.endswith(".py") or "/" in hook or "\\" in hook:
        path = Path(hook).resolve()
        if not path.exists():
            raise ConfigError(f"hook file not found: {path}")
        spec = importlib.util.spec_from_file_location("retrieval_diff_user_hook", path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ConfigError(f"cannot load hook from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    try:
        return importlib.import_module(hook)
    except ImportError as exc:
        raise ConfigError(f"cannot import hook module {hook!r}: {exc}") from exc


def load_context(hook: str, *, queries: Iterable[str] | None = None) -> ProjectContext:
    """Invoke a project's ``build_context`` hook and return its context.

    Args:
        hook: A dotted module path (``my_pkg.rdiff_hook``) or a path to a ``.py``
            file exposing :data:`HOOK_FACTORY_NAME`.
        queries: Optional queries to pass to the hook; hooks may ignore this and
            supply their own.

    Returns:
        The :class:`ProjectContext` produced by the hook.

    Raises:
        ConfigError: If the module or factory is missing, or the returned object
            is not a :class:`ProjectContext`.
    """
    module = _import_hook_module(hook)
    factory = getattr(module, HOOK_FACTORY_NAME, None)
    if factory is None or not callable(factory):
        raise ConfigError(
            f"hook {hook!r} must define a callable {HOOK_FACTORY_NAME}() returning a ProjectContext"
        )
    context = factory(list(queries) if queries is not None else None)
    if not isinstance(context, ProjectContext):
        raise ConfigError(
            f"{HOOK_FACTORY_NAME}() must return a ProjectContext, got {type(context).__name__}"
        )
    return context


__all__ = [
    "HOOK_FACTORY_NAME",
    "ConfigError",
    "ProjectConfig",
    "ProjectContext",
    "load_context",
    "load_project_config",
    "load_queries",
]
