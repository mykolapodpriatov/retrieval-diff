"""retrieval-diff: a git-style diff for RAG retrievers.

Snapshot a retriever's top-K output per query into a versioned ``retrieval.lock``,
diff two snapshots to see which chunks were added/removed/reranked, attribute
each change to a single config axis by held-fixed replay, and gate CI on
retrieval regressions.

Public API surface:

* :class:`~retrieval_diff.types.Snapshot`, :class:`~retrieval_diff.types.SnapshotDiff`,
  :class:`~retrieval_diff.types.QueryDiff`, :class:`~retrieval_diff.types.ChangeKind`.
* :func:`~retrieval_diff.snapshot.snapshot`, lockfile :func:`~retrieval_diff.lockfile.load`/
  :func:`~retrieval_diff.lockfile.save`.
* :func:`~retrieval_diff.diff.diff_snapshots`.
* :func:`~retrieval_diff.attribution.attribute`.
* :class:`~retrieval_diff.budget.RegressionBudget`,
  :func:`~retrieval_diff.budget.evaluate_budget`.
"""

from __future__ import annotations

from retrieval_diff.attribution import attribute
from retrieval_diff.budget import BudgetReport, RegressionBudget, evaluate_budget
from retrieval_diff.diff import EmptyIntersectionError, KMismatchError, diff_snapshots
from retrieval_diff.fingerprint import ConfigFingerprint, index_content_hash
from retrieval_diff.lockfile import LockfileError, load, save
from retrieval_diff.snapshot import snapshot
from retrieval_diff.types import (
    AxisAttribution,
    ChangeKind,
    ChangeRef,
    QueryDiff,
    QueryResult,
    ScoredHit,
    Snapshot,
    SnapshotDiff,
)

__version__ = "0.1.0"

__all__ = [
    "AxisAttribution",
    "BudgetReport",
    "ChangeKind",
    "ChangeRef",
    "ConfigFingerprint",
    "EmptyIntersectionError",
    "KMismatchError",
    "LockfileError",
    "QueryDiff",
    "QueryResult",
    "RegressionBudget",
    "ScoredHit",
    "Snapshot",
    "SnapshotDiff",
    "__version__",
    "attribute",
    "diff_snapshots",
    "evaluate_budget",
    "index_content_hash",
    "load",
    "save",
    "snapshot",
]
