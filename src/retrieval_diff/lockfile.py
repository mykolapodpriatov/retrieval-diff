"""Read, write, and validate the versioned ``retrieval.lock`` file.

The lockfile is JSON with **sorted keys and stable formatting** so that diffs in
git are minimal and reviewable, and so a re-save of an unchanged snapshot is
byte-identical. Scores are stored at full float precision (not the ``.9g``
digest form) so that ``score_eps`` comparisons remain meaningful, while the
*fingerprint digest* uses the canonical form for cross-platform stability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.snapshot import SNAPSHOT_VERSION
from retrieval_diff.types import QueryResult, ScoredHit, Snapshot

#: The maximum lockfile schema version this library understands.
SUPPORTED_VERSION = SNAPSHOT_VERSION


class LockfileError(Exception):
    """Raised when a lockfile cannot be parsed or is incompatible."""


def _fingerprint_to_json(fp: ConfigFingerprint) -> dict[str, Any]:
    """Serialize a fingerprint to a plain, sorted-key JSON dict."""
    return {
        "alpha": fp.alpha,
        "chunk_params": dict(sorted(fp.chunk_params.items())),
        "digest": fp.digest(),
        "embedding_model": fp.embedding_model,
        "index_content_hash": fp.index_content_hash,
        "reranker": fp.reranker,
    }


def _fingerprint_from_json(data: dict[str, Any]) -> ConfigFingerprint:
    """Deserialize a fingerprint, ignoring the stored (derived) digest."""
    try:
        return ConfigFingerprint(
            embedding_model=data.get("embedding_model"),
            chunk_params=dict(data.get("chunk_params") or {}),
            index_content_hash=data.get("index_content_hash"),
            reranker=data.get("reranker"),
            alpha=data.get("alpha"),
        )
    except Exception as exc:  # pydantic ValidationError or similar
        raise LockfileError(f"invalid fingerprint in lockfile: {exc}") from exc


def snapshot_to_dict(snap: Snapshot) -> dict[str, Any]:
    """Return the canonical, sorted-key dict representation of a snapshot."""
    results: dict[str, Any] = {}
    for query in sorted(snap.results):
        qr = snap.results[query]
        results[query] = {
            "hits": [{"id": hit.id, "rank": hit.rank, "score": hit.score} for hit in qr.hits],
            "query": qr.query,
        }
    return {
        "created_label": snap.created_label,
        "fingerprint": _fingerprint_to_json(snap.fingerprint),
        "k": snap.k,
        "results": results,
        "version": snap.version,
    }


def dumps(snap: Snapshot) -> str:
    """Serialize a snapshot to a stable, sorted-key JSON string (trailing newline)."""
    return (
        json.dumps(
            snapshot_to_dict(snap),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def save(snap: Snapshot, path: str | Path) -> None:
    """Write a snapshot to ``path`` as a stable lockfile."""
    Path(path).write_text(dumps(snap), encoding="utf-8")


def _validate_version(version: Any) -> int:
    """Validate the lockfile version, rejecting unknown future versions clearly."""
    if not isinstance(version, int):
        raise LockfileError(f"lockfile 'version' must be an integer, got {version!r}")
    if version > SUPPORTED_VERSION:
        raise LockfileError(
            f"lockfile version {version} is newer than supported version "
            f"{SUPPORTED_VERSION}; upgrade retrieval-diff to read it"
        )
    if version < 1:
        raise LockfileError(f"lockfile version {version} is invalid (must be >= 1)")
    return version


def loads(text: str) -> Snapshot:
    """Parse a lockfile JSON string into a :class:`Snapshot`.

    Raises:
        LockfileError: On malformed JSON, an unsupported version, or a schema
            violation. The message is actionable rather than a raw traceback.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LockfileError(f"lockfile is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LockfileError("lockfile root must be a JSON object")

    version = _validate_version(data.get("version"))

    label = data.get("created_label")
    if not isinstance(label, str) or not label:
        raise LockfileError("lockfile 'created_label' must be a non-empty string")

    k = data.get("k")
    if not isinstance(k, int) or k <= 0:
        raise LockfileError("lockfile 'k' must be a positive integer")

    fingerprint = _fingerprint_from_json(data.get("fingerprint") or {})

    raw_results = data.get("results")
    if not isinstance(raw_results, dict):
        raise LockfileError("lockfile 'results' must be a JSON object")

    results: dict[str, QueryResult] = {}
    for query, payload in raw_results.items():
        results[query] = _parse_query_result(query, payload)

    return Snapshot(
        version=version,
        created_label=label,
        fingerprint=fingerprint,
        k=k,
        results=results,
    )


def _parse_query_result(query: str, payload: Any) -> QueryResult:
    """Parse one query's recorded result block."""
    if not isinstance(payload, dict):
        raise LockfileError(f"results[{query!r}] must be a JSON object")
    raw_hits = payload.get("hits")
    if not isinstance(raw_hits, list):
        raise LockfileError(f"results[{query!r}].hits must be a list")
    hits: list[ScoredHit] = []
    for entry in raw_hits:
        if not isinstance(entry, dict):
            raise LockfileError(f"results[{query!r}] has a non-object hit")
        try:
            hits.append(
                ScoredHit(
                    id=str(entry["id"]),
                    score=float(entry["score"]),
                    rank=int(entry["rank"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LockfileError(f"results[{query!r}] has a malformed hit: {exc}") from exc
    _validate_ranks(query, hits)
    return QueryResult(query=query, hits=tuple(hits))


def _validate_ranks(query: str, hits: list[ScoredHit]) -> None:
    """Reject hand-edited ranks that are not a contiguous 0-based sequence.

    A snapshot's ranks must be exactly ``0..len(hits)-1`` in some order; a
    non-contiguous set (e.g. ``[0, 2, 4]`` from a hand-edit) or a duplicate would
    silently corrupt the ``rank_delta`` computed against this lock, so it is a
    hard error rather than a quietly accepted value.
    """
    ranks = [hit.rank for hit in hits]
    if sorted(ranks) != list(range(len(hits))):
        raise LockfileError(
            f"results[{query!r}] ranks must be a contiguous 0-based sequence "
            f"(0..{len(hits) - 1}); got {ranks}"
        )


def load(path: str | Path) -> Snapshot:
    """Load and validate a lockfile from ``path``."""
    p = Path(path)
    if not p.exists():
        raise LockfileError(f"lockfile not found: {p}")
    return loads(p.read_text(encoding="utf-8"))
