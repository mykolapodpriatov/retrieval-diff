"""Causal attribution of retrieval changes via held-fixed replay.

Given two fingerprints, a :class:`~retrieval_diff.retrievers.factory.RetrieverFactory`,
the query set, and an observed :class:`~retrieval_diff.types.SnapshotDiff`, this
module attributes each per-chunk change to a single config axis.

Method (held-fixed replay, §3.5): for each axis that **differs** between the two
fingerprints **and** is replayable, build a retriever with *only that axis*
changed to its new value (all others held at old). Re-snapshot, diff against the
old snapshot, and record which observed changes that single-axis replay
reproduces. A change is:

* ``confirmed`` -- reproduced by **exactly one** differing replayable axis.
* ``ambiguous`` -- reproduced by **two or more** axes independently.
* ``not_attributable`` -- reproduced by **none** (an interaction effect) or the
  responsible axis is not replayable / no factory was provided.

``None`` is a **legitimate, distinct config value**, not an "exclude me" marker.
``reranker=None`` means "no reranker" and ``alpha=None`` means "not hybrid";
those are real states, so a genuine ``None -> value`` swap that changed
retrieval is fully attributable. Differing axes are therefore computed by
**canonical comparison** (``None`` vs ``None`` does not differ; ``None`` vs a
value does), and an axis is a replay candidate iff it both differs and is in
:meth:`factory.replayable_axes`.

Attributability is decided by replayability and by whether the factory can
actually build the single-axis replay config -- never by a blanket ``None``
test. If the factory cannot build a configuration (e.g. it raises for a
``None`` or unknown ``embedding_model``), that replay reproduces nothing and the
change degrades to ``not_attributable``. Attribution is a best-effort
diagnostic, so **any** exception from the build+snapshot path is caught and
turned into ``not_attributable`` rather than crashing the caller.

Exactly one replay per differing replayable axis -- no combinatorial search.
"""

from __future__ import annotations

from collections.abc import Iterable

from retrieval_diff.diff import DEFAULT_SCORE_EPS, diff_snapshots
from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.retrievers.factory import RetrieverFactory
from retrieval_diff.snapshot import snapshot
from retrieval_diff.types import (
    AxisAttribution,
    ChangeKind,
    ChangeRef,
    Snapshot,
    SnapshotDiff,
)

#: ChangeKinds that represent an actual movement worth attributing.
_ATTRIBUTABLE_KINDS = frozenset({ChangeKind.ADDED, ChangeKind.REMOVED, ChangeKind.REORDERED})

#: Replay label used for the synthetic single-axis snapshots.
_REPLAY_LABEL = "attribution-replay"


def _build_config_with_axis(
    old_fp: ConfigFingerprint, new_fp: ConfigFingerprint, axis: str
) -> ConfigFingerprint:
    """Return ``old_fp`` with a single ``axis`` overridden to its new value."""
    overrides: dict[str, object] = {}
    overrides[axis] = getattr(new_fp, axis)
    return old_fp.model_copy(update=overrides)


def _observed_changes(diff: SnapshotDiff) -> list[ChangeRef]:
    """Enumerate every attributable per-chunk change in a diff, deterministically."""
    refs: list[ChangeRef] = []
    for query in sorted(diff.per_query):
        qd = diff.per_query[query]
        for cid in sorted(qd.kinds):
            for kind in sorted(qd.kinds[cid], key=lambda k: k.value):
                if kind in _ATTRIBUTABLE_KINDS:
                    refs.append(ChangeRef(query=query, chunk_id=cid, kind=kind))
    return refs


def _change_keys_of_diff(diff: SnapshotDiff) -> set[tuple[str, str, ChangeKind]]:
    """Return the set of ``(query, id, kind)`` attributable changes in a diff."""
    keys: set[tuple[str, str, ChangeKind]] = set()
    for query, qd in diff.per_query.items():
        for cid, kinds in qd.kinds.items():
            for kind in kinds:
                if kind in _ATTRIBUTABLE_KINDS:
                    keys.add((query, cid, kind))
    return keys


def attribute(
    old_fp: ConfigFingerprint,
    new_fp: ConfigFingerprint,
    observed: SnapshotDiff,
    *,
    factory: RetrieverFactory | None,
    queries: Iterable[str],
    k: int,
    corpus: object | None = None,
    score_eps: float = DEFAULT_SCORE_EPS,
    baseline: SnapshotDiff | None = None,
) -> list[AxisAttribution]:
    """Attribute each observed change to a single config axis.

    Args:
        old_fp: The baseline fingerprint.
        new_fp: The candidate fingerprint.
        observed: The observed old-vs-new diff whose changes are attributed.
        factory: A factory able to rebuild retrievers for replay; if ``None``,
            every change is ``not_attributable``.
        queries: The query set to replay (must match the diffed set).
        k: The snapshot K to replay at.
        corpus: Optional raw corpus for build-time axes.
        score_eps: Score tolerance for the replay diffs.
        baseline: Optional pre-built old snapshot replay (the retriever at
            ``old_fp``) used as the diff baseline. When omitted the factory
            builds it from ``old_fp``.

    Returns:
        One :class:`AxisAttribution` per attributable change, in deterministic
        ``(query, id, kind)`` order. The number of factory replays is at most the
        number of differing replayable axes (bounded; no combinatorial search).
        A misconfigured factory (one that raises while building) never crashes
        attribution -- the offending axis simply reproduces nothing and every
        otherwise-unexplained change is reported ``not_attributable``.
    """
    refs = _observed_changes(observed)

    if factory is None:
        note = "no factory provided; attribution unavailable"
        return [
            AxisAttribution(
                change_ref=ref,
                axis="",
                confidence="not_attributable",
                evidence={"reason": note},
            )
            for ref in refs
        ]

    query_list = list(queries)
    # ``differing_axes`` already compares canonically, so ``None`` is treated as
    # a distinct value: ``None`` vs ``None`` does not differ; ``None`` vs a value
    # does. A real ``None -> value`` swap (e.g. reranker None -> "cross-encoder")
    # is thus a genuine differing axis, never silently dropped. Attributability
    # is decided purely by replayability plus whether the factory can build the
    # single-axis replay config below -- not by any blanket ``None`` test.
    differing = old_fp.differing_axes(new_fp)
    replayable = set(factory.replayable_axes())
    replay_axes = [axis for axis in differing if axis in replayable]
    non_replayable = [axis for axis in differing if axis not in replay_axes]

    # Build the baseline (old config) once. A factory that cannot reconstruct the
    # baseline (e.g. it raises for a None or unknown embedding_model) must not
    # crash attribution: catch the failure and fall back to "nothing
    # reproduces", so every change is not_attributable rather than an exception.
    old_snapshot = None
    if replay_axes and refs:
        old_snapshot = _try_snapshot(factory, old_fp, corpus, query_list, k)
        if old_snapshot is None:
            replay_axes = []
            non_replayable = list(differing)

    # For each differing replayable axis, replay with only that axis changed and
    # record which observed changes it reproduces. A build that raises (e.g. an
    # unknown model id) reproduces nothing for that axis.
    axis_reproduced: dict[str, set[tuple[str, str, ChangeKind]]] = {}
    for axis in replay_axes:
        config = _build_config_with_axis(old_fp, new_fp, axis)
        replay_snapshot = _try_snapshot(factory, config, corpus, query_list, k)
        if replay_snapshot is None:
            continue
        assert old_snapshot is not None  # guaranteed when replay_axes non-empty
        replay_diff = diff_snapshots(old_snapshot, replay_snapshot, score_eps=score_eps)
        axis_reproduced[axis] = _change_keys_of_diff(replay_diff)

    results: list[AxisAttribution] = []
    for ref in refs:
        key = (ref.query, ref.chunk_id, ref.kind)
        explaining = sorted(axis for axis, changes in axis_reproduced.items() if key in changes)
        results.append(_verdict(ref, explaining, non_replayable, differing))
    return results


def _try_snapshot(
    factory: RetrieverFactory,
    config: ConfigFingerprint,
    corpus: object | None,
    query_list: list[str],
    k: int,
) -> Snapshot | None:
    """Build and snapshot ``config`` via the factory, or ``None`` if it cannot.

    Attribution is a best-effort diagnostic, not a correctness gate, so **any**
    exception from the build+snapshot path is swallowed and reported as ``None``
    -- a ``ValueError`` for a ``None``/unknown embedding model or a corpus-less
    build-time axis, but equally a custom factory's ``KeyError``/``RuntimeError``
    /``TypeError`` -- so that an un-buildable replay config degrades to
    ``not_attributable`` instead of crashing the caller. The guard wraps only the
    build+snapshot call (the narrowest reasonable scope), not the rest of
    attribution.
    """
    try:
        retriever = factory.build(config, corpus=corpus)
        return snapshot(retriever, query_list, k, label=_REPLAY_LABEL)
    except Exception:
        return None


def _verdict(
    ref: ChangeRef,
    explaining: list[str],
    non_replayable: list[str],
    differing: list[str],
) -> AxisAttribution:
    """Render the attribution verdict for a single change.

    Args:
        ref: The change being attributed.
        explaining: Replayable axes whose single-axis replay reproduced it.
        non_replayable: Differing axes that could not be replayed.
        differing: All differing axes (for evidence).

    Returns:
        The :class:`AxisAttribution` verdict.
    """
    evidence: dict[str, object] = {
        "differing_axes": list(differing),
        "explaining_axes": list(explaining),
        "non_replayable_axes": list(non_replayable),
    }
    if len(explaining) == 1:
        return AxisAttribution(
            change_ref=ref,
            axis=explaining[0],
            confidence="confirmed",
            evidence=evidence,
        )
    if len(explaining) >= 2:
        return AxisAttribution(
            change_ref=ref,
            axis="",
            confidence="ambiguous",
            evidence=evidence,
        )
    # No single replayable axis reproduced the change.
    reason = (
        "interaction effect: no single axis reproduces this change"
        if not non_replayable
        else (
            "responsible axis was not replayed "
            "(not replayable, or the factory could not build that single-axis config)"
        )
    )
    evidence["reason"] = reason
    return AxisAttribution(
        change_ref=ref,
        axis="",
        confidence="not_attributable",
        evidence=evidence,
    )


def attribution_index(
    attributions: Iterable[AxisAttribution],
) -> dict[tuple[str, str, ChangeKind], AxisAttribution]:
    """Index attributions by ``(query, id, kind)`` for convenient lookup."""
    return {(a.change_ref.query, a.change_ref.chunk_id, a.change_ref.kind): a for a in attributions}
