"""Tests for held-fixed-replay causal attribution.

These use a purpose-built deterministic retriever whose ranking is an explicit
function of the config axes, so each verdict (confirmed / ambiguous /
not_attributable) can be constructed exactly. A separate test exercises the real
in-memory hybrid factory to confirm the wiring works end to end.
"""

from __future__ import annotations

from collections.abc import Sequence

from rdiff_testkit import make_snapshot
from retrieval_diff.attribution import attribute, attribution_index
from retrieval_diff.diff import diff_snapshots
from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.retrievers.factory import BUILD_TIME_AXES, MemoryRetrieverFactory
from retrieval_diff.retrievers.memory import Corpus, HashingEmbedder, KeywordReranker
from retrieval_diff.snapshot import snapshot
from retrieval_diff.types import ChangeKind, ScoredHit, SnapshotDiff

QUERY = "q"


class _ScriptedRetriever:
    """A retriever whose hit list is a pure function of the config fingerprint.

    The mapping is supplied as ``{(embedding_model, reranker, alpha_key): [ids]}``
    keyed by the salient axes for a scenario, giving exact control over which
    axis change produces which ranking.
    """

    def __init__(self, ids: Sequence[str], fingerprint: ConfigFingerprint) -> None:
        self._ids = list(ids)
        self._fp = fingerprint

    def search(self, query: str, k: int) -> list[ScoredHit]:
        return [
            ScoredHit(id=cid, score=1.0 - 0.01 * rank, rank=rank)
            for rank, cid in enumerate(self._ids[:k])
        ]

    def fingerprint(self) -> ConfigFingerprint:
        return self._fp


class _ScriptedFactory:
    """Factory returning scripted retrievers per a rules table.

    ``rules`` maps a frozenset of "changed-from-old" axis names to the id order
    that configuration produces. This lets a test say: "if only embedding
    changed -> order X; if only reranker changed -> order Y; if both -> order Z".
    """

    def __init__(
        self,
        old_fp: ConfigFingerprint,
        rules: dict[frozenset[str], list[str]],
        replayable: set[str],
    ) -> None:
        self._old = old_fp
        self._rules = rules
        self._replayable = replayable
        self.build_calls: list[frozenset[str]] = []

    def replayable_axes(self) -> set[str]:
        return set(self._replayable)

    def build(self, config: ConfigFingerprint, corpus: object | None = None) -> _ScriptedRetriever:
        changed = frozenset(self._old.differing_axes(config))
        self.build_calls.append(changed)
        ids = self._rules[changed]
        return _ScriptedRetriever(ids, config)


def _two_axis_setup(
    rules: dict[frozenset[str], list[str]],
    *,
    replayable: set[str] | None = None,
) -> tuple[ConfigFingerprint, ConfigFingerprint, SnapshotDiff, _ScriptedFactory]:
    """Build old/new fingerprints differing on embedding_model + reranker."""
    old_fp = ConfigFingerprint(embedding_model="m1", reranker="r1", alpha=0.5)
    new_fp = ConfigFingerprint(embedding_model="m2", reranker="r2", alpha=0.5)
    old = make_snapshot(
        {QUERY: [(cid, 1.0 - 0.01 * i) for i, cid in enumerate(rules[frozenset()])]},
        k=4,
        fingerprint=old_fp,
    )
    new = make_snapshot(
        {
            QUERY: [
                (cid, 1.0 - 0.01 * i)
                for i, cid in enumerate(rules[frozenset({"embedding_model", "reranker"})])
            ]
        },
        k=4,
        fingerprint=new_fp,
    )
    diff = diff_snapshots(old, new)
    factory = _ScriptedFactory(
        old_fp,
        rules,
        replayable if replayable is not None else {"embedding_model", "reranker", "alpha"},
    )
    return old_fp, new_fp, diff, factory


def test_disjoint_changes_each_confirmed() -> None:
    """Each axis's replay explains a disjoint change -> both confirmed."""
    # old: [a, b, c]; embedding-only swaps a<->b; reranker-only swaps b/c order.
    rules = {
        frozenset(): ["a", "b", "c"],
        frozenset({"embedding_model"}): ["b", "a", "c"],  # moves a and b
        frozenset({"reranker"}): ["a", "c", "b"],  # moves b and c
        frozenset({"embedding_model", "reranker"}): ["b", "c", "a"],  # combined
    }
    old_fp, new_fp, diff, factory = _two_axis_setup(rules)
    attrs = attribute(old_fp, new_fp, diff, factory=factory, queries=[QUERY], k=4)
    index = attribution_index(attrs)

    # 'a' moves rank 0->2 (reordered). embedding-only moves a (0->1), combined
    # moves a (0->2). reranker-only leaves a at 0. So only... both differ on a's
    # final rank; the *change kind* reordered for 'a' is reproduced by embedding
    # (a is reordered there) but not reranker (a stays). -> confirmed embedding.
    a_key = (QUERY, "a", ChangeKind.REORDERED)
    assert index[a_key].confidence == "confirmed"
    assert index[a_key].axis == "embedding_model"

    # 'c' moves 2->1 (reordered). reranker-only reorders c; embedding-only leaves
    # c at rank 2. -> confirmed reranker.
    c_key = (QUERY, "c", ChangeKind.REORDERED)
    assert index[c_key].confidence == "confirmed"
    assert index[c_key].axis == "reranker"


def test_change_explained_by_both_is_ambiguous() -> None:
    """A change reproduced by two single-axis replays is ambiguous."""
    # Both embedding-only and reranker-only move 'a' from rank 0 to rank 1.
    rules = {
        frozenset(): ["a", "b"],
        frozenset({"embedding_model"}): ["b", "a"],
        frozenset({"reranker"}): ["b", "a"],
        frozenset({"embedding_model", "reranker"}): ["b", "a"],
    }
    old_fp, new_fp, diff, factory = _two_axis_setup(rules)
    attrs = attribute(old_fp, new_fp, diff, factory=factory, queries=[QUERY], k=4)
    index = attribution_index(attrs)
    a_key = (QUERY, "a", ChangeKind.REORDERED)
    assert index[a_key].confidence == "ambiguous"
    assert index[a_key].axis == ""
    assert sorted(index[a_key].evidence["explaining_axes"]) == [  # type: ignore[arg-type]
        "embedding_model",
        "reranker",
    ]


def test_interaction_effect_not_attributable() -> None:
    """A change no single axis reproduces (only the combination) is not attributable."""
    # Neither single-axis replay reorders anything; only the combination swaps.
    rules = {
        frozenset(): ["a", "b"],
        frozenset({"embedding_model"}): ["a", "b"],  # no change alone
        frozenset({"reranker"}): ["a", "b"],  # no change alone
        frozenset({"embedding_model", "reranker"}): ["b", "a"],  # only together
    }
    old_fp, new_fp, diff, factory = _two_axis_setup(rules)
    attrs = attribute(old_fp, new_fp, diff, factory=factory, queries=[QUERY], k=4)
    index = attribution_index(attrs)
    a_key = (QUERY, "a", ChangeKind.REORDERED)
    assert index[a_key].confidence == "not_attributable"
    assert "interaction" in index[a_key].evidence["reason"]  # type: ignore[operator]


def test_non_replayable_axis_is_not_attributable() -> None:
    """A differing axis the factory cannot replay yields not_attributable."""
    rules = {
        frozenset(): ["a", "b"],
        frozenset({"reranker"}): ["b", "a"],
        frozenset({"embedding_model", "reranker"}): ["b", "a"],
    }
    # Only 'reranker' is replayable; embedding differs but is not replayable.
    old_fp, new_fp, diff, factory = _two_axis_setup(rules, replayable={"reranker"})
    attrs = attribute(old_fp, new_fp, diff, factory=factory, queries=[QUERY], k=4)
    index = attribution_index(attrs)
    a_key = (QUERY, "a", ChangeKind.REORDERED)
    # reranker-only reproduces the swap -> confirmed reranker (embedding is moot).
    assert index[a_key].confidence == "confirmed"
    assert index[a_key].axis == "reranker"
    assert "embedding_model" in index[a_key].evidence["non_replayable_axes"]  # type: ignore[operator]


def test_no_factory_all_not_attributable() -> None:
    """Without a factory every change is not_attributable with a clear note."""
    old = make_snapshot({QUERY: [("a", 0.9), ("b", 0.8)]}, k=2)
    new = make_snapshot({QUERY: [("b", 0.9), ("a", 0.8)]}, k=2)
    diff = diff_snapshots(old, new)
    attrs = attribute(old.fingerprint, new.fingerprint, diff, factory=None, queries=[QUERY], k=2)
    assert attrs
    assert all(a.confidence == "not_attributable" for a in attrs)
    assert all("no factory" in a.evidence["reason"] for a in attrs)  # type: ignore[operator]


def test_none_to_value_axis_is_not_attributable_when_factory_cannot_build() -> None:
    """A None -> value swap on an axis the factory cannot build degrades gracefully.

    ``None`` is a legitimate, distinct value, so attributability is decided by
    whether the factory can actually build the single-axis replay -- not by a
    blanket None test. ``embedding_model`` goes None -> "m1"; building the
    baseline requires ``embedding_model=None``, which MemoryRetrieverFactory
    rejects (it can only reconstruct known model ids). The baseline build fails,
    so the change is not_attributable rather than crashing or being falsely
    confirmed.
    """
    corpus = Corpus.from_mapping(
        {"d1": "vector search", "d2": "lazy dog", "d3": "brown fox", "d4": "grey wolf"}
    )
    # embedding_model goes None -> "m1"; it is the only differing axis and it is
    # build-time replayable, but the factory cannot build embedding_model=None.
    old_fp = ConfigFingerprint(embedding_model=None, alpha=0.5)
    new_fp = ConfigFingerprint(embedding_model="m1", alpha=0.5)
    assert old_fp.differing_axes(new_fp) == ["embedding_model"]
    assert "embedding_model" in BUILD_TIME_AXES
    old = make_snapshot({QUERY: [("a", 0.9), ("b", 0.8)]}, k=4, fingerprint=old_fp)
    new = make_snapshot({QUERY: [("b", 0.9), ("a", 0.8)]}, k=4, fingerprint=new_fp)
    diff = diff_snapshots(old, new)
    embedder = HashingEmbedder(dim=16, model_id="m1")
    factory = MemoryRetrieverFactory(corpus, embedders={"m1": embedder})
    # embedding_model is in replayable_axes (corpus held), so it IS a replay
    # candidate -- the None test no longer excludes it. It degrades only because
    # the baseline build (embedding_model=None) raises.
    assert "embedding_model" in factory.replayable_axes()
    attrs = attribute(old_fp, new_fp, diff, factory=factory, queries=[QUERY], k=4)
    index = attribution_index(attrs)
    a_key = (QUERY, "a", ChangeKind.REORDERED)
    assert index[a_key].confidence == "not_attributable"
    assert index[a_key].axis == ""


def test_none_to_value_reranker_swap_is_confirmed() -> None:
    """A real reranker None -> value swap that changed retrieval IS attributed.

    ``reranker=None`` ("no reranker") is a legitimate baseline state, not an
    "exclude me" marker. The factory can build ``reranker=None`` (a search-time
    axis), so the single-axis replay reproduces the observed swap and the change
    is confirmed to the reranker axis -- the over-corrected blanket-None
    exclusion no longer suppresses it.
    """
    # reranker goes None -> "r2"; it is the only differing axis, and the factory
    # CAN reconstruct both reranker=None (baseline) and reranker="r2".
    old_fp = ConfigFingerprint(embedding_model="m1", reranker=None, alpha=0.5)
    new_fp = ConfigFingerprint(embedding_model="m1", reranker="r2", alpha=0.5)
    old = make_snapshot({QUERY: [("a", 0.9), ("b", 0.8)]}, k=4, fingerprint=old_fp)
    new = make_snapshot({QUERY: [("b", 0.9), ("a", 0.8)]}, k=4, fingerprint=new_fp)
    diff = diff_snapshots(old, new)
    # The reranker axis genuinely differs (None vs a value), per canonical compare.
    assert old_fp.differing_axes(new_fp) == ["reranker"]
    rules = {
        frozenset(): ["a", "b"],  # baseline (reranker=None) build succeeds
        frozenset({"reranker"}): ["b", "a"],  # single-axis replay reproduces the swap
    }
    factory = _ScriptedFactory(old_fp, rules, {"reranker"})
    attrs = attribute(old_fp, new_fp, diff, factory=factory, queries=[QUERY], k=4)
    index = attribution_index(attrs)
    a_key = (QUERY, "a", ChangeKind.REORDERED)
    # None -> value is a legitimate, buildable swap -> confirmed to reranker.
    assert index[a_key].confidence == "confirmed"
    assert index[a_key].axis == "reranker"
    # The single-axis reranker replay WAS attempted (no longer suppressed).
    assert frozenset({"reranker"}) in factory.build_calls


class _RaisingFactory:
    """A factory whose ``build`` always raises a non-ValueError exception.

    Exercises the broadened best-effort guard: a custom factory/snapshot path
    that raises ``RuntimeError`` (not just ``ValueError``) must degrade to
    not_attributable rather than crashing attribution.
    """

    def __init__(self, replayable: set[str]) -> None:
        self._replayable = replayable

    def replayable_axes(self) -> set[str]:
        return set(self._replayable)

    def build(self, config: ConfigFingerprint, corpus: object | None = None) -> _ScriptedRetriever:
        raise RuntimeError("boom: custom factory failed to build")


def test_factory_build_raising_runtimeerror_is_not_attributable() -> None:
    """A factory whose build raises RuntimeError degrades to not_attributable.

    The guard around build+snapshot catches any Exception (attribution is a
    best-effort diagnostic), so a non-ValueError failure on the very first
    (baseline) build does not propagate.
    """
    old_fp = ConfigFingerprint(embedding_model="m1", reranker="r1", alpha=0.5)
    new_fp = ConfigFingerprint(embedding_model="m1", reranker="r2", alpha=0.5)
    assert old_fp.differing_axes(new_fp) == ["reranker"]
    old = make_snapshot({QUERY: [("a", 0.9), ("b", 0.8)]}, k=2, fingerprint=old_fp)
    new = make_snapshot({QUERY: [("b", 0.9), ("a", 0.8)]}, k=2, fingerprint=new_fp)
    diff = diff_snapshots(old, new)
    factory = _RaisingFactory({"reranker"})
    # Must not raise despite factory.build raising RuntimeError for every config.
    attrs = attribute(old_fp, new_fp, diff, factory=factory, queries=[QUERY], k=2)
    assert attrs
    assert all(a.confidence == "not_attributable" for a in attrs)
    assert all(a.axis == "" for a in attrs)


def test_search_time_replay_with_none_embedding_does_not_raise() -> None:
    """A search-time replay whose factory cannot build (embedding_model=None).

    MemoryRetrieverFactory.build raises ValueError when embedding_model is None.
    With alpha as the only differing (replayable) axis, the baseline build fails;
    attribution must catch it and return not_attributable rather than crashing.
    """
    corpus = Corpus.from_mapping(
        {"d1": "vector search", "d2": "lazy dog", "d3": "brown fox", "d4": "grey wolf"}
    )
    # alpha differs (search-time, replayable); embedding_model is None in BOTH,
    # so the factory's build() raises ValueError for any config it is handed.
    old_fp = ConfigFingerprint(embedding_model=None, alpha=0.2)
    new_fp = ConfigFingerprint(embedding_model=None, alpha=0.8)
    assert old_fp.differing_axes(new_fp) == ["alpha"]
    old = make_snapshot({QUERY: [("a", 0.9), ("b", 0.8)]}, k=2, fingerprint=old_fp)
    new = make_snapshot({QUERY: [("b", 0.9), ("a", 0.8)]}, k=2, fingerprint=new_fp)
    diff = diff_snapshots(old, new)
    embedder = HashingEmbedder(dim=16, model_id="e1")
    factory = MemoryRetrieverFactory(corpus, embedders={"e1": embedder})
    # Must not raise despite factory.build(old_fp) hitting embedding_model=None.
    attrs = attribute(old_fp, new_fp, diff, factory=factory, queries=[QUERY], k=2)
    assert attrs
    assert all(a.confidence == "not_attributable" for a in attrs)


def test_replays_are_bounded_by_differing_replayable_axes() -> None:
    """The factory is built once per differing replayable axis, plus the baseline."""
    rules = {
        frozenset(): ["a", "b", "c"],
        frozenset({"embedding_model"}): ["b", "a", "c"],
        frozenset({"reranker"}): ["a", "c", "b"],
        frozenset({"embedding_model", "reranker"}): ["b", "c", "a"],
    }
    old_fp, new_fp, diff, factory = _two_axis_setup(rules)
    attribute(old_fp, new_fp, diff, factory=factory, queries=[QUERY], k=4)
    # 2 differing replayable axes -> 1 baseline build + 2 single-axis replays.
    single_axis_builds = [c for c in factory.build_calls if len(c) <= 1]
    assert len(single_axis_builds) <= 1 + 2  # baseline (empty) + 2 axes
    # Each single-axis replay changed exactly one axis from old.
    replays = [c for c in factory.build_calls if len(c) == 1]
    assert {frozenset({"embedding_model"}), frozenset({"reranker"})} == set(replays)


# --- end-to-end with the real hybrid factory --------------------------------


def test_build_time_axis_without_corpus_not_attributable() -> None:
    """An embedding swap with a corpus-less factory is not attributable."""
    corpus = Corpus.from_mapping(
        {
            "d1": "vector search over dense embeddings",
            "d2": "lazy dog sleeps in the sun",
            "d3": "the quick brown fox",
            "d4": "wolves and foxes roam the forest",
        }
    )
    # Different embedding dimensionality -> genuinely different vectors/rankings,
    # while the corpus (and thus index_content_hash) is identical, isolating the
    # embedding_model axis.
    e1 = HashingEmbedder(dim=16, model_id="e1")
    e2 = HashingEmbedder(dim=128, model_id="e2")

    from retrieval_diff.retrievers.memory import HybridRetriever

    query = "fox in the forest"
    old_r = HybridRetriever(corpus, e1, alpha=1.0)  # pure dense to surface the swap
    new_r = HybridRetriever(corpus, e2, alpha=1.0)
    old = snapshot(old_r, [query], 3, label="o")
    new = snapshot(new_r, [query], 3, label="n")
    diff = diff_snapshots(old, new)
    # Sanity: the embedding swap must actually move something.
    assert any(
        kinds != {ChangeKind.UNCHANGED}
        for qd in diff.per_query.values()
        for kinds in qd.kinds.values()
    )
    # Only the embedding_model axis differs (content hash unchanged).
    assert old.fingerprint.differing_axes(new.fingerprint) == ["embedding_model"]

    # Factory holds the embedders but NOT the corpus -> build-time axes unreplayable.
    factory = MemoryRetrieverFactory(corpus, embedders={"e1": e1, "e2": e2}, hold_corpus=False)
    assert not (BUILD_TIME_AXES & factory.replayable_axes())

    attrs = attribute(
        old.fingerprint,
        new.fingerprint,
        diff,
        factory=factory,
        queries=[query],
        k=3,
    )
    # embedding_model differs but is build-time with no corpus -> every observed
    # change must be not_attributable (responsible axis is not replayable).
    assert attrs
    assert all(a.confidence == "not_attributable" for a in attrs)
    assert all(
        "not replayable" in a.evidence["reason"]  # type: ignore[operator]
        for a in attrs
    )


def test_real_hybrid_factory_confirms_reranker_change() -> None:
    """An end-to-end reranker-only change is confirmed via the real factory."""
    corpus = Corpus.from_mapping(
        {
            "d1": "dense vector search retrieval",
            "d2": "the lazy dog and the brown fox",
            "d3": "vector embeddings power semantic search",
            "d4": "a wolf howls at the moon",
        }
    )
    embedder = HashingEmbedder(dim=32, model_id="e1")
    rr = KeywordReranker(keyword="vector", boost=5.0, reranker_id="rr-vec")
    # A distinct no-op reranker on the *old* side (keyword matches nothing in the
    # corpus) makes the reranker axis a value->value swap, not None->value. A
    # None-valued axis is unprovided and excluded from attribution by contract,
    # so both sides must carry a reranker for this swap to be attributable.
    rr_noop = KeywordReranker(keyword="zzzznotpresent", boost=5.0, reranker_id="rr-noop")

    from retrieval_diff.retrievers.memory import HybridRetriever

    old_r = HybridRetriever(corpus, embedder, alpha=0.5, reranker=rr_noop)
    new_r = HybridRetriever(corpus, embedder, alpha=0.5, reranker=rr)
    query = "the fox"
    old = snapshot(old_r, [query], 4, label="o")
    new = snapshot(new_r, [query], 4, label="n")
    diff = diff_snapshots(old, new)
    # Only the reranker axis differs (value->value), and it must actually move
    # something for this test to be meaningful.
    assert old.fingerprint.differing_axes(new.fingerprint) == ["reranker"]
    assert any(
        kinds != {ChangeKind.UNCHANGED}
        for qd in diff.per_query.values()
        for kinds in qd.kinds.values()
    )

    factory = MemoryRetrieverFactory(
        corpus, embedders={"e1": embedder}, rerankers={"rr-vec": rr, "rr-noop": rr_noop}
    )
    attrs = attribute(old.fingerprint, new.fingerprint, diff, factory=factory, queries=[query], k=4)
    # Only the reranker axis differs -> every reproduced change is confirmed to it.
    confirmed = [a for a in attrs if a.confidence == "confirmed"]
    assert confirmed
    assert all(a.axis == "reranker" for a in confirmed)


def test_real_hybrid_factory_confirms_none_to_value_reranker() -> None:
    """An end-to-end reranker None -> value swap is confirmed via the real factory.

    ``reranker=None`` ("no reranker") is a legitimate baseline the factory can
    rebuild, so adding a reranker that genuinely reorders results is attributed
    to the reranker axis -- not suppressed as "unprovided". This is the case the
    earlier blanket-None exclusion wrongly made impossible to attribute.
    """
    corpus = Corpus.from_mapping(
        {
            "d1": "dense vector search retrieval",
            "d2": "the lazy dog and the brown fox",
            "d3": "vector embeddings power semantic search",
            "d4": "a wolf howls at the moon",
        }
    )
    embedder = HashingEmbedder(dim=32, model_id="e1")
    rr = KeywordReranker(keyword="vector", boost=5.0, reranker_id="rr-vec")

    from retrieval_diff.retrievers.memory import HybridRetriever

    # Old side has NO reranker (reranker=None); new side adds the vector-boosting
    # reranker. The boost pulls the "vector" docs up, so the ranking moves.
    old_r = HybridRetriever(corpus, embedder, alpha=0.5, reranker=None)
    new_r = HybridRetriever(corpus, embedder, alpha=0.5, reranker=rr)
    query = "the fox"
    old = snapshot(old_r, [query], 4, label="o")
    new = snapshot(new_r, [query], 4, label="n")
    diff = diff_snapshots(old, new)
    # The reranker axis differs as None -> "rr-vec" (a genuine, distinct value).
    assert old.fingerprint.reranker is None
    assert old.fingerprint.differing_axes(new.fingerprint) == ["reranker"]
    # The swap must actually move something for this test to be meaningful.
    assert any(
        kinds != {ChangeKind.UNCHANGED}
        for qd in diff.per_query.values()
        for kinds in qd.kinds.values()
    )

    # The factory can rebuild reranker=None (baseline) and reranker="rr-vec".
    factory = MemoryRetrieverFactory(corpus, embedders={"e1": embedder}, rerankers={"rr-vec": rr})
    attrs = attribute(old.fingerprint, new.fingerprint, diff, factory=factory, queries=[query], k=4)
    # Only the reranker axis differs -> every reproduced change is confirmed to it,
    # and nothing degrades to not_attributable on account of the None baseline.
    confirmed = [a for a in attrs if a.confidence == "confirmed"]
    assert confirmed
    assert all(a.axis == "reranker" for a in confirmed)
    assert not any(a.confidence == "not_attributable" for a in attrs)
