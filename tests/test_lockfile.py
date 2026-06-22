"""Tests for the lockfile roundtrip, byte-stability, and canonical digest."""

from __future__ import annotations

import pytest

from rdiff_testkit import make_snapshot
from retrieval_diff.fingerprint import ConfigFingerprint, index_content_hash
from retrieval_diff.lockfile import (
    SUPPORTED_VERSION,
    LockfileError,
    dumps,
    load,
    loads,
    save,
)
from retrieval_diff.snapshot import snapshot
from retrieval_diff.types import QueryResult, ScoredHit, Snapshot


def _fp(**overrides: object) -> ConfigFingerprint:
    base: dict[str, object] = {
        "embedding_model": "embed-v1",
        "chunk_params": {"size": 512, "overlap_ratio": 0.1},
        "index_content_hash": "abc123",
        "reranker": None,
        "alpha": 0.3,
    }
    base.update(overrides)
    return ConfigFingerprint(**base)  # type: ignore[arg-type]


# --- digest determinism -----------------------------------------------------


def test_digest_is_deterministic_across_constructions() -> None:
    """Two independently built identical fingerprints share a digest."""
    assert _fp().digest() == _fp().digest()


def test_digest_changes_when_any_axis_changes() -> None:
    """Changing any axis changes the digest."""
    base = _fp().digest()
    assert _fp(embedding_model="embed-v2").digest() != base
    assert _fp(reranker="rr").digest() != base
    assert _fp(alpha=0.31).digest() != base
    assert _fp(index_content_hash="zzz").digest() != base
    assert _fp(chunk_params={"size": 256, "overlap_ratio": 0.1}).digest() != base


def test_none_axis_is_distinct_from_set_axis() -> None:
    """An unset (None) axis must not collide with any concrete value."""
    none_reranker = _fp(reranker=None).digest()
    set_reranker = _fp(reranker="none").digest()
    assert none_reranker != set_reranker


def test_float_canonicalization_inside_chunk_params() -> None:
    """Floats nested in chunk_params are canonicalized (0.1 == 0.10)."""
    a = _fp(chunk_params={"overlap_ratio": 0.1})
    b = _fp(chunk_params={"overlap_ratio": 0.10})
    assert a.digest() == b.digest()


def test_alpha_canonicalization_stable_precision() -> None:
    """alpha is compared at .9g precision; tiny representational noise agrees."""
    a = _fp(alpha=0.1 + 0.2)  # 0.30000000000000004
    b = _fp(alpha=0.3)
    # .9g rounds both to 0.3 -> identical digest.
    assert a.digest() == b.digest()


def test_bool_in_chunk_params_not_treated_as_float() -> None:
    """Booleans in chunk_params survive canonicalization as booleans."""
    a = _fp(chunk_params={"flag": True})
    b = _fp(chunk_params={"flag": True})
    c = _fp(chunk_params={"flag": False})
    assert a.digest() == b.digest()
    assert a.digest() != c.digest()


def test_chunk_params_rejects_non_primitive() -> None:
    """Nested objects in chunk_params are rejected at construction."""
    with pytest.raises(ValueError, match="JSON primitive"):
        ConfigFingerprint(chunk_params={"bad": {"nested": 1}})  # type: ignore[dict-item]


# --- differing_axes type-faithful comparison --------------------------------


def test_differing_axes_distinguishes_bool_from_int_in_chunk_params() -> None:
    """A chunk_params change True -> 1 is a real differing axis (not masked).

    Python equality treats ``True == 1``, so a plain ``==`` comparison would
    report no differing axis even though the two configs hash differently.
    differing_axes compares canonical JSON, which is type-faithful.
    """
    a = ConfigFingerprint(chunk_params={"flag": True})
    b = ConfigFingerprint(chunk_params={"flag": 1})
    # The two genuinely differ (distinct digests), so the axis must be reported.
    assert a.digest() != b.digest()
    assert a.differing_axes(b) == ["chunk_params"]
    assert b.differing_axes(a) == ["chunk_params"]


def test_differing_axes_distinguishes_false_from_zero_in_chunk_params() -> None:
    """A chunk_params change False -> 0 is a real differing axis (not masked)."""
    a = ConfigFingerprint(chunk_params={"flag": False})
    b = ConfigFingerprint(chunk_params={"flag": 0})
    assert a.digest() != b.digest()
    assert a.differing_axes(b) == ["chunk_params"]


def test_differing_axes_ignores_float_precision_noise() -> None:
    """Values equal at .9g precision are not reported as differing."""
    a = ConfigFingerprint(alpha=0.1 + 0.2, chunk_params={"ratio": 0.1})
    b = ConfigFingerprint(alpha=0.3, chunk_params={"ratio": 0.10})
    assert a.differing_axes(b) == []


def test_differing_axes_true_equals_true_no_diff() -> None:
    """Identical bool chunk params are not reported as differing."""
    a = ConfigFingerprint(chunk_params={"flag": True})
    b = ConfigFingerprint(chunk_params={"flag": True})
    assert a.differing_axes(b) == []


# --- index_content_hash -----------------------------------------------------


def test_index_content_hash_is_order_independent() -> None:
    """Corpus content hash is invariant to chunk ordering."""
    a = index_content_hash([("id1", "alpha"), ("id2", "beta")])
    b = index_content_hash([("id2", "beta"), ("id1", "alpha")])
    assert a == b


def test_index_content_hash_uses_utf8_byte_sort_for_non_ascii_ids() -> None:
    """Non-ASCII chunk ids hash via UTF-8 byte sort (stable, deterministic)."""
    chunks = [("zürich", "text-a"), ("ångström", "text-b"), ("éclair", "text-c")]
    first = index_content_hash(chunks)
    shuffled = [chunks[2], chunks[0], chunks[1]]
    assert index_content_hash(shuffled) == first
    # Distinct content yields a distinct hash.
    assert index_content_hash([("zürich", "different"), *chunks[1:]]) != first


def test_index_content_hash_separates_id_and_text() -> None:
    """The unit separator prevents id/text boundary collisions."""
    a = index_content_hash([("ab", "c")])
    b = index_content_hash([("a", "bc")])
    assert a != b


def test_index_content_hash_id_text_boundary_unambiguous_with_separator_in_id() -> None:
    """An id that itself contains the unit separator cannot collide with text.

    A delimiter-only framing would feed identical bytes for ``(id="a\\x1f",
    text="")`` and ``(id="a", text="\\x1f")`` and collide. Length-prefix framing
    keeps the id/text boundary unambiguous regardless of the bytes inside either.
    """
    a = index_content_hash([("a\x1f", "")])
    b = index_content_hash([("a", "\x1f")])
    assert a != b


def test_index_content_hash_multichunk_boundary_unambiguous() -> None:
    """Adjacent chunks cannot be confused for a single longer concatenated chunk."""
    two = index_content_hash([("a", "x"), ("b", "y")])
    # One chunk whose bytes concatenate the two chunks' field bytes must differ.
    spliced = index_content_hash([("axb", "y")])
    assert two != spliced


# --- roundtrip & byte-stability ---------------------------------------------


def test_snapshot_lock_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A snapshot survives save/load unchanged."""
    snap = make_snapshot(
        {"q1": [("a", 0.9), ("b", 0.5)], "q2": [("c", 0.7)]},
        k=3,
        label="sha-roundtrip",
        fingerprint=_fp(),
    )
    path = tmp_path / "retrieval.lock"
    save(snap, path)
    loaded = load(path)
    assert loaded == snap


def test_resave_is_byte_identical(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Re-saving an unchanged snapshot produces byte-identical output."""
    snap = make_snapshot(
        {"q2": [("c", 0.7)], "q1": [("b", 0.5), ("a", 0.9)]},  # unsorted input
        k=2,
        label="sha-stable",
        fingerprint=_fp(),
    )
    first = dumps(snap)
    reloaded = loads(first)
    second = dumps(reloaded)
    assert first == second


def test_lock_keys_are_sorted(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Queries are emitted in sorted order for reviewable git diffs."""
    snap = make_snapshot(
        {"zebra": [("z", 0.1)], "apple": [("a", 0.2)]},
        k=1,
        label="sha-sorted",
    )
    text = dumps(snap)
    assert text.index('"apple"') < text.index('"zebra"')


def test_non_ascii_query_and_id_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Non-ASCII queries/ids roundtrip (ensure_ascii=False in the lock body)."""
    snap = Snapshot(
        version=1,
        created_label="sha-unicode",
        fingerprint=_fp(),
        k=1,
        results={
            "café résumé": QueryResult(
                query="café résumé",
                hits=(ScoredHit(id="naïve", score=0.5, rank=0),),
            )
        },
    )
    path = tmp_path / "u.lock"
    save(snap, path)
    assert load(path) == snap


# --- version validation -----------------------------------------------------


def test_future_version_raises_clear_error() -> None:
    """A future lockfile version is rejected with an actionable message."""
    snap = make_snapshot({"q": [("a", 0.5)]}, k=1, label="sha")
    text = dumps(snap).replace(
        f'"version": {SUPPORTED_VERSION}', f'"version": {SUPPORTED_VERSION + 99}'
    )
    with pytest.raises(LockfileError, match="newer than supported"):
        loads(text)


def test_invalid_version_zero_raises() -> None:
    """Version 0 is rejected."""
    snap = make_snapshot({"q": [("a", 0.5)]}, k=1, label="sha")
    text = dumps(snap).replace('"version": 1', '"version": 0')
    with pytest.raises(LockfileError, match="invalid"):
        loads(text)


def test_malformed_json_raises_lockfile_error() -> None:
    """Garbage input raises a LockfileError, not a raw JSONDecodeError."""
    with pytest.raises(LockfileError, match="not valid JSON"):
        loads("{not json")


def test_missing_label_raises() -> None:
    """An empty created_label is rejected on load."""
    snap = make_snapshot({"q": [("a", 0.5)]}, k=1, label="sha")
    text = dumps(snap).replace('"created_label": "sha"', '"created_label": ""')
    with pytest.raises(LockfileError, match="created_label"):
        loads(text)


def test_load_missing_file_raises() -> None:
    """Loading a nonexistent path raises a clear error."""
    with pytest.raises(LockfileError, match="not found"):
        load("/nonexistent/retrieval.lock")


def test_stored_digest_matches_fingerprint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The lockfile records the fingerprint digest for human review."""
    snap = make_snapshot({"q": [("a", 0.5)]}, k=1, label="sha", fingerprint=_fp())
    path = tmp_path / "d.lock"
    save(snap, path)
    body = path.read_text(encoding="utf-8")
    assert _fp().digest() in body


def test_snapshot_helper_requires_label() -> None:
    """snapshot() rejects an empty label (no clock fallback)."""
    from retrieval_diff.retrievers.memory import StaticRetriever

    retriever = StaticRetriever(hits_by_query={"q": ()})
    with pytest.raises(ValueError, match="label is required"):
        snapshot(retriever, ["q"], 3, label="")


# --- malformed-lock error paths ---------------------------------------------


def test_root_must_be_object() -> None:
    """A non-object JSON root is rejected."""
    with pytest.raises(LockfileError, match="must be a JSON object"):
        loads("[1, 2, 3]")


def test_version_must_be_integer() -> None:
    """A non-integer version is rejected."""
    with pytest.raises(LockfileError, match="must be an integer"):
        loads('{"version": "1", "created_label": "s", "k": 1, "results": {}}')


def test_k_must_be_positive_integer() -> None:
    """A non-positive K is rejected."""
    with pytest.raises(LockfileError, match="'k' must be a positive integer"):
        loads('{"version": 1, "created_label": "s", "k": 0, "results": {}}')


def test_results_must_be_object() -> None:
    """A non-object results block is rejected."""
    with pytest.raises(LockfileError, match="'results' must be a JSON object"):
        loads('{"version": 1, "created_label": "s", "k": 1, "results": []}')


def test_query_result_must_be_object() -> None:
    """A non-object per-query payload is rejected."""
    with pytest.raises(LockfileError, match="must be a JSON object"):
        loads('{"version": 1, "created_label": "s", "k": 1, "results": {"q": 5}}')


def test_hits_must_be_list() -> None:
    """A non-list hits field is rejected."""
    with pytest.raises(LockfileError, match="hits must be a list"):
        loads('{"version": 1, "created_label": "s", "k": 1, "results": {"q": {"hits": {}}}}')


def test_non_object_hit_rejected() -> None:
    """A non-object hit entry is rejected."""
    with pytest.raises(LockfileError, match="non-object hit"):
        loads('{"version": 1, "created_label": "s", "k": 1, "results": {"q": {"hits": [42]}}}')


def test_malformed_hit_missing_field_rejected() -> None:
    """A hit missing the score field is rejected with a clear message."""
    with pytest.raises(LockfileError, match="malformed hit"):
        loads(
            '{"version": 1, "created_label": "s", "k": 1, '
            '"results": {"q": {"hits": [{"id": "a", "rank": 0}]}}}'
        )


def test_negative_rank_in_hit_rejected() -> None:
    """A negative rank violates the ScoredHit constraint and is reported."""
    with pytest.raises(LockfileError, match="malformed hit"):
        loads(
            '{"version": 1, "created_label": "s", "k": 1, '
            '"results": {"q": {"hits": [{"id": "a", "rank": -1, "score": 0.5}]}}}'
        )


def test_non_contiguous_ranks_rejected() -> None:
    """Hand-edited non-contiguous ranks (e.g. 0,2,4) are rejected.

    Non-contiguous ranks load silently otherwise and corrupt rank_delta against
    this lock; the loader requires ranks to be exactly 0..n-1 in some order.
    """
    text = (
        '{"version": 1, "created_label": "s", "k": 3, "results": {"q": {"hits": ['
        '{"id": "a", "rank": 0, "score": 0.9}, '
        '{"id": "b", "rank": 2, "score": 0.8}, '
        '{"id": "c", "rank": 4, "score": 0.7}]}}}'
    )
    with pytest.raises(LockfileError, match="contiguous 0-based sequence"):
        loads(text)


def test_duplicate_ranks_rejected() -> None:
    """Duplicate ranks (not a 0-based permutation) are rejected."""
    text = (
        '{"version": 1, "created_label": "s", "k": 2, "results": {"q": {"hits": ['
        '{"id": "a", "rank": 0, "score": 0.9}, '
        '{"id": "b", "rank": 0, "score": 0.8}]}}}'
    )
    with pytest.raises(LockfileError, match="contiguous 0-based sequence"):
        loads(text)


def test_permuted_contiguous_ranks_accepted() -> None:
    """Ranks that are a 0-based permutation (1,0) load fine (order-independent)."""
    text = (
        '{"version": 1, "created_label": "s", "k": 2, "results": {"q": {"hits": ['
        '{"id": "a", "rank": 1, "score": 0.9}, '
        '{"id": "b", "rank": 0, "score": 0.8}]}}}'
    )
    snap = loads(text)
    assert {h.rank for h in snap.results["q"].hits} == {0, 1}


def test_missing_fingerprint_defaults_to_empty() -> None:
    """A lock with no fingerprint block loads with an empty fingerprint."""
    snap = loads('{"version": 1, "created_label": "s", "k": 1, "results": {}}')
    assert snap.fingerprint == ConfigFingerprint()


def test_invalid_fingerprint_chunk_params_rejected() -> None:
    """A fingerprint with a nested-object chunk param is rejected on load."""
    text = (
        '{"version": 1, "created_label": "s", "k": 1, "results": {}, '
        '"fingerprint": {"chunk_params": {"bad": {"nested": 1}}}}'
    )
    with pytest.raises(LockfileError, match="invalid fingerprint"):
        loads(text)


def test_snapshot_to_dict_is_round_trippable_via_loads() -> None:
    """snapshot_to_dict output parses back through loads to an equal snapshot."""
    import json as _json

    from retrieval_diff.lockfile import snapshot_to_dict

    snap = make_snapshot({"q": [("a", 0.5)]}, k=2, label="rt", fingerprint=_fp())
    text = _json.dumps(snapshot_to_dict(snap))
    assert loads(text) == snap
