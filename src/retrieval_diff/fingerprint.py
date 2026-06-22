"""Configuration fingerprints and the canonical deterministic digest.

A :class:`ConfigFingerprint` captures the identity of a retriever configuration
along the four axes that can change retrieval output between two commits:

* ``embedding_model`` -- the embedding model id (build-time axis).
* ``chunk_params`` -- chunking parameters (build-time axis).
* ``index_content_hash`` -- a hash of the indexed corpus (build-time axis).
* ``reranker`` -- the reranker id (search-time axis).
* ``alpha`` -- the hybrid fusion weight (search-time axis).

The digest is **canonical and deterministic** across Python versions and
platforms: every float (including floats nested inside ``chunk_params``) is
serialized with ``format(x, ".9g")`` and optional axes are serialized as JSON
``null`` when unset so that "absent" is distinct from any concrete value.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, field_validator

#: Float canonicalization precision used everywhere a float is hashed.
FLOAT_FORMAT: Final[str] = ".9g"

#: The four config axes, in canonical (sorted) order.
AXES: Final[tuple[str, ...]] = (
    "alpha",
    "chunk_params",
    "embedding_model",
    "index_content_hash",
    "reranker",
)

#: JSON primitive types permitted as ``chunk_params`` values.
_ChunkParamValue = str | int | float | bool | None


def canonicalize_float(value: float) -> str:
    """Return the canonical string form of a float for hashing.

    Uses ``format(value, ".9g")`` so that values such as ``0.3`` hash
    identically across platforms and Python builds.
    """
    return format(value, FLOAT_FORMAT)


def _canonicalize(value: Any) -> Any:
    """Recursively canonicalize a JSON-compatible value for digesting.

    Floats become their ``.9g`` string form. Booleans are preserved as
    booleans (``bool`` is a subclass of ``int`` and must not be treated as a
    float). Mappings and sequences are canonicalized element-wise with sorted
    keys so the result is order-independent.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return canonicalize_float(value)
    if isinstance(value, Mapping):
        return {str(k): _canonicalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    return value


def index_content_hash(chunks: Iterable[tuple[str, str]]) -> str:
    """Compute a deterministic SHA-256 over indexed ``(id, text)`` chunks.

    Chunks are sorted by their **UTF-8-byte-encoded id**. Each field (id, then
    text) is **length-prefixed** with its byte length as an 8-byte big-endian
    integer before its bytes, which makes every field and chunk boundary
    unambiguous *regardless of the bytes inside id or text*. A delimiter byte
    alone is insufficient: because an id may itself contain ``\\x1f``,
    ``(id="a\\x1f", text="")`` and ``(id="a", text="\\x1f")`` would feed identical
    bytes to a delimiter-framed hash and collide. Length framing cannot collide.
    The hash is stable regardless of corpus ordering or the platform's default
    string collation.

    Args:
        chunks: An iterable of ``(chunk_id, chunk_text)`` pairs.

    Returns:
        The hex SHA-256 digest of the canonicalized corpus content.
    """
    encoded: list[tuple[bytes, bytes]] = [
        (cid.encode("utf-8"), text.encode("utf-8")) for cid, text in chunks
    ]
    encoded.sort(key=lambda pair: pair[0])
    hasher = hashlib.sha256()
    for id_bytes, text_bytes in encoded:
        hasher.update(len(id_bytes).to_bytes(8, "big"))
        hasher.update(id_bytes)
        hasher.update(len(text_bytes).to_bytes(8, "big"))
        hasher.update(text_bytes)
    return hasher.hexdigest()


class ConfigFingerprint(BaseModel):
    """An immutable, hashable fingerprint of a retriever configuration.

    Axes set to ``None`` are treated as "unknown/unprovided" and are excluded
    from causal attribution (they cannot have "changed" in a meaningful way).
    All axes participate in :meth:`digest`, serialized as JSON ``null`` when
    unset, so an absent axis never collides with a concrete value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    embedding_model: str | None = None
    # Typed as ``Any`` so the JSON-primitive contract is enforced by the
    # validator below with a single, clear error message (rather than pydantic's
    # per-member union errors). Conceptually the value type is ``_ChunkParamValue``.
    chunk_params: dict[str, Any] = {}
    index_content_hash: str | None = None
    reranker: str | None = None
    alpha: float | None = None

    @field_validator("chunk_params")
    @classmethod
    def _validate_chunk_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Reject ``chunk_params`` values that are not JSON primitives."""
        for key, val in value.items():
            if val is not None and not isinstance(val, (str, int, float, bool)):
                raise ValueError(
                    f"chunk_params[{key!r}] must be a JSON primitive "
                    f"(str/int/float/bool/None), got {type(val).__name__}"
                )
        return value

    def canonical_dict(self) -> dict[str, Any]:
        """Return the canonical, digest-ready dict for this fingerprint.

        Every float is rendered via :func:`canonicalize_float`; every optional
        axis is present (as ``None``) so absence is encoded explicitly.
        """
        return {
            "alpha": (None if self.alpha is None else canonicalize_float(self.alpha)),
            "chunk_params": _canonicalize(self.chunk_params),
            "embedding_model": self.embedding_model,
            "index_content_hash": self.index_content_hash,
            "reranker": self.reranker,
        }

    def digest(self) -> str:
        """Return the canonical SHA-256 digest of this fingerprint.

        The digest is computed over
        ``json.dumps(canonical, sort_keys=True, separators=(",",":"), ensure_ascii=True)``
        and is stable across Python versions and platforms.
        """
        payload = json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def differing_axes(self, other: ConfigFingerprint) -> list[str]:
        """Return the sorted list of axes whose values differ from ``other``.

        Each axis is compared by the **canonical JSON serialization** of its
        canonicalized value, not by Python ``==``. This matters because Python
        equality conflates ``True == 1`` and ``False == 0`` and treats ``1 ==
        1.0``, so a ``chunk_params`` change like ``{"flag": True}`` ->
        ``{"flag": 1}`` (which yields a different :meth:`digest`) would otherwise
        report no differing axis. Floats are canonicalized to their ``.9g`` form
        first, so values equal at that precision are not reported as differing.
        """
        diffs: list[str] = []
        for axis in AXES:
            mine = getattr(self, axis)
            theirs = getattr(other, axis)
            if _canonical_axis_repr(mine) != _canonical_axis_repr(theirs):
                diffs.append(axis)
        return diffs


def _canonical_axis_repr(value: Any) -> str:
    """Return the canonical JSON string for an axis value, for exact comparison.

    Floats (``alpha`` and any float nested in ``chunk_params``) are rendered via
    :func:`canonicalize_float` by :func:`_canonicalize` first, so values equal at
    ``.9g`` precision compare equal. Serializing to JSON makes the comparison
    type-faithful: ``True`` and ``1`` produce distinct strings, so a bool/int
    swap is detected rather than masked by Python's ``True == 1``.
    """
    return json.dumps(_canonicalize(value), sort_keys=True, ensure_ascii=True)
