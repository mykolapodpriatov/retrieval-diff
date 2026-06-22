"""Deterministic, offline in-memory retrievers for tests and demos.

These retrievers require no network and no model downloads. A
:class:`HashingEmbedder` maps text to a fixed-dimension vector using a seeded,
per-token hash, so embeddings are byte-reproducible across platforms and Python
versions. Dense, BM25, and hybrid retrievers are provided, plus an optional
deterministic reranker.

The hybrid retriever fuses normalized dense and BM25 scores with a configurable
``alpha`` and exposes every config axis through its fingerprint, which makes it
the workhorse for the attribution tests.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from retrieval_diff.fingerprint import ConfigFingerprint, index_content_hash
from retrieval_diff.types import ScoredHit

#: Token pattern for the deterministic tokenizer (lowercase alphanumerics).
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize ``text`` into lowercase alphanumeric tokens, deterministically."""
    return _TOKEN_RE.findall(text.lower())


def _token_seed(token: str) -> int:
    """Return a stable 64-bit seed for a token via SHA-256.

    Using a cryptographic hash (rather than Python's salted ``hash``) keeps the
    embedding reproducible across processes and platforms.
    """
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class HashingEmbedder:
    """A deterministic bag-of-tokens hashing embedder.

    Each token contributes a fixed pseudo-random unit-ish vector (seeded by the
    token's SHA-256) summed into the document vector, which is then L2
    normalized. No training, no network, fully reproducible.
    """

    def __init__(self, dim: int = 64, model_id: str = "hashing-embedder-v1") -> None:
        """Initialize the embedder.

        Args:
            dim: Embedding dimensionality.
            model_id: An identifier surfaced in the fingerprint; changing it
                signals a (simulated) model swap without changing the vectors.
        """
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.model_id = model_id

    def _token_vector(self, token: str) -> np.ndarray:
        """Return the deterministic vector for a single token."""
        rng = np.random.default_rng(_token_seed(token))
        return rng.standard_normal(self.dim)

    def embed(self, text: str) -> np.ndarray:
        """Return the L2-normalized embedding for ``text``."""
        tokens = tokenize(text)
        vector = np.zeros(self.dim, dtype=np.float64)
        for token in tokens:
            vector += self._token_vector(token)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm


def _rank_hits(scores: Mapping[str, float], k: int) -> list[ScoredHit]:
    """Convert a ``{id: score}`` map into the rank-ordered top-``k`` hits.

    Ties are broken by ascending chunk id so the ordering is deterministic.
    """
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        ScoredHit(id=cid, score=float(score), rank=rank)
        for rank, (cid, score) in enumerate(ordered[:k])
    ]


def _minmax_normalize(scores: Mapping[str, float]) -> dict[str, float]:
    """Min-max normalize scores into ``[0, 1]``; a flat map becomes all zeros."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if math.isclose(hi, lo):
        return dict.fromkeys(scores, 0.0)
    span = hi - lo
    return {cid: (score - lo) / span for cid, score in scores.items()}


@dataclass(frozen=True)
class Corpus:
    """An ordered collection of ``(id, text)`` chunks."""

    chunks: tuple[tuple[str, str], ...]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> Corpus:
        """Build a corpus from an ``{id: text}`` mapping (sorted by id)."""
        return cls(tuple(sorted(mapping.items())))

    def ids(self) -> list[str]:
        """Return chunk ids in stored order."""
        return [cid for cid, _ in self.chunks]

    def content_hash(self) -> str:
        """Return the deterministic content hash of this corpus."""
        return index_content_hash(self.chunks)


class DenseRetriever:
    """A deterministic dense retriever over a :class:`Corpus`.

    Scores are cosine similarities between the query and chunk embeddings.
    """

    def __init__(self, corpus: Corpus, embedder: HashingEmbedder) -> None:
        self.corpus = corpus
        self.embedder = embedder
        self._matrix = (
            np.vstack([embedder.embed(text) for _, text in corpus.chunks])
            if corpus.chunks
            else np.zeros((0, embedder.dim))
        )
        self._ids = corpus.ids()

    def raw_scores(self, query: str) -> dict[str, float]:
        """Return ``{id: cosine_similarity}`` for every chunk."""
        if not self._ids:
            return {}
        qv = self.embedder.embed(query)
        sims = self._matrix @ qv
        return {cid: float(sims[i]) for i, cid in enumerate(self._ids)}

    def search(self, query: str, k: int) -> list[ScoredHit]:
        """Return the rank-ordered top-``k`` dense hits for ``query``."""
        return _rank_hits(self.raw_scores(query), k)

    def fingerprint(self) -> ConfigFingerprint:
        """Return the dense retriever's fingerprint."""
        return ConfigFingerprint(
            embedding_model=self.embedder.model_id,
            chunk_params={},
            index_content_hash=self.corpus.content_hash(),
            reranker=None,
            alpha=None,
        )


class BM25Retriever:
    """A deterministic BM25 retriever over a :class:`Corpus`."""

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self._ids = corpus.ids()
        tokenized = [tokenize(text) for _, text in corpus.chunks]
        # rank_bm25 requires a non-empty corpus; guard the empty case.
        self._bm25: BM25Okapi | None = BM25Okapi(tokenized) if tokenized else None

    def raw_scores(self, query: str) -> dict[str, float]:
        """Return ``{id: bm25_score}`` for every chunk."""
        if self._bm25 is None:
            return {}
        scores = self._bm25.get_scores(tokenize(query))
        return {cid: float(scores[i]) for i, cid in enumerate(self._ids)}

    def search(self, query: str, k: int) -> list[ScoredHit]:
        """Return the rank-ordered top-``k`` BM25 hits for ``query``."""
        return _rank_hits(self.raw_scores(query), k)

    def fingerprint(self) -> ConfigFingerprint:
        """Return the BM25 retriever's fingerprint."""
        return ConfigFingerprint(
            embedding_model="bm25",
            chunk_params={},
            index_content_hash=self.corpus.content_hash(),
            reranker=None,
            alpha=None,
        )


@dataclass(frozen=True)
class KeywordReranker:
    """A deterministic reranker that boosts chunks containing a keyword.

    The reranker adds ``boost`` to the fused score of any chunk whose text
    contains ``keyword`` (case-insensitive). It is intentionally simple so its
    effect on ranking is easy to reason about in attribution tests.
    """

    keyword: str
    boost: float = 0.5
    reranker_id: str = "keyword-reranker-v1"

    def apply(self, corpus: Corpus, scores: Mapping[str, float]) -> dict[str, float]:
        """Return rescored ``{id: score}`` with the keyword boost applied."""
        text_by_id = dict(corpus.chunks)
        needle = self.keyword.lower()
        out: dict[str, float] = {}
        for cid, score in scores.items():
            text = text_by_id.get(cid, "")
            bonus = self.boost if needle in text.lower() else 0.0
            out[cid] = score + bonus
        return out


class HybridRetriever:
    """A deterministic hybrid retriever fusing dense and BM25 scores.

    Final score = ``alpha * dense_norm + (1 - alpha) * bm25_norm`` after min-max
    normalization, with an optional keyword reranker applied last. Every config
    axis (embedding model, chunk params, content hash, reranker, alpha) is
    reflected in the fingerprint.
    """

    def __init__(
        self,
        corpus: Corpus,
        embedder: HashingEmbedder,
        *,
        alpha: float = 0.5,
        reranker: KeywordReranker | None = None,
        chunk_params: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.corpus = corpus
        self.embedder = embedder
        self.alpha = alpha
        self.reranker = reranker
        self.chunk_params: dict[str, str | int | float | bool | None] = (
            dict(chunk_params) if chunk_params else {}
        )
        self._dense = DenseRetriever(corpus, embedder)
        self._bm25 = BM25Retriever(corpus)

    def raw_scores(self, query: str) -> dict[str, float]:
        """Return the fused (and optionally reranked) ``{id: score}`` map."""
        dense_norm = _minmax_normalize(self._dense.raw_scores(query))
        bm25_norm = _minmax_normalize(self._bm25.raw_scores(query))
        ids = self.corpus.ids()
        fused = {
            cid: self.alpha * dense_norm.get(cid, 0.0)
            + (1.0 - self.alpha) * bm25_norm.get(cid, 0.0)
            for cid in ids
        }
        if self.reranker is not None:
            fused = self.reranker.apply(self.corpus, fused)
        return fused

    def search(self, query: str, k: int) -> list[ScoredHit]:
        """Return the rank-ordered top-``k`` hybrid hits for ``query``."""
        return _rank_hits(self.raw_scores(query), k)

    def fingerprint(self) -> ConfigFingerprint:
        """Return the hybrid retriever's fingerprint across all axes."""
        return ConfigFingerprint(
            embedding_model=self.embedder.model_id,
            chunk_params=dict(self.chunk_params),
            index_content_hash=self.corpus.content_hash(),
            reranker=(None if self.reranker is None else self.reranker.reranker_id),
            alpha=self.alpha,
        )


@dataclass
class StaticRetriever:
    """A retriever that replays a fixed, pre-recorded set of hits.

    Useful for diff/budget tests that need precise control over scores and ranks
    without modelling an index. ``hits_by_query`` is consumed verbatim (already
    rank-ordered); ``search`` truncates to ``k``.
    """

    hits_by_query: Mapping[str, Sequence[ScoredHit]]
    config: ConfigFingerprint = field(default_factory=ConfigFingerprint)

    def search(self, query: str, k: int) -> list[ScoredHit]:
        """Return up to ``k`` pre-recorded hits for ``query``."""
        return list(self.hits_by_query.get(query, ()))[:k]

    def fingerprint(self) -> ConfigFingerprint:
        """Return the static retriever's fingerprint."""
        return self.config


__all__ = [
    "BM25Retriever",
    "Corpus",
    "DenseRetriever",
    "HashingEmbedder",
    "HybridRetriever",
    "KeywordReranker",
    "StaticRetriever",
    "tokenize",
]
