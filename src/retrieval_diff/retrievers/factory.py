"""Retriever factories for held-fixed replay during attribution.

A :class:`RetrieverFactory` knows how to reconstruct a retriever for an
arbitrary :class:`~retrieval_diff.fingerprint.ConfigFingerprint`, changing one
config axis at a time. Attribution uses this to ask "if only the embedding model
changed, would this chunk still have moved?".

Axes split into two kinds:

* **search-time** (``alpha``, ``reranker``) -- cheap; re-query/re-wrap the
  existing index.
* **build-time** (``embedding_model``, ``chunk_params``, ``index_content``) --
  need the **raw corpus** to re-index. If the corpus is unavailable, those axes
  are *not replayable* and the change is reported ``not_attributable``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from retrieval_diff.fingerprint import ConfigFingerprint
from retrieval_diff.retrievers import Retriever
from retrieval_diff.retrievers.memory import (
    Corpus,
    HashingEmbedder,
    HybridRetriever,
    KeywordReranker,
)

#: Axes that can be replayed without re-indexing (search-time).
SEARCH_TIME_AXES: frozenset[str] = frozenset({"alpha", "reranker"})

#: Axes that require the raw corpus to replay (build-time).
BUILD_TIME_AXES: frozenset[str] = frozenset(
    {"embedding_model", "chunk_params", "index_content_hash"}
)


@runtime_checkable
class RetrieverFactory(Protocol):
    """Reconstructs retrievers under arbitrary configs for held-fixed replay."""

    def replayable_axes(self) -> set[str]:
        """Return the set of axes this factory can reconstruct.

        Build-time axes are only included when the factory holds the raw corpus
        needed to re-index.
        """
        ...

    def build(self, config: ConfigFingerprint, corpus: object | None = None) -> Retriever:
        """Rebuild a retriever under ``config``.

        Args:
            config: The target fingerprint to reconstruct.
            corpus: Optional raw corpus; required to vary build-time axes.

        Returns:
            A retriever whose ``fingerprint()`` matches ``config`` on every axis
            the factory can control.

        Raises:
            ValueError: If a build-time axis is requested without a corpus.
        """
        ...


class MemoryRetrieverFactory:
    """A :class:`RetrieverFactory` over the in-memory hybrid retriever.

    The factory is constructed with the raw corpus, the embedders it can
    reconstruct (keyed by model id), and the rerankers it can reconstruct (keyed
    by reranker id). It can therefore replay every axis whose target value it
    knows how to materialize.
    """

    def __init__(
        self,
        corpus: Corpus,
        *,
        embedders: dict[str, HashingEmbedder],
        rerankers: dict[str, KeywordReranker] | None = None,
        hold_corpus: bool = True,
    ) -> None:
        """Initialize the factory.

        Args:
            corpus: The raw corpus used to re-index for build-time axes.
            embedders: Embedders by model id the factory can swap in.
            rerankers: Rerankers by reranker id the factory can swap in.
            hold_corpus: If ``False``, the factory behaves as if it lacks the
                corpus, making build-time axes non-replayable (used to test the
                "build-time axis without corpus" path).
        """
        self._corpus = corpus
        self._embedders = dict(embedders)
        self._rerankers = dict(rerankers or {})
        self._hold_corpus = hold_corpus

    def replayable_axes(self) -> set[str]:
        """Return the axes this factory can reconstruct.

        Search-time axes are always replayable. Build-time axes are replayable
        only when the factory holds the corpus.
        """
        axes: set[str] = set(SEARCH_TIME_AXES)
        if self._hold_corpus:
            axes |= set(BUILD_TIME_AXES)
        return axes

    def _resolve_corpus(self, corpus: object | None) -> Corpus:
        """Return the corpus to index with, preferring an explicit argument."""
        if isinstance(corpus, Corpus):
            return corpus
        if not self._hold_corpus:
            raise ValueError(
                "build-time axis requested but no corpus is available "
                "(factory was constructed with hold_corpus=False)"
            )
        return self._corpus

    def build(self, config: ConfigFingerprint, corpus: object | None = None) -> Retriever:
        """Build a hybrid retriever matching ``config``.

        Args:
            config: The target fingerprint.
            corpus: Optional explicit corpus override.

        Returns:
            A :class:`HybridRetriever` configured per ``config``.

        Raises:
            ValueError: If the embedding model or reranker in ``config`` is
                unknown to the factory, or a build-time axis is needed without a
                corpus.
        """
        active_corpus = self._resolve_corpus(corpus)

        model_id = config.embedding_model
        if model_id is None or model_id not in self._embedders:
            raise ValueError(f"factory cannot reconstruct embedding_model={model_id!r}")
        embedder = self._embedders[model_id]

        reranker: KeywordReranker | None = None
        if config.reranker is not None:
            if config.reranker not in self._rerankers:
                raise ValueError(f"factory cannot reconstruct reranker={config.reranker!r}")
            reranker = self._rerankers[config.reranker]

        alpha = 0.5 if config.alpha is None else config.alpha
        return HybridRetriever(
            active_corpus,
            embedder,
            alpha=alpha,
            reranker=reranker,
            chunk_params=config.chunk_params,
        )


__all__ = [
    "BUILD_TIME_AXES",
    "SEARCH_TIME_AXES",
    "MemoryRetrieverFactory",
    "RetrieverFactory",
]
