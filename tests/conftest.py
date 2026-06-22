"""Shared pytest fixtures for the offline, deterministic test suite.

Module-level builders live in :mod:`rdiff_testkit` (uniquely named to avoid a
``tests`` namespace collision); this file only declares fixtures. Everything is
network-free: a deterministic hashing embedder plus in-memory retrievers.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure this directory is importable as a flat module path so ``import
# rdiff_testkit`` resolves here regardless of any unrelated ``tests`` package
# that may already be on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from rdiff_testkit import CORPUS_TEXT
from retrieval_diff.retrievers.memory import (
    Corpus,
    HashingEmbedder,
    HybridRetriever,
    KeywordReranker,
)


@pytest.fixture
def corpus() -> Corpus:
    """Return the shared in-memory corpus."""
    return Corpus.from_mapping(CORPUS_TEXT)


@pytest.fixture
def embedder() -> HashingEmbedder:
    """Return the default deterministic hashing embedder."""
    return HashingEmbedder(dim=48, model_id="embed-v1")


@pytest.fixture
def alt_embedder() -> HashingEmbedder:
    """Return an alternate embedder id (simulated model swap, same vectors)."""
    return HashingEmbedder(dim=48, model_id="embed-v2")


@pytest.fixture
def reranker() -> KeywordReranker:
    """Return a deterministic keyword reranker boosting 'vector'."""
    return KeywordReranker(keyword="vector", boost=0.6, reranker_id="rr-vector")


@pytest.fixture
def hybrid(corpus: Corpus, embedder: HashingEmbedder) -> HybridRetriever:
    """Return a baseline hybrid retriever over the shared corpus."""
    return HybridRetriever(corpus, embedder, alpha=0.5)
