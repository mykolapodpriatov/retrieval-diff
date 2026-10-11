"""BM25 setup when a chunk has no alphanumeric tokens."""

from retrieval_diff.retrievers.memory import BM25Retriever, Corpus


def test_punctuation_only_corpus_does_not_crash() -> None:
    corpus = Corpus((("a", "???"), ("b", "!!!")))
    retriever = BM25Retriever(corpus)
    assert retriever.raw_scores("???") == {}
    assert retriever.search("???", 2) == []


def test_punctuation_chunk_does_not_drop_a_real_hit() -> None:
    # Two documents make this term's idf zero in rank_bm25, so a third
    # document is what shows the real chunk still outranks the punctuation.
    corpus = Corpus(
        (
            ("empty", "???"),
            ("real", "alpha beta token"),
            ("other", "gamma delta"),
        )
    )
    retriever = BM25Retriever(corpus)
    scores = retriever.raw_scores("alpha")
    assert set(scores) == {"empty", "real", "other"}
    assert scores["real"] > scores["empty"]
    assert scores["real"] > scores["other"]
