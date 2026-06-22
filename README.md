# retrieval-diff

> Git-style diff for your RAG retriever — see exactly which chunks a query change, model swap, or reindex added, dropped, or reranked, and why.

![status](https://img.shields.io/badge/status-early%20development-orange) ![language](https://img.shields.io/badge/language-Python-blue) ![license](https://img.shields.io/badge/license-MIT-green)

Snapshots a retriever's output (top-K chunk ids, scores, ranks) for a set of queries into a versioned `retrieval.lock` file, then diffs two snapshots to show added/removed/reordered chunks per query, score deltas, and rank churn.

## Why

You can unit-test application code, but retrieval quality usually changes invisibly. This turns a retrieval regression into a reviewable diff that gates CI.

## Features

- Versioned, human-readable `retrieval.lock` of per-query top-K ids/scores/ranks
- Per-query chunk-level diff: added/removed/reordered, score deltas, rank churn
- Causal attribution by held-fixed replay (embedding vs chunking vs index vs reranker)
- pytest assertions + a GitHub Action that fails CI on retrieval regressions
- Adapters for FAISS/Chroma/Qdrant/pgvector; local or cloud embeddings

## How it works

Define a query set, snapshot it into a lockfile, and commit it. On a PR the tool re-runs retrieval, diffs against the committed lock, and attributes each change to a single variable by replaying with the others held fixed.

## Tech stack

- Python
- pytest
- FAISS / Chroma / Qdrant / pgvector
- sentence-transformers / Ollama
- OpenAI / Voyage / Gemini embeddings
- rich

## Status & roadmap

🚧 **Early development.** This repository is being built in the open; the scaffold and design are in place and the implementation is landing incrementally.

- [ ] Snapshot/lockfile format + per-query chunk diff
- [ ] Held-fixed causal attribution engine
- [ ] pytest assertions + GitHub Action CI gate
- [ ] HTML diff report; per-query drift budgets

## Installation

> Coming soon.

## License

[MIT](LICENSE) © 2026 Mykola Podpriatov
