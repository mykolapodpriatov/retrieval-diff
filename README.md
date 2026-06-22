# retrieval-diff

> Git-style diff for your RAG retriever — see exactly which chunks a query change, model swap, or reindex added, dropped, or reranked, and why.

![status](https://img.shields.io/badge/status-early%20development-orange) ![language](https://img.shields.io/badge/language-Python-blue) ![license](https://img.shields.io/badge/license-MIT-green)

Snapshot a retriever's output (top-K chunk ids, scores, ranks) for a set of queries into a versioned `retrieval.lock`, then diff two snapshots to show added/removed/reranked chunks per query, score deltas, and rank churn. Attribute each change to a single config axis by held-fixed replay, and gate CI on retrieval regressions.

## Why

You can unit-test application code, but retrieval quality usually changes invisibly when you bump an embedding model, change chunking, or reindex. retrieval-diff turns a retrieval regression into a reviewable diff and a CI gate.

## Features

- **Versioned, reviewable lockfile** — a `retrieval.lock` of per-query top-K ids/scores/ranks with sorted keys and a deterministic, cross-platform config fingerprint, so a change in *git* is minimal and human-readable.
- **Per-query chunk diff** — added / removed / reordered / score-changed per chunk, with rank and score deltas and a normalized rank-churn metric in `[0, 1]`.
- **Causal attribution** — held-fixed replay attributes each change to one axis (`embedding_model`, `chunk_params`, `index_content`, `reranker`, `alpha`), reporting `confirmed`, `ambiguous`, or `not_attributable`.
- **CI gate** — a pytest assertion and a GitHub Action fail the build when a golden chunk drops, churn exceeds a cap, or the query set silently drifts.
- **Deterministic & offline by default** — a built-in hashing embedder and in-memory retrievers make the whole pipeline reproducible with no downloads, no network, and no clock or RNG inside the library.

## Install

```bash
pip install retrieval-diff
```

Optional backend adapters (FAISS / Chroma / Qdrant / pgvector) are import-guarded behind extras:

```bash
pip install "retrieval-diff[faiss]"   # or [chroma], [qdrant], [pg]
```

## Quickstart

Wire your retriever through a small project hook (this mirrors how you already build it), declared in `pyproject.toml`:

```toml
[tool.retrieval_diff]
hook = "myproject.rdiff_hook"   # exposes build_context() -> ProjectContext
queries = "queries.jsonl"
k = 10

[tool.retrieval_diff.budget]
max_golden_rank_drop = 2
max_churn = 0.5
query_set_change_fails = true
```

```python
# myproject/rdiff_hook.py
from retrieval_diff.config import ProjectContext

def build_context(queries):
    retriever = build_my_retriever()           # your existing code
    return ProjectContext(
        retriever=retriever,
        queries=("first query", "second query"),
        goldens={"first query": {"doc-42"}},   # the chunks CI must protect
    )
```

Then drive it from the CLI:

```bash
# Snapshot the baseline into a lockfile and commit it (--label is required).
retrieval-diff snapshot --out retrieval.lock --label "$(git rev-parse HEAD)"

# On a PR: re-run retrieval, diff vs the committed lock, fail on a regression.
retrieval-diff check --lock retrieval.lock

# Inspect the change, or attribute it to a config axis.
retrieval-diff diff OLD.lock NEW.lock --format term
retrieval-diff attribute OLD.lock NEW.lock
```

Gate it from pytest:

```python
from retrieval_diff.budget import RegressionBudget
from retrieval_diff.pytest_plugin import assert_no_regression

def test_retrieval_does_not_regress():
    assert_no_regression(
        "retrieval.lock", build_my_retriever(), QUERIES,
        RegressionBudget(max_golden_rank_drop=2, max_churn=0.5),
        goldens={"first query": {"doc-42"}},
    )
```

## How it works

You define a query set, snapshot it into a lockfile, and commit it. On a PR the tool re-runs retrieval, diffs against the committed lock over the **intersection** of the query sets (reporting any query-set drift rather than hiding it), and gates CI on a regression budget. Attribution rebuilds intermediate retrievers with one config axis changed at a time and checks which axis reproduces each observed change.

A few deliberate design choices:

- **Deterministic digest.** The config fingerprint is hashed canonically — every float (including those inside `chunk_params`) is serialized with `format(x, ".9g")`, optional axes are encoded as `null` (so "absent" never collides with a value), and the content hash sorts chunks by their UTF-8-byte id. The digest is identical across Python versions and platforms.
- **Explicit churn metric.** `churn = min(1, Σ|Δrank| / (|common| · (K-1)))`, with documented fallbacks for `K ≤ 1` and an empty common set, so thresholds are meaningful and reproducible. Both snapshots must share the same `K`.
- **No hidden nondeterminism.** Snapshots carry a caller-supplied `created_label`; the library never reads the wall clock or an RNG.

## Example

A runnable, fully offline walkthrough (snapshot → diff → check → attribution, including a simulated regression) lives in [`examples/`](examples/):

```bash
cd examples && python run_demo.py
```

## GitHub Action

A composite action runs `retrieval-diff check`, sets the exit code, and optionally posts the diff as a PR comment (token-gated; the comment is skipped without a token):

```yaml
- uses: mykolapodpriatov/retrieval-diff/src/retrieval_diff/action@main
  with:
    lock: retrieval.lock
    github-token: ${{ secrets.GITHUB_TOKEN }}
```

## Status & roadmap

🚧 **Early development**, built in the open and landing incrementally.

- [x] Snapshot/lockfile format + per-query chunk diff
- [x] Held-fixed causal attribution engine
- [x] pytest assertion + GitHub Action CI gate
- [x] Rich terminal + Markdown reports
- [ ] Concrete FAISS / Chroma / Qdrant / pgvector adapters (interfaces are in place)

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy src
pytest
```

## License

[MIT](LICENSE) © 2026 Mykola Podpriatov
