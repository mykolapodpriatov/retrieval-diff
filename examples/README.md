# retrieval-diff example: snapshot -> diff -> check (fully offline)

This directory is a runnable, network-free walkthrough of the core loop. It uses
the built-in deterministic in-memory retriever, so it needs no models, no API
keys, and no vector database.

## Files

- `rdiff_hook.py` -- the project hook: builds the in-memory retriever, query set,
  golden chunks, and an attribution factory. This mirrors how you would wire your
  own retriever in a real project.
- `pyproject.toml` -- a minimal `[tool.retrieval_diff]` section pointing at the
  hook and declaring `k` and a regression budget.
- `queries.jsonl` -- the committed query set.
- `run_demo.py` -- runs snapshot -> diff -> check end to end and prints the
  output. It also simulates a regression (an embedding swap) so you can see the
  gate fail.

## Run it

From this directory:

```bash
python run_demo.py
```

Or drive the CLI directly:

```bash
# 1) Snapshot the baseline retriever into a lockfile (label is required).
retrieval-diff snapshot --out retrieval.lock --label baseline --config pyproject.toml

# 2) Diff two lockfiles (here, the baseline against itself -> no changes).
retrieval-diff diff retrieval.lock retrieval.lock --format term

# 3) Gate: re-run retrieval and fail on a regression beyond the budget.
retrieval-diff check --lock retrieval.lock --config pyproject.toml
```

`run_demo.py` is also exercised by the test suite, so it is guaranteed to stay
working.
