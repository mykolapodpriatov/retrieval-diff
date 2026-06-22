"""GitHub Action support for retrieval-diff.

The composite action (``action.yml``) installs the package and runs
:mod:`retrieval_diff.action.entrypoint`, which executes ``retrieval-diff check``
and -- when a token is present -- posts the Markdown diff as a PR comment. The
comment step is skipped entirely without a token, so the action is safe to use
on forks and in token-less contexts.
"""

from __future__ import annotations
