"""Adapters for external vector stores (behind optional extras).

Each adapter wraps an existing index/client as a
:class:`~retrieval_diff.retrievers.Retriever` so retrieval-diff can snapshot it.
The heavy client libraries are **import-guarded**: importing this package never
pulls in FAISS/Chroma/Qdrant/pgvector. Instantiating an adapter raises a clear
:class:`MissingDependencyError` listing the extra to install (e.g.
``pip install "retrieval-diff[faiss]"``).

Every adapter -- FAISS, Chroma, Qdrant and pgvector -- implements its real
query path.
"""

from __future__ import annotations


class MissingDependencyError(ImportError):
    """Raised when an adapter's optional backend dependency is not installed."""

    def __init__(self, backend: str, extra: str) -> None:
        self.backend = backend
        self.extra = extra
        super().__init__(
            f"the {backend!r} adapter requires the optional '{extra}' extra; "
            f'install it with: pip install "retrieval-diff[{extra}]"'
        )


def require(module: str, *, backend: str, extra: str) -> object:
    """Import ``module`` or raise a clear :class:`MissingDependencyError`.

    Args:
        module: The importable backend module name.
        backend: Human-readable backend name for the error.
        extra: The pip extra that provides the dependency.

    Returns:
        The imported module object.

    Raises:
        MissingDependencyError: If the module cannot be imported.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised via adapter ctors
        raise MissingDependencyError(backend, extra) from exc


__all__ = ["MissingDependencyError", "require"]
