"""Embedder abstraction and registry for pluggable text embedding models.

Usage::

    from patent.dataset.embedders import get_embedder

    embedder = get_embedder("sentence-transformers:all-MiniLM-L6-v2")
    vectors = embedder.encode(["hello world", "foo bar"])
    print(embedder.embedding_dim)  # 384
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class Embedder(ABC):
    """Abstract text embedding model.

    Subclasses must implement :meth:`encode` and expose
    :attr:`embedding_dim`.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Base initialiser — subclasses override with their own kwargs."""

    @abstractmethod
    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Encode *texts* into a ``(n, embedding_dim)`` float32 array."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Output dimensionality."""
        ...

    def stop_pool(self) -> None:
        """Release resources (no-op by default; override in subclasses)."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.embedding_dim})"


# ---------------------------------------------------------------------------
# Sentence-Transformers implementation
# ---------------------------------------------------------------------------


@dataclass
class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by a `sentence-transformers` model.

    Parameters
    ----------
    model_name : str
        Any model name accepted by ``SentenceTransformer``, e.g.
        ``"all-MiniLM-L6-v2"``, ``"all-mpnet-base-v2"``.
    device : str | None
        Torch device (``"cpu"``, ``"cuda"``, …).  ``None`` means auto.
    """

    model_name: str = "all-MiniLM-L6-v2"
    device: str | None = None
    _pool: Any = None  # opaque handle from sentence-transformers

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._dim: int = self._model.get_sentence_embedding_dimension() or -1
        self._pool = self._model.start_multi_process_pool()

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        return self._model.encode(
            texts,
            pool=self._pool,
            show_progress_bar=show_progress,
        )

    def stop_pool(self) -> None:
        """Release the multi-process pool (called after all chunks)."""
        if self._pool is not None:
            self._model.stop_multi_process_pool(self._pool)
            self._pool = None


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, type[Embedder]] = {
    "sentence-transformers": SentenceTransformerEmbedder,
}


def register_embedder(protocol: str, cls: type[Embedder]) -> None:
    """Register a new embedder backend."""
    _REGISTRY[protocol] = cls


def get_embedder(spec: str, **kwargs) -> Embedder:
    """Create an embedder from a spec string.

    Format: ``"<protocol>:<model>"``

    Examples
    --------
    >>> get_embedder("sentence-transformers:all-MiniLM-L6-v2")
    >>> get_embedder("sentence-transformers:all-mpnet-base-v2", device="cuda")
    """
    if ":" not in spec:
        raise ValueError(
            f"Invalid embedder spec {spec!r}. Expected '<protocol>:<model>' "
            f"(e.g. 'sentence-transformers:all-MiniLM-L6-v2')"
        )
    protocol, _, model = spec.partition(":")
    cls = _REGISTRY.get(protocol)
    if cls is None:
        raise ValueError(f"Unknown embedding protocol {protocol!r}. Available: {list(_REGISTRY)}")
    return cls(model_name=model, **kwargs)
