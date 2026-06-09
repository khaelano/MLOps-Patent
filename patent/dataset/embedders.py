"""Embedder abstraction and registry for pluggable text embedding models.

Usage::

    from patent.dataset.embedders import get_embedder

    # Candle backend (any HuggingFace model)
    embedder = get_embedder("embed-anything:MongoDB/mdbr-leaf-mt")

    # ONNX quantized backend (5× faster, pre-configured models)
    embedder = get_embedder("embed-anything-onnx:AllMiniLML6V2Q")

    vectors = embedder.encode(["hello world", "foo bar"])
    print(embedder.embedding_dim)  # 384

Environment variables
---------------------
``HF_HOME``
    HuggingFace cache directory (set by ``patent.config`` to
    ``<project>/.hf_cache`` — overridable in CI).
``HF_TOKEN``
    HuggingFace API token for gated / private models (set via
    ``.env`` or CI secrets).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Any

from loguru import logger
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
# Shared encoding helper (progress logging)
# ---------------------------------------------------------------------------


def _encode_batched(
    model: Any,
    texts: list[str],
    config: Any,
    *,
    log_batch: int = 10_000,
    model_label: str = "embed",
) -> np.ndarray:
    """Encode *texts* in sub-batches with INFO-level progress logging.

    EmbedAnything's ``embed_query`` processes all texts in one call with
    internal batching, but gives no visibility into progress.  This helper
    splits the work into smaller chunks so the CLI shows per-batch timing.

    Parameters
    ----------
    model : EmbeddingModel
    texts : list[str]
    config : TextEmbedConfig
    log_batch : int
        Number of texts per progress-log line (default 10 000).
    model_label : str
        Human label for the log prefix.
    """
    total = len(texts)
    if total <= log_batch:
        logger.info("[{}] encoding {} texts", model_label, total)
        results = model.embed_query(texts, config=config)
        return np.array([r.embedding for r in results], dtype=np.float32)

    n_batches = (total + log_batch - 1) // log_batch
    logger.info(
        "[{}] encoding {} texts in {} batch(es) of ≤{}",
        model_label,
        total,
        n_batches,
        log_batch,
    )
    all_embeddings: list[np.ndarray] = []

    for bi in range(n_batches):
        start = bi * log_batch
        end = min(start + log_batch, total)
        chunk = texts[start:end]

        results = model.embed_query(chunk, config=config)
        arr = np.array([r.embedding for r in results], dtype=np.float32)
        all_embeddings.append(arr)

        pct = end / total * 100
        logger.info(
            "[{}] batch {}/{}  rows {}-{} ({})  {:.0f}%",
            model_label,
            bi + 1,
            n_batches,
            start,
            end,
            end - start,
            pct,
        )

    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# EmbedAnything — Candle backend (any HuggingFace model)
# ---------------------------------------------------------------------------


@dataclass
class EmbedAnythingEmbedder(Embedder):
    """Embedder backed by EmbedAnything's Candle inference engine.

    Parameters
    ----------
    model_name : str
        Any HuggingFace model ID supported by Candle, e.g.
        ``"MongoDB/mdbr-leaf-mt"``, ``"jinaai/jina-embeddings-v2-small-en"``.
    batch_size : int
        Internal batch size for the inference pipeline (default 32).
    dtype : str
        Quantization dtype: ``"INT8"`` (~40% faster), ``"F16"``, ``"BF16"``,
        or empty string for FP32 (default).
    """

    model_name: str = "MongoDB/mdbr-leaf-mt"
    batch_size: int = 32
    dtype: str = "INT8"

    def __post_init__(self) -> None:
        from embed_anything import Dtype, EmbeddingModel, TextEmbedConfig

        from patent.config import HF_TOKEN

        quant_dtype: Any = None
        if self.dtype:
            quant_dtype = getattr(Dtype, self.dtype)
        self._model = EmbeddingModel.from_pretrained_hf(
            model_id=self.model_name,
            dtype=quant_dtype,
            token=HF_TOKEN,
        )
        self._config = TextEmbedConfig(chunk_size=256, batch_size=self.batch_size)

        dummy = self._model.embed_query(["dim"], config=self._config)
        self._dim: int = len(dummy[0].embedding)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        return _encode_batched(
            self._model,
            texts,
            self._config,
            log_batch=10_000,
            model_label=f"Candle/{self.model_name}",
        )

    def stop_pool(self) -> None:
        pass


# ---------------------------------------------------------------------------
# EmbedAnything — ONNX quantized backend (pre-configured models, fast)
# ---------------------------------------------------------------------------


@dataclass
class EmbedAnythingONNXEmbedder(Embedder):
    """Embedder backed by EmbedAnything's ONNX runtime with quantization.

    Uses pre-configured ONNX models that are tested and optimized.
    Available models (384d, quantized 4-bit): ``AllMiniLML6V2Q``,
    ``AllMiniLML12V2Q``, ``BGESmallENV15Q``, ``NomicEmbedTextV15Q``,
    ``GTEBaseENV15Q``.

    Parameters
    ----------
    model_name : str
        Member name from ``embed_anything.ONNXModel``, e.g.
        ``"AllMiniLML6V2Q"``.
    batch_size : int
        Internal batch size (default 32).
    dtype : str
        ONNX quantization dtype: ``"Q4F16"`` (default), ``"F16"``, ``"INT8"``.
    """

    model_name: str = "AllMiniLML6V2Q"
    batch_size: int = 32
    dtype: str = "Q4F16"

    def __post_init__(self) -> None:
        from embed_anything import (
            Dtype,
            EmbeddingModel,
            ONNXModel,
            TextEmbedConfig,
            WhichModel,
        )

        from patent.config import HF_TOKEN

        # The ONNX path does not expose a ``token`` kwarg — the underlying
        # HuggingFace download reads HF_TOKEN from the environment.
        if HF_TOKEN:
            os.environ.setdefault("HF_TOKEN", HF_TOKEN)

        onnx_model = getattr(ONNXModel, self.model_name)
        quant_dtype = getattr(Dtype, self.dtype)
        self._model = EmbeddingModel.from_pretrained_onnx(
            WhichModel.Bert, model_name=onnx_model, dtype=quant_dtype
        )
        self._config = TextEmbedConfig(chunk_size=256, batch_size=self.batch_size)

        dummy = self._model.embed_query(["dim"], config=self._config)
        self._dim: int = len(dummy[0].embedding)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        return _encode_batched(
            self._model,
            texts,
            self._config,
            log_batch=10_000,
            model_label=f"ONNX/{self.model_name}",
        )

    def stop_pool(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, type[Embedder]] = {
    "embed-anything": EmbedAnythingEmbedder,
    "embed-anything-onnx": EmbedAnythingONNXEmbedder,
}


def register_embedder(protocol: str, cls: type[Embedder]) -> None:
    """Register a new embedder backend."""
    _REGISTRY[protocol] = cls


def get_embedder(spec: str, **kwargs) -> Embedder:
    """Create an embedder from a spec string.

    Format: ``"<protocol>:<model>"``

    Examples
    --------
    >>> get_embedder("embed-anything:MongoDB/mdbr-leaf-mt")
    >>> get_embedder("embed-anything-onnx:AllMiniLML6V2Q")
    """
    if ":" not in spec:
        raise ValueError(
            f"Invalid embedder spec {spec!r}. Expected '<protocol>:<model>' "
            f"(e.g. 'embed-anything-onnx:AllMiniLML6V2Q')"
        )
    protocol, _, model = spec.partition(":")
    cls = _REGISTRY.get(protocol)
    if cls is None:
        raise ValueError(f"Unknown embedding protocol {protocol!r}. Available: {list(_REGISTRY)}")
    return cls(model_name=model, **kwargs)
