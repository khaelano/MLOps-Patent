"""MLflow pyfunc wrapper for the LSHiForest anomaly detection model.

Bundles the LSHiForest model with its embedder into a single MLflow
pyfunc model so that ``mlflow models build-docker`` can produce a
self-contained serving image.

Usage in training::

    from patent.modeling.pyfunc_model import LSHiForestPyfuncModel
    mlflow.pyfunc.log_model(
        "pyfunc_model",
        python_model=LSHiForestPyfuncModel(embedder_spec="embed-anything-onnx:AllMiniLML6V2Q"),
        artifacts={"model.lshif": "/path/to/model.lshif"},
        code_paths=["patent"],
        pip_requirements=["embed-anything>=0.7.0", "numpy", "loguru"],
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from patent.lshiforest import LSHiForest

from patent.lshiforest._scoring import rescale_scores

# Default embedder spec used when none is provided to the constructor.
DEFAULT_EMBEDDER_SPEC = "embed-anything-onnx:AllMiniLML6V2Q"


class LSHiForestPyfuncModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc model that encodes text with a bundled embedder and scores
    anomalies with a trained LSHiForest.

    Parameters
    ----------
    embedder_spec : str
        Embedder spec string ``"<protocol>:<model>"`` passed to
        :func:`patent.dataset.embedders.get_embedder`.  The embedder is
        instantiated **at ``load_context`` time** so the ONNX / Candle
        model is downloaded once and cached on the container volume.
    """

    def __init__(self, embedder_spec: str = DEFAULT_EMBEDDER_SPEC) -> None:
        self.embedder_spec = embedder_spec
        self._model: LSHiForest | None = None
        self._embedder: Any = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the LSHiForest model and embedder from the artifact context.

        Called once by MLflow when the model is loaded into memory.  The
        ``.lshif`` file is expected in the artifacts directory under the
        key ``"model.lshif"``.
        """
        from patent.dataset.embedders import get_embedder
        from patent.lshiforest import LSHiForest

        # ── Load LSHiForest from the artifact ──────────────────────────
        # ``context.artifacts["model.lshif"]`` is the path to the
        # .lshif file that was passed via the ``artifacts`` dict in
        # ``mlflow.pyfunc.log_model()``.
        self._model = LSHiForest.load(context.artifacts["model.lshif"])

        # ── Load embedder (downloads ONNX model if needed) ─────────────
        self._embedder = get_embedder(self.embedder_spec)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Score a batch of texts for anomaly.

        Parameters
        ----------
        model_input : pd.DataFrame
            Must contain a ``"texts"`` column where each row is a string
            (title + abstract concatenation, same format as training).

        Returns
        -------
        pd.DataFrame
            Columns: ``scores`` (raw LSHiForest anomaly scores ∈ [0, 1]),
            ``rescaled_scores`` (percentile-rescaled scores ∈ [0, 1]).
        """
        if self._model is None or self._embedder is None:
            raise RuntimeError("Model or embedder not loaded.  Call load_context first.")

        texts = model_input["texts"].tolist()
        if not texts:
            return pd.DataFrame({"scores": [], "rescaled_scores": []})

        # ── Embed ──────────────────────────────────────────────────────
        embeddings = self._embedder.encode(texts, show_progress=False)
        X = np.asarray(embeddings, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # ── Score ──────────────────────────────────────────────────────
        raw_scores = self._model.score(X)
        rescaled = rescale_scores(raw_scores)

        return pd.DataFrame(
            {
                "scores": raw_scores.tolist(),
                "rescaled_scores": rescaled.tolist(),
            }
        )
