from functools import partial
import gc

from loguru import logger
import mlflow
import numpy as np
import optuna
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split


def compute_composite_score(
    diagnostics: dict,
    weights: dict = None,
    stability_threshold: float = 0.05,
    rank_threshold: float = 0.80,
) -> float:
    """
    Combine diagnostics into a single minimizable score.
    """
    w = weights or {
        "stability_relative": 0.5,
        "top10_consistency": 0.4,
        "stability_absolute": 0.1,
    }

    cv = np.clip(diagnostics["stability_relative"], 0, 1)
    rank_instability = np.clip(1.0 - diagnostics["top10_consistency"], 0, 1)

    score_min = diagnostics["score_min"]
    score_max = diagnostics["score_max"]
    score_span = max(score_max - score_min, 1e-8)
    abs_stability_norm = np.clip(diagnostics["stability"] / score_span, 0, 1)

    if cv > stability_threshold:
        cv_penalty = cv * (1 + 2 * (cv - stability_threshold))
    else:
        cv_penalty = cv

    if rank_instability > (1 - rank_threshold):
        rank_penalty = rank_instability * (1 + 3 * (rank_instability - (1 - rank_threshold)))
    else:
        rank_penalty = rank_instability

    composite = (
        w["stability_relative"] * cv_penalty
        + w["top10_consistency"] * rank_penalty
        + w["stability_absolute"] * abs_stability_norm
    )

    return float(np.clip(composite, 0, 10))


def evaluate_params(params, X_train: np.ndarray, X_val: np.ndarray, n_bootstrap=3):
    scores = []
    n_samples = min(50000, int(0.8 * len(X_train)))

    for i in range(n_bootstrap):
        rng = np.random.default_rng(seed=i)
        idx = rng.choice(len(X_train), size=n_samples, replace=True)
        X_boot = X_train[idx]

        model = IsolationForest(**params, random_state=i)
        model.fit(X_boot)

        score = -model.score_samples(X_val)
        scores.append(score)

        del X_boot
        del model
        gc.collect()

    scores = np.array(scores)
    mean_scores = np.mean(scores, axis=0)
    std_per_sample = np.std(scores, axis=0)

    evaluation = {
        "stability": float(np.mean(std_per_sample)),
        "stability_relative": float(
            np.mean(std_per_sample) / (np.mean(np.abs(mean_scores)) + 1e-8)
        ),
        "score_max": float(np.min(mean_scores)),
        "score_min": float(np.max(mean_scores)),
        "top10_consistency": float(np.corrcoef(scores[:, :100].reshape(len(scores), -1))[0, 1])
        if len(scores) > 1
        else 1.0,
    }

    composite = compute_composite_score(evaluation)
    evaluation["composite"] = composite

    return evaluation


def objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    X_val: np.ndarray,
    experiment_id: str,
    parent_run_id: str,
):
    logger.trace(f"Starting trial no-{trial.number}")

    tags = {"mlflow.parentRunId": parent_run_id}

    with mlflow.start_run(
        experiment_id=experiment_id, run_name=f"trial-no-{trial.number}", tags=tags
    ):
        params = {
            "max_samples": trial.suggest_int("max_samples", 64, 256, step=64),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
            "contamination": trial.suggest_float("contamination", 0.01, 0.1, step=0.01),
            "n_jobs": 1,
        }

        model = IsolationForest(**params, random_state=42)
        model.fit(X_train)

        evaluation = evaluate_params(params, X_train=X_train, X_val=X_val)
        logger.trace(f"Evaluation for run no-{trial.number}: {evaluation}")

        mlflow.log_params(params)
        mlflow.log_metrics(evaluation)

        return evaluation["composite"]


def optimize_iforest(train_data: np.ndarray, experiment_id: int, run_name="isolated-forest"):
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as parent_run:
        # Split training data
        logger.trace("Splitting training data")
        X_train, X_val = train_test_split(
            train_data, test_size=0.02, random_state=42, shuffle=True
        )

        # Start hypertune
        logger.info("Starting iforest hyperparameter tuning")
        partial_objective = partial(
            objective,
            X_train=X_train,
            X_val=X_val,
            experiment_id=str(parent_run.info.experiment_id),
            parent_run_id=parent_run.info.run_id,
        )
        study = optuna.create_study(direction="minimize", study_name=run_name)
        study.optimize(
            partial_objective,
            n_trials=50,
            show_progress_bar=True,
            n_jobs=4,
        )
        logger.debug("Finished iforest hyperparamer tuning")

        params = study.best_params
        metrics = study.best_trial.user_attrs

        # Start model training with full dataset
        logger.info("Starting full model training")
        model = IsolationForest(
            **params,
            random_state=42,
            n_jobs=4,
        )
        model.fit(train_data)
        logger.debug("Finished full model training")

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model)

        logger.info(f"Best params: {params}")
        logger.info(f"Best metrics: {metrics}")

        return {"params": params, "metrics": metrics, "model": model}
