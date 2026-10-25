"""
Task 3.1 - Evaluation metrics on real BRL scale.

Responsibilities:
- MSE, MAE, RMSE, R2, WAPE.
- Convert log1p predictions back to BRL exactly once when requested.
- Compare 4 ML models and 4 baselines.
- Export metrics tables to results/metrics/.

This module does not train/fine-tune any model.
"""

from __future__ import annotations
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

DEFAULT_METRICS_DIR = Path("results/metrics")


def inverse_log_target(values: Sequence[float]) -> np.ndarray:
    """Inverse-transform values produced by log1p(target)."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Prediction contains NaN or infinite values.")
    return np.expm1(values)


def ensure_brl_scale(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    predictions_are_log: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return actual and predicted revenue on BRL scale.

    y_true is expected to already be BRL.
    y_pred is inverse-transformed only when predictions_are_log=True.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("y_true/y_pred contains NaN or infinite values.")

    if (y_true < 0).any():
        raise ValueError("Actual revenue cannot be negative.")

    if predictions_are_log:
        y_pred = inverse_log_target(y_pred)

    # Revenue cannot be negative after inverse transformation.
    y_pred = np.maximum(y_pred, 0.0)
    return y_true, y_pred


def mse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2(y_true, y_pred) -> float:
    # R2 is undefined for a constant target.
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def wape(y_true, y_pred) -> float:
    """
    Weighted Absolute Percentage Error:
        sum(|actual - pred|) / sum(|actual|)

    If total actual revenue is zero, WAPE is undefined and returns NaN.
    """
    denominator = float(np.sum(np.abs(y_true)))
    numerator = float(np.sum(np.abs(y_true - y_pred)))
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def calculate_metrics(
    y_true,
    y_pred,
    *,
    predictions_are_log: bool = False,
    model_name: str | None = None,
    model_type: str | None = None,
) -> dict:
    """Calculate all Task 3 metrics on BRL scale."""
    y_true, y_pred = ensure_brl_scale(
        y_true,
        y_pred,
        predictions_are_log=predictions_are_log,
    )

    result = {
        "model": model_name or "model",
        "type": model_type or "model",
        "scale": "BRL",
        "MSE": mse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "N": int(len(y_true)),
    }
    return result


def evaluate_prediction_dict(
    y_true,
    predictions: Mapping[str, Sequence[float]],
    *,
    model_type: str = "model",
    predictions_are_log: bool | Mapping[str, bool] = False,
) -> pd.DataFrame:
    """Evaluate a dictionary {model_name: predictions}."""
    rows = []

    for name, pred in predictions.items():
        if isinstance(predictions_are_log, Mapping):
            is_log = bool(predictions_are_log.get(name, False))
        else:
            is_log = bool(predictions_are_log)

        rows.append(
            calculate_metrics(
                y_true,
                pred,
                predictions_are_log=is_log,
                model_name=name,
                model_type=model_type,
            )
        )

    return pd.DataFrame(rows)


def compare_models_and_baselines(
    y_true,
    model_predictions: Mapping[str, Sequence[float]],
    baseline_predictions: Mapping[str, Sequence[float]],
    *,
    model_predictions_are_log: bool | Mapping[str, bool] = False,
    baseline_predictions_are_log: bool | Mapping[str, bool] = False,
    output_dir=DEFAULT_METRICS_DIR,
    filename: str = "model_baseline_metrics.csv",
) -> pd.DataFrame:
    """
    Produce one table for 4 ML models + 4 baselines.

    The function validates the requested 4 + 4 structure but does not
    require exact model names.
    """
    if len(model_predictions) != 4:
        raise ValueError(
            f"Expected 4 ML models, received {len(model_predictions)}."
        )
    if len(baseline_predictions) != 4:
        raise ValueError(
            f"Expected 4 baselines, received {len(baseline_predictions)}."
        )

    model_df = evaluate_prediction_dict(
        y_true,
        model_predictions,
        model_type="model",
        predictions_are_log=model_predictions_are_log,
    )
    baseline_df = evaluate_prediction_dict(
        y_true,
        baseline_predictions,
        model_type="baseline",
        predictions_are_log=baseline_predictions_are_log,
    )

    result = pd.concat(
        [model_df, baseline_df],
        ignore_index=True,
    )

    # Lower is better for errors; higher is better for R2.
    result["MAE_rank"] = result["MAE"].rank(method="min")
    result["RMSE_rank"] = result["RMSE"].rank(method="min")
    result["WAPE_rank"] = result["WAPE"].rank(method="min")
    result["R2_rank"] = result["R2"].rank(
        method="min",
        ascending=False,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / filename, index=False)

    return result


def save_metrics_json(
    metrics_df: pd.DataFrame,
    output_dir=DEFAULT_METRICS_DIR,
    filename: str = "model_baseline_metrics.json",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    metrics_df.to_json(path, orient="records", indent=2)
    return path
