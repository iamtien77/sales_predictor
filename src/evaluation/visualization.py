"""
Task 3.2 - 18 scientific evaluation figures.

All figures are saved under results/figures/.
The plotting functions are intentionally data-driven: no model is trained here.
"""

from __future__ import annotations
from pathlib import Path
from typing import Mapping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_FIGURE_DIR = Path("results/figures")

FIGURE_NAMES = [
    "01_actual_vs_predicted.png",
    "02_residuals.png",
    "03_residual_distribution.png",
    "04_absolute_error_distribution.png",
    "05_error_over_index.png",
    "06_actual_over_index.png",
    "07_predicted_over_index.png",
    "08_actual_vs_predicted_time_order.png",
    "09_error_boxplot.png",
    "10_absolute_error_boxplot.png",
    "11_feature_importance.png",
    "12_model_error_comparison.png",
    "13_mae_comparison.png",
    "14_rmse_comparison.png",
    "15_wape_comparison.png",
    "16_r2_comparison.png",
    "17_top_20_errors.png",
    "18_error_vs_actual.png",
]


def _save(fig, output_dir, filename: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _base_arrays(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(actual) != len(pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not np.isfinite(actual).all() or not np.isfinite(pred).all():
        raise ValueError("Plot input contains NaN or infinite values.")
    return actual, pred


def plot_actual_vs_predicted(y_true, y_pred, output_dir=DEFAULT_FIGURE_DIR):
    actual, pred = _base_arrays(y_true, y_pred)
    fig = plt.figure(figsize=(8, 7))
    plt.scatter(actual, pred, alpha=0.65)
    low = min(actual.min(), pred.min())
    high = max(actual.max(), pred.max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual Revenue (BRL)")
    plt.ylabel("Predicted Revenue (BRL)")
    plt.title("Actual vs Predicted Revenue")
    return _save(fig, output_dir, FIGURE_NAMES[0])


def plot_residuals(y_true, y_pred, output_dir=DEFAULT_FIGURE_DIR):
    actual, pred = _base_arrays(y_true, y_pred)
    residual = actual - pred
    fig = plt.figure(figsize=(9, 6))
    plt.scatter(pred, residual, alpha=0.65)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Revenue (BRL)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals Plot")
    return _save(fig, output_dir, FIGURE_NAMES[1])


def plot_residual_distribution(y_true, y_pred, output_dir=DEFAULT_FIGURE_DIR):
    actual, pred = _base_arrays(y_true, y_pred)
    residual = actual - pred
    fig = plt.figure(figsize=(9, 6))
    plt.hist(residual, bins=30)
    plt.xlabel("Residual (BRL)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    return _save(fig, output_dir, FIGURE_NAMES[2])


def plot_absolute_error_distribution(y_true, y_pred, output_dir=DEFAULT_FIGURE_DIR):
    actual, pred = _base_arrays(y_true, y_pred)
    error = np.abs(actual - pred)
    fig = plt.figure(figsize=(9, 6))
    plt.hist(error, bins=30)
    plt.xlabel("Absolute Error (BRL)")
    plt.ylabel("Frequency")
    plt.title("Absolute Error Distribution")
    return _save(fig, output_dir, FIGURE_NAMES[3])


def plot_error_over_index(y_true, y_pred, output_dir=DEFAULT_FIGURE_DIR):
    actual, pred = _base_arrays(y_true, y_pred)
    error = actual - pred
    fig = plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(error)), error)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Observation Index")
    plt.ylabel("Residual (BRL)")
    plt.title("Residual Error Over Observations")
    return _save(fig, output_dir, FIGURE_NAMES[4])


def plot_actual_over_index(y_true, output_dir=DEFAULT_FIGURE_DIR):
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(actual)), actual)
    plt.xlabel("Observation Index")
    plt.ylabel("Actual Revenue (BRL)")
    plt.title("Actual Revenue Over Observations")
    return _save(fig, output_dir, FIGURE_NAMES[5])


def plot_predicted_over_index(y_pred, output_dir=DEFAULT_FIGURE_DIR):
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(pred)), pred)
    plt.xlabel("Observation Index")
    plt.ylabel("Predicted Revenue (BRL)")
    plt.title("Predicted Revenue Over Observations")
    return _save(fig, output_dir, FIGURE_NAMES[6])


def plot_actual_vs_predicted_time_order(
    y_true,
    y_pred,
    output_dir=DEFAULT_FIGURE_DIR,
):
    actual, pred = _base_arrays(y_true, y_pred)
    fig = plt.figure(figsize=(11, 6))
    x = np.arange(len(actual))
    plt.plot(x, actual, label="Actual")
    plt.plot(x, pred, label="Predicted")
    plt.xlabel("Observation / Week Order")
    plt.ylabel("Revenue (BRL)")
    plt.title("Actual vs Predicted in Time Order")
    plt.legend()
    return _save(fig, output_dir, FIGURE_NAMES[7])


def plot_error_boxplot(y_true, y_pred, output_dir=DEFAULT_FIGURE_DIR):
    actual, pred = _base_arrays(y_true, y_pred)
    error = actual - pred
    fig = plt.figure(figsize=(7, 6))
    plt.boxplot(error)
    plt.ylabel("Residual (BRL)")
    plt.title("Residual Error Boxplot")
    return _save(fig, output_dir, FIGURE_NAMES[8])


def plot_absolute_error_boxplot(
    y_true,
    y_pred,
    output_dir=DEFAULT_FIGURE_DIR,
):
    actual, pred = _base_arrays(y_true, y_pred)
    error = np.abs(actual - pred)
    fig = plt.figure(figsize=(7, 6))
    plt.boxplot(error)
    plt.ylabel("Absolute Error (BRL)")
    plt.title("Absolute Error Boxplot")
    return _save(fig, output_dir, FIGURE_NAMES[9])


def plot_feature_importance(
    feature_importance_df: pd.DataFrame | None,
    output_dir=DEFAULT_FIGURE_DIR,
    top_n: int = 20,
):
    """
    Accepts columns:
      feature + importance
    or feature + importance + model.
    """
    fig = plt.figure(figsize=(10, 7))

    if feature_importance_df is None or feature_importance_df.empty:
        plt.text(
            0.5,
            0.5,
            "Feature importance not supplied by the model.",
            ha="center",
            va="center",
        )
        plt.axis("off")
        return _save(fig, output_dir, FIGURE_NAMES[10])

    df = feature_importance_df.copy()
    required = {"feature", "importance"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing feature importance columns: {missing}")

    if "model" in df.columns:
        # Aggregate across models so the figure remains one of the 18 required figures.
        df = (
            df.groupby("feature", as_index=False)["importance"]
            .mean()
        )

    df = (
        df.sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)
    )

    plt.barh(df["feature"].astype(str), df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importance")
    return _save(fig, output_dir, FIGURE_NAMES[10])


def plot_model_error_comparison(
    metrics_df: pd.DataFrame,
    output_dir=DEFAULT_FIGURE_DIR,
):
    required = {"model", "MAE", "RMSE", "WAPE"}
    missing = required - set(metrics_df.columns)
    if missing:
        raise KeyError(f"Missing metrics columns: {missing}")

    fig = plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics_df))
    width = 0.25

    plt.bar(x - width, metrics_df["MAE"], width, label="MAE")
    plt.bar(x, metrics_df["RMSE"], width, label="RMSE")
    plt.bar(x + width, metrics_df["WAPE"], width, label="WAPE")
    plt.xticks(x, metrics_df["model"], rotation=45, ha="right")
    plt.ylabel("Metric Value")
    plt.title("Model and Baseline Error Comparison")
    plt.legend()
    return _save(fig, output_dir, FIGURE_NAMES[11])


def _metric_comparison(metrics_df, metric, title, filename_index, output_dir):
    required = {"model", metric}
    missing = required - set(metrics_df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    fig = plt.figure(figsize=(11, 6))
    plt.bar(metrics_df["model"].astype(str), metrics_df[metric])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    return _save(fig, output_dir, FIGURE_NAMES[filename_index])


def plot_mae_comparison(metrics_df, output_dir=DEFAULT_FIGURE_DIR):
    return _metric_comparison(
        metrics_df, "MAE", "MAE Comparison", 12, output_dir
    )


def plot_rmse_comparison(metrics_df, output_dir=DEFAULT_FIGURE_DIR):
    return _metric_comparison(
        metrics_df, "RMSE", "RMSE Comparison", 13, output_dir
    )


def plot_wape_comparison(metrics_df, output_dir=DEFAULT_FIGURE_DIR):
    return _metric_comparison(
        metrics_df, "WAPE", "WAPE Comparison", 14, output_dir
    )


def plot_r2_comparison(metrics_df, output_dir=DEFAULT_FIGURE_DIR):
    return _metric_comparison(
        metrics_df, "R2", "R² Comparison", 15, output_dir
    )


def plot_top_20_errors(
    error_df: pd.DataFrame | None,
    output_dir=DEFAULT_FIGURE_DIR,
):
    fig = plt.figure(figsize=(11, 7))

    if error_df is None or error_df.empty:
        plt.text(0.5, 0.5, "Top 20 errors not supplied.", ha="center", va="center")
        plt.axis("off")
        return _save(fig, output_dir, FIGURE_NAMES[16])

    required = {"absolute_error"}
    missing = required - set(error_df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    data = error_df.head(20).copy()
    labels = (
        data["product_id"].astype(str) + " | " + data["week"].astype(str)
        if {"product_id", "week"} <= set(data.columns)
        else data.index.astype(str)
    )
    plt.barh(labels[::-1], data["absolute_error"].iloc[::-1])
    plt.xlabel("Absolute Error (BRL)")
    plt.ylabel("Product | Week")
    plt.title("Top 20 Largest Absolute Errors")
    return _save(fig, output_dir, FIGURE_NAMES[16])


def plot_error_vs_actual(y_true, y_pred, output_dir=DEFAULT_FIGURE_DIR):
    actual, pred = _base_arrays(y_true, y_pred)
    error = np.abs(actual - pred)

    fig = plt.figure(figsize=(9, 6))
    plt.scatter(actual, error, alpha=0.65)
    plt.xlabel("Actual Revenue (BRL)")
    plt.ylabel("Absolute Error (BRL)")
    plt.title("Absolute Error vs Actual Revenue")
    return _save(fig, output_dir, FIGURE_NAMES[17])


def validate_18_figures(output_dir=DEFAULT_FIGURE_DIR) -> list[Path]:
    """Fail loudly if the required 18 files are not present."""
    output_dir = Path(output_dir)
    missing = [
        name for name in FIGURE_NAMES
        if not (output_dir / name).exists()
    ]
    if missing:
        raise RuntimeError(
            "Missing required figures: " + ", ".join(missing)
        )
    return [output_dir / name for name in FIGURE_NAMES]


def generate_all_figures(
    y_true,
    y_pred,
    metrics_df: pd.DataFrame,
    *,
    error_df: pd.DataFrame | None = None,
    feature_importance_df: pd.DataFrame | None = None,
    output_dir=DEFAULT_FIGURE_DIR,
) -> list[Path]:
    """Generate exactly the 18 required figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove only the 18 files owned by this module; do not delete unrelated files.
    for filename in FIGURE_NAMES:
        path = output_dir / filename
        if path.exists():
            path.unlink()

    generated = [
        plot_actual_vs_predicted(y_true, y_pred, output_dir),
        plot_residuals(y_true, y_pred, output_dir),
        plot_residual_distribution(y_true, y_pred, output_dir),
        plot_absolute_error_distribution(y_true, y_pred, output_dir),
        plot_error_over_index(y_true, y_pred, output_dir),
        plot_actual_over_index(y_true, output_dir),
        plot_predicted_over_index(y_pred, output_dir),
        plot_actual_vs_predicted_time_order(y_true, y_pred, output_dir),
        plot_error_boxplot(y_true, y_pred, output_dir),
        plot_absolute_error_boxplot(y_true, y_pred, output_dir),
        plot_feature_importance(feature_importance_df, output_dir),
        plot_model_error_comparison(metrics_df, output_dir),
        plot_mae_comparison(metrics_df, output_dir),
        plot_rmse_comparison(metrics_df, output_dir),
        plot_wape_comparison(metrics_df, output_dir),
        plot_r2_comparison(metrics_df, output_dir),
        plot_top_20_errors(error_df, output_dir),
        plot_error_vs_actual(y_true, y_pred, output_dir),
    ]

    validate_18_figures(output_dir)
    if len(generated) != 18:
        raise RuntimeError(f"Expected 18 figures, generated {len(generated)}.")

    return generated
