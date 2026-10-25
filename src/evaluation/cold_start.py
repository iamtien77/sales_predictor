"""
Evaluation of new products versus products with historical sales.

This file evaluates predictions already produced by Task 2 / predictor.
It does NOT train or redefine the cold-start algorithm.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.evaluation.metrics import calculate_metrics
DEFAULT_OUTPUT_DIR = Path("results/metrics")


def evaluate_cold_start(
    df: pd.DataFrame,
    *,
    actual_col: str = "actual",
    prediction_col: str = "prediction",
    history_col: str = "has_history",
    prediction_is_log: bool = False,
    output_dir=DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Compare New Product (no history) and Historical Product (has history).

    history_col may be bool or 0/1.
    """
    required = {
        actual_col,
        prediction_col,
        history_col,
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Cold-start evaluation data is empty.")

    if df[actual_col].isna().any() or df[prediction_col].isna().any():
        raise ValueError("Actual/prediction contains NaN.")

    rows = []

    for has_history, group in df.groupby(history_col, dropna=False):
        label = (
            "Historical"
            if bool(has_history)
            else "New Product"
        )

        result = calculate_metrics(
            group[actual_col].to_numpy(),
            group[prediction_col].to_numpy(),
            predictions_are_log=prediction_is_log,
            model_name=label,
            model_type="cold_start_group",
        )
        result["group"] = label
        result["has_history"] = bool(has_history)
        rows.append(result)

    result_df = pd.DataFrame(rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(
        output_dir / "cold_start_evaluation.csv",
        index=False,
    )

    return result_df
