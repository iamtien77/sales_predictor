"""
Task 3.3 - Error analysis and Top 20 largest absolute errors.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
DEFAULT_METRICS_DIR = Path("results/metrics")


def build_error_table(
    df: pd.DataFrame,
    *,
    actual_col: str = "actual",
    prediction_col: str = "prediction",
    product_col: str = "product_id",
    week_col: str = "week",
) -> pd.DataFrame:
    """Create an error table while preserving product and week identifiers."""
    required = {
        actual_col,
        prediction_col,
        product_col,
        week_col,
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    result = df.copy()
    result["error"] = (
        result[actual_col].astype(float)
        - result[prediction_col].astype(float)
    )
    result["absolute_error"] = result["error"].abs()

    result["percentage_error"] = np.where(
        result[actual_col].astype(float) != 0,
        result["absolute_error"]
        / result[actual_col].abs()
        * 100.0,
        np.nan,
    )

    result["error_direction"] = np.select(
        [
            result["error"] > 0,
            result["error"] < 0,
            result["error"] == 0,
        ],
        [
            "Under-prediction",
            "Over-prediction",
            "Exact",
        ],
        default="Unknown",
    )

    result["error_group"] = result.apply(
        classify_error,
        axis=1,
    )

    result["possible_cause"] = result.apply(
        identify_possible_cause,
        axis=1,
    )

    return result


def classify_error(row) -> str:
    actual = float(row["actual"])
    prediction = float(row["prediction"])

    if actual == 0 and prediction == 0:
        return "Correct zero"

    if actual == 0 and prediction > 0:
        return "Zero-demand false positive"

    if actual > 0 and prediction == 0:
        return "Zero-prediction false negative"

    if actual > 0:
        ratio = prediction / actual
        if ratio >= 2:
            return "Strong over-prediction"
        if ratio <= 0.5:
            return "Strong under-prediction"

    return "Moderate/normal deviation"


def identify_possible_cause(row) -> str:
    """
    Heuristic interpretation only.

    The function must not claim a causal business explanation when the
    corresponding evidence is absent from the row.
    """
    actual = float(row["actual"])
    prediction = float(row["prediction"])

    if actual == 0 and prediction > 0:
        return "Sparse/zero-demand observation may be difficult to distinguish"

    if actual > 0 and prediction == 0:
        return "Observed demand was missed by the prediction"

    if actual > 0 and prediction >= 2 * actual:
        return "Possible demand overestimation or sensitivity after inverse transform"

    if actual > 0 and prediction <= 0.5 * actual:
        return "Possible demand underestimation or unobserved demand spike"

    return "No single cause can be established from prediction values alone"


def extract_top_20_errors(
    df: pd.DataFrame,
    *,
    actual_col: str = "actual",
    prediction_col: str = "prediction",
    product_col: str = "product_id",
    week_col: str = "week",
) -> pd.DataFrame:
    """Return exactly up to 20 largest absolute errors in descending order."""
    errors = build_error_table(
        df,
        actual_col=actual_col,
        prediction_col=prediction_col,
        product_col=product_col,
        week_col=week_col,
    )

    errors = errors.sort_values(
        ["absolute_error", product_col, week_col],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    errors.insert(0, "rank", np.arange(1, len(errors) + 1))
    return errors.head(20)


def summarize_error_groups(error_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate error groups for reporting."""
    if error_df.empty:
        return pd.DataFrame(
            columns=[
                "error_group",
                "count",
                "mean_absolute_error",
                "max_absolute_error",
            ]
        )

    return (
        error_df.groupby("error_group", dropna=False)
        .agg(
            count=("absolute_error", "size"),
            mean_absolute_error=("absolute_error", "mean"),
            max_absolute_error=("absolute_error", "max"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )


def summarize_error_statistics(error_df: pd.DataFrame) -> dict:
    """Return overall statistical characteristics of the error table."""
    if error_df.empty:
        return {
            "n": 0,
            "mean_error": None,
            "median_error": None,
            "mean_absolute_error": None,
            "max_absolute_error": None,
            "zero_actual_count": 0,
            "over_prediction_count": 0,
            "under_prediction_count": 0,
        }

    return {
        "n": int(len(error_df)),
        "mean_error": float(error_df["error"].mean()),
        "median_error": float(error_df["error"].median()),
        "mean_absolute_error": float(error_df["absolute_error"].mean()),
        "max_absolute_error": float(error_df["absolute_error"].max()),
        "zero_actual_count": int((error_df["actual"] == 0).sum()),
        "over_prediction_count": int(
            (error_df["error"] < 0).sum()
        ),
        "under_prediction_count": int(
            (error_df["error"] > 0).sum()
        ),
    }


def save_error_analysis(
    error_df: pd.DataFrame,
    output_dir=DEFAULT_METRICS_DIR,
) -> dict[str, Path]:
    """Save Top 20 and group/statistical summaries."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top20 = error_df.head(20).copy()
    groups = summarize_error_groups(error_df)
    statistics = summarize_error_statistics(error_df)

    top_path = output_dir / "top_20_errors.csv"
    group_path = output_dir / "error_groups.csv"
    stats_path = output_dir / "error_statistics.json"

    top20.to_csv(top_path, index=False)
    groups.to_csv(group_path, index=False)

    import json
    stats_path.write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "top20": top_path,
        "groups": group_path,
        "statistics": stats_path,
    }
