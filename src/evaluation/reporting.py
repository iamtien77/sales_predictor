"""
Task 3 reporting.

Combines metrics, Top 20 errors, cold-start results, figures and
scenario results supplied by My's deployment layer.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
import pandas as pd

DEFAULT_OUTPUT_DIR = Path("results/metrics")


def collect_figure_files(
    figure_dir="results/figures",
) -> list[str]:
    figure_dir = Path(figure_dir)
    return sorted(
        str(path)
        for path in figure_dir.glob("*.png")
    )


def build_report(
    *,
    metrics_df: pd.DataFrame | None = None,
    top_errors_df: pd.DataFrame | None = None,
    cold_start_df: pd.DataFrame | None = None,
    error_groups_df: pd.DataFrame | None = None,
    feature_importance_df: pd.DataFrame | None = None,
    scenario_results: Any = None,
    figure_dir="results/figures",
) -> dict:
    """Build a JSON-serializable Task 3 report."""
    report = {
        "task": "Task 3 - Evaluation, Experiments and Reporting",
        "scale": "BRL",
        "metrics": (
            metrics_df.to_dict(orient="records")
            if metrics_df is not None else []
        ),
        "top_20_errors": (
            top_errors_df.head(20).to_dict(orient="records")
            if top_errors_df is not None else []
        ),
        "cold_start": (
            cold_start_df.to_dict(orient="records")
            if cold_start_df is not None else []
        ),
        "error_groups": (
            error_groups_df.to_dict(orient="records")
            if error_groups_df is not None else []
        ),
        "feature_importance": (
            feature_importance_df.to_dict(orient="records")
            if feature_importance_df is not None else []
        ),
        "figures": collect_figure_files(figure_dir),
        "scenarios": scenario_results if scenario_results is not None else [],
    }
    return report


def save_report(
    report: Mapping[str, Any],
    output_dir=DEFAULT_OUTPUT_DIR,
    json_filename="task3_report.json",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / json_filename
    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    # Flatten the major tables into separate CSV artifacts.
    paths = {"json": json_path}

    if report.get("metrics"):
        p = output_dir / "metrics_summary.csv"
        pd.DataFrame(report["metrics"]).to_csv(p, index=False)
        paths["metrics"] = p

    if report.get("top_20_errors"):
        p = output_dir / "top_20_errors_report.csv"
        pd.DataFrame(report["top_20_errors"]).to_csv(p, index=False)
        paths["top20"] = p

    if report.get("cold_start"):
        p = output_dir / "cold_start_report.csv"
        pd.DataFrame(report["cold_start"]).to_csv(p, index=False)
        paths["cold_start"] = p

    if report.get("error_groups"):
        p = output_dir / "error_groups_report.csv"
        pd.DataFrame(report["error_groups"]).to_csv(p, index=False)
        paths["error_groups"] = p

    return paths
