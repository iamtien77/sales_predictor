"""
Task 3.5 - Experiment orchestration.

This module coordinates evaluation; it does not retrain Task 1-2 models.
"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Mapping, Sequence, Any
import pandas as pd

from src.evaluation.metrics import compare_models_and_baselines
from src.evaluation.error_analysis import (
    extract_top_20_errors,
    save_error_analysis,
    summarize_error_groups,
)
from src.evaluation.cold_start import evaluate_cold_start
from src.evaluation.visualization import (
    generate_all_figures,
    validate_18_figures,
)
from src.evaluation.reporting import build_report, save_report

logger = logging.getLogger(__name__)


def _select_prediction(
    model_predictions: Mapping[str, Sequence[float]],
    preferred_model: str | None = None,
):
    if not model_predictions:
        raise ValueError("No model predictions supplied.")
    if preferred_model is not None:
        if preferred_model not in model_predictions:
            raise KeyError(f"Unknown preferred model: {preferred_model}")
        return preferred_model, model_predictions[preferred_model]
    name = next(iter(model_predictions))
    return name, model_predictions[name]


def run_task3(
    *,
    evaluation_df: pd.DataFrame,
    y_true,
    model_predictions: Mapping[str, Sequence[float]],
    baseline_predictions: Mapping[str, Sequence[float]],
    feature_importance_df: pd.DataFrame | None = None,
    cold_start_df: pd.DataFrame | None = None,
    scenario_results: Any = None,
    predictions_are_log: bool | Mapping[str, bool] = False,
    baseline_predictions_are_log: bool | Mapping[str, bool] = False,
    preferred_model: str | None = None,
    output_dir="results",
    fast_mode: bool = False,
) -> dict:
    """
    Run the complete Task 3 evaluation chain.

    The caller supplies predictions from the existing Task 1-2 / deployment
    pipeline. This function does not fit or alter models.
    """
    start = time.perf_counter()
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    figure_dir = output_dir / "figures"

    logger.info(
        "Task 3 started: mode=%s, n_rows=%d",
        "fast" if fast_mode else "full",
        len(evaluation_df),
    )

    # 1. Metrics
    metrics_df = compare_models_and_baselines(
        y_true,
        model_predictions,
        baseline_predictions,
        model_predictions_are_log=predictions_are_log,
        baseline_predictions_are_log=baseline_predictions_are_log,
        output_dir=metrics_dir,
    )

    # 2. Select the model used for detailed error visualization.
    selected_name, selected_prediction = _select_prediction(
        model_predictions,
        preferred_model,
    )

    # 3. Error table. The evaluation dataframe must preserve product/week.
    error_input = evaluation_df.copy()
    if "actual" not in error_input.columns:
        error_input["actual"] = y_true
    if "prediction" not in error_input.columns:
        error_input["prediction"] = selected_prediction

    top20 = extract_top_20_errors(error_input)

    # Full error table for group statistics.
    from .error_analysis import build_error_table
    full_errors = build_error_table(error_input)
    error_paths = save_error_analysis(full_errors, metrics_dir)
    error_groups = summarize_error_groups(full_errors)

    # 4. Cold-start evaluation, if supplied by the pipeline.
    cold_start_result = None
    if cold_start_df is not None:
        cold_start_result = evaluate_cold_start(
            cold_start_df,
            output_dir=metrics_dir,
        )

    # 5. 18 required figures.
    figures = generate_all_figures(
        y_true,
        selected_prediction,
        metrics_df,
        error_df=top20,
        feature_importance_df=feature_importance_df,
        output_dir=figure_dir,
    )
    validate_18_figures(figure_dir)

    # 6. Consolidated report.
    report = build_report(
        metrics_df=metrics_df,
        top_errors_df=top20,
        cold_start_df=cold_start_result,
        error_groups_df=error_groups,
        feature_importance_df=feature_importance_df,
        scenario_results=scenario_results,
        figure_dir=figure_dir,
    )
    report_paths = save_report(report, metrics_dir)

    elapsed = time.perf_counter() - start
    logger.info(
        "Task 3 finished in %.3f seconds; selected_model=%s; figures=%d",
        elapsed,
        selected_name,
        len(figures),
    )

    return {
        "metrics": metrics_df,
        "top_errors": top20,
        "error_groups": error_groups,
        "cold_start": cold_start_result,
        "figures": figures,
        "report": report,
        "report_paths": report_paths,
        "error_paths": error_paths,
        "runtime_seconds": elapsed,
        "mode": "fast" if fast_mode else "full",
        "selected_model": selected_name,
    }
