import numpy as np
import pandas as pd

from src.evaluation.experiments import run_task3
from src.evaluation.visualization import FIGURE_NAMES


def _test_data():
    n = 25
    y_true = np.arange(100.0, 100.0 + n)

    models = {
        "Model_1": y_true * 0.98,
        "Model_2": y_true * 1.01,
        "Model_3": y_true * 0.95,
        "Model_4": y_true * 1.03,
    }

    baselines = {
        "Baseline_1": np.full(n, y_true.mean()),
        "Baseline_2": np.roll(y_true, 1),
        "Baseline_3": np.full(n, 100.0),
        "Baseline_4": np.full(n, 120.0),
    }

    evaluation_df = pd.DataFrame(
        {
            "product_id": [f"P{i % 5}" for i in range(n)],
            "week": [f"W{i + 1}" for i in range(n)],
            "actual": y_true,
        }
    )

    return evaluation_df, y_true, models, baselines


def test_full_pipeline_run_fast_mode(tmp_path):
    df, y_true, models, baselines = _test_data()

    result = run_task3(
        evaluation_df=df,
        y_true=y_true,
        model_predictions=models,
        baseline_predictions=baselines,
        output_dir=tmp_path,
        fast_mode=True,
    )

    assert result["mode"] == "fast"
    assert result["runtime_seconds"] >= 0
    assert result["metrics"].shape[0] == 8


def test_artifacts_generation(tmp_path):
    df, y_true, models, baselines = _test_data()

    result = run_task3(
        evaluation_df=df,
        y_true=y_true,
        model_predictions=models,
        baseline_predictions=baselines,
        output_dir=tmp_path,
        fast_mode=True,
    )

    figures = result["figures"]

    assert len(figures) == 18
    assert all(path.exists() for path in figures)

    assert set(
        result["metrics"]["model"]
    ) == set(models) | set(baselines)

    top20 = result["top_errors"]
    assert len(top20) == 20
    assert list(top20["rank"]) == list(range(1, 21))
    assert top20["absolute_error"].is_monotonic_decreasing
    assert "product_id" in top20.columns
    assert "week" in top20.columns

    assert (tmp_path / "metrics" / "metrics_summary.csv").exists()
    assert (tmp_path / "metrics" / "task3_report.json").exists()

    assert set(
        p.name for p in (tmp_path / "figures").glob("*.png")
    ) == set(FIGURE_NAMES)


def test_cli_argument_parsing():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "full"], default="full")

    assert parser.parse_args(["--mode", "fast"]).mode == "fast"
    assert parser.parse_args(["--mode", "full"]).mode == "full"
