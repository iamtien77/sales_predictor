"""Kiểm thử xuất báo cáo JSON tổng hợp 3 giai đoạn (trước, trong, sau train)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.builder import build_weekly_sales_dataset
from src.data.feature_engineering import WeeklySalesFeatureEngineer
from src.data.preprocessing import CustomDataPipeline
from src.evaluation import run_report
from src.models.regressors import DecisionTreeRegressorFromScratch


def _reject_nonstandard_json_constant(constant: str):
    raise ValueError(f"Invalid JSON constant detected: {constant}")


def _build_synthetic_transactions(n_weeks: int = 16) -> pd.DataFrame:
    weeks = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")
    rows = []
    for i, week in enumerate(weeks):
        for prod in ["p1", "p2", "p3"]:
            if prod == "p1" and i % 4 == 2:
                continue
            rows.append({
                "product_id": prod,
                "order_id": f"{prod}_{i}",
                "order_item_id": 1,
                "price": float(50.0 + i * 5.0 + (10.0 if prod == "p2" else 0.0)),
                config.TIMESTAMP_COLUMN: week + pd.Timedelta(days=1),
                "category": "books" if prod != "p3" else "electronics",
            })
    return pd.DataFrame(rows)


def test_to_json_safe_and_strict_save(tmp_path):
    df = pd.DataFrame({
        "Model": ["A", "B"],
        "RMSE": [np.float64(12.5), np.nan],
        "InfVal": [np.inf, -np.inf],
    })
    payload = {
        "int_val": np.int64(7),
        "bool_val": np.bool_(True),
        "ts": pd.Timestamp("2020-05-04"),
        "path": config.METRICS_DIR / "dummy.csv",
        "arr": np.array([1.0, np.nan, 3.0]),
        "table": df,
    }
    out_path = tmp_path / "strict_report.json"
    written = run_report.save_pipeline_report(payload, filepath=out_path)
    assert written == out_path
    assert out_path.exists()
    assert not out_path.with_suffix(".json.tmp").exists()

    raw_text = out_path.read_text(encoding="utf-8")
    parsed = json.loads(raw_text, parse_constant=_reject_nonstandard_json_constant)
    assert parsed["int_val"] == 7
    assert parsed["bool_val"] is True
    assert parsed["ts"] == "2020-05-04T00:00:00"
    assert parsed["arr"] == [1.0, None, 3.0]
    assert parsed["table"][1]["RMSE"] is None
    assert parsed["table"][0]["InfVal"] is None


def test_summarize_pipeline_split_and_model_helpers():
    tx = _build_synthetic_transactions(n_weeks=14)
    panel = build_weekly_sales_dataset(tx)
    split_summary = run_report.summarize_dataset_split(panel, total_rows=len(panel))
    assert split_summary["n_rows"] == len(panel)
    assert split_summary["ratio"] == pytest.approx(1.0)
    assert split_summary["n_unique_products"] == 3
    assert split_summary["zero_sales_count"] >= 1
    assert split_summary["target_stats_brl"]["count"] == len(panel)

    pipe = CustomDataPipeline(
        num_cols=config.WEEKLY_NUMERICAL_COLS,
        cat_cols=config.WEEKLY_CATEGORICAL_COLS,
        outlier_cols=[],
        feature_engineer=WeeklySalesFeatureEngineer(),
    )
    X_train, y_train = pipe.fit_transform(panel)
    pipe_summary = run_report.summarize_pipeline(pipe)
    assert pipe_summary["n_features"] == len(pipe.feature_names_)
    assert len(pipe_summary["scaler_stats"]) == len(config.WEEKLY_NUMERICAL_COLS)

    tree = DecisionTreeRegressorFromScratch(max_depth=3, min_samples_leaf=2).fit(X_train, y_train)
    hparams = run_report.extract_model_hyperparameters(tree)
    assert hparams["max_depth"] == 3
    assert hparams["min_samples_leaf"] == 2

    importances = run_report.extract_feature_importances(tree, pipe.feature_names_, top_n=5)
    assert len(importances) <= 5
    assert importances[0]["rank"] == 1

    res_summary = run_report.summarize_residual_errors(y_train, tree.predict(X_train))
    assert res_summary["n_samples"] == len(y_train)
    assert res_summary["non_finite_count"] == 0


def _patch_paths_to_tmp(monkeypatch, tmp_path: Path, raw_csv: Path):
    import main
    processed_dir = tmp_path / "processed"
    results_dir = tmp_path / "results"
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    saved_models_dir = results_dir / "saved_models"
    nb_results = tmp_path / "nb_results"
    nb_metrics = nb_results / "metrics"
    nb_figures = nb_results / "figures"
    nb_models = nb_results / "saved_models"
    for d in [processed_dir, metrics_dir, figures_dir, saved_models_dir, nb_metrics, nb_figures, nb_models]:
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "RAW_DATA_FILE", raw_csv)
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(config, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(config, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(config, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(config, "SAVED_MODELS_DIR", saved_models_dir)
    monkeypatch.setattr(config, "NOTEBOOK_RESULTS_DIR", nb_results)
    monkeypatch.setattr(config, "NOTEBOOK_METRICS_DIR", nb_metrics)
    monkeypatch.setattr(config, "NOTEBOOK_FIGURES_DIR", nb_figures)
    monkeypatch.setattr(config, "NOTEBOOK_SAVED_MODELS_DIR", nb_models)
    monkeypatch.setattr(config, "CACHE_X_TRAIN", processed_dir / "X_train.npz")
    monkeypatch.setattr(config, "CACHE_Y_TRAIN", processed_dir / "y_train.npy")
    monkeypatch.setattr(config, "CACHE_X_VAL", processed_dir / "X_val.npz")
    monkeypatch.setattr(config, "CACHE_Y_VAL", processed_dir / "y_val.npy")
    monkeypatch.setattr(config, "CACHE_X_TEST", processed_dir / "X_test.npz")
    monkeypatch.setattr(config, "CACHE_Y_TEST", processed_dir / "y_test.npy")
    monkeypatch.setattr(config, "CACHE_FEATURE_NAMES", processed_dir / "feature_names.json")
    monkeypatch.setattr(config, "CACHE_PIPELINE", processed_dir / "preprocessor_pipeline.pkl")
    monkeypatch.setattr(config, "PIPELINE_REPORT_JSON", metrics_dir / "pipeline_run_report.json")
    return main


def test_main_pipeline_writes_three_stage_json_end_to_end(tmp_path, monkeypatch):
    raw_csv = tmp_path / "synthetic_raw.csv"
    _build_synthetic_transactions(n_weeks=18).to_csv(raw_csv, index=False)
    main = _patch_paths_to_tmp(monkeypatch, tmp_path, raw_csv)

    # Tránh ghi đè README.md gốc hoặc phụ thuộc file Olist raw lớn cho cold_start
    import src.evaluation.cold_start as cold_start_eval
    import src.evaluation.reporting as reporting_mod
    monkeypatch.setattr(reporting_mod, "sync_readme", lambda: None)

    def _fake_cold_start():
        df_eval = pd.DataFrame([{
            "Model": "ColdStartProductRegressor",
            "MAE_BRL": 10.0,
            "RMSE_BRL": 15.0,
            "R2_Score": 0.8,
            "Test_Count": 3,
        }])
        manifest = {"train_count": 5, "test_count": 3, "purged_count": 0}
        return None, df_eval, manifest

    monkeypatch.setattr(cold_start_eval, "train_and_evaluate", _fake_cold_start)
    monkeypatch.setattr(sys, "argv", ["main.py", "--fast-mode", "--skip-eda", "--skip-tuning", "--force-reprocess"])

    main.main()

    report_path = config.PIPELINE_REPORT_JSON
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"), parse_constant=_reject_nonstandard_json_constant)

    assert set(["run_metadata", "pre_training", "training", "post_training"]).issubset(data.keys())
    assert data["run_metadata"]["status"] == "completed"
    assert data["run_metadata"]["error"] is None
    assert data["run_metadata"]["cli_args"]["skip_tuning"] is True

    pre = data["pre_training"]
    assert pre["raw_data"]["n_rows"] > 0
    assert pre["eda"]["executed"] is False
    assert pre["temporal_split"]["train"]["date_max"] < pre["temporal_split"]["val"]["date_min"]
    assert pre["preprocessing"]["loaded_from_cache"] is False
    assert pre["preprocessing"]["pipeline"]["n_features"] > 0

    trn = data["training"]
    assert len(trn["models"]) == 9
    assert trn["hyperparameter_tuning"]["executed"] is False
    assert trn["validation_selection"]["selected_model"] in {m["Model"] for m in trn["validation_selection"]["validation_comparison"]}

    post = data["post_training"]
    assert len(post["test_evaluation"]["model_comparison"]) == 10
    assert len(post["error_analysis"]["top_20_worst_predictions"]) > 0
    assert post["cold_start_proxy"]["manifest"]["train_count"] == 5
    assert len(post["scenario_demos"]["marketing_campaign_demo"]["results"]) == 5
    assert post["readme_synced"] is True

    # Lần chạy thứ 2 không có --force-reprocess phải nạp cache nhưng vẫn xuất đủ JSON
    monkeypatch.setattr(sys, "argv", ["main.py", "--fast-mode", "--skip-eda", "--skip-tuning"])
    main.main()
    cached_data = json.loads(report_path.read_text(encoding="utf-8"), parse_constant=_reject_nonstandard_json_constant)
    assert cached_data["run_metadata"]["status"] == "completed"
    assert cached_data["pre_training"]["preprocessing"]["loaded_from_cache"] is True
    assert cached_data["pre_training"]["preprocessing"]["pipeline"]["n_features"] == pre["preprocessing"]["pipeline"]["n_features"]


def test_step_7_tuning_populates_report_context(tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path)
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.log1p(np.arange(20, dtype=float))
    ctx = {}
    _, params = main.step_7_hyperparameter_tuning(
        X, y, X[:4], y[:4], skip_tuning=False, fast_mode=False, report_context=ctx
    )
    assert ctx["executed"] is True
    assert ctx["trials_count"] == 18
    assert len(ctx["tuning_history"]) == 18
    assert set(ctx["best_params_per_family"].keys()) == {"Random Forest", "Decision Tree", "Gradient Boosting"}
    assert ctx["best_params_overall"] == params


def test_main_pipeline_saves_partial_report_on_failure(tmp_path, monkeypatch):
    raw_csv = tmp_path / "synthetic_raw.csv"
    _build_synthetic_transactions(n_weeks=18).to_csv(raw_csv, index=False)
    main = _patch_paths_to_tmp(monkeypatch, tmp_path, raw_csv)
    monkeypatch.setattr(sys, "argv", ["main.py", "--fast-mode", "--skip-eda", "--skip-tuning"])

    def _boom(*args, **kwargs):
        raise RuntimeError("Simulated training failure")

    monkeypatch.setattr(main, "step_6_train_models", _boom)

    with pytest.raises(RuntimeError, match="Simulated training failure"):
        main.main()

    assert config.PIPELINE_REPORT_JSON.exists()
    data = json.loads(
        config.PIPELINE_REPORT_JSON.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_json_constant,
    )
    assert data["run_metadata"]["status"] == "failed"
    assert data["run_metadata"]["error"]["type"] == "RuntimeError"
    assert "pre_training" in data and "preprocessing" in data["pre_training"]
