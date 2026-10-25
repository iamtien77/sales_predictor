"""
main.py - Điểm vào (Entry Point) Điều Phối Toàn Bộ Quy Trình Dự Án Sales Predictor
====================================================================================
Thực thi trơn tru khép kín 10 bước chuẩn mực (From-Scratch 100% thuần NumPy):
  1. Nạp và Khám phá Dữ liệu Giao dịch Olist (src.data.loader)
  2. Làm sạch và kiểm tra nguồn item (src.data.builder)
  3. Tạo weekly panel, lag và rolling từ lịch sử (src.data.builder)
  4. Trực quan hóa Khám phá Dữ liệu EDA Đa chiều (src.evaluation.visualization)
  5. Phân chia Chuỗi thời gian 8:1:1 & Chuẩn hóa Ma trận Đặc trưng (Temporal Split)
  6. Huấn luyện 4 Thuật toán Hồi quy From-Scratch (src.models.regressors)
  7. Tối ưu Siêu tham số GridSearchCV trên Validation 10% (src.models.tuning)
  8. Nghiệm thu Độc lập trên Test 10% & Xuất Bảng So sánh Chỉ số (src.evaluation.metrics)
  9. Phân tích Phần dư & Mổ xẻ Top 20 Đơn hàng Sai lệch Lớn nhất (src.evaluation.error_analysis)
  10. Đóng gói Model Persistence & Demo 3 Kịch bản Kinh doanh (src.deployment.predictor)
"""
import os
import sys
import time
import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp

# 1. Đảm bảo nạp đúng package nội bộ src và cấu hình Unicode chuẩn UTF-8
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src import config
from src.utils.logger import setup_logger, log_step, print_section_header
from src.data import loader, preprocessing, feature_engineering, builder
from src.models import regressors, tuning, persistence
from src.evaluation import metrics, visualization, error_analysis, cold_start, reporting
from src.deployment import predictor

logger = setup_logger("sales_predictor")


def preprocessing_fingerprint(df: pd.DataFrame) -> str:
    """Vô hiệu hóa cache khi dữ liệu hoặc mã tiền xử lý thay đổi."""
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    digest.update(str(list(df.columns)).encode())
    for relative in ["src/config.py", "src/data/preprocessing.py", "src/data/feature_engineering.py", "src/data/builder.py", "main.py"]:
        digest.update((ROOT_DIR / relative).read_bytes())
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    """Thiết lập các cờ tham số dòng lệnh CLI cho pipeline"""
    parser = argparse.ArgumentParser(
        description="Bộ Điều phối Pipeline Hồi quy Dự đoán Doanh số From-Scratch (Olist Sales Predictor)"
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Chạy ở chế độ tăng tốc (dùng tham số nhỏ hơn để nghiệm thu nhanh toàn bộ pipeline trong < 30s)"
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Bỏ qua bước vẽ và xuất biểu đồ EDA ban đầu"
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Bỏ qua bước GridSearchCV để sử dụng cấu hình mô hình tối ưu đã biết"
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Buộc tiền xử lý lại từ tệp raw thay vì tận dụng ma trận đặc trưng trong cache data/processed"
    )
    return parser.parse_args()


def step_1_load_and_explore(data_path: Path) -> pd.DataFrame:
    """BƯỚC 1: Nạp và Khám phá Dữ liệu Giao dịch Olist"""
    print_section_header("Nạp và Khám phá Dữ liệu Giao dịch", step_num=1)
    df = loader.load_data(data_path)
    logger.info(f"Tập dữ liệu nạp thành công: {df.shape[0]:,} dòng × {df.shape[1]} cột")
    return df


def step_2_clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """BƯỚC 2: Làm sạch nguồn item trước aggregation, chưa fit thống kê."""
    print_section_header("Làm sạch và kiểm tra dữ liệu nguồn", step_num=2)
    cleaned_df = builder.clean_weekly_source_data(df)
    logger.info(f"Nguồn hợp lệ: {len(cleaned_df):,} dòng sản phẩm; loại {len(df)-len(cleaned_df):,} dòng; không kẹp biến mục tiêu.")
    return cleaned_df


def step_3_feature_engineering(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """BƯỚC 3: Aggregation sản phẩm–tuần và lịch sử lag/rolling."""
    print_section_header("Tạo weekly panel và đặc trưng lịch sử", step_num=3)
    featured_df = builder.build_weekly_sales_dataset(cleaned_df)
    logger.info(f"Dữ liệu tuần: {len(featured_df):,} mẫu sản phẩm–tuần; mục tiêu là doanh thu tuần kế tiếp (BRL); giữ nguyên tuần không bán.")
    return featured_df


def step_4_eda_visualizations(df: pd.DataFrame) -> None:
    """BƯỚC 4: Trực quan hóa Khám phá Dữ liệu EDA Đa chiều (8 Biểu đồ chuẩn mực)"""
    print_section_header("Trực quan hóa Khám phá Dữ liệu EDA", step_num=4)
    figs_dir = config.FIGURES_DIR
    nb_figs_dir = config.NOTEBOOK_FIGURES_DIR
    figs_dir.mkdir(parents=True, exist_ok=True)
    nb_figs_dir.mkdir(parents=True, exist_ok=True)

    if "last_week_sales" in df:
        import matplotlib.pyplot as plt
        visualization.plot_target_sales_distribution(df, save_path=figs_dir / "weekly_target_distribution.png")
        totals = df.groupby(config.TIMESTAMP_COLUMN)[config.TARGET_COLUMN].sum()
        fig, ax = plt.subplots(figsize=(10, 4))
        totals.plot(ax=ax, title="Train weekly product revenue", ylabel="BRL")
        fig.tight_layout()
        fig.savefig(figs_dir / "weekly_revenue.png", dpi=config.FIGURE_DPI)
        plt.close(fig)
        return

    eda_tasks = [
        ("Phân bố biến mục tiêu Doanh số (Sales & Log1p)", visualization.plot_target_sales_distribution, "eda_target_sales_distribution.png"),
        ("Biểu đồ Hộp phát hiện Ngoại lai (Outliers Boxplots)", visualization.plot_outliers_boxplots, "eda_outliers_boxplots.png"),
        ("Phân tích Mùa vụ & Xu hướng Thời gian (Seasonal Trends)", visualization.plot_seasonal_and_time_trends, "eda_seasonal_and_time_trends.png"),
        ("Mối quan hệ Hai biến Bivariate (Price/Freight vs Sales)", visualization.plot_bivariate_relationships, "eda_bivariate_relationships.png"),
        ("Phân tích Khách hàng & CLV/RFM Demographics", visualization.plot_customer_demographics, "eda_customer_demographics.png"),
        ("Top Danh mục Hàng hóa Đóng góp Doanh thu Cao nhất", visualization.plot_top_categories_sales, "eda_top_categories_sales.png"),
        ("Bản đồ Nhiệt Ma trận Tương quan (Correlation Heatmap)", visualization.plot_correlation_heatmap, "eda_correlation_heatmap.png"),
    ]

    for title, plot_fn, filename in eda_tasks:
        logger.info(f"Đang xuất biểu đồ EDA: [{title}]...")
        plot_fn(df, save_path=[figs_dir / filename, nb_figs_dir / filename])

    logger.info(f"✓ Đã lưu toàn bộ 7 biểu đồ EDA vào: {figs_dir}")


def step_5_split_and_pipeline(
    df: pd.DataFrame,
    force_reprocess: bool = False
) -> Tuple[Any, Any, Any, np.ndarray, np.ndarray, np.ndarray, Any]:
    """BƯỚC 5: Phân chia Chuỗi thời gian 8:1:1 & Chuẩn hóa Ma trận Đặc trưng"""
    print_section_header("Phân chia Chuỗi Thời Gian 8:1:1 & Fit Pipeline", step_num=5)

    # Kiểm tra xem đã có cache data đã xử lý chưa
    fingerprint = preprocessing_fingerprint(df)
    manifest_path = config.PROCESSED_DATA_DIR / "cache_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    has_cache = (
        config.CACHE_X_TRAIN.exists() and config.CACHE_Y_TRAIN.exists() and
        config.CACHE_X_VAL.exists() and config.CACHE_Y_VAL.exists() and
        config.CACHE_X_TEST.exists() and config.CACHE_Y_TEST.exists() and
        config.CACHE_PIPELINE.exists() and config.CACHE_FEATURE_NAMES.exists()
        and manifest.get("fingerprint") == fingerprint
    )

    if has_cache and not force_reprocess:
        logger.info("📁 Đã tìm thấy ma trận đặc trưng hợp lệ trong cache data/processed. Tải trực tiếp...")
        t0 = time.time()
        X_train = sp.load_npz(config.CACHE_X_TRAIN)
        y_train = np.load(config.CACHE_Y_TRAIN)
        X_val = sp.load_npz(config.CACHE_X_VAL)
        y_val = np.load(config.CACHE_Y_VAL)
        X_test = sp.load_npz(config.CACHE_X_TEST)
        y_test = np.load(config.CACHE_Y_TEST)
        with open(config.CACHE_PIPELINE, "rb") as f:
            import pickle
            pipeline = pickle.load(f)
        logger.info(f"✓ Nạp xong ma trận trong {time.time() - t0:.2f}s: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test, pipeline

    logger.info("Thực hiện phân chia Temporal Split 8:1:1 theo order_purchase_timestamp...")
    # Chia theo tuần target riêng biệt, không chia theo số dòng sản phẩm.
    train_df, val_df, test_df = preprocessing.weekly_split_data(df)

    logger.info(f"  • Tập Train (80%): {train_df.shape[0]:,} dòng")
    logger.info(f"  • Tập Val   (10%): {val_df.shape[0]:,} dòng")
    logger.info(f"  • Tập Test  (10%): {test_df.shape[0]:,} dòng")

    # Xuất biểu đồ kiểm chứng phân chia chuỗi thời gian
    logger.info("Đang xuất biểu đồ: [Phân bố Mẫu Phân chia Chuỗi thời gian (Temporal Split 8:1:1)]...")
    visualization.plot_temporal_split_distribution(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        save_path=[config.FIGURES_DIR / "eda_temporal_split_distribution.png", config.NOTEBOOK_FIGURES_DIR / "eda_temporal_split_distribution.png"]
    )

    # Fit Pipeline trên Train
    logger.info("Khởi tạo và Fit CustomDataPipeline trên tập Train...")
    pipeline = preprocessing.CustomDataPipeline(
        num_cols=config.WEEKLY_NUMERICAL_COLS, cat_cols=config.WEEKLY_CATEGORICAL_COLS,
        outlier_cols=[], use_log_target=True,
        feature_engineer=feature_engineering.WeeklySalesFeatureEngineer())
    pipeline.fit(train_df)
    pipeline.forecast_granularity_ = "product_week"
    pipeline.default_forecast_week_ = str(df[config.TIMESTAMP_COLUMN].max() + pd.Timedelta(weeks=1))

    res_tr = pipeline.transform(train_df)
    X_train = res_tr[0] if isinstance(res_tr, tuple) else res_tr
    res_v = pipeline.transform(val_df)
    X_val = res_v[0] if isinstance(res_v, tuple) else res_v
    res_te = pipeline.transform(test_df)
    X_test = res_te[0] if isinstance(res_te, tuple) else res_te

    y_train = np.log1p(np.maximum(0.0, train_df[config.TARGET_COLUMN].values.astype(float)))
    y_val = np.log1p(np.maximum(0.0, val_df[config.TARGET_COLUMN].values.astype(float)))
    y_test = np.log1p(np.maximum(0.0, test_df[config.TARGET_COLUMN].values.astype(float)))

    # Lưu cache
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_val, X_test = (sp.csr_matrix(X) for X in (X_train, X_val, X_test))
    sp.save_npz(config.CACHE_X_TRAIN, X_train)
    sp.save_npz(config.CACHE_X_VAL, X_val)
    sp.save_npz(config.CACHE_X_TEST, X_test)
    np.save(config.CACHE_Y_TRAIN, y_train)
    np.save(config.CACHE_Y_VAL, y_val)
    np.save(config.CACHE_Y_TEST, y_test)
    with open(config.CACHE_PIPELINE, "wb") as f:
        import pickle
        pickle.dump(pipeline, f)
    config.CACHE_FEATURE_NAMES.write_text(json.dumps(pipeline.feature_names_, ensure_ascii=False), encoding="utf-8")
    manifest_path.write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")

    # Đồng bộ artifacts phục vụ Notebook 04 và 05
    test_df[["product_id", config.TIMESTAMP_COLUMN]].to_csv(config.PROCESSED_DATA_DIR / "weekly_test_keys.csv.gz", index=False)
    positions = list(np.flatnonzero(test_df[config.TARGET_COLUMN].to_numpy() > 0)[:3]) + list(np.flatnonzero(test_df[config.TARGET_COLUMN].to_numpy() == 0)[:2])
    demo_rows = test_df.iloc[positions]
    demo_payload = {
        "fingerprint": fingerprint,
        "positions": [int(i) for i in positions],
        "rows": json.loads(demo_rows.to_json(orient="records", date_format="iso"))
    }
    (config.PROCESSED_DATA_DIR / "weekly_demo_rows.json").write_text(json.dumps(demo_payload, ensure_ascii=False), encoding="utf-8")
    logger.info(f"✓ Đã lưu cache tiền xử lý & artifacts notebook vào: {config.PROCESSED_DATA_DIR}")

    return X_train, X_val, X_test, y_train, y_val, y_test, pipeline


def step_6_train_models(
    X_train: Any,
    y_train: np.ndarray,
    fast_mode: bool = False,
    pipeline: Any = None
) -> Dict[str, Any]:
    """BƯỚC 6: Huấn luyện các Thuật toán Hồi quy From-Scratch (bao gồm Two-Stage Hurdle)"""
    print_section_header("Huấn luyện các Thuật Toán Hồi Quy From Scratch", step_num=6)

    n_trees = 10 if fast_mode else 15
    models = {
        "Linear Regression": regressors.LinearRegressionFromScratch(fit_intercept=True, solver="normal_equation"),
        "Decision Tree": regressors.DecisionTreeRegressorFromScratch(max_depth=5, min_samples_split=20, min_samples_leaf=10, n_quantiles=16),
        "Random Forest": regressors.RandomForestRegressorFromScratch(n_estimators=n_trees, max_depth=6, min_samples_split=20, min_samples_leaf=10, max_features="sqrt", random_state=42),
        "Gradient Boosting": regressors.GradientBoostingRegressorFromScratch(n_estimators=n_trees, learning_rate=0.1, max_depth=3, n_quantiles=16, random_state=42),
        "Two-Stage Hurdle": regressors.TwoStageHurdleRegressorFromScratch(
            classifier_max_depth=4 if fast_mode else 5,
            regressor_max_depth=5 if fast_mode else 6,
            min_samples_leaf=10,
            n_quantiles=16,
            random_state=42
        )
    }

    trained_models = {}
    for name, cls in [("Zero Baseline", regressors.ZeroRegressorFromScratch),
                      ("Revenue Mean Baseline", regressors.RevenueMeanRegressorFromScratch)]:
        trained_models[name] = {"model": cls().fit(X_train, y_train), "fit_time": 0.0}
    baseline = regressors.MeanRegressorFromScratch().fit(X_train, y_train)
    trained_models["Mean Baseline"] = {"model": baseline, "fit_time": 0.0}
    if pipeline is not None:
        lag_idx = pipeline.num_cols.index("last_week_sales")
        trained_models["Previous Week Baseline"] = {
            "model": regressors.PreviousWeekRegressorFromScratch(
                lag_idx, pipeline.scaler.mean_[lag_idx], pipeline.scaler.std_[lag_idx]),
            "fit_time": 0.0}
    if fast_mode and X_train.shape[0] > 20000:
        indices = np.random.default_rng(config.RANDOM_STATE).choice(X_train.shape[0], 20000, replace=False)
        X_train, y_train = X_train[indices], y_train[indices]
        logger.info("Chế độ nhanh: mô hình học trên 20.000 dòng Train được lấy mẫu tái lập; giữ đầy đủ Validation/Test.")
    for name, model in models.items():
        logger.info(f"► Đang huấn luyện: [{name}]...")
        t0 = time.time()
        model.fit(X_train, y_train)
        dur = time.time() - t0
        logger.info(f"  ✓ [{name}] hoàn tất trong {dur:.2f}s")
        trained_models[name] = {
            "model": model,
            "fit_time": dur
        }

    # Đồng bộ candidates phục vụ Notebook 04
    try:
        manifest_path = config.PROCESSED_DATA_DIR / "cache_manifest.json"
        cache_fingerprint = json.loads(manifest_path.read_text(encoding="utf-8")).get("fingerprint", "")
    except Exception:
        cache_fingerprint = ""
    candidate_bundle = {
        "trained": trained_models,
        "pipeline": pipeline,
        "fingerprint": cache_fingerprint,
        "fast_mode": fast_mode
    }
    config.NOTEBOOK_SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with (config.NOTEBOOK_SAVED_MODELS_DIR / "weekly_candidates.pkl").open("wb") as handle:
        import pickle
        pickle.dump(candidate_bundle, handle)

    return trained_models


def step_7_hyperparameter_tuning(
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
    skip_tuning: bool = False,
    fast_mode: bool = False
) -> Any:
    """BƯỚC 7: Tối ưu Siêu tham số GridSearchCV trên Validation 10%"""
    print_section_header("Tối ưu Siêu Tham Số GridSearchCV trên Validation 10%", step_num=7)
    if fast_mode and X_train.shape[0] > 20000:
        indices = np.random.default_rng(config.RANDOM_STATE).choice(X_train.shape[0], 20000, replace=False)
        X_train, y_train = X_train[indices], y_train[indices]

    if skip_tuning:
        logger.info("⏩ Bỏ qua bước GridSearch theo cấu hình CLI. Khởi tạo mô hình Random Forest tối ưu mặc định.")
        best_rf = regressors.RandomForestRegressorFromScratch(
            n_estimators=20 if not fast_mode else 10,
            max_depth=8,
            min_samples_split=10,
            max_features="sqrt",
            random_state=42
        )
        best_rf.fit(X_train, y_train)
        return best_rf, {"n_estimators": 10 if fast_mode else 20, "max_depth": 8, "min_samples_split": 10}

    param_grid = {
        "n_estimators": [10, 20] if not fast_mode else [10],
        "max_depth": [5, 8],
        "min_samples_split": [10, 20]
    }
    logger.info(f"Bắt đầu GridSearchCV From-Scratch cho Random Forest với lưới: {param_grid}")
    t0 = time.time()
    tune_res = tuning.grid_search_from_scratch(
        model_class=regressors.RandomForestRegressorFromScratch,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        scoring="rmse_brl",
        verbose=True
    )
    logger.info(f"✓ Hoàn tất Tuning trong {time.time() - t0:.2f}s! Best Params: {tune_res['best_params']} | Val RMSE: {tune_res['best_score']:.4f}")
    searches = [("Random Forest", tune_res)]
    for name, cls, grid in [
        ("Decision Tree", regressors.DecisionTreeRegressorFromScratch,
         {"max_depth": [3, 5, 8], "min_samples_leaf": [10, 30]}),
        ("Gradient Boosting", regressors.GradientBoostingRegressorFromScratch,
         {"n_estimators": [10] if fast_mode else [15, 30], "max_depth": [2, 3], "learning_rate": [0.1]})]:
        logger.info(f"Tìm tham số cho {name}: {grid}")
        searches.append((name, tuning.grid_search_from_scratch(cls, grid, X_train, y_train,
                         X_val=X_val, y_val=y_val, scoring="rmse_brl", verbose=True)))
    history = [{"Model": name, **entry["params"], "Validation_RMSE_BRL": entry["score"]}
               for name, search in searches for entry in search["cv_results"]]
    pd.DataFrame(history).to_csv(config.METRICS_DIR / "tuning_history.csv", index=False)
    name, winner = min(searches, key=lambda pair: pair[1]["best_score"])
    return winner["best_estimator"], {**winner["best_params"], "model_family": name}




def _predict_log(models: Dict[str, Any], X_test: Any) -> Dict[str, np.ndarray]:
    """Predict in the model's native log1p target space."""
    out = {}
    for name, model in models.items():
        pred = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
        if not np.isfinite(pred).all():
            raise ValueError(f"Model {name} returned NaN/Inf predictions.")
        out[name] = pred
    return out


def _feature_importance_frame(model: Any, feature_names: list[str]) -> pd.DataFrame | None:
    """Normalize model feature importance to the evaluation module contract."""
    values = getattr(model, "feature_importances_", None)
    if values is None or not feature_names:
        return None
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) != len(feature_names):
        logger.warning(
            "Feature importance length (%d) != feature count (%d); skipping figure.",
            len(values), len(feature_names),
        )
        return None
    return pd.DataFrame({"feature": feature_names, "importance": values})


def _build_evaluation_frame(test_df: pd.DataFrame, y_true_brl: np.ndarray,
                            selected_prediction_brl: np.ndarray) -> pd.DataFrame:
    """Build the error-analysis input while preserving product_id and week."""
    if len(test_df) != len(y_true_brl):
        raise ValueError("Test dataframe and target length are inconsistent.")
    week = test_df[config.TIMESTAMP_COLUMN]
    return pd.DataFrame({
        "product_id": test_df["product_id"].astype(str).to_numpy(),
        "week": week.astype(str).to_numpy(),
        "actual": np.asarray(y_true_brl, dtype=float),
        "prediction": np.asarray(selected_prediction_brl, dtype=float),
    })


def _load_optional_cold_start(path: str | None) -> pd.DataFrame | None:
    """Load cold-start predictions already handed over by Task 2/My.

    Expected columns: actual, prediction, has_history. No cold-start model is
    trained or recreated here.
    """
    if not path:
        return None
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Cold-start evaluation file not found: {source}")
    if source.suffix.lower() == ".parquet":
        df = pd.read_parquet(source)
    else:
        df = pd.read_csv(source)
    required = {"actual", "prediction", "has_history"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cold-start file missing columns: {sorted(missing)}")
    return df


def _run_my_step10_scenarios(model: Any, pipeline: Any, *, scenario_file: str | None = None) -> dict:
    """Coordinate My's deployment functions without duplicating their logic.

    predictor.py is the service boundary. If the functions are not yet present,
    the result records that fact instead of silently implementing a second copy.
    """
    if scenario_file:
        source = Path(scenario_file)
        if not source.exists():
            raise FileNotFoundError(f"Scenario file not found: {source}")
        if source.suffix.lower() == ".json":
            return json.loads(source.read_text(encoding="utf-8"))
        if source.suffix.lower() == ".csv":
            return pd.read_csv(source).to_dict(orient="records")

    results: dict[str, Any] = {}
    new_spec = {
        "product_category_name": "informatica_acessorios",
        "price": 180.0,
        "freight_value": 22.5,
        "product_weight_g": 650.0,
        "ad_spend": 350.0,
    }

    fn = getattr(predictor, "predict_scenario_new_product", None)
    if callable(fn):
        results["new_product_launch"] = fn(
            product_spec=new_spec, model=model, pipeline=pipeline
        )
    else:
        results["new_product_launch"] = {
            "status": "not_available",
            "message": "My's predictor.predict_scenario_new_product is not implemented yet.",
        }

    fn = getattr(predictor, "predict_scenario_marketing_campaign", None)
    if callable(fn):
        results["marketing_campaign"] = fn(
            base_product=new_spec,
            ad_spend_levels=[100.0, 500.0, 1000.0, 2000.0, 5000.0],
            model=model,
            pipeline=pipeline,
        )
        if isinstance(results["marketing_campaign"], pd.DataFrame):
            results["marketing_campaign"] = results["marketing_campaign"].to_dict(orient="records")
    else:
        results["marketing_campaign"] = {
            "status": "not_available",
            "message": "My's predictor.predict_scenario_marketing_campaign is not implemented yet.",
        }

    fn = getattr(predictor, "calculate_discount_scenario", None)
    if callable(fn):
        promo = fn(
            original_price=float(new_spec["price"]),
            discount_rates=[0.0, 0.1, 0.2, 0.3, 0.5],
            assumed_units=10,
            reference_revenue=float(new_spec["price"]) * 10,
        )
        results["black_friday_discount"] = (
            promo.to_dict(orient="records") if isinstance(promo, pd.DataFrame) else promo
        )
    else:
        results["black_friday_discount"] = {
            "status": "not_available",
            "message": "My's predictor.calculate_discount_scenario is not implemented yet.",
        }

    return results


def step_8_to_10_task3(
    *,
    trained_dict: Dict[str, Any],
    best_tuned_model: Any,
    best_params: dict,
    X_test: Any,
    y_test_log: np.ndarray,
    pipeline: Any,
    feature_names: list[str],
    cleaned_df: pd.DataFrame,
    cold_start_file: str | None = None,
    scenario_file: str | None = None,
    fast_mode: bool = False,
) -> dict:
    """Huyền's Task 3 orchestration for evaluation, figures and error analysis.

    Task 1-2 training/pipeline code is untouched. This function consumes their
    fitted artifacts and delegates business prediction scenarios to My's
    predictor service.
    """
    print_section_header("TASK 3 - Đánh giá, Trực quan hóa & Phân tích Sai số", step_num=8)
    task3_start = time.perf_counter()

    # Exactly 4 ML models + 4 baselines for T3.1.
    ml_names = ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
    baseline_names = ["Zero Baseline", "Revenue Mean Baseline", "Mean Baseline", "Previous Week Baseline"]

    missing_ml = [name for name in ml_names if name not in trained_dict]
    missing_baseline = [name for name in baseline_names if name not in trained_dict]
    if missing_ml or missing_baseline:
        raise RuntimeError(
            f"Task 3 requires 4 ML + 4 baselines. Missing ML={missing_ml}, baselines={missing_baseline}"
        )

    evaluation_models = {name: trained_dict[name]["model"] for name in ml_names}
    baseline_models = {name: trained_dict[name]["model"] for name in baseline_names}
    model_predictions_log = _predict_log(evaluation_models, X_test)
    baseline_predictions_log = _predict_log(baseline_models, X_test)

    y_true_brl = metrics.inverse_log_target(y_test_log)
    y_true_brl = np.maximum(y_true_brl, 0.0)

    # Select the detailed model using validation only, never by peeking at test.
    candidates = {name: item["model"] for name, item in trained_dict.items()}
    candidates["Tuned Model"] = best_tuned_model
    val_df = preprocessing.weekly_split_data(cleaned_df)[1]
    val_true_brl = np.maximum(val_df[config.TARGET_COLUMN].to_numpy(dtype=float), 0.0)
    val_scores = {}
    for name, model in candidates.items():
        val_X = pipeline.transform(val_df)
        if isinstance(val_X, tuple):
            val_X = val_X[0]
        pred_log = np.asarray(model.predict(val_X), dtype=float)
        val_scores[name] = metrics.rmse(val_true_brl, np.maximum(0.0, np.expm1(pred_log)))
    selected_name = min(val_scores, key=val_scores.get)
    selected_model = candidates[selected_name]

    # T3.1 - metrics on real BRL scale; inverse log exactly once inside metrics.
    metrics_df = metrics.compare_models_and_baselines(
        y_true_brl,
        model_predictions_log,
        baseline_predictions_log,
        model_predictions_are_log=True,
        baseline_predictions_are_log=True,
        output_dir=config.METRICS_DIR,
    )
    metrics.save_metrics_json(metrics_df, output_dir=config.METRICS_DIR)

    # Add optional Hurdle/Tuned evaluation without changing the required 4+4 table.
    extra_rows = []
    for name in ["Two-Stage Hurdle", "Tuned Model"]:
        if name in candidates:
            pred_log = np.asarray(candidates[name].predict(X_test), dtype=float)
            extra_rows.append(metrics.calculate_metrics(
                y_true_brl, pred_log, predictions_are_log=True,
                model_name=name, model_type="additional_model"
            ))
    if extra_rows:
        pd.DataFrame(extra_rows).to_csv(config.METRICS_DIR / "additional_model_metrics.csv", index=False)

    selected_pred_log = np.asarray(selected_model.predict(X_test), dtype=float)
    selected_pred_brl = np.maximum(0.0, np.expm1(selected_pred_log))
    evaluation_df = _build_evaluation_frame(
        preprocessing.weekly_split_data(cleaned_df)[2], y_true_brl, selected_pred_brl
    )

    # T3.3 - Top 20 + grouping + possible causes.
    top20 = error_analysis.extract_top_20_errors(evaluation_df)
    full_errors = error_analysis.build_error_table(evaluation_df)
    error_analysis.save_error_analysis(full_errors, config.METRICS_DIR)
    error_groups = error_analysis.summarize_error_groups(full_errors)

    # T3.2 - exactly 18 figures.
    feature_importance_df = _feature_importance_frame(selected_model, feature_names)
    figures = visualization.generate_all_figures(
        y_true_brl,
        selected_pred_brl,
        metrics_df,
        error_df=top20,
        feature_importance_df=feature_importance_df,
        output_dir=config.FIGURES_DIR,
    )
    visualization.validate_18_figures(config.FIGURES_DIR)

    # Cold-start: evaluate handed-over predictions only; no new algorithm.
    cold_start_df = _load_optional_cold_start(cold_start_file)
    cold_start_result = None
    if cold_start_df is not None:
        cold_start_result = cold_start.evaluate_cold_start(
            cold_start_df, output_dir=config.METRICS_DIR
        )

    # T3.4 coordination at step 10: call My's deployment service boundary.
    scenario_results = _run_my_step10_scenarios(
        selected_model, pipeline, scenario_file=scenario_file
    )

    # Reporting receives scenario output from My without importing listing logic.
    report = reporting.build_report(
        metrics_df=metrics_df,
        top_errors_df=top20,
        cold_start_df=cold_start_result,
        error_groups_df=error_groups,
        feature_importance_df=feature_importance_df,
        scenario_results=scenario_results,
        figure_dir=config.FIGURES_DIR,
    )
    report_paths = reporting.save_report(report, output_dir=config.METRICS_DIR)

    # Save a compact validation-selection record.
    pd.DataFrame([
        {"Model": name, "Validation_RMSE_BRL": score, "Selected": name == selected_name}
        for name, score in val_scores.items()
    ]).to_csv(config.METRICS_DIR / "validation_comparison.csv", index=False)

    # Persistence for My's predictor: save the already-fitted pipeline/model only.
    target_pkl = config.SAVED_MODELS_DIR / "best_sales_model.pkl"
    metadata = {
        "target": "next_week_product_revenue_brl",
        "evaluation": "rolling_origin_one_week_ahead",
        "forecast_granularity": "product_week",
        "target_scale": "BRL",
        "selected_model": selected_name,
        "best_params": best_params,
        "n_features": int(X_test.shape[1]),
        "task3_mode": "fast" if fast_mode else "full",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    persistence.save_artifacts(
        pipeline=pipeline,
        model=selected_model,
        filepath=target_pkl,
        metadata=metadata,
    )

    elapsed = time.perf_counter() - task3_start
    manifest = {
        "task": "Task 3",
        "mode": "fast" if fast_mode else "full",
        "runtime_seconds": elapsed,
        "selected_model": selected_name,
        "metrics_rows": int(len(metrics_df)),
        "required_model_count": 4,
        "required_baseline_count": 4,
        "figure_count": len(figures),
        "top20_count": len(top20),
        "cold_start_evaluated": cold_start_result is not None,
        "scenario_results_received": bool(scenario_results),
    }
    (config.METRICS_DIR / "task3_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    logger.info(
        "Task 3 hoàn tất: selected=%s, metrics=%d rows, figures=%d, top20=%d, runtime=%.2fs",
        selected_name, len(metrics_df), len(figures), len(top20), elapsed,
    )
    return {
        "metrics": metrics_df,
        "top20": top20,
        "error_groups": error_groups,
        "cold_start": cold_start_result,
        "figures": figures,
        "report": report,
        "report_paths": report_paths,
        "scenario_results": scenario_results,
        "selected_model": selected_name,
        "runtime_seconds": elapsed,
    }


def parse_task3_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sales Predictor - Task 3 evaluation orchestrator"
    )
    parser.add_argument("--mode", choices=["fast", "full"], default="full")
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--force-reprocess", action="store_true")
    parser.add_argument("--cold-start-file", default=None,
                        help="CSV/Parquet bàn giao từ Task 2/My: actual,prediction,has_history")
    parser.add_argument("--scenario-file", default=None,
                        help="JSON/CSV kết quả kịch bản bàn giao từ My (nếu đã chạy riêng)")
    return parser.parse_args()


def main() -> None:
    """Điểm vào duy nhất: giữ nguyên Task 1-2, Huyền điều phối Task 3."""
    args = parse_task3_arguments()
    fast_mode = args.mode == "fast"
    started = time.perf_counter()

    logger.info(
        "Run configuration: mode=%s skip_eda=%s skip_tuning=%s force_reprocess=%s",
        args.mode, args.skip_eda, args.skip_tuning, args.force_reprocess,
    )

    # ==================== TASK 1-2: GIỮ NGUYÊN ====================
    raw_df = step_1_load_and_explore(config.RAW_DATA_FILE)
    cleaned_items = step_2_clean_and_preprocess(raw_df)
    cleaned_df = step_3_feature_engineering(cleaned_items)

    if not args.skip_eda:
        eda_train, _, _ = preprocessing.weekly_split_data(cleaned_df)
        step_4_eda_visualizations(eda_train)

    X_train, X_val, X_test, y_train, y_val, y_test, pipeline = step_5_split_and_pipeline(
        df=cleaned_df, force_reprocess=args.force_reprocess
    )
    feature_names = []
    if config.CACHE_FEATURE_NAMES.exists():
        feature_names = json.loads(config.CACHE_FEATURE_NAMES.read_text(encoding="utf-8"))

    trained_dict = step_6_train_models(
        X_train, y_train, fast_mode=fast_mode, pipeline=pipeline
    )
    best_tuned_model, best_params = step_7_hyperparameter_tuning(
        X_train, y_train, X_val, y_val,
        skip_tuning=args.skip_tuning, fast_mode=fast_mode
    )

    # ==================== TASK 3: HUYỀN ====================
    task3 = step_8_to_10_task3(
        trained_dict=trained_dict,
        best_tuned_model=best_tuned_model,
        best_params=best_params,
        X_test=X_test,
        y_test_log=y_test,
        pipeline=pipeline,
        feature_names=feature_names,
        cleaned_df=cleaned_df,
        cold_start_file=args.cold_start_file,
        scenario_file=args.scenario_file,
        fast_mode=fast_mode,
    )

    total = time.perf_counter() - started
    print("\n" + "=" * 80)
    print("TASK 3 HOÀN TẤT")
    print(f"Mô hình chi tiết: {task3['selected_model']}")
    print(f"Metrics: {len(task3['metrics'])} dòng (4 ML + 4 baseline)")
    print(f"Biểu đồ: {len(task3['figures'])}/18")
    print(f"Top 20: {len(task3['top20'])}")
    print(f"Tổng runtime: {total:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
