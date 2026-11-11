"""
main.py - Điểm vào (Entry Point) Điều Phối Toàn Bộ Quy Trình Dự Án Sales Predictor
====================================================================================
Thực thi trơn tru khép kín 10 bước chuẩn mực (From-Scratch 100% thuần NumPy):
  1. Nạp và Khám phá Dữ liệu Giao dịch Olist (src.data.loader)
  2. Lọc và kiểm tra hợp lệ dữ liệu giao dịch thô (src.data.builder)
  3. Tổng hợp bảng chuỗi thời gian sản phẩm-tuần (src.data.builder)
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
from typing import Dict, Any, Tuple, Optional
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
from src.evaluation import metrics, visualization, error_analysis, run_report
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
    """BƯỚC 2: Lọc và kiểm tra hợp lệ dữ liệu giao dịch thô (Data Sanitization, chưa tiền xử lý ML)."""
    print_section_header("Lọc và kiểm tra hợp lệ dữ liệu giao dịch thô", step_num=2)
    cleaned_df = builder.clean_weekly_source_data(df)
    logger.info(f"Nguồn hợp lệ: {len(cleaned_df):,} dòng sản phẩm; loại {len(df)-len(cleaned_df):,} dòng; không kẹp biến mục tiêu.")
    return cleaned_df

# Alias rõ nghĩa đúng bản chất kỹ thuật
step_2_sanitize_source_data = step_2_clean_and_preprocess


def step_3_feature_engineering(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """BƯỚC 3: Tổng hợp bảng chuỗi thời gian sản phẩm-tuần (Weekly Panel Construction, chưa fit tiền xử lý)."""
    print_section_header("Tổng hợp bảng chuỗi thời gian sản phẩm-tuần", step_num=3)
    featured_df = builder.build_weekly_sales_dataset(cleaned_df)
    logger.info(f"Dữ liệu tuần: {len(featured_df):,} mẫu sản phẩm-tuần; mục tiêu là doanh thu tuần kế tiếp (BRL); giữ nguyên tuần không bán.")
    return featured_df

# Alias rõ nghĩa đúng bản chất kỹ thuật
step_3_build_weekly_panel = step_3_feature_engineering


def step_4_eda_visualizations(
    df: pd.DataFrame,
    report_context: Optional[Dict[str, Any]] = None
) -> None:
    """BƯỚC 4: Trực quan hóa Khám phá Dữ liệu EDA trên Toàn bộ Bảng Dữ liệu Panel"""
    print_section_header("Trực quan hóa Khám phá Dữ liệu EDA (Toàn bộ Dữ liệu)", step_num=4)
    figs_dir = config.FIGURES_DIR
    nb_figs_dir = config.NOTEBOOK_FIGURES_DIR
    figs_dir.mkdir(parents=True, exist_ok=True)
    nb_figs_dir.mkdir(parents=True, exist_ok=True)

    if "last_week_sales" in df:
        import matplotlib.pyplot as plt
        visualization.plot_target_sales_distribution(
            df,
            save_path=[figs_dir / "weekly_target_distribution.png", nb_figs_dir / "weekly_target_distribution.png"]
        )
        totals = df.groupby(config.TIMESTAMP_COLUMN)[config.TARGET_COLUMN].sum()
        fig, ax = plt.subplots(figsize=(10, 4))
        totals.plot(ax=ax, title="Weekly product revenue across time (Full Panel)", ylabel="BRL")
        fig.tight_layout()
        fig.savefig(figs_dir / "weekly_revenue.png", dpi=config.FIGURE_DPI)
        fig.savefig(nb_figs_dir / "weekly_revenue.png", dpi=config.FIGURE_DPI)
        plt.close(fig)
        if report_context is not None:
            report_context["figures_saved"] = [
                run_report.format_report_path(figs_dir / "weekly_target_distribution.png"),
                run_report.format_report_path(figs_dir / "weekly_revenue.png"),
            ]
        logger.info(f"✓ Đã lưu biểu đồ EDA toàn cảnh vào: {figs_dir}")
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

    saved_figures = []
    for title, plot_fn, filename in eda_tasks:
        logger.info(f"Đang xuất biểu đồ EDA: [{title}]...")
        plot_fn(df, save_path=[figs_dir / filename, nb_figs_dir / filename])
        saved_figures.append(run_report.format_report_path(figs_dir / filename))

    if report_context is not None:
        report_context["figures_saved"] = saved_figures

    logger.info(f"✓ Đã lưu toàn bộ 7 biểu đồ EDA vào: {figs_dir}")


def step_split_temporal_data(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Phân chia Chuỗi thời gian 8:1:1 theo order_purchase_timestamp TRƯỚC khi tiền xử lý."""
    print_section_header("Phân chia Chuỗi Thời Gian 8:1:1 (Temporal Split)", step_num="Split")
    logger.info("Thực hiện phân chia Temporal Split 8:1:1 theo mốc thời gian order_purchase_timestamp...")
    train_df, val_df, test_df = preprocessing.weekly_split_data(df)

    logger.info(f"  • Tập Train (80%): {train_df.shape[0]:,} dòng (từ {train_df[config.TIMESTAMP_COLUMN].min()} đến {train_df[config.TIMESTAMP_COLUMN].max()})")
    logger.info(f"  • Tập Val   (10%): {val_df.shape[0]:,} dòng (từ {val_df[config.TIMESTAMP_COLUMN].min()} đến {val_df[config.TIMESTAMP_COLUMN].max()})")
    logger.info(f"  • Tập Test  (10%): {test_df.shape[0]:,} dòng (từ {test_df[config.TIMESTAMP_COLUMN].min()} đến {test_df[config.TIMESTAMP_COLUMN].max()})")

    # Xuất biểu đồ kiểm chứng phân chia chuỗi thời gian
    logger.info("Đang xuất biểu đồ: [Phân bố Mẫu Phân chia Chuỗi thời gian (Temporal Split 8:1:1)]...")
    visualization.plot_temporal_split_distribution(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        save_path=[config.FIGURES_DIR / "eda_temporal_split_distribution.png", config.NOTEBOOK_FIGURES_DIR / "eda_temporal_split_distribution.png"]
    )
    return train_df, val_df, test_df


def step_5_split_and_pipeline(
    df: Optional[pd.DataFrame] = None,
    force_reprocess: bool = False,
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    report_context: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Any, Any, np.ndarray, np.ndarray, np.ndarray, Any]:
    """BƯỚC 5: Tiền xử lý & Kỹ nghệ Đặc trưng (Fit DUY NHẤT trên Train, Transform sang Val & Test)"""
    print_section_header("Fit CustomDataPipeline trên Train & Transform Val/Test", step_num=5)

    if train_df is None or val_df is None or test_df is None:
        if df is None:
            raise ValueError("Cần cung cấp df hoặc đủ bộ 3 (train_df, val_df, test_df).")
        logger.info("Chưa chia trước tập dữ liệu. Thực hiện phân chia Temporal Split 8:1:1...")
        train_df, val_df, test_df = preprocessing.weekly_split_data(df)
        source_df_for_fp = df
    else:
        source_df_for_fp = df if df is not None else pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Kiểm tra xem đã có cache data đã xử lý chưa
    fingerprint = preprocessing_fingerprint(source_df_for_fp)
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
        if report_context is not None:
            report_context.update({"loaded_from_cache": True, "fingerprint": fingerprint})
        logger.info(f"✓ Nạp xong ma trận trong {time.time() - t0:.2f}s: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test, pipeline

    logger.info(f"  • Tập Train (80%): {train_df.shape[0]:,} dòng")
    logger.info(f"  • Tập Val   (10%): {val_df.shape[0]:,} dòng")
    logger.info(f"  • Tập Test  (10%): {test_df.shape[0]:,} dòng")

    # Fit Pipeline CHỈ TRÊN Train
    logger.info("Khởi tạo và Fit CustomDataPipeline DUY NHẤT trên tập Train (tránh rò rỉ dữ liệu)...")
    pipeline = preprocessing.CustomDataPipeline(
        num_cols=config.WEEKLY_NUMERICAL_COLS, cat_cols=config.WEEKLY_CATEGORICAL_COLS,
        outlier_cols=[], use_log_target=True,
        feature_engineer=feature_engineering.WeeklySalesFeatureEngineer())
    pipeline.fit(train_df)
    pipeline.forecast_granularity_ = "product_week"
    max_ts = max(train_df[config.TIMESTAMP_COLUMN].max(), test_df[config.TIMESTAMP_COLUMN].max())
    pipeline.default_forecast_week_ = str(max_ts + pd.Timedelta(weeks=1))

    logger.info("Transform độc lập sang Train, Val và Test bằng tham số đã học từ Train...")
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
    if report_context is not None:
        report_context.update({"loaded_from_cache": False, "fingerprint": fingerprint})
    logger.info(f"✓ Đã lưu cache tiền xử lý & artifacts notebook vào: {config.PROCESSED_DATA_DIR}")

    return X_train, X_val, X_test, y_train, y_val, y_test, pipeline



def step_6_train_models(
    X_train: Any,
    y_train: np.ndarray,
    fast_mode: bool = False,
    pipeline: Any = None,
    report_context: Optional[Dict[str, Any]] = None
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

    train_rows_available = int(X_train.shape[0])
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
    sampling_applied = bool(fast_mode and X_train.shape[0] > 20000)
    if sampling_applied:
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

    if report_context is not None:
        report_context.update({
            "fast_mode": fast_mode,
            "train_rows_available": train_rows_available,
            "train_rows_used": int(X_train.shape[0]),
            "sampling_applied": sampling_applied,
            "sampling_seed": config.RANDOM_STATE if sampling_applied else None,
        })

    return trained_models


def step_7_hyperparameter_tuning(
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
    skip_tuning: bool = False,
    fast_mode: bool = False,
    report_context: Optional[Dict[str, Any]] = None
) -> Any:
    """BƯỚC 7: Tối ưu Siêu tham số GridSearchCV trên Validation 10%"""
    print_section_header("Tối ưu Siêu Tham Số GridSearchCV trên Validation 10%", step_num=7)
    t_tune_start = time.perf_counter()
    if fast_mode and X_train.shape[0] > 20000:
        indices = np.random.default_rng(config.RANDOM_STATE).choice(X_train.shape[0], 20000, replace=False)
        X_train, y_train = X_train[indices], y_train[indices]

    if skip_tuning:
        logger.info("⏩ Bỏ qua bước GridSearch theo cấu hình CLI. Khởi tạo mô hình Random Forest tối ưu mặc định.")
        default_params = {"n_estimators": 10 if fast_mode else 20, "max_depth": 8, "min_samples_split": 10}
        best_rf = regressors.RandomForestRegressorFromScratch(
            n_estimators=default_params["n_estimators"],
            max_depth=default_params["max_depth"],
            min_samples_split=default_params["min_samples_split"],
            max_features="sqrt",
            random_state=42
        )
        best_rf.fit(X_train, y_train)
        if report_context is not None:
            val_sales = np.maximum(0.0, np.expm1(y_val))
            val_pred = np.maximum(0.0, np.expm1(best_rf.predict(X_val)))
            report_context.update({
                "executed": False,
                "scoring": "rmse_brl",
                "duration_seconds": round(time.perf_counter() - t_tune_start, 4),
                "train_rows_used": int(X_train.shape[0]),
                "param_grids": {},
                "trials_count": 0,
                "tuning_history": [],
                "best_params_per_family": {},
                "best_model_family": "Random Forest",
                "best_params_overall": default_params,
                "best_val_rmse_brl": metrics.compute_rmse(val_sales, val_pred),
            })
        return best_rf, default_params

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
    grids_searched = {"Random Forest": param_grid}
    for name, cls, grid in [
        ("Decision Tree", regressors.DecisionTreeRegressorFromScratch,
         {"max_depth": [3, 5, 8], "min_samples_leaf": [10, 30]}),
        ("Gradient Boosting", regressors.GradientBoostingRegressorFromScratch,
         {"n_estimators": [10] if fast_mode else [15, 30], "max_depth": [2, 3], "learning_rate": [0.1]})]:
        logger.info(f"Tìm tham số cho {name}: {grid}")
        grids_searched[name] = grid
        searches.append((name, tuning.grid_search_from_scratch(cls, grid, X_train, y_train,
                         X_val=X_val, y_val=y_val, scoring="rmse_brl", verbose=True)))
    history = [{"Model": name, **entry["params"], "Validation_RMSE_BRL": entry["score"]}
               for name, search in searches for entry in search["cv_results"]]
    pd.DataFrame(history).to_csv(config.METRICS_DIR / "tuning_history.csv", index=False)
    name, winner = min(searches, key=lambda pair: pair[1]["best_score"])
    best_overall = {**winner["best_params"], "model_family": name}
    if report_context is not None:
        report_context.update({
            "executed": True,
            "scoring": "rmse_brl",
            "duration_seconds": round(time.perf_counter() - t_tune_start, 4),
            "train_rows_used": int(X_train.shape[0]),
            "param_grids": grids_searched,
            "trials_count": len(history),
            "tuning_history": history,
            "best_params_per_family": {
                fam: {"best_params": s["best_params"], "best_val_rmse_brl": s["best_score"]}
                for fam, s in searches
            },
            "best_model_family": name,
            "best_params_overall": best_overall,
            "best_val_rmse_brl": winner["best_score"],
        })
    return winner["best_estimator"], best_overall


def step_8_evaluate_and_compare(
    trained_dict: Dict[str, Any],
    best_tuned_model: Any,
    X_test: Any,
    y_test: np.ndarray,
    report_context: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """BƯỚC 8: Nghiệm thu Độc lập trên Test 10% & Xuất Bảng So sánh Chỉ số"""
    print_section_header("Nghiệm thu Độc lập trên Test 10% & Đánh giá Chỉ số", step_num=8)

    # Báo sai số theo BRL; bản thân mô hình dự đoán log1p(sales).
    y_test = np.maximum(0.0, np.expm1(y_test))
    results_eval = {}
    for name, item in trained_dict.items():
        m = item["model"]
        y_pred = np.maximum(0.0, np.expm1(m.predict(X_test)))
        metrics_dict = metrics.compute_extended_metrics(y_test, y_pred)
        results_eval[name] = {
            "model": m,
            "metrics": metrics_dict,
            "y_pred": y_pred,
            "fit_time": item["fit_time"]
        }

    # Bổ sung kết quả của Tuned Random Forest
    y_pred_tuned = np.maximum(0.0, np.expm1(best_tuned_model.predict(X_test)))
    tuned_metrics = metrics.compute_extended_metrics(y_test, y_pred_tuned)
    results_eval["Tuned Model"] = {
        "model": best_tuned_model,
        "metrics": tuned_metrics,
        "y_pred": y_pred_tuned,
        "fit_time": 0.0
    }

    summary_df = metrics.build_summary_table(results_eval)
    summary_df["Target_Scale"] = "BRL"
    print("\n" + "=" * 70)
    print("🏆 BẢNG TỔNG HỢP SO SÁNH CHỈ SỐ HỒI QUY TRÊN TẬP TEST SET ĐỘC LẬP:")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70 + "\n")

    # Lưu kết quả
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")

    # Đánh giá nghiệp vụ phân nhóm và đo lường sai lệch thiếu/thừa (Under/Over-forecast)
    subgroup_dfs = []
    for m_name in ["Tuned Model", "Two-Stage Hurdle"]:
        if m_name in results_eval:
            sg_df = metrics.evaluate_business_subgroups(
                y_true=y_test,
                y_pred=results_eval[m_name]["y_pred"],
                model_name=m_name
            )
            subgroup_dfs.append(sg_df)
    all_subgroups_df = None
    if subgroup_dfs:
        all_subgroups_df = pd.concat(subgroup_dfs, ignore_index=True)
        subgroup_csv = config.METRICS_DIR / "business_subgroup_evaluation.csv"
        all_subgroups_df.to_csv(subgroup_csv, index=False, encoding="utf-8-sig")
        print("\n" + "=" * 70)
        print("📦 BẢNG ĐÁNH GIÁ NGHIỆP VỤ PHÂN NHÓM & DỰ BÁO THIẾU/THỪA (TEST SET):")
        print("=" * 70)
        print(all_subgroups_df.to_string(index=False))
        print("=" * 70 + "\n")

    # Trực quan hóa so sánh
    figs_dir = config.FIGURES_DIR
    logger.info("Xuất các biểu đồ đánh giá mô hình chuẩn dpi=300...")
    visualization.plot_model_comparison(summary_df, save_path=figs_dir / "model_comparison.png")
    visualization.plot_actual_vs_predicted(results_eval, y_test, save_path=figs_dir / "actual_vs_predicted.png")
    visualization.plot_residuals(results_eval, y_test, save_path=figs_dir / "residuals_plot.png")

    if report_context is not None:
        report_context.update({
            "target_scale": "BRL",
            "n_test_samples": int(len(y_test)),
            "model_comparison": summary_df.to_dict(orient="records"),
            "business_subgroup_evaluation": all_subgroups_df.to_dict(orient="records") if all_subgroups_df is not None else [],
            "artifacts_saved": [
                run_report.format_report_path(config.METRICS_DIR / "model_comparison.csv"),
                *(
                    [run_report.format_report_path(config.METRICS_DIR / "business_subgroup_evaluation.csv")]
                    if all_subgroups_df is not None else []
                ),
            ],
            "figures_saved": [
                run_report.format_report_path(figs_dir / "model_comparison.png"),
                run_report.format_report_path(figs_dir / "actual_vs_predicted.png"),
                run_report.format_report_path(figs_dir / "residuals_plot.png"),
            ],
        })

    return summary_df



def step_9_error_analysis(
    best_model: Any,
    X_test: Any,
    y_test: np.ndarray,
    feature_names: list,
    report_context: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """BƯỚC 9: Phân tích Phần dư & Mổ xẻ Top 20 Đơn hàng Sai lệch Lớn nhất"""
    print_section_header("Phân tích Phần dư & Top 20 Worst Predictions", step_num=9)

    y_pred = best_model.predict(X_test)
    worst_df = error_analysis.find_worst_predictions(
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        n=20,
        save_path=config.METRICS_DIR / "worst_predictions.csv"
    )

    logger.info(f"Đã lưu danh sách 20 ca sai số lớn nhất vào: {config.METRICS_DIR / 'worst_predictions.csv'}")

    # Vẽ feature importance
    artifacts_saved = [run_report.format_report_path(config.METRICS_DIR / "worst_predictions.csv")]
    if hasattr(best_model, "feature_importances_") and feature_names:
        logger.info("Xuất biểu đồ Feature Importance của mô hình tốt nhất...")
        visualization.plot_feature_importance(
            model=best_model,
            feature_names=feature_names,
            top_n=15,
            save_path=config.FIGURES_DIR / "feature_importance.png"
        )
        artifacts_saved.append(run_report.format_report_path(config.FIGURES_DIR / "feature_importance.png"))

    if report_context is not None:
        report_context.update({
            "selected_model_class": best_model.__class__.__name__,
            "residual_summary": run_report.summarize_residual_errors(y_test, y_pred),
            "top_20_worst_predictions": worst_df.to_dict(orient="records"),
            "top_feature_importances": run_report.extract_feature_importances(
                best_model, feature_names=feature_names, top_n=15
            ),
            "artifacts_saved": artifacts_saved,
        })

    return worst_df


def step_10_persistence_and_demo(
    best_model: Any,
    pipeline: Any,
    best_params: dict,
    X_test: Any,
    y_test: np.ndarray,
    report_context: Optional[Dict[str, Any]] = None
) -> None:
    """BƯỚC 10: Đóng gói Model Persistence & Demo 3 Kịch bản Kinh doanh"""
    print_section_header("Đóng gói Persistence & Demo 3 Kịch bản Kinh doanh", step_num=10)

    # Fixed candidate; temporal split purges labels overlapping Test origins.
    from src.evaluation.cold_start import train_and_evaluate
    cold_start_model, cold_eval_df, cold_manifest = train_and_evaluate()
    print("First-sale seven-day proxy (not listing-date cold start):")
    print(cold_eval_df.to_string(index=False))

    # 2. Đóng gói Artifact
    config.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.NOTEBOOK_SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target_pkl = config.SAVED_MODELS_DIR / "best_sales_model.pkl"
    nb_target_pkl = config.NOTEBOOK_SAVED_MODELS_DIR / "best_sales_model.pkl"

    metadata = {
        "target": "next_week_product_revenue_brl",
        "evaluation": "rolling_origin_one_week_ahead",
        "forecast_granularity": "product_week",
        "synthetic_marketing_used": False,
        "best_params": best_params,
        "n_features": int(X_test.shape[1]),
        "has_cold_start_model": cold_start_model is not None,
        "cold_start_evaluation": cold_manifest,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    persistence.save_artifacts(pipeline=pipeline, model=best_model, filepath=target_pkl, metadata=metadata, cold_start_model=cold_start_model)
    persistence.save_artifacts(pipeline=pipeline, model=best_model, filepath=nb_target_pkl, metadata=metadata, cold_start_model=cold_start_model)
    logger.info(f"💾 Đã đóng gói thành công Artifact vào: {target_pkl}")

    # 3. Demo kiểm chứng mẫu Test
    print("\n► DEMO 1: KIỂM CHỨNG SUY LUẬN TRÊN 5 MẪU TẬP TEST SET (ĐƠN VỊ: BRL):")
    demo_df = predictor.demo_prediction(model=best_model, pipeline=pipeline, X_test=X_test, y_test=y_test, n_samples=5)
    print(demo_df.to_string(index=False))

    # 4. Proxy sau giao dịch đầu; giá dưới đây giả định đã quan sát tại giao dịch đầu.
    print("\n► DEMO 2: PROXY 7 NGÀY TỪ GIAO DỊCH ĐẦU (BAO GỒM GIAO DỊCH ĐÃ BIẾT):")
    new_spec = {
        "product_category_name": "informatica_acessorios",
        "price": 180.0,
        "freight_value": 22.5,
        "product_weight_g": 650.0,
        "ad_spend": 350.0
    }
    forecast = predictor.predict_scenario_new_product(
        product_spec=new_spec,
        model=best_model,
        pipeline=pipeline,
        cold_start_model=cold_start_model
    )
    for k, v in forecast.items():
        print(f"  • {k:<25}: {v}")
    
    # Minh họa cùng proxy sau giao dịch đầu, chưa kiểm chứng trước niêm yết.
    cold_start_res = predictor.predict_cold_start_product(new_spec, cold_start_model=cold_start_model)
    print(f"  • {'cold_start_launch_sales':<25}: {cold_start_res['predicted_launch_sales_BRL']} BRL ({cold_start_res['estimated_launch_units']} sản phẩm) - Động cơ: {cold_start_res['model_engine']}")

    # 4. Kịch bản 2: Biến thiên Ngân sách Marketing
    print("\n► DEMO 3: KỊCH BẢN BIẾN THIÊN NGÂN SÁCH MARKETING (AD SPEND SCALING):")
    ad_spend_levels = [100.0, 500.0, 1000.0, 2000.0, 5000.0]
    mkt_df = predictor.predict_scenario_marketing_campaign(
        base_product=new_spec,
        ad_spend_levels=ad_spend_levels,
        model=best_model,
        pipeline=pipeline
    )
    print(mkt_df.to_string(index=False))

    print("\n► DEMO 3.1: MÔ PHỎNG PHẢN HỒI TIẾP THỊ KINH TẾ LƯỢNG (MARKETING UPLIFT & ROAS):")
    mkt_uplift_df = predictor.simulate_marketing_response(
        base_product=new_spec,
        ad_spend_levels=ad_spend_levels,
        model=best_model,
        pipeline=pipeline,
        elasticity=0.15
    )
    cols_to_show = ["Ad_Spend_BRL", "Predicted_Sales_BRL", "Incremental_Sales_BRL", "Sales_Growth_%", "ROAS", "Marketing_ROI_%"]
    print(mkt_uplift_df[cols_to_show].to_string(index=False))

    # 5. Kịch bản 3: Bảng tính Doanh thu theo Giả định Sản lượng (Discount Scenario Calculation)
    print("\n► DEMO 4: BẢNG TÍNH DOANH THU THEO GIẢ ĐỊNH SẢN LƯỢNG (DISCOUNT SCENARIO):")
    print("  (Lưu ý: Phép tính giả định số học, không phải dự báo tác động can thiệp của ML)")
    discount_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
    promo_df = predictor.calculate_discount_scenario(
        original_price=float(new_spec.get("price", 180.0)),
        discount_rates=discount_rates,
        assumed_units=10,
        reference_revenue=float(new_spec.get("price", 180.0)) * 10
    )
    print(promo_df.to_string(index=False))

    if report_context is not None:
        report_context.update({
            "cold_start_proxy": {
                "manifest": cold_manifest,
                "evaluation": cold_eval_df.to_dict(orient="records"),
            },
            "persistence": {
                "saved_artifact_paths": [
                    run_report.format_report_path(target_pkl),
                    run_report.format_report_path(nb_target_pkl),
                ],
                "metadata": metadata,
            },
            "scenario_demos": {
                "test_samples_demo": demo_df.to_dict(orient="records"),
                "new_product_proxy_demo": {
                    "product_spec": new_spec,
                    "weekly_proxy_forecast": forecast,
                    "cold_start_result": cold_start_res,
                },
                "marketing_campaign_demo": {
                    "base_product": new_spec,
                    "ad_spend_levels": ad_spend_levels,
                    "results": mkt_df.to_dict(orient="records"),
                },
                "marketing_uplift_demo": {
                    "base_product": new_spec,
                    "ad_spend_levels": ad_spend_levels,
                    "elasticity": 0.15,
                    "results": mkt_uplift_df.to_dict(orient="records"),
                },
                "discount_scenario_demo": {
                    "original_price": float(new_spec.get("price", 180.0)),
                    "discount_rates": discount_rates,
                    "assumed_units": 10,
                    "results": promo_df.to_dict(orient="records"),
                },
            },
        })


def main() -> None:
    """Hàm điều phối trung tâm của toàn bộ hệ thống"""
    args = parse_arguments()
    t_global_start = time.time()
    report = run_report.create_initial_report(
        cli_args={
            "fast_mode": bool(args.fast_mode),
            "skip_eda": bool(args.skip_eda),
            "skip_tuning": bool(args.skip_tuning),
            "force_reprocess": bool(args.force_reprocess),
        },
        random_state=config.RANDOM_STATE,
    )
    run_status = "completed"
    run_error: Optional[BaseException] = None

    print("\n" + "#" * 80)
    print("🚀 BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE PRODUCT SALES PREDICTOR (FROM-SCRATCH NUMPY) 🚀".center(80))
    print("#" * 80)
    logger.info(f"Cấu hình thực thi: Fast Mode={args.fast_mode}, Skip EDA={args.skip_eda}, Skip Tuning={args.skip_tuning}, Force Reprocess={args.force_reprocess}")

    try:
        # Bước 1: Nạp & Khám phá Dữ liệu
        t_step = time.perf_counter()
        raw_df = step_1_load_and_explore(config.RAW_DATA_FILE)
        report["pre_training"]["raw_data"] = {
            "file_path": run_report.format_report_path(config.RAW_DATA_FILE),
            "n_rows": int(raw_df.shape[0]),
            "n_cols": int(raw_df.shape[1]),
            "columns": list(raw_df.columns),
            "duration_seconds": round(time.perf_counter() - t_step, 4),
        }

        # Bước 2 & 3: Lọc giao dịch thô & Tổng hợp bảng chuỗi thời gian tuần (Weekly Panel)
        t_step = time.perf_counter()
        cleaned_items = step_2_clean_and_preprocess(raw_df)
        report["pre_training"]["sanitization"] = {
            "input_rows": int(len(raw_df)),
            "valid_rows": int(len(cleaned_items)),
            "dropped_rows": int(len(raw_df) - len(cleaned_items)),
            "duration_seconds": round(time.perf_counter() - t_step, 4),
        }

        t_step = time.perf_counter()
        cleaned_df = step_3_feature_engineering(cleaned_items)
        panel_summary = run_report.summarize_dataset_split(
            cleaned_df, total_rows=len(cleaned_df)
        )
        panel_summary.update({
            "n_cols": int(cleaned_df.shape[1]),
            "columns": list(cleaned_df.columns),
            "duration_seconds": round(time.perf_counter() - t_step, 4),
        })
        report["pre_training"]["weekly_panel"] = panel_summary

        # Bước 4: Khám phá Dữ liệu EDA trên TOÀN BỘ dữ liệu panel
        eda_ctx: Dict[str, Any] = {}
        t_step = time.perf_counter()
        if not args.skip_eda:
            logger.info("Thực hiện Khám phá Dữ liệu EDA trên toàn bộ bảng dữ liệu sản phẩm-tuần...")
            step_4_eda_visualizations(cleaned_df, report_context=eda_ctx)
            report["pre_training"]["eda"] = {
                "executed": True,
                "scope": "full_panel",
                "duration_seconds": round(time.perf_counter() - t_step, 4),
                "figures_saved": eda_ctx.get("figures_saved", []),
            }
        else:
            logger.info("⏩ Bỏ qua bước EDA theo cờ --skip-eda")
            report["pre_training"]["eda"] = {
                "executed": False,
                "scope": "full_panel",
                "duration_seconds": 0.0,
                "figures_saved": [],
            }

        # Phân chia Chuỗi thời gian 8:1:1 TRƯỚC khi tiền xử lý đặc trưng
        t_step = time.perf_counter()
        train_df, val_df, test_df = step_split_temporal_data(cleaned_df)
        total_panel_rows = len(cleaned_df)
        report["pre_training"]["temporal_split"] = {
            "method": "weekly_temporal_split",
            "target_ratios": {
                "train": config.TRAIN_SIZE,
                "val": config.VAL_SIZE,
                "test": config.TEST_SIZE,
            },
            "total_rows": int(total_panel_rows),
            "duration_seconds": round(time.perf_counter() - t_step, 4),
            "train": run_report.summarize_dataset_split(train_df, total_rows=total_panel_rows),
            "val": run_report.summarize_dataset_split(val_df, total_rows=total_panel_rows),
            "test": run_report.summarize_dataset_split(test_df, total_rows=total_panel_rows),
            "figures_saved": [
                run_report.format_report_path(config.FIGURES_DIR / "eda_temporal_split_distribution.png"),
                run_report.format_report_path(config.NOTEBOOK_FIGURES_DIR / "eda_temporal_split_distribution.png"),
            ],
        }

        # Bước 5: Tiền xử lý & Kỹ nghệ đặc trưng (Fit DUY NHẤT trên Train, Transform sang Val & Test)
        step5_ctx: Dict[str, Any] = {}
        t_step = time.perf_counter()
        X_train, X_val, X_test, y_train, y_val, y_test, pipeline = step_5_split_and_pipeline(
            df=cleaned_df,
            force_reprocess=args.force_reprocess,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            report_context=step5_ctx,
        )
        step5_dur = round(time.perf_counter() - t_step, 4)

        # Nạp danh sách tên đặc trưng nếu có
        feature_names = []
        if config.CACHE_FEATURE_NAMES.exists():
            with open(config.CACHE_FEATURE_NAMES, "r", encoding="utf-8") as f:
                feature_names = json.load(f)
        elif hasattr(pipeline, "feature_names_"):
            feature_names = list(pipeline.feature_names_)

        report["pre_training"]["preprocessing"] = {
            "loaded_from_cache": step5_ctx.get("loaded_from_cache"),
            "fingerprint": step5_ctx.get("fingerprint"),
            "duration_seconds": step5_dur,
            "pipeline": run_report.summarize_pipeline(pipeline),
            "matrices_shape": {
                "X_train": list(X_train.shape),
                "X_val": list(X_val.shape),
                "X_test": list(X_test.shape),
            },
            "target_log_stats": {
                "train": run_report.summarize_numeric_values(y_train),
                "val": run_report.summarize_numeric_values(y_val),
                "test": run_report.summarize_numeric_values(y_test),
            },
            "target_brl_stats": {
                "train": run_report.summarize_numeric_values(np.maximum(0.0, np.expm1(y_train))),
                "val": run_report.summarize_numeric_values(np.maximum(0.0, np.expm1(y_val))),
                "test": run_report.summarize_numeric_values(np.maximum(0.0, np.expm1(y_test))),
            },
        }

        # Bước 6: Huấn luyện các mô hình From Scratch
        step6_ctx: Dict[str, Any] = {}
        t_step = time.perf_counter()
        trained_dict = step_6_train_models(
            X_train, y_train, fast_mode=args.fast_mode, pipeline=pipeline, report_context=step6_ctx
        )
        step6_ctx["duration_seconds"] = round(time.perf_counter() - t_step, 4)
        report["training"]["sampling"] = step6_ctx
        report["training"]["models"] = run_report.build_trained_models_summary(trained_dict)

        # Bước 7: Tối ưu Siêu tham số trên Validation 10%
        step7_ctx: Dict[str, Any] = {}
        best_tuned_model, best_params = step_7_hyperparameter_tuning(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            skip_tuning=args.skip_tuning,
            fast_mode=args.fast_mode,
            report_context=step7_ctx,
        )
        report["training"]["hyperparameter_tuning"] = step7_ctx

        # Chỉ chọn mô hình bằng Validation trước khi xem kết quả Test độc lập.
        candidates = {name: item["model"] for name, item in trained_dict.items()}
        candidates["Tuned Model"] = best_tuned_model
        val_sales = np.maximum(0.0, np.expm1(y_val))
        val_extended = {}
        for name, model in candidates.items():
            val_pred_brl = np.maximum(0.0, np.expm1(model.predict(X_val)))
            val_extended[name] = {
                "metrics": metrics.compute_extended_metrics(val_sales, val_pred_brl),
                "fit_time": trained_dict[name]["fit_time"] if name in trained_dict else 0.0,
            }
        val_scores = {name: item["metrics"]["RMSE"] for name, item in val_extended.items()}
        selected_name = min(val_scores, key=val_scores.get)
        pd.DataFrame([{"Model": name, "Validation_RMSE_BRL": score,
                       "Selected": name == selected_name} for name, score in val_scores.items()]).to_csv(
            config.METRICS_DIR / "validation_comparison.csv", index=False)
        selected_model = candidates[selected_name]
        selected_params = best_params if selected_name == "Tuned Model" else {}
        selected_params = {**selected_params, "selected_model": selected_name,
                           "validation_rmse_brl": val_scores[selected_name],
                           "training_mode": "fast" if args.fast_mode else "full",
                           "skip_tuning": args.skip_tuning,
                           "train_rows_available": int(X_train.shape[0])}
        logger.info(f"Chọn mô hình bằng Validation RMSE (BRL): {selected_name} | {val_scores[selected_name]:.4f}")

        val_summary_df = metrics.build_summary_table(val_extended)
        val_summary_df["Selected"] = val_summary_df["Model"].eq(selected_name)
        report["training"]["validation_selection"] = {
            "selection_metric": "Validation_RMSE_BRL",
            "selected_model": selected_name,
            "selected_model_class": selected_model.__class__.__name__,
            "selected_params": selected_params,
            "validation_comparison": val_summary_df.to_dict(orient="records"),
        }

        # Bước 8: Đánh giá độc lập trên Test 10% & Xuất biểu đồ so sánh
        step8_ctx: Dict[str, Any] = {}
        t_step = time.perf_counter()
        step_8_evaluate_and_compare(
            trained_dict=trained_dict,
            best_tuned_model=best_tuned_model,
            X_test=X_test,
            y_test=y_test,
            report_context=step8_ctx,
        )
        step8_ctx["duration_seconds"] = round(time.perf_counter() - t_step, 4)
        report["post_training"]["test_evaluation"] = step8_ctx

        # Bước 9: Phân tích Phần dư & Top 20 Worst Predictions
        step9_ctx: Dict[str, Any] = {}
        t_step = time.perf_counter()
        step_9_error_analysis(
            best_model=selected_model,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            report_context=step9_ctx,
        )
        step9_ctx["duration_seconds"] = round(time.perf_counter() - t_step, 4)
        report["post_training"]["error_analysis"] = step9_ctx

        # Bước 10: Đóng gói Persistence & Demo 3 Kịch bản Kinh doanh
        step10_ctx: Dict[str, Any] = {}
        t_step = time.perf_counter()
        step_10_persistence_and_demo(
            best_model=selected_model,
            pipeline=pipeline,
            best_params=selected_params,
            X_test=X_test,
            y_test=y_test,
            report_context=step10_ctx,
        )
        step10_dur = round(time.perf_counter() - t_step, 4)
        report["post_training"]["cold_start_proxy"] = step10_ctx.get("cold_start_proxy", {})
        report["post_training"]["persistence"] = step10_ctx.get("persistence", {})
        report["post_training"]["scenario_demos"] = step10_ctx.get("scenario_demos", {})
        report["post_training"]["step_10_duration_seconds"] = step10_dur

        from src.evaluation.reporting import sync_readme
        sync_readme()
        report["post_training"]["readme_synced"] = True
    except Exception as exc:
        run_status = "failed"
        run_error = exc
        raise
    finally:
        total_time = time.time() - t_global_start
        run_report.finalize_report(
            report,
            status=run_status,
            total_duration_seconds=total_time,
            error=run_error,
        )
        report_path = run_report.save_pipeline_report(report)
        logger.info(f"📄 Đã lưu báo cáo JSON tổng hợp pipeline (trước/trong/sau train) vào: {report_path}")

    print("\n" + "#" * 80)
    print(f"🎉 TOÀN BỘ PIPELINE ĐÃ HOÀN TẤT THÀNH CÔNG TRONG {total_time:.2f} GIÂY! 🎉".center(80))
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
