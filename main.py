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
from src.evaluation import metrics, visualization, error_analysis
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


def step_8_evaluate_and_compare(
    trained_dict: Dict[str, Any],
    best_tuned_model: Any,
    X_test: Any,
    y_test: np.ndarray
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

    return summary_df



def step_9_error_analysis(
    best_model: Any,
    X_test: Any,
    y_test: np.ndarray,
    feature_names: list
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
    if hasattr(best_model, "feature_importances_") and feature_names:
        logger.info("Xuất biểu đồ Feature Importance của mô hình tốt nhất...")
        visualization.plot_feature_importance(
            model=best_model,
            feature_names=feature_names,
            top_n=15,
            save_path=config.FIGURES_DIR / "feature_importance.png"
        )

    return worst_df


def step_10_persistence_and_demo(
    best_model: Any,
    pipeline: Any,
    best_params: dict,
    X_test: Any,
    y_test: np.ndarray
) -> None:
    """BƯỚC 10: Đóng gói Model Persistence & Demo 3 Kịch bản Kinh doanh"""
    print_section_header("Đóng gói Persistence & Demo 3 Kịch bản Kinh doanh", step_num=10)

    # 1. Đóng gói Artifact
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
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    persistence.save_artifacts(pipeline=pipeline, model=best_model, filepath=target_pkl, metadata=metadata)
    persistence.save_artifacts(pipeline=pipeline, model=best_model, filepath=nb_target_pkl, metadata=metadata)
    logger.info(f"💾 Đã đóng gói thành công Artifact vào: {target_pkl}")

    # 2. Demo kiểm chứng mẫu Test
    print("\n► DEMO 1: KIỂM CHỨNG SUY LUẬN TRÊN 5 MẪU TẬP TEST SET (ĐƠN VỊ: BRL):")
    demo_df = predictor.demo_prediction(model=best_model, pipeline=pipeline, X_test=X_test, y_test=y_test, n_samples=5)
    print(demo_df.to_string(index=False))

    # 3. Kịch bản 1: Ra mắt Sản phẩm Mới
    print("\n► DEMO 2: KỊCH BẢN RA MẮT SẢN PHẨM MỚI (NEW PRODUCT LAUNCH):")
    new_spec = {
        "product_category_name": "informatica_acessorios",
        "price": 180.0,
        "freight_value": 22.5,
        "product_weight_g": 650.0,
        "ad_spend": 350.0
    }
    forecast = predictor.predict_scenario_new_product(product_spec=new_spec, model=best_model, pipeline=pipeline)
    for k, v in forecast.items():
        print(f"  • {k:<25}: {v}")

    # 4. Kịch bản 2: Biến thiên Ngân sách Marketing
    print("\n► DEMO 3: KỊCH BẢN BIẾN THIÊN NGÂN SÁCH MARKETING (AD SPEND SCALING):")
    mkt_df = predictor.predict_scenario_marketing_campaign(
        base_product=new_spec,
        ad_spend_levels=[100.0, 500.0, 1000.0, 2000.0, 5000.0],
        model=best_model,
        pipeline=pipeline
    )
    print(mkt_df.to_string(index=False))

    # 5. Kịch bản 3: Bảng tính Doanh thu theo Giả định Sản lượng (Discount Scenario Calculation)
    print("\n► DEMO 4: BẢNG TÍNH DOANH THU THEO GIẢ ĐỊNH SẢN LƯỢNG (DISCOUNT SCENARIO):")
    print("  (Lưu ý: Phép tính giả định số học, không phải dự báo tác động can thiệp của ML)")
    promo_df = predictor.calculate_discount_scenario(
        original_price=float(new_spec.get("price", 180.0)),
        discount_rates=[0.0, 0.1, 0.2, 0.3, 0.5],
        assumed_units=10,
        reference_revenue=float(new_spec.get("price", 180.0)) * 10
    )
    print(promo_df.to_string(index=False))


def main() -> None:
    """Hàm điều phối trung tâm của toàn bộ hệ thống"""
    args = parse_arguments()
    t_global_start = time.time()

    print("\n" + "#" * 80)
    print("🚀 BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE PRODUCT SALES PREDICTOR (FROM-SCRATCH NUMPY) 🚀".center(80))
    print("#" * 80)
    logger.info(f"Cấu hình thực thi: Fast Mode={args.fast_mode}, Skip EDA={args.skip_eda}, Skip Tuning={args.skip_tuning}, Force Reprocess={args.force_reprocess}")

    # Bước 1: Nạp & Khám phá Dữ liệu
    raw_df = step_1_load_and_explore(config.RAW_DATA_FILE)

    # Bước 2 & 3: Tiền xử lý & Kỹ nghệ đặc trưng
    cleaned_items = step_2_clean_and_preprocess(raw_df)
    cleaned_df = step_3_feature_engineering(cleaned_items)

    # Bước 4: EDA Biểu đồ phân tích
    if not args.skip_eda:
        eda_train, _, _ = preprocessing.weekly_split_data(cleaned_df)
        step_4_eda_visualizations(eda_train)
    else:
        logger.info("⏩ Bỏ qua bước EDA theo cờ --skip-eda")

    # Bước 5: Phân chia Temporal 8:1:1 & Fit Data Pipeline
    X_train, X_val, X_test, y_train, y_val, y_test, pipeline = step_5_split_and_pipeline(
        df=cleaned_df,
        force_reprocess=args.force_reprocess
    )

    # Nạp danh sách tên đặc trưng nếu có
    feature_names = []
    if config.CACHE_FEATURE_NAMES.exists():
        with open(config.CACHE_FEATURE_NAMES, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

    # Bước 6: Huấn luyện 4 mô hình From Scratch
    trained_dict = step_6_train_models(X_train, y_train, fast_mode=args.fast_mode, pipeline=pipeline)

    # Bước 7: Tối ưu Siêu tham số trên Validation 10%
    best_tuned_model, best_params = step_7_hyperparameter_tuning(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        skip_tuning=args.skip_tuning,
        fast_mode=args.fast_mode
    )

    # Chỉ chọn mô hình bằng Validation trước khi xem kết quả Test độc lập.
    candidates = {name: item["model"] for name, item in trained_dict.items()}
    candidates["Tuned Model"] = best_tuned_model
    val_sales = np.maximum(0.0, np.expm1(y_val))
    val_scores = {name: metrics.compute_rmse(val_sales, np.maximum(0.0, np.expm1(model.predict(X_val))))
                  for name, model in candidates.items()}
    selected_name = min(val_scores, key=val_scores.get)
    pd.DataFrame([{"Model": name, "Validation_RMSE_BRL": score,
                   "Selected": name == selected_name} for name, score in val_scores.items()]).to_csv(
        config.METRICS_DIR / "validation_comparison.csv", index=False)
    selected_model = candidates[selected_name]
    selected_params = best_params if selected_name == "Tuned Model" else {}
    selected_params = {**selected_params, "selected_model": selected_name,
                       "validation_rmse_brl": val_scores[selected_name]}
    logger.info(f"Chọn mô hình bằng Validation RMSE (BRL): {selected_name} | {val_scores[selected_name]:.4f}")

    # Bước 8: Đánh giá độc lập trên Test 10% & Xuất biểu đồ so sánh
    step_8_evaluate_and_compare(
        trained_dict=trained_dict,
        best_tuned_model=best_tuned_model,
        X_test=X_test,
        y_test=y_test
    )

    # Bước 9: Phân tích Phần dư & Top 20 Worst Predictions
    step_9_error_analysis(
        best_model=selected_model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names
    )

    # Bước 10: Đóng gói Persistence & Demo 3 Kịch bản Kinh doanh
    step_10_persistence_and_demo(
        best_model=selected_model,
        pipeline=pipeline,
        best_params=selected_params,
        X_test=X_test,
        y_test=y_test
    )

    total_time = time.time() - t_global_start
    print("\n" + "#" * 80)
    print(f"🎉 TOÀN BỘ PIPELINE ĐÃ HOÀN TẤT THÀNH CÔNG RỰC RỠ TRONG {total_time:.2f} GIÂY! 🎉".center(80))
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
