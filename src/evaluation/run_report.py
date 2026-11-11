"""
run_report.py - Tổng hợp và xuất báo cáo JSON cho toàn bộ vòng đời chạy pipeline
================================================================================
Ghi lại đầy đủ kết quả 3 giai đoạn trong một file JSON chuẩn (strict RFC 8259):
  - pre_training: Nạp dữ liệu thô, làm sạch, weekly panel, EDA, Temporal Split 8:1:1, Preprocessing
  - training: Lấy mẫu, cấu hình & thời gian huấn luyện mô hình/baseline, Tuning, Validation Selection
  - post_training: Đánh giá Test (BRL), phân nhóm nghiệp vụ, phân tích lỗi, Cold-Start, Persistence & Demo
"""
import inspect
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src import config


def format_report_path(path: Any) -> Optional[str]:
    """Chuẩn hóa đường dẫn sang dạng tương đối (nếu thuộc thư mục dự án) để portable."""
    if path is None:
        return None
    p = Path(path)
    try:
        return p.resolve().relative_to(config.BASE_DIR.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def to_json_safe(value: Any) -> Any:
    """Chuyển đổi đệ quy kiểu NumPy/Pandas/Path và loại bỏ NaN/Infinity thành kiểu JSON hợp lệ."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Path):
        return format_report_path(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return [to_json_safe(record) for record in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return to_json_safe(value.to_dict())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        fval = float(value)
        return fval if math.isfinite(fval) else None
    if isinstance(value, np.ndarray):
        return [to_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def summarize_numeric_values(values: Any) -> Dict[str, Any]:
    """Tính thống kê tóm tắt phân bố cho một mảng/cột số học 1D."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
            "sum": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
    }


def summarize_dataset_split(
    df: pd.DataFrame,
    timestamp_col: str = config.TIMESTAMP_COLUMN,
    target_col: str = config.TARGET_COLUMN,
    total_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Tạo bản tóm tắt kích thước, khoảng thời gian và phân bố doanh thu cho một DataFrame."""
    n_rows = int(len(df))
    denom = total_rows if total_rows is not None and total_rows > 0 else n_rows
    ratio = float(n_rows / denom) if denom > 0 else 0.0

    date_min = None
    date_max = None
    n_unique_weeks = None
    if timestamp_col in df.columns and n_rows > 0:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce").dropna()
        if not ts.empty:
            date_min = ts.min().isoformat()
            date_max = ts.max().isoformat()
            n_unique_weeks = int(ts.nunique())

    zero_count = None
    zero_ratio = None
    target_stats = None
    if target_col in df.columns and n_rows > 0:
        sales = pd.to_numeric(df[target_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if sales.size > 0:
            zero_count = int(np.sum(sales == 0.0))
            zero_ratio = float(np.mean(sales == 0.0))
            target_stats = summarize_numeric_values(sales)

    n_products = int(df["product_id"].nunique()) if "product_id" in df.columns else None

    return {
        "n_rows": n_rows,
        "ratio": ratio,
        "n_unique_products": n_products,
        "n_unique_weeks": n_unique_weeks,
        "date_min": date_min,
        "date_max": date_max,
        "zero_sales_count": zero_count,
        "zero_sales_ratio": zero_ratio,
        "target_stats_brl": target_stats,
    }


def extract_model_hyperparameters(model: Any) -> Dict[str, Any]:
    """Trích xuất tham số cấu hình khởi tạo (__init__) của mô hình From-Scratch."""
    params: Dict[str, Any] = {}
    try:
        sig = inspect.signature(model.__init__)
    except (TypeError, ValueError):
        return params

    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        val = getattr(model, name, param.default if param.default is not inspect.Parameter.empty else None)
        params[name] = to_json_safe(val)
    return params


def summarize_pipeline(pipeline: Any) -> Dict[str, Any]:
    """Trích xuất cấu hình và tham số đã fit trên Train của CustomDataPipeline."""
    if pipeline is None:
        return {}

    num_cols = list(getattr(pipeline, "num_cols", []) or [])
    cat_cols = list(getattr(pipeline, "cat_cols", []) or [])
    feature_names = list(getattr(pipeline, "feature_names_", []) or [])

    imputer = getattr(pipeline, "imputer", None)
    scaler = getattr(pipeline, "scaler", None)

    scaler_stats: Dict[str, Dict[str, Any]] = {}
    if scaler is not None:
        means = getattr(scaler, "mean_", None)
        stds = getattr(scaler, "std_", None)
        if means is not None and stds is not None and len(means) == len(num_cols):
            for col, m_val, s_val in zip(num_cols, means, stds):
                scaler_stats[col] = {"mean": float(m_val), "std": float(s_val)}

    return {
        "use_log_target": bool(getattr(pipeline, "use_log_target", True)),
        "forecast_granularity": getattr(pipeline, "forecast_granularity_", None),
        "default_forecast_week": getattr(pipeline, "default_forecast_week_", None),
        "n_num_cols": len(num_cols),
        "n_cat_cols": len(cat_cols),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "outlier_cols": list(getattr(pipeline, "outlier_cols", []) or []),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "imputer": {
            "num_medians": dict(getattr(imputer, "num_medians_", {}) or {}) if imputer is not None else {},
            "cat_modes": dict(getattr(imputer, "cat_modes_", {}) or {}) if imputer is not None else {},
        },
        "scaler_stats": scaler_stats,
    }


def summarize_residual_errors(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, Any]:
    """Tóm tắt phân bố sai số và tỷ lệ dự báo thiếu/thừa trên thang BRL."""
    y_t = np.asarray(y_true_log, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred_log, dtype=np.float64).ravel()
    valid = np.isfinite(y_t) & np.isfinite(y_p)
    y_t_valid = y_t[valid]
    y_p_valid = y_p[valid]

    true_brl = np.maximum(0.0, np.expm1(np.clip(y_t_valid, -1.0, 500.0)))
    pred_brl = np.maximum(0.0, np.expm1(np.clip(y_p_valid, -1.0, 500.0)))

    signed_err = pred_brl - true_brl
    abs_err = np.abs(true_brl - pred_brl)
    n = int(true_brl.size)

    return {
        "n_samples": n,
        "non_finite_count": int(y_t.size - n),
        "mean_error_brl": float(np.mean(signed_err)) if n > 0 else None,
        "std_error_brl": float(np.std(signed_err)) if n > 0 else None,
        "over_forecast_rate": float(np.mean(signed_err > 1e-9)) if n > 0 else None,
        "under_forecast_rate": float(np.mean(signed_err < -1e-9)) if n > 0 else None,
        "abs_error_stats_brl": summarize_numeric_values(abs_err),
    }


def build_trained_models_summary(trained_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Tổng hợp metadata cấu hình và thời gian huấn luyện của từng mô hình/baseline."""
    records = []
    for name, item in trained_dict.items():
        model = item["model"]
        role = "baseline" if "Baseline" in name else "model"
        records.append({
            "name": name,
            "role": role,
            "class_name": model.__class__.__name__,
            "fit_time_seconds": float(item.get("fit_time", 0.0)),
            "hyperparameters": extract_model_hyperparameters(model),
        })
    return records


def extract_feature_importances(
    model: Any,
    feature_names: Optional[List[str]] = None,
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """Trích xuất Top N đặc trưng quan trọng nhất nếu mô hình hỗ trợ feature_importances_."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    imp_arr = np.asarray(importances, dtype=np.float64).ravel()
    if imp_arr.size == 0:
        return []

    if feature_names is not None and len(feature_names) == imp_arr.size:
        names = list(feature_names)
    else:
        names = [f"Feature_{i}" for i in range(imp_arr.size)]

    top_indices = np.argsort(imp_arr)[::-1][:top_n]
    return [
        {
            "rank": rank,
            "feature": names[int(idx)],
            "importance": float(imp_arr[int(idx)]),
        }
        for rank, idx in enumerate(top_indices, 1)
    ]


def create_initial_report(cli_args: Dict[str, Any], random_state: int) -> Dict[str, Any]:
    """Khởi tạo cấu trúc báo cáo 3 giai đoạn cho một lần chạy pipeline."""
    return {
        "run_metadata": {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "total_duration_seconds": None,
            "status": "running",
            "error": None,
            "cli_args": cli_args,
            "random_state": int(random_state),
        },
        "pre_training": {},
        "training": {},
        "post_training": {},
    }


def finalize_report(
    report: Dict[str, Any],
    status: str,
    total_duration_seconds: float,
    error: Optional[BaseException] = None,
) -> Dict[str, Any]:
    """Cập nhật trạng thái kết thúc, tổng thời gian thực thi và lỗi (nếu có)."""
    meta = report.setdefault("run_metadata", {})
    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["total_duration_seconds"] = round(float(total_duration_seconds), 4)
    meta["status"] = status
    if error is not None:
        meta["error"] = {"type": type(error).__name__, "message": str(error)}
    return report


def save_pipeline_report(report: Dict[str, Any], filepath: Optional[Path] = None) -> Path:
    """Tuần tự hóa báo cáo an toàn theo chuẩn JSON (strict) và ghi nguyên tử ra đĩa."""
    target_path = Path(filepath) if filepath is not None else config.PIPELINE_REPORT_JSON
    target_path.parent.mkdir(parents=True, exist_ok=True)

    safe_payload = to_json_safe(report)
    serialized = json.dumps(safe_payload, ensure_ascii=False, indent=2, allow_nan=False)

    tmp_path = target_path.with_suffix(".json.tmp")
    tmp_path.write_text(serialized + "\n", encoding="utf-8")
    tmp_path.replace(target_path)
    return target_path
