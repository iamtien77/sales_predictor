"""
metrics.py - Tính toán và tổng hợp các chỉ số đánh giá Hồi quy thuần NumPy
=============================================================================
Cung cấp:
  - compute_metrics(y_true, y_pred): Bộ 4 chỉ số tiêu chuẩn (MSE, MAE, RMSE, R²)
  - compute_wape(y_true, y_pred): Weighted Absolute Percentage Error (Chuẩn bán lẻ e-commerce)
  - compute_supply_chain_cost(y_true, y_pred): Ma trận chi phí tồn kho vs đứt hàng
  - compute_extended_metrics(y_true, y_pred): Bộ chỉ số mở rộng gồm cả tài chính
  - print_regression_reports(results)
  - build_summary_table(all_results)
"""
import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.regressors import (
    compute_mse,
    compute_mae,
    compute_rmse,
    compute_r2,
    compute_regression_metrics,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính toán 4 chỉ số hồi quy tiêu chuẩn: MSE, MAE, RMSE, R²"""
    return compute_regression_metrics(y_true, y_pred)


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error (WAPE):
      WAPE = sum(|y_true - y_pred|) / sum(y_true)
    Chỉ số tiêu chuẩn vàng trong chuỗi cung ứng thương mại điện tử,
    không bị lỗi chia cho 0 khi doanh thu thực tế bằng 0 (khắc phục nhược điểm của MAPE).
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    sum_actual = float(np.sum(y_t))
    sum_abs_err = float(np.sum(np.abs(y_t - y_p)))
    if sum_actual <= 1e-9:
        return 0.0 if sum_abs_err <= 1e-9 else 1.0
    return float(sum_abs_err / sum_actual)


def compute_supply_chain_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    understock_penalty: float = 10.0,
    overstock_holding_cost: float = 2.0
) -> Dict[str, float]:
    """
    Mô phỏng ma trận tổn thất tài chính thực tế trong quản trị tồn kho:
      - Dự báo thiếu (y_pred < y_true): Mất cơ hội bán hàng, phạt understock_penalty (mặc định 10 BRL / BRL thiếu)
      - Dự báo thừa (y_pred > y_true): Chi phí lưu kho và vốn ứ đọng, phạt overstock_holding_cost (mặc định 2 BRL / BRL thừa)
    
    Returns:
        Dict: {"total_cost", "understock_cost", "overstock_cost", "cost_per_item"}
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    diff = y_p - y_t

    # Dự báo thừa: diff > 0
    overstock_errors = np.maximum(0.0, diff)
    overstock_cost = float(np.sum(overstock_errors * overstock_holding_cost))

    # Dự báo thiếu: diff < 0
    understock_errors = np.maximum(0.0, -diff)
    understock_cost = float(np.sum(understock_errors * understock_penalty))

    total_cost = understock_cost + overstock_cost
    n_samples = max(1, len(y_t))

    return {
        "total_cost": total_cost,
        "understock_cost": understock_cost,
        "overstock_cost": overstock_cost,
        "cost_per_item": float(total_cost / n_samples),
    }


def compute_extended_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    understock_penalty: float = 10.0,
    overstock_holding_cost: float = 2.0
) -> Dict[str, float]:
    """Bộ chỉ số toàn diện tích hợp cả toán học hồi quy và tài chính chuỗi cung ứng."""
    base_m = compute_regression_metrics(y_true, y_pred)
    wape = compute_wape(y_true, y_pred)
    cost = compute_supply_chain_cost(
        y_true, y_pred,
        understock_penalty=understock_penalty,
        overstock_holding_cost=overstock_holding_cost
    )
    base_m["WAPE"] = wape
    base_m["Total_Cost_BRL"] = cost["total_cost"]
    base_m["Cost_Per_Item_BRL"] = cost["cost_per_item"]
    return base_m


def print_regression_reports(results: Dict[str, Dict[str, Any]]) -> None:
    """In báo cáo đánh giá hồi quy cho từng mô hình ra màn hình console"""
    print("\n" + "=" * 70)
    print(" 📊 BÁO CÁO HIỆU NĂNG CÁC MÔ HÌNH HỒI QUY (EVALUATION REPORT)")
    print("=" * 70)
    for model_name, res in results.items():
        m = res.get("metrics", {})
        print(f"\n🔹 MÔ HÌNH: {model_name.upper()}")
        print(f"   • MSE  (Mean Squared Error)     : {m.get('MSE', 0.0):.6f}")
        print(f"   • MAE  (Mean Absolute Error)    : {m.get('MAE', 0.0):.6f}")
        print(f"   • RMSE (Root Mean Squared Error): {m.get('RMSE', 0.0):.6f}")
        print(f"   • R²   (R-squared Score)        : {m.get('R2', 0.0):.4f} ({m.get('R2', 0.0)*100:.2f}%)")
        if "WAPE" in m:
            print(f"   • WAPE (Weighted Abs Pct Error) : {m.get('WAPE', 0.0):.4f} ({m.get('WAPE', 0.0)*100:.2f}%)")
        if "Cost_Per_Item_BRL" in m:
            print(f"   • Phí tổn kho/hụt hàng trung bình: {m.get('Cost_Per_Item_BRL', 0.0):.2f} BRL/item")
    print("=" * 70 + "\n")


def build_summary_table(all_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Tạo bảng DataFrame tổng hợp các chỉ số đánh giá, tự động sắp xếp theo RMSE tăng dần
    (Mô hình có RMSE thấp nhất được xếp đầu tiên làm Best Model).
    """
    rows = []
    for model_name, res in all_results.items():
        m = res.get("metrics", {})
        fit_time = res.get("fit_time", None)
        row = {
            "Model": model_name,
            "MSE": float(m.get("MSE", np.nan)),
            "MAE": float(m.get("MAE", np.nan)),
            "RMSE": float(m.get("RMSE", np.nan)),
            "R2": float(m.get("R2", np.nan)),
        }
        if "WAPE" in m:
            row["WAPE"] = float(m.get("WAPE", np.nan))
        if "Cost_Per_Item_BRL" in m:
            row["Cost_Per_Item_BRL"] = float(m.get("Cost_Per_Item_BRL", np.nan))
        if fit_time is not None:
            row["Fit Time (s)"] = round(float(fit_time), 2)
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    if "RMSE" in df_summary.columns:
        df_summary = df_summary.sort_values(by="RMSE", ascending=True).reset_index(drop=True)
    return df_summary


def compute_under_over_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Tính tổng sai lệch dự báo thiếu (Underforecast), thừa (Overforecast) và độ lệch tổng (Total Bias).
    
    Parameters:
    -----------
    y_true: Doanh thu thực tế (đơn vị BRL)
    y_pred: Doanh thu dự báo (đơn vị BRL)
    
    Returns:
    --------
    Dict chứa: total_underforecast, total_overforecast, total_bias
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()

    underforecast = np.maximum(yt - yp, 0.0)
    overforecast = np.maximum(yp - yt, 0.0)

    return {
        "total_underforecast": float(np.sum(underforecast)),
        "total_overforecast": float(np.sum(overforecast)),
        "total_bias": float(np.sum(yp) - np.sum(yt))
    }


def compute_asymmetric_penalty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    under_weight: float = 1.0,
    over_weight: float = 1.0
) -> float:
    """
    Tính điểm phạt bất đối xứng giả định (Asymmetric Penalty Score).
    Trong chuỗi cung ứng bán lẻ, phạt thiếu hàng thường lớn hơn phạt tồn hàng (ví dụ tỷ lệ 2:1 hoặc 5:1).
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()

    underforecast = np.maximum(yt - yp, 0.0)
    overforecast = np.maximum(yp - yt, 0.0)

    weighted_error = (under_weight * underforecast + over_weight * overforecast).mean()
    return float(weighted_error)


def evaluate_business_subgroups(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_positive_mask: Optional[np.ndarray] = None,
    is_zero_mask: Optional[np.ndarray] = None,
    is_unseen_mask: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> pd.DataFrame:
    """
    Đánh giá chi tiết hiệu năng mô hình trên 4 phân nhóm kinh doanh:
    1. Toàn bộ mẫu sản phẩm-tuần (All Samples)
    2. Tuần thực sự có phát sinh doanh số (Positive Sales Weeks: y > 0)
    3. Tuần không có doanh số (Zero Sales Weeks: y == 0)
    4. Nhóm sản phẩm chưa xuất hiện trong tập Train (Unseen Products)
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()

    if is_positive_mask is None:
        is_positive_mask = (yt > 0.0)
    if is_zero_mask is None:
        is_zero_mask = (yt <= 0.0)

    groups = [
        ("Toàn bộ sản phẩm-tuần (All Samples)", np.ones_like(yt, dtype=bool)),
        ("Tuần có doanh thu (y > 0)", is_positive_mask),
        ("Tuần không có doanh thu (y == 0)", is_zero_mask),
    ]
    if is_unseen_mask is not None:
        groups.append(("Sản phẩm chưa thấy ở Train (Unseen)", is_unseen_mask))

    rows = []
    for grp_name, mask in groups:
        n_samples = int(np.sum(mask))
        if n_samples == 0:
            continue

        sub_yt = yt[mask]
        sub_yp = yp[mask]

        rmse = compute_rmse(sub_yt, sub_yp)
        mae = compute_mae(sub_yt, sub_yp)
        var_y = float(np.var(sub_yt))
        r2 = compute_r2(sub_yt, sub_yp) if var_y > 1e-12 else np.nan

        uo = compute_under_over_forecast(sub_yt, sub_yp)
        p_1_1 = compute_asymmetric_penalty(sub_yt, sub_yp, 1.0, 1.0)
        p_2_1 = compute_asymmetric_penalty(sub_yt, sub_yp, 2.0, 1.0)
        p_5_1 = compute_asymmetric_penalty(sub_yt, sub_yp, 5.0, 1.0)

        rows.append({
            "Model": model_name,
            "Group": grp_name,
            "Sample_Count": n_samples,
            "RMSE_BRL": round(rmse, 2),
            "MAE_BRL": round(mae, 2),
            "R2_Score": round(r2, 4) if np.isfinite(r2) else "N/A (Var=0)",
            "Total_Underforecast_BRL": round(uo["total_underforecast"], 2),
            "Total_Overforecast_BRL": round(uo["total_overforecast"], 2),
            "Total_Bias_BRL": round(uo["total_bias"], 2),
            "Penalty_1_1": round(p_1_1, 2),
            "Penalty_2_1": round(p_2_1, 2),
            "Penalty_5_1": round(p_5_1, 2),
        })

    return pd.DataFrame(rows)

