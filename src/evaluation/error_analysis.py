"""
error_analysis.py - Phân tích Lỗi Dự đoán Chuyên sâu (Error Analysis) From Scratch
==================================================================================
Nghiệm thu theo chuẩn mực kỹ thuật Task 3:
  - find_worst_predictions: Trích xuất Top N đơn hàng dự đoán sai lệch nhiều nhất
  - analyze_residuals: Phân tích tương quan giữa phần dư và các thuộc tính đầu vào
  - compare_error_patterns: Đối sánh phân bố sai số giữa các thuật toán hồi quy
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import scipy.sparse as sp

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config


def find_worst_predictions(
    X_test: Any,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n: int = 20,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Trích xuất Top N mẫu có sai số tuyệt đối lớn nhất |y - ŷ| trên tập Test.
    Hỗ trợ phát hiện các ngoại lai hoặc nhóm sản phẩm/đơn hàng dị biệt gây nhiễu.
    """
    y_t = np.asarray(y_test, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()

    true_sales = np.maximum(0.0, np.expm1(y_t))
    pred_sales = np.maximum(0.0, np.expm1(y_p))
    abs_error = np.abs(true_sales - pred_sales)
    rel_error_pct = (abs_error / np.maximum(true_sales, 1e-4)) * 100.0

    worst_indices = np.argsort(abs_error)[::-1][:n]

    records = []
    for rank, idx in enumerate(worst_indices, 1):
        rec = {
            "Rank": rank,
            "Sample_Index": int(idx),
            "Actual_Sales (log)": round(float(y_t[idx]), 4),
            "Predicted_Sales (log)": round(float(y_p[idx]), 4),
            "Absolute_Error_BRL": round(float(abs_error[idx]), 4),
            "Relative_Error (%)": f"{rel_error_pct[idx]:,.1f}%",
            "Actual_Sales (BRL)": round(float(true_sales[idx]), 2),
            "Predicted_Sales (BRL)": round(float(pred_sales[idx]), 2),
        }
        records.append(rec)

    worst_df = pd.DataFrame(records)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        worst_df.to_csv(p, index=False, encoding="utf-8-sig")

    return worst_df


def analyze_residuals(
    X_test: Any,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_correlated: int = 10
) -> pd.DataFrame:
    """
    Phân tích tương quan Pearson giữa phần dư e = y - ŷ với các đặc trưng đầu vào.
    Nếu phần dư có tương quan cao với một đặc trưng nào đó, chứng tỏ mô hình
    đang bỏ sót mối quan hệ phi tuyến của đặc trưng đó.
    """
    y_t = np.asarray(y_test, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    residuals = y_t - y_p

    if sp.issparse(X_test):
        X_mat = X_test.toarray()
    else:
        X_mat = np.asarray(X_test, dtype=float)

    n_feats = X_mat.shape[1]
    if feature_names is None or len(feature_names) != n_feats:
        names = [f"Feature_{i}" for i in range(n_feats)]
    else:
        names = list(feature_names)

    corrs = []
    res_std = np.std(residuals)
    if res_std > 1e-12:
        for j in range(n_feats):
            col = X_mat[:, j]
            col_std = np.std(col)
            if col_std > 1e-12:
                r = float(np.corrcoef(col, residuals)[0, 1])
                corrs.append({"Feature": names[j], "Correlation_With_Residuals": r, "Abs_Corr": abs(r)})
            else:
                corrs.append({"Feature": names[j], "Correlation_With_Residuals": 0.0, "Abs_Corr": 0.0})
    else:
        for name in names:
            corrs.append({"Feature": name, "Correlation_With_Residuals": 0.0, "Abs_Corr": 0.0})

    corr_df = pd.DataFrame(corrs).sort_values(by="Abs_Corr", ascending=False).head(top_correlated).reset_index(drop=True)
    return corr_df


def compare_error_patterns(
    results_dict: Dict[str, Dict[str, Any]],
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    So sánh ma trận tương quan giữa phần dư của các mô hình khác nhau.
    Nếu tương quan giữa 2 mô hình thấp, việc kết hợp Ensemble giữa chúng sẽ đạt hiệu quả vượt trội.
    """
    y_t = np.asarray(y_test, dtype=float).ravel()
    res_data = {}

    for m_name, res in results_dict.items():
        y_p = np.asarray(res["y_pred"], dtype=float).ravel()
        res_data[m_name] = y_t - y_p

    df_res = pd.DataFrame(res_data)
    return df_res.corr()
