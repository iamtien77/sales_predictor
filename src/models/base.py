"""
base.py - Lớp cơ sở và các hàm tiện ích cốt lõi cho mô hình From-Scratch thuần NumPy
======================================================================================
Cung cấp:
  - BaseRegressorFromScratch: Lớp cơ sở trừu tượng chuẩn mực
  - _ensure_dense_2d: Chuẩn hóa ma trận đầu vào (hỗ trợ scipy.sparse và dense ndarray)
  - _safe_expm1: Chuyển đổi ngược log về tiền tệ an toàn, chống tràn số (overflow)
  - compute_mse, compute_mae, compute_rmse, compute_r2, compute_regression_metrics
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp


def _ensure_dense_2d(X: Any) -> np.ndarray:
    """Chuyển đổi dữ liệu ma trận đầu vào (kể cả sparse CSR) thành numpy 2D ndarray float64."""
    if sp.issparse(X):
        return X.toarray().astype(np.float64)
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _safe_expm1(y: np.ndarray, clip_max: Optional[float] = None) -> np.ndarray:
    """
    Nghịch đảo an toàn từ không gian log y = log(1 + sales) về thang tiền tệ BRL:
    sales = max(0, exp(y) - 1). Kẹp biên trên để chống nổ số học.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    if clip_max is not None:
        y_arr = np.clip(y_arr, -1.0, float(clip_max))
    return np.maximum(0.0, np.expm1(y_arr))


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error: MSE = (1/n) * sum((y - y_pred)^2)"""
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean((y_t - y_p) ** 2))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error: MAE = (1/n) * sum(|y - y_pred|)"""
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean(np.abs(y_t - y_p)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error: RMSE = sqrt(MSE)"""
    return float(np.sqrt(compute_mse(y_true, y_pred)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R-squared Score: R2 = 1 - (sum((y - y_pred)^2) / sum((y - mean(y))^2)).
    Nếu mẫu có phương sai bằng 0, trả về 0.0.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tổng hợp 4 chỉ số hồi quy tiêu chuẩn: MSE, MAE, RMSE, R²"""
    return {
        "MSE": compute_mse(y_true, y_pred),
        "MAE": compute_mae(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "R2": compute_r2(y_true, y_pred),
    }


class BaseRegressorFromScratch(ABC):
    """Lớp cơ sở trừu tượng cho tất cả các mô hình hồi quy From-Scratch thuần NumPy."""

    @abstractmethod
    def fit(self, X: Any, y: Any) -> "BaseRegressorFromScratch":
        """Huấn luyện mô hình từ dữ liệu X và nhãn y."""
        pass

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Dự báo biến mục tiêu liên tục cho tập dữ liệu X."""
        pass
