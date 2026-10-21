"""
baselines.py - Hệ thống các mô hình hồi quy cơ sở (Baselines From-Scratch)
=============================================================================
Cung cấp 4 mô hình tham chiếu chuẩn mực (Rule 10: Baseline before complex models):
  1. ZeroRegressorFromScratch: Luôn dự báo 0 BRL (thích hợp với dữ liệu thưa zero-sales)
  2. RevenueMeanRegressorFromScratch: Dự báo trung bình doanh thu thực tế của tập Train
  3. MeanRegressorFromScratch: Dự báo hằng số bằng trung bình nhãn trong không gian log
  4. PreviousWeekRegressorFromScratch: Mô hình quán tính chuỗi thời gian (Persistence lag-1)
"""
from typing import Any
import numpy as np
import scipy.sparse as sp

from src.models.base import BaseRegressorFromScratch, _ensure_dense_2d


class ZeroRegressorFromScratch(BaseRegressorFromScratch):
    """Mốc so sánh không bán hàng, cần thiết khi đa số tuần có doanh thu 0."""

    def fit(self, X: Any, y: Any) -> "ZeroRegressorFromScratch":
        return self

    def predict(self, X: Any) -> np.ndarray:
        n_samples = X.shape[0]
        return np.zeros(n_samples, dtype=np.float64)


class RevenueMeanRegressorFromScratch(BaseRegressorFromScratch):
    """Dự báo trung bình doanh thu Train, trả về log để dùng chung giao diện."""

    def __init__(self):
        self.value_: float = 0.0

    def fit(self, X: Any, y: Any) -> "RevenueMeanRegressorFromScratch":
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        # Doanh thu thực tế: exp(y) - 1
        real_sales = np.maximum(0.0, np.expm1(y_arr))
        mean_sales = float(np.mean(real_sales))
        self.value_ = float(np.log1p(mean_sales))
        return self

    def predict(self, X: Any) -> np.ndarray:
        n_samples = X.shape[0]
        return np.full(n_samples, self.value_, dtype=np.float64)


class MeanRegressorFromScratch(BaseRegressorFromScratch):
    """Constant baseline fitted in the same target space as the other regressors."""

    def __init__(self):
        self.mean_: float = 0.0

    def fit(self, X: Any, y: Any) -> "MeanRegressorFromScratch":
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        self.mean_ = float(np.mean(y_arr))
        return self

    def predict(self, X: Any) -> np.ndarray:
        n_samples = X.shape[0]
        return np.full(n_samples, self.mean_, dtype=np.float64)


class PreviousWeekRegressorFromScratch(BaseRegressorFromScratch):
    """Persistence baseline: predict last completed week's revenue in log space."""

    def __init__(self, column_index: int, mean: float, std: float):
        self.column_index = column_index
        self.mean = mean
        self.std = std

    def fit(self, X: Any, y: Any) -> "PreviousWeekRegressorFromScratch":
        return self

    def predict(self, X: Any) -> np.ndarray:
        column = X[:, self.column_index]
        values = column.toarray().ravel() if sp.issparse(column) else np.asarray(column).ravel()
        # Nghịch đảo Z-Score về tiền tệ gốc rồi lấy log1p
        denorm_sales = np.maximum(0.0, values * (self.std + 1e-8) + self.mean)
        return np.log1p(denorm_sales)
