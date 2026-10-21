"""
hurdle.py - Mô hình Hai giai đoạn thuần NumPy (Two-Stage Hurdle Regressor From-Scratch)
==========================================================================================
Đặc trị bài toán Doanh thu thưa (Zero-Inflated Demand với >90% tuần bằng 0 BRL):
  - Giai đoạn 1 (Classification Stage): Ước lượng xác suất phát sinh giao dịch P(sales > 0 | x).
  - Giai đoạn 2: Hồi quy log1p doanh thu trên các mẫu dương.
  - Dự báo hợp nhất (không mặc nhiên là kỳ vọng số học):
      sales_hat = P(sales > 0 | x) * expm1(predicted_positive_log_sales)
      log_sales_hat = log(1 + sales_hat)
"""
from typing import Any, Optional
import numpy as np

from src.models.base import BaseRegressorFromScratch, _ensure_dense_2d, _safe_expm1
from src.models.tree import DecisionTreeRegressorFromScratch


class TwoStageHurdleRegressorFromScratch(BaseRegressorFromScratch):
    """
    Mô hình Hai giai đoạn (Hurdle / Two-Stage Model) thuần túy From Scratch bằng NumPy:
      1. Stage 1 (Occurrence Model): Dự báo xác suất có đơn hàng p_i = P(y > 0 | x).
         Sử dụng DecisionTreeRegressor học trên nhãn nhị phân z = 1(y > 0). Tại mỗi nút lá,
         giá trị trung bình mẫu chính là xác suất thực nghiệm rơi vào lớp dương: p in [0, 1].
      2. Stage 2 (Severity Model): Mô hình hồi quy DecisionTree chỉ huấn luyện trên tập các mẫu
         có doanh thu dương (y > 0).
      3. Kết hợp dự báo: sales_hat = p(x) * max(0, exp(y_pos_hat) - 1).
    """

    def __init__(
        self,
        classifier_max_depth: int = 5,
        regressor_max_depth: int = 6,
        min_samples_leaf: int = 5,
        n_quantiles: int = 16,
        random_state: int = 42
    ):
        self.classifier_max_depth = classifier_max_depth
        self.regressor_max_depth = regressor_max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_quantiles = n_quantiles
        self.random_state = random_state

        self.classifier_: Optional[DecisionTreeRegressorFromScratch] = None
        self.regressor_: Optional[DecisionTreeRegressorFromScratch] = None
        self.pos_fallback_value_: float = 0.0

    def fit(self, X: Any, y: Any) -> "TwoStageHurdleRegressorFromScratch":
        X_mat = _ensure_dense_2d(X)
        y_vec = np.asarray(y, dtype=np.float64).ravel()

        # Tạo nhãn nhị phân: 1 nếu doanh thu > 0, ngược lại 0
        is_positive = (y_vec > 0.0).astype(np.float64)

        # Giai đoạn 1: Huấn luyện bộ ước lượng xác suất bán được hàng
        self.classifier_ = DecisionTreeRegressorFromScratch(
            max_depth=self.classifier_max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_quantiles=self.n_quantiles,
            random_state=self.random_state
        )
        self.classifier_.fit(X_mat, is_positive)

        # Giai đoạn 2: Lọc các mẫu dương và huấn luyện bộ hồi quy điều kiện
        pos_mask = y_vec > 0.0
        n_pos = int(np.sum(pos_mask))
        y_pos = y_vec[pos_mask] if n_pos > 0 else np.array([], dtype=np.float64)

        if n_pos >= max(10, self.min_samples_leaf):
            X_pos = X_mat[pos_mask]
            self.pos_fallback_value_ = float(np.mean(y_pos))

            self.regressor_ = DecisionTreeRegressorFromScratch(
                max_depth=self.regressor_max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_quantiles=self.n_quantiles,
                random_state=self.random_state + 1
            )
            self.regressor_.fit(X_pos, y_pos)
        else:
            # Nếu số lượng mẫu dương quá ít để fit cây, fallback về trung bình của chính các mẫu dương
            if n_pos > 0:
                self.pos_fallback_value_ = float(np.mean(y_pos))
            else:
                self.pos_fallback_value_ = 0.0
            self.regressor_ = None

        return self

    def predict_proba_sale(self, X: Any) -> np.ndarray:
        """Dự báo xác suất phát sinh giao dịch P(sales > 0 | x) in [0, 1]."""
        if self.classifier_ is None:
            raise RuntimeError("Mô hình chưa được fit! Vui lòng gọi fit(X, y) trước.")
        probs = self.classifier_.predict(X)
        return np.clip(probs, 0.0, 1.0)

    def predict_positive_sales(self, X: Any) -> np.ndarray:
        """
        Dự báo độ lớn doanh thu điều kiện khi có đơn hàng (thang đo tiền tệ BRL).
        
        Lưu ý kỹ thuật & thống kê:
        Do cây hồi quy học trên nhãn log1p(sales), phép nghịch đảo expm1(y_hat_log)
        trả về nghịch đảo log của dự báo trong không gian log-transform.
        Theo Bất đẳng thức Jensen, giá trị này thấp hơn kỳ vọng số học E[Y|X],
        tạo thiên lệch bảo thủ (conservative bias) giúp mô hình ít bị méo bởi ngoại lai cực lớn.
        """
        if self.classifier_ is None:
            raise RuntimeError("Mô hình chưa được fit! Vui lòng gọi fit(X, y) trước.")

        if self.regressor_ is not None:
            pos_log_preds = self.regressor_.predict(X)
            pos_money = np.maximum(0.0, np.expm1(pos_log_preds))
        else:
            pos_money = np.full(X.shape[0], np.maximum(0.0, np.expm1(self.pos_fallback_value_)))

        return pos_money

    def predict(self, X: Any) -> np.ndarray:
        """
        Dự báo chỉ số doanh thu kết hợp và trả về dưới dạng log tự nhiên log1p(demand).
        
        Đại lượng kết hợp: demand = P(sale > 0 | x) * expm1(ŷ_log_pos(x)).
        Được so sánh bằng RMSE tiền tệ trên Validation; nhánh dương tối ưu MSE log,
        không phải trực tiếp tối ưu MSE tiền tệ của mô hình kết hợp.
        """
        prob_sale = self.predict_proba_sale(X)
        pos_sales = self.predict_positive_sales(X)

        # Doanh thu dự báo kết hợp: P(sale > 0) * Pos_Demand
        expected_sales = prob_sale * pos_sales
        expected_sales = np.maximum(0.0, expected_sales)

        return np.log1p(expected_sales)
