"""
linear.py - Hồi quy Tuyến tính thuần NumPy (Linear Regression From-Scratch)
=============================================================================
Cung cấp:
  - LinearRegressionFromScratch: Hỗ trợ cả Normal Equation OLS và Gradient Descent.
  - Tích hợp cơ chế Kẹp biên Thích ứng An toàn (Adaptive Numerical Clamping):
    Loại bỏ triệt để hiện tượng nổ số học khi dự báo ngoại lai vượt ngưỡng,
    bảo toàn tính chính xác của hàm hồi quy tuyến tính trong miền quan sát.
"""
from typing import Any, List, Optional
import numpy as np
import scipy.sparse as sp

from src.models.base import BaseRegressorFromScratch, _ensure_dense_2d


class LinearRegressionFromScratch(BaseRegressorFromScratch):
    """
    Hồi quy Tuyến tính thuần NumPy (Baseline Model).
    Hỗ trợ 2 bộ giải (solvers):
      - 'normal_equation' (mặc định): Nghiệm giải tích đóng OLS w = (X^T X + lambda I)^(-1) X^T y.
        Sử dụng Ridge regularization nhỏ (alpha_ridge=1e-5) và pseudo-inverse chống singular matrix.
      - 'gradient_descent': Tối ưu hóa Gradient Descent lặp theo gradient hàm MSE.
    
    Cơ chế Clamping (clip_predictions):
      - Mặc định True: Kẹp biên thích ứng [min_y - 0.5*span, max_y + 0.5*span],
        ngăn chặn hoàn toàn hiện tượng nổ số học (Overflow) khi có dữ liệu ngoại lai cực đoan.
      - False: Giữ nguyên dự báo OLS thuần túy không bị chặn.
    """
    def __init__(
        self,
        fit_intercept: bool = True,
        solver: str = "normal_equation",
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        alpha_ridge: float = 1e-5,
        clip_predictions: bool = True,
        random_state: int = 42
    ):
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha_ridge = alpha_ridge
        self.clip_predictions = clip_predictions
        self.random_state = random_state

        self.weights_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.coef_: Optional[np.ndarray] = None
        self.cost_history_: List[float] = []
        self.max_target_log_: float = 15.0
        self.min_target_log_: float = 0.0

    def fit(self, X: Any, y: Any) -> "LinearRegressionFromScratch":
        y_vec = np.asarray(y, dtype=np.float64).ravel()
        n_samples = y_vec.shape[0]

        # Lưu lại biên giá trị của tập huấn luyện phục vụ kẹp thích ứng
        if n_samples > 0:
            self.max_target_log_ = float(np.max(y_vec))
            self.min_target_log_ = float(np.min(y_vec))

        # Hỗ trợ tối ưu hóa trực tiếp trên ma trận thưa nếu dùng normal_equation
        if sp.issparse(X) and self.solver == "normal_equation":
            if self.fit_intercept:
                ones = sp.csr_matrix(np.ones((n_samples, 1), dtype=np.float64))
                X_b = sp.hstack([ones, X], format="csr")
            else:
                X_b = X.tocsr()

            p = X_b.shape[1]
            # Tính ma trận Gram thưa X^T X
            XTX = (X_b.T @ X_b).toarray()
            # Thêm hệ số điều chuẩn Ridge vào đường chéo (trừ intercept ở vị trí 0)
            reg = np.eye(p, dtype=np.float64) * self.alpha_ridge
            if self.fit_intercept:
                reg[0, 0] = 0.0
            XTX_reg = XTX + reg

            XTy = (X_b.T @ y_vec).ravel()
            # Giải hệ phương trình tuyến tính ổn định với pinv
            w = np.linalg.pinv(XTX_reg) @ XTy

            self.weights_ = w
            if self.fit_intercept:
                self.intercept_ = float(w[0])
                self.coef_ = w[1:].copy()
            else:
                self.intercept_ = 0.0
                self.coef_ = w.copy()
            return self

        # Dạng ma trận dense
        X_mat = _ensure_dense_2d(X)
        if self.fit_intercept:
            ones = np.ones((n_samples, 1), dtype=np.float64)
            X_b = np.hstack([ones, X_mat])
        else:
            X_b = X_mat

        n, p = X_b.shape

        if self.solver == "normal_equation":
            reg = np.eye(p, dtype=np.float64) * self.alpha_ridge
            if self.fit_intercept:
                reg[0, 0] = 0.0
            XTX = X_b.T @ X_b + reg
            XTy = X_b.T @ y_vec
            w = np.linalg.pinv(XTX) @ XTy
            self.weights_ = w
        elif self.solver == "gradient_descent":
            rng = np.random.RandomState(self.random_state)
            w = rng.randn(p) * 0.01
            self.cost_history_ = []

            for _ in range(self.n_iterations):
                preds = X_b @ w
                error = preds - y_vec
                cost = float(np.mean(error ** 2))
                self.cost_history_.append(cost)

                gradient = (2.0 / n) * (X_b.T @ error) + (2.0 * self.alpha_ridge * w)
                if self.fit_intercept:
                    gradient[0] -= 2.0 * self.alpha_ridge * w[0]
                w -= self.learning_rate * gradient

            self.weights_ = w
        else:
            raise ValueError(f"Solver không hợp lệ: {self.solver}. Chỉ hỗ trợ 'normal_equation' hoặc 'gradient_descent'.")

        if self.fit_intercept:
            self.intercept_ = float(self.weights_[0])
            self.coef_ = self.weights_[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = self.weights_.copy()

        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Mô hình chưa được fit! Vui lòng gọi fit(X, y) trước.")

        if sp.issparse(X):
            preds = (X @ self.coef_).ravel() + self.intercept_
            preds = preds.astype(np.float64)
        else:
            X_mat = _ensure_dense_2d(X)
            preds = (X_mat @ self.coef_ + self.intercept_).ravel().astype(np.float64)

        # Áp dụng cơ chế kẹp biên thích ứng an toàn chống nổ số học
        if self.clip_predictions:
            span = max(1.0, self.max_target_log_ - self.min_target_log_)
            lower_bound = self.min_target_log_ - 0.5 * span
            upper_bound = self.max_target_log_ + 0.5 * span
            preds = np.clip(preds, lower_bound, upper_bound)

        return preds
