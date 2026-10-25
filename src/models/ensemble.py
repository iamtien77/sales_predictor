"""
ensemble.py - Các phương pháp hồi quy tổ hợp thuần NumPy (Ensemble Regressors)
================================================================================
Cung cấp:
  - RandomForestRegressorFromScratch (Ensemble Bagging + Feature Subsampling)
  - GradientBoostingRegressorFromScratch (Ensemble Boosting tuần tự trên Residuals)
"""
from typing import Any, List, Optional, Union
import numpy as np

from src.models.base import BaseRegressorFromScratch, _ensure_dense_2d
from src.models.tree import DecisionTreeRegressorFromScratch


class RandomForestRegressorFromScratch(BaseRegressorFromScratch):
    """
    Rừng Ngẫu Nhiên Hồi Quy (Ensemble Bagging) thuần NumPy.
    - Bootstrap Sampling: Lấy mẫu ngẫu nhiên có hoàn lại n mẫu cho từng cây.
    - Feature Subsampling: Chỉ chọn k = sqrt(d) đặc trưng ở mỗi nút cây con.
    - Dự đoán: Lấy trung bình cộng dự đoán của B cây thành phần.
    """

    def __init__(
        self,
        n_estimators: int = 30,
        max_depth: Optional[int] = 6,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: Union[str, int, float] = "sqrt",
        bootstrap: bool = True,
        n_quantiles: int = 16,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_quantiles = n_quantiles
        self.random_state = random_state

        self.trees_: List[DecisionTreeRegressorFromScratch] = []
        self.n_features_in_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any) -> "RandomForestRegressorFromScratch":
        X_mat = _ensure_dense_2d(X)
        y_vec = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_mat.shape

        self.n_features_in_ = n_features
        self.trees_ = []
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_estimators):
            tree_seed = int(rng.randint(0, 1_000_000))
            tree = DecisionTreeRegressorFromScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_quantiles=self.n_quantiles,
                random_state=tree_seed
            )

            if self.bootstrap:
                boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
                tree.fit(X_mat[boot_idx], y_vec[boot_idx])
            else:
                tree.fit(X_mat, y_vec)

            self.trees_.append(tree)

        # Tính tổng hợp feature importances từ tất cả các cây
        all_importances = [t.feature_importances_ for t in self.trees_ if t.feature_importances_ is not None]
        if all_importances:
            mean_imp = np.mean(all_importances, axis=0)
            total = np.sum(mean_imp)
            self.feature_importances_ = mean_imp / total if total > 0 else np.zeros(n_features)
        else:
            self.feature_importances_ = np.zeros(n_features)

        return self

    def predict(self, X: Any) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("Mô hình chưa được fit! Vui lòng gọi fit(X, y) trước.")
        X_mat = _ensure_dense_2d(X)
        all_preds = np.zeros((len(self.trees_), X_mat.shape[0]), dtype=np.float64)

        for idx, tree in enumerate(self.trees_):
            all_preds[idx] = tree.predict(X_mat)

        return np.mean(all_preds, axis=0)


class GradientBoostingRegressorFromScratch(BaseRegressorFromScratch):
    """
    Tăng cường Độ dốc Hồi quy (Gradient Boosting) thuần túy From Scratch.
    - Khởi tạo giá trị cơ sở F_0(x) = mean(y).
    - Vòng lặp m = 1..M:
        1. Tính pseudo-residuals: r = y - F_{m-1}(x)
        2. Fit cây con DecisionTreeRegressorFromScratch(max_depth=3) trên (X, r)
        3. Cập nhật F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
    """

    def __init__(
        self,
        n_estimators: int = 30,
        learning_rate: float = 0.05,
        max_depth: Optional[int] = 3,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: Optional[Union[str, int, float]] = None,
        n_quantiles: int = 16,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth if max_depth is not None else 3
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_quantiles = n_quantiles
        self.random_state = random_state

        self.init_value_: float = 0.0
        self.trees_: List[DecisionTreeRegressorFromScratch] = []
        self.n_features_in_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None
        self.train_loss_history_: List[float] = []

    def fit(self, X: Any, y: Any) -> "GradientBoostingRegressorFromScratch":
        X_mat = _ensure_dense_2d(X)
        y_vec = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_mat.shape

        self.n_features_in_ = n_features
        self.init_value_ = float(np.mean(y_vec))
        self.trees_ = []
        self.train_loss_history_ = []

        current_preds = np.full(n_samples, self.init_value_, dtype=np.float64)
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_estimators):
            residuals = y_vec - current_preds
            loss = float(np.mean(residuals ** 2))
            self.train_loss_history_.append(loss)

            tree_seed = int(rng.randint(0, 1_000_000))
            tree = DecisionTreeRegressorFromScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_quantiles=self.n_quantiles,
                random_state=tree_seed
            )
            tree.fit(X_mat, residuals)

            tree_preds = tree.predict(X_mat)
            current_preds += self.learning_rate * tree_preds
            self.trees_.append(tree)

        # Tính Feature Importances tổng hợp có trọng số
        all_importances = [t.feature_importances_ for t in self.trees_ if t.feature_importances_ is not None]
        if all_importances:
            mean_imp = np.mean(all_importances, axis=0)
            total = np.sum(mean_imp)
            self.feature_importances_ = mean_imp / total if total > 0 else np.zeros(n_features)
        else:
            self.feature_importances_ = np.zeros(n_features)

        return self

    def predict(self, X: Any) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("Mô hình chưa được fit! Vui lòng gọi fit(X, y) trước.")
        X_mat = _ensure_dense_2d(X)
        preds = np.full(X_mat.shape[0], self.init_value_, dtype=np.float64)

        for tree in self.trees_:
            preds += self.learning_rate * tree.predict(X_mat)

        return preds
