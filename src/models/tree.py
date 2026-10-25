"""
tree.py - Cây Quyết định Hồi quy thuần NumPy (Decision Tree Regressor From-Scratch)
====================================================================================
Cung cấp:
  - DecisionTreeNode: Cấu trúc nút biểu diễn cây nhị phân đệ quy
  - DecisionTreeRegressorFromScratch: Cây hồi quy tối ưu hóa theo Variance Reduction,
    tích hợp kỹ thuật 16 Quantiles ứng viên giúp duyệt siêu tốc trên tập dữ liệu lớn.
"""
from typing import Any, List, Optional, Tuple, Union
import numpy as np

from src.models.base import BaseRegressorFromScratch, _ensure_dense_2d


class DecisionTreeNode:
    """Nút biểu diễn cây quyết định hồi quy."""

    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["DecisionTreeNode"] = None,
        right: Optional["DecisionTreeNode"] = None,
        value: Optional[float] = None,
        variance_reduction: float = 0.0,
        n_samples: int = 0
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.variance_reduction = variance_reduction
        self.n_samples = n_samples

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeRegressorFromScratch(BaseRegressorFromScratch):
    """
    Cây quyết định hồi quy thuần túy From Scratch (Cài đặt đệ quy theo Variance Reduction).
    Tối ưu hóa:
      - Sử dụng phân vị Quantiles (n_quantiles) để giới hạn số ngưỡng duyệt, tránh nghẽn CPU trên dataset lớn.
      - Hỗ trợ subsampling đặc trưng ngẫu nhiên (max_features) phục vụ cho Random Forest.
      - Vectorized Variance Gain calculation & Batch mask prediction.
    """

    def __init__(
        self,
        max_depth: Optional[int] = 5,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: Optional[Union[str, int, float]] = None,
        n_quantiles: int = 16,
        random_state: int = 42
    ):
        self.max_depth = max_depth if max_depth is not None else 20
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_quantiles = n_quantiles
        self.random_state = random_state

        self.root_: Optional[DecisionTreeNode] = None
        self.n_features_in_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None
        self._rng = np.random.RandomState(self.random_state)

    def _get_feature_indices(self, total_features: int) -> np.ndarray:
        """Chọn tập thuộc tính ngẫu nhiên khi phân nhánh."""
        if self.max_features is None:
            return np.arange(total_features)
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(total_features)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(total_features)))
        elif isinstance(self.max_features, int):
            k = min(total_features, max(1, self.max_features))
        elif isinstance(self.max_features, float) and 0.0 < self.max_features <= 1.0:
            k = max(1, int(self.max_features * total_features))
        else:
            k = total_features

        return self._rng.choice(total_features, size=k, replace=False)

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Tìm thuộc tính và ngưỡng cắt làm giảm phương sai (Variance Reduction) lớn nhất:
          Delta Var = Var(y) - (n_L/n * Var(y_L) + n_R/n * Var(y_R))
        """
        n = y.shape[0]
        if n < self.min_samples_split:
            return None, None, 0.0

        current_var = float(np.var(y))
        if current_var < 1e-10:
            return None, None, 0.0

        best_gain = 0.0
        best_feat = None
        best_thresh = None

        for feat_idx in feature_indices:
            x_col = X[:, feat_idx]
            unique_vals = np.unique(x_col)
            if unique_vals.shape[0] <= 1:
                continue

            # Giới hạn số ngưỡng bằng Quantiles để tăng tốc tính toán
            if unique_vals.shape[0] > self.n_quantiles:
                percentiles = np.linspace(5, 95, self.n_quantiles)
                thresholds = np.percentile(unique_vals, percentiles)
                thresholds = np.unique(thresholds)
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = x_col <= thresh
                n_left = int(np.sum(left_mask))
                n_right = n - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                var_left = float(np.var(y[left_mask]))
                var_right = float(np.var(y[~left_mask]))

                gain = current_var - ((n_left / n) * var_left + (n_right / n) * var_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = float(thresh)

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionTreeNode:
        """Xây dựng cây nhị phân đệ quy."""
        n_samples = y.shape[0]

        # Điều kiện dừng: đạt max_depth, không đủ mẫu split, hoặc phương sai triệt tiêu
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.var(y) < 1e-10:
            return DecisionTreeNode(value=float(np.mean(y)), n_samples=n_samples)

        feat_indices = self._get_feature_indices(X.shape[1])
        best_feat, best_thresh, best_gain = self._best_split(X, y, feat_indices)

        if best_feat is None or best_gain <= 0.0:
            return DecisionTreeNode(value=float(np.mean(y)), n_samples=n_samples)

        left_mask = X[:, best_feat] <= best_thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], depth=depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth=depth + 1)

        return DecisionTreeNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            variance_reduction=best_gain,
            n_samples=n_samples
        )

    def fit(self, X: Any, y: Any) -> "DecisionTreeRegressorFromScratch":
        X_mat = _ensure_dense_2d(X)
        y_vec = np.asarray(y, dtype=np.float64).ravel()

        self._rng = np.random.RandomState(self.random_state)
        self.n_features_in_ = X_mat.shape[1]
        self.root_ = self._build_tree(X_mat, y_vec, depth=0)

        # Tính toán feature importances dựa trên lượng giảm phương sai tích lũy
        importances = np.zeros(self.n_features_in_, dtype=np.float64)

        def _accumulate_importance(node: Optional[DecisionTreeNode]):
            if node is None or node.is_leaf:
                return
            if node.feature_idx is not None:
                importances[node.feature_idx] += node.variance_reduction * node.n_samples
            _accumulate_importance(node.left)
            _accumulate_importance(node.right)

        _accumulate_importance(self.root_)
        total_imp = np.sum(importances)
        if total_imp > 0.0:
            self.feature_importances_ = importances / total_imp
        else:
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=np.float64)

        return self

    def _predict_row(self, x: np.ndarray, node: DecisionTreeNode) -> float:
        """Duyệt một mẫu qua cây đến nút lá."""
        if node.is_leaf or node.threshold is None or node.feature_idx is None:
            return float(node.value)
        if x[node.feature_idx] <= node.threshold:
            return self._predict_row(x, node.left)  # type: ignore
        else:
            return self._predict_row(x, node.right)  # type: ignore

    def predict(self, X: Any) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("Mô hình chưa được fit! Vui lòng gọi fit(X, y) trước.")
        X_mat = _ensure_dense_2d(X)
        preds = np.zeros(X_mat.shape[0], dtype=np.float64)

        # Phân nhánh theo batch masks đệ quy để tăng tốc tối đa
        def _predict_batch(indices: np.ndarray, node: DecisionTreeNode):
            if indices.size == 0:
                return
            if node.is_leaf or node.threshold is None or node.feature_idx is None:
                preds[indices] = node.value
                return
            sub_x = X_mat[indices, node.feature_idx]
            left_mask = sub_x <= node.threshold
            _predict_batch(indices[left_mask], node.left)  # type: ignore
            _predict_batch(indices[~left_mask], node.right)  # type: ignore

        all_indices = np.arange(X_mat.shape[0])
        _predict_batch(all_indices, self.root_)
        return preds
