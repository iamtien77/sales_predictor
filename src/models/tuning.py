"""
tuning.py - Tối ưu Siêu tham số (Hyperparameter Tuning) thuần túy From Scratch
================================================================================
Không sử dụng Scikit-Learn:
  - grid_search_from_scratch: Tìm kiếm lưới trên tập Validation (10%) độc lập
  - tune_decision_tree
  - tune_random_forest
  - tune_gradient_boosting
  - tune_all
"""
import sys
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.models.regressors import (
    LinearRegressionFromScratch,
    DecisionTreeRegressorFromScratch,
    RandomForestRegressorFromScratch,
    GradientBoostingRegressorFromScratch,
    compute_rmse,
    compute_r2,
    compute_mae
)


def grid_search_from_scratch(
    model_class: Any,
    param_grid: Dict[str, List[Any]],
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    scoring: str = "rmse",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Tìm kiếm siêu tham số trên lưới (Grid Search) hoàn toàn thuần túy From Scratch.
    Đánh giá khách quan trên tập Validation độc lập để phòng chống Overfitting và Data Leakage.

    Args:
        model_class: Lớp mô hình (ví dụ: DecisionTreeRegressorFromScratch)
        param_grid: Dict chứa danh sách giá trị cần thử nghiệm cho từng tham số
        X_train, y_train: Dữ liệu huấn luyện
        X_val, y_val: Dữ liệu kiểm định (Validation Set)
        scoring: Tiêu chí tối ưu ('rmse', 'mae', 'r2')
        verbose: In tiến trình nếu True

    Returns:
        Dict chứa:
          - 'best_estimator': Mô hình tốt nhất đã fit
          - 'best_params': Bộ tham số đạt điểm cao nhất
          - 'best_score': Giá trị metric tốt nhất
          - 'cv_results': Bảng tổng hợp lịch sử thử nghiệm
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    best_score = float("inf") if scoring in ["rmse", "mae", "rmse_brl"] else float("-inf")
    best_params = None
    best_estimator = None
    history = []

    for idx, combo in enumerate(combinations):
        current_params = dict(zip(param_names, combo))
        model = model_class(**current_params)

        # Fit trên tập Train
        model.fit(X_train, y_train)

        # Predict trên tập Validation
        val_preds = model.predict(X_val)

        # Đánh giá theo metric
        if scoring == "rmse_brl":
            score = compute_rmse(np.maximum(0.0, np.expm1(y_val)), np.maximum(0.0, np.expm1(val_preds)))
            is_better = score < best_score
        elif scoring == "rmse":
            score = compute_rmse(y_val, val_preds)
            is_better = score < best_score
        elif scoring == "mae":
            score = compute_mae(y_val, val_preds)
            is_better = score < best_score
        elif scoring == "r2":
            score = compute_r2(y_val, val_preds)
            is_better = score > best_score
        else:
            raise ValueError(f"Chỉ số scoring không được hỗ trợ: {scoring}")

        history.append({
            "iteration": idx + 1,
            "params": current_params,
            "score": score
        })

        if verbose:
            print(f"[{idx + 1}/{len(combinations)}] Tham số: {current_params} => {scoring.upper()}: {score:.4f}")

        if is_better:
            best_score = score
            best_params = current_params
            best_estimator = model

    return {
        "best_estimator": best_estimator,
        "best_params": best_params,
        "best_score": best_score,
        "cv_results": history
    }


def _resolve_train_val_split(
    X: Any,
    y: Any,
    X_val: Optional[Any] = None,
    y_val: Optional[Any] = None,
    val_ratio: float = 0.1
) -> Tuple[Any, Any, Any, Any]:
    """Tự động phân tách Validation nếu người dùng không truyền sẵn X_val, y_val"""
    if X_val is not None and y_val is not None:
        return X, y, X_val, y_val

    # Cắt cuối mảng theo quy tắc chuỗi thời gian
    n = len(y)
    split_idx = int(n * (1.0 - val_ratio))
    X_tr = X[:split_idx]
    y_tr = y[:split_idx]
    X_v = X[split_idx:]
    y_v = y[split_idx:]
    return X_tr, y_tr, X_v, y_v


def tune_decision_tree(
    X_train: Any,
    y_train: Any,
    X_val: Optional[Any] = None,
    y_val: Optional[Any] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    scoring: str = "rmse"
) -> Dict[str, Any]:
    """Tối ưu siêu tham số cho DecisionTreeRegressorFromScratch"""
    X_tr, y_tr, X_v, y_v = _resolve_train_val_split(X_train, y_train, X_val, y_val)
    if param_grid is None:
        param_grid = config.PARAM_GRIDS.get("decision_tree", {
            "max_depth": [3, 5, 8],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [2, 5]
        })

    return grid_search_from_scratch(
        model_class=DecisionTreeRegressorFromScratch,
        param_grid=param_grid,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_v,
        y_val=y_v,
        scoring=scoring
    )


def tune_random_forest(
    X_train: Any,
    y_train: Any,
    X_val: Optional[Any] = None,
    y_val: Optional[Any] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    scoring: str = "rmse"
) -> Dict[str, Any]:
    """Tối ưu siêu tham số cho RandomForestRegressorFromScratch"""
    X_tr, y_tr, X_v, y_v = _resolve_train_val_split(X_train, y_train, X_val, y_val)
    if param_grid is None:
        param_grid = config.PARAM_GRIDS.get("random_forest", {
            "n_estimators": [20, 30],
            "max_depth": [5, 8],
            "min_samples_split": [5, 10]
        })

    return grid_search_from_scratch(
        model_class=RandomForestRegressorFromScratch,
        param_grid=param_grid,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_v,
        y_val=y_v,
        scoring=scoring
    )


def tune_gradient_boosting(
    X_train: Any,
    y_train: Any,
    X_val: Optional[Any] = None,
    y_val: Optional[Any] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    scoring: str = "rmse"
) -> Dict[str, Any]:
    """Tối ưu siêu tham số cho GradientBoostingRegressorFromScratch"""
    X_tr, y_tr, X_v, y_v = _resolve_train_val_split(X_train, y_train, X_val, y_val)
    if param_grid is None:
        param_grid = config.PARAM_GRIDS.get("gradient_boosting", {
            "n_estimators": [20, 30],
            "learning_rate": [0.03, 0.05],
            "max_depth": [3, 4]
        })

    return grid_search_from_scratch(
        model_class=GradientBoostingRegressorFromScratch,
        param_grid=param_grid,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_v,
        y_val=y_v,
        scoring=scoring
    )


def tune_all(
    X_train: Any,
    y_train: Any,
    X_val: Optional[Any] = None,
    y_val: Optional[Any] = None,
    scoring: str = "rmse"
) -> Dict[str, Dict[str, Any]]:
    """Tối ưu toàn diện 3 mô hình phi tuyến và trả về kết quả đối sánh"""
    X_tr, y_tr, X_v, y_v = _resolve_train_val_split(X_train, y_train, X_val, y_val)
    return {
        "decision_tree": tune_decision_tree(X_tr, y_tr, X_v, y_v, scoring=scoring),
        "random_forest": tune_random_forest(X_tr, y_tr, X_v, y_v, scoring=scoring),
        "gradient_boosting": tune_gradient_boosting(X_tr, y_tr, X_v, y_v, scoring=scoring),
    }
