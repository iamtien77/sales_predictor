"""
regressors.py - Facade API & Điểm tập hợp toàn bộ các mô hình hồi quy From-Scratch thuần NumPy
================================================================================================
Tuân thủ nguyên tắc "From Scratch" 100% (Nói KHÔNG với Scikit-Learn).
Thiết kế theo mẫu Facade Pattern (Deep Modules Architecture):
  - Tách bạch các module con chuyên biệt:
      * src.models.base: Lớp cơ sở và các hàm tính metrics MSE, MAE, RMSE, R²
      * src.models.baselines: 4 Mô hình đối chứng cơ sở (Zero, RevMean, LogMean, PreviousWeek)
      * src.models.linear: LinearRegressionFromScratch (tích hợp Numerical Clamping chống nổ số)
      * src.models.tree: DecisionTreeRegressorFromScratch & DecisionTreeNode (Variance Reduction)
      * src.models.ensemble: RandomForestRegressorFromScratch & GradientBoostingRegressorFromScratch
      * src.models.hurdle: TwoStageHurdleRegressorFromScratch (Đặc trị bài toán >90% tuần doanh thu 0 BRL)
  - Duy trì 100% tương thích ngược (Backward Compatibility) với toàn bộ main.py, app.py, notebooks và tests.

Cung cấp:
  - compute_mse, compute_mae, compute_rmse, compute_r2, compute_regression_metrics
  - get_models()
  - train_and_evaluate(models, X_train, X_test, y_train, y_test)
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import scipy.sparse as sp

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Re-export các lớp cơ sở và hàm tính toán metrics
from src.models.base import (
    BaseRegressorFromScratch,
    _ensure_dense_2d,
    _safe_expm1,
    compute_mse,
    compute_mae,
    compute_rmse,
    compute_r2,
    compute_regression_metrics,
)

# Re-export 4 Mô hình Baseline
from src.models.baselines import (
    ZeroRegressorFromScratch,
    RevenueMeanRegressorFromScratch,
    MeanRegressorFromScratch,
    PreviousWeekRegressorFromScratch,
)

# Re-export Mô hình Tuyến tính có Clamping
from src.models.linear import LinearRegressionFromScratch

# Re-export Mô hình Cây Quyết định
from src.models.tree import (
    DecisionTreeNode,
    DecisionTreeRegressorFromScratch,
)

# Re-export Mô hình Tổ hợp (Ensemble)
from src.models.ensemble import (
    RandomForestRegressorFromScratch,
    GradientBoostingRegressorFromScratch,
)

# Re-export Mô hình Hai giai đoạn (Hurdle)
from src.models.hurdle import TwoStageHurdleRegressorFromScratch

# Re-export Mô hình Sản phẩm Mới (Cold-Start)
from src.models.cold_start import (
    CategoryMedianColdStartBaseline,
    ColdStartProductRegressor,
)


# =====================================================================
# API TIỆN ÍCH DỰ ÁN (PROJECT APIS)
# =====================================================================
def get_models() -> Dict[str, Any]:
    """
    Trả về dict 4 mô hình hồi quy From Scratch được cấu hình sẵn:
      - linear_regression: Baseline Normal Equation OLS (có kẹp biên an toàn)
      - decision_tree: Cây quyết định (max_depth=5)
      - random_forest: Rừng ngẫu nhiên (30 cây, max_depth=6)
      - gradient_boosting: Tăng cường độ dốc (30 cây, lr=0.05, max_depth=3)
    """
    return {
        "linear_regression": LinearRegressionFromScratch(fit_intercept=True, solver="normal_equation"),
        "decision_tree": DecisionTreeRegressorFromScratch(max_depth=5, min_samples_split=10, min_samples_leaf=5),
        "random_forest": RandomForestRegressorFromScratch(n_estimators=30, max_depth=6, min_samples_split=10),
        "gradient_boosting": GradientBoostingRegressorFromScratch(n_estimators=30, learning_rate=0.05, max_depth=3),
    }


def train_and_evaluate(
    models: Dict[str, Any],
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any
) -> Dict[str, Dict[str, Any]]:
    """
    Huấn luyện tuần tự các mô hình, dự đoán trên tập Test và tính toán metrics (MSE, MAE, RMSE, R²).

    Returns:
        Dict[str, Dict]: {
            model_name: {
                "model": fitted_model,
                "metrics": {"MSE": ..., "MAE": ..., "RMSE": ..., "R2": ...},
                "y_pred": np.ndarray
            }
        }
    """
    results = {}
    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        m = compute_regression_metrics(y_test, y_pred)

        results[name] = {
            "model": model,
            "metrics": m,
            "y_pred": y_pred
        }

    return results
