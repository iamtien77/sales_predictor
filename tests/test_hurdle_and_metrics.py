"""
Unit tests for TwoStageHurdleRegressor, Linear Clamping and Supply Chain Metrics
=================================================================================
Kiểm thử toàn diện các cải tiến kiến trúc và điểm nghẽn đã khắc phục:
  1. Kiểm thử LinearRegressionFromScratch có Clamping an toàn chống nổ số học
  2. Kiểm thử TwoStageHurdleRegressorFromScratch trên dữ liệu zero-inflated (>90% số 0)
  3. Kiểm thử WAPE (Weighted Absolute Percentage Error)
  4. Kiểm thử compute_supply_chain_cost (Understock vs Overstock Cost)
  5. Kiểm thử 100% Backward Compatibility của Facade src.models.regressors
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Kiểm tra Backward Compatibility: Import từ regressors.py
from src.models.regressors import (
    BaseRegressorFromScratch,
    LinearRegressionFromScratch,
    DecisionTreeNode,
    DecisionTreeRegressorFromScratch,
    RandomForestRegressorFromScratch,
    GradientBoostingRegressorFromScratch,
    TwoStageHurdleRegressorFromScratch,
    ZeroRegressorFromScratch,
    RevenueMeanRegressorFromScratch,
    MeanRegressorFromScratch,
    PreviousWeekRegressorFromScratch,
    compute_mse,
    compute_mae,
    compute_rmse,
    compute_r2,
    compute_regression_metrics,
    get_models,
)
from src.evaluation.metrics import (
    compute_wape,
    compute_supply_chain_cost,
    compute_extended_metrics,
)


class TestHurdleAndMetrics(unittest.TestCase):
    """Kiểm thử mô hình Hurdle, tính ổn định của Linear Regression và chỉ số chuỗi cung ứng."""

    def setUp(self):
        np.random.seed(42)
        # Giả lập tập dữ liệu zero-inflated (85% số 0)
        self.n_samples = 300
        self.n_features = 10
        self.X = np.random.randn(self.n_samples, self.n_features)
        
        # 85% tuần bán 0 BRL, 15% tuần có đơn hàng từ 10 đến 200 BRL
        is_sale = np.random.rand(self.n_samples) < 0.15
        sales = np.zeros(self.n_samples)
        sales[is_sale] = np.random.uniform(10.0, 200.0, size=int(np.sum(is_sale)))
        self.y = np.log1p(sales)

    def test_linear_regression_clamping_prevents_overflow(self):
        """Kiểm tra cơ chế kẹp biên chống nổ số học của Linear Regression."""
        model = LinearRegressionFromScratch(fit_intercept=True, clip_predictions=True)
        model.fit(self.X, self.y)

        # Tạo mẫu ngoại lai cực đoan (outlier X = 100.0)
        X_outlier = np.full((5, self.n_features), 100.0)
        preds = model.predict(X_outlier)

        # Dự báo phải là số hữu hạn và không vượt quá cận an toàn
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertTrue(np.all(preds >= 0.0))
        self.assertTrue(np.all(preds <= model.max_target_log_ * 1.2))

        # Kiểm tra chuyển đổi tiền tệ không bị Overflow
        sales_preds = np.expm1(preds)
        self.assertTrue(np.all(np.isfinite(sales_preds)))
        self.assertTrue(np.all(sales_preds < 1e8))

    def test_two_stage_hurdle_regressor_workflow(self):
        """Kiểm tra mô hình Hai giai đoạn TwoStageHurdleRegressorFromScratch."""
        hurdle = TwoStageHurdleRegressorFromScratch(
            classifier_max_depth=3,
            regressor_max_depth=4,
            min_samples_leaf=3
        )
        hurdle.fit(self.X, self.y)

        # Kiểm tra xác suất có đơn hàng P(sale > 0) thuộc [0, 1]
        probs = hurdle.predict_proba_sale(self.X)
        self.assertEqual(probs.shape, (self.n_samples,))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

        # Kiểm tra doanh thu điều kiện dương không âm
        pos_sales = hurdle.predict_positive_sales(self.X)
        self.assertEqual(pos_sales.shape, (self.n_samples,))
        self.assertTrue(np.all(pos_sales >= 0.0))

        # Kiểm tra dự báo tổng hợp log không âm
        preds = hurdle.predict(self.X)
        self.assertEqual(preds.shape, (self.n_samples,))
        self.assertTrue(np.all(preds >= 0.0))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_compute_wape_correctness(self):
        """Kiểm tra độ chính xác của chỉ số WAPE."""
        y_true = np.array([100.0, 0.0, 50.0, 0.0])
        y_pred = np.array([90.0, 10.0, 60.0, 0.0])

        # Sai số tuyệt đối: |100-90| + |0-10| + |50-60| + |0-0| = 10 + 10 + 10 + 0 = 30
        # Tổng thực tế: 100 + 0 + 50 + 0 = 150
        # WAPE = 30 / 150 = 0.20 (20%)
        wape = compute_wape(y_true, y_pred)
        self.assertAlmostEqual(wape, 0.20, places=5)

    def test_compute_supply_chain_cost_correctness(self):
        """Kiểm tra tính toán ma trận chi phí tồn kho vs hụt hàng."""
        y_true = np.array([100.0, 20.0])
        y_pred = np.array([80.0, 30.0])
        # Mẫu 1: dự báo thiếu 20 BRL (understock) -> phạt 20 * 10 = 200 BRL
        # Mẫu 2: dự báo thừa 10 BRL (overstock) -> phạt 10 * 2 = 20 BRL
        # Tổng chi phí = 220 BRL
        cost_info = compute_supply_chain_cost(y_true, y_pred, understock_penalty=10.0, overstock_holding_cost=2.0)
        self.assertAlmostEqual(cost_info["understock_cost"], 200.0, places=4)
        self.assertAlmostEqual(cost_info["overstock_cost"], 20.0, places=4)
        self.assertAlmostEqual(cost_info["total_cost"], 220.0, places=4)
        self.assertAlmostEqual(cost_info["cost_per_item"], 110.0, places=4)

    def test_backward_compatibility_facade(self):
        """Xác nhận Facade regressors.py cung cấp đầy đủ các class và hàm cũ."""
        models = get_models()
        self.assertIn("linear_regression", models)
        self.assertIn("decision_tree", models)
        self.assertIn("random_forest", models)
        self.assertIn("gradient_boosting", models)

        # Kiểm tra kế thừa BaseRegressorFromScratch
        for name, m in models.items():
            self.assertIsInstance(m, BaseRegressorFromScratch)

    def test_under_over_forecast_and_penalties(self):
        """Kiểm tra tính đúng đắn của hàm under/over forecast và asymmetric penalties."""
        from src.evaluation.metrics import (
            compute_under_over_forecast,
            compute_asymmetric_penalty,
            evaluate_business_subgroups
        )
        y_true = np.array([100.0, 0.0, 50.0])
        y_pred = np.array([80.0, 10.0, 50.0])
        # Mẫu 1: 100 - 80 = under 20
        # Mẫu 2: 10 - 0 = over 10
        # Mẫu 3: 50 - 50 = 0
        uo = compute_under_over_forecast(y_true, y_pred)
        self.assertAlmostEqual(uo["total_underforecast"], 20.0, places=4)
        self.assertAlmostEqual(uo["total_overforecast"], 10.0, places=4)
        self.assertAlmostEqual(uo["total_bias"], -10.0, places=4)

        # Penalty 1:1 = (20 + 10) / 3 = 10.0
        p_1_1 = compute_asymmetric_penalty(y_true, y_pred, 1.0, 1.0)
        self.assertAlmostEqual(p_1_1, 10.0, places=4)

        # Penalty 2:1 = (2*20 + 1*10) / 3 = 50 / 3 = 16.6667
        p_2_1 = compute_asymmetric_penalty(y_true, y_pred, 2.0, 1.0)
        self.assertAlmostEqual(p_2_1, 50.0 / 3.0, places=4)

        # Subgroup evaluation
        sub_df = evaluate_business_subgroups(y_true, y_pred, model_name="MockModel")
        self.assertIsInstance(sub_df, pd.DataFrame)
        self.assertEqual(len(sub_df), 3)  # All, Positive, Zero
        self.assertIn("Total_Underforecast_BRL", sub_df.columns)
        self.assertIn("Total_Overforecast_BRL", sub_df.columns)
        self.assertIn("Penalty_5_1", sub_df.columns)


if __name__ == "__main__":
    unittest.main()

