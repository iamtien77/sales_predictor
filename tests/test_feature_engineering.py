"""
Unit tests for feature engineering module (Lab 2: Product Sales Predictor)
Kiểm thử toàn diện kiến trúc kỹ nghệ và chọn lọc đặc trưng chuẩn mực:
- 6 nhóm đặc trưng cao cấp và cơ chế chống rò rỉ dữ liệu (SalesFeatureEngineer)
- Theo dõi lát cắt ranh giới đặc trưng (feature_slices_)
- Kiểm toán tương quan mục tiêu (audit_feature_target_correlation)
- Phát hiện đa cộng tuyến nghiêm trọng (detect_multicollinear_pairs)
- Bộ chọn lọc đặc trưng Hồi quy From-Scratch (SalesFeatureSelectorScratch)
- Quét Top-K đa mục tiêu trên Validation set (sweep_top_k)
- Tích hợp toàn diện vào CustomDataPipeline
"""

import sys
import unittest
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.feature_engineering import (
    SalesFeatureEngineer,
    SalesFeatureSelectorScratch,
    audit_feature_target_correlation,
    detect_multicollinear_pairs,
    engineer_all_features,
    add_pricing_and_freight_features,
    add_logistics_features,
    add_cyclical_calendar_features,
    add_reputation_and_customer_features,
    add_target_transformations,
)
from src.data.preprocessing import CustomDataPipeline


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        """Khởi tạo tập dữ liệu giả lập có đầy đủ các cột đa chiều."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2017-01-01", periods=n, freq="D")
        
        prices = np.random.uniform(20.0, 300.0, n)
        freight = np.random.uniform(5.0, 40.0, n)
        
        self.df_train = pd.DataFrame({
            "order_purchase_timestamp": dates,
            "order_estimated_delivery_date": dates + pd.Timedelta(days=15),
            "price": prices,
            "freight_value": freight,
            "discount_rate": np.random.uniform(0.0, 0.3, n),
            "discounted_price": prices * 0.9,  # Tạo tương quan cao để test multicollinearity
            "ad_spend": np.random.uniform(10.0, 100.0, n),
            "product_length_cm": np.random.uniform(10.0, 50.0, n),
            "product_height_cm": np.random.uniform(5.0, 30.0, n),
            "product_width_cm": np.random.uniform(10.0, 40.0, n),
            "product_weight_g": np.random.uniform(200.0, 4000.0, n),
            "customer_state": np.random.choice(["SP", "RJ", "MG"], n),
            "seller_state": np.random.choice(["SP", "PR"], n),
            "sales_lag_1": np.random.uniform(10.0, 100.0, n),
            "sales_lag_7": np.random.uniform(10.0, 100.0, n),
            "sales_rolling_mean_7": np.random.uniform(15.0, 90.0, n),
            "sales_rolling_std_7": np.random.uniform(1.0, 10.0, n),
            "month": np.random.randint(1, 13, n),
            "day_of_week": np.random.randint(0, 7, n),
            "is_weekend": np.random.choice([0, 1], n),
            "is_holiday": np.random.choice([0, 1], n),
            "is_peak_shopping_season": np.random.choice([0, 1], n),
            "recency_days": np.random.randint(1, 365, n),
            "customer_lifetime_value": np.random.uniform(50.0, 1500.0, n),
            "previous_purchases": np.random.randint(0, 5, n),
            "category": np.random.choice(["eletronicos", "beleza_saude"], n),
            "rating": np.random.uniform(1.0, 5.0, n),
            "num_reviews": np.random.randint(1, 50, n),
            "sales": prices * 1.2 + np.random.normal(0, 5, n)
        })

        # Tập Test giả lập có danh mục mới chưa từng thấy trong Train (Unseen Category)
        self.df_test = pd.DataFrame({
            "order_purchase_timestamp": pd.date_range("2017-05-01", periods=20, freq="D"),
            "order_estimated_delivery_date": pd.date_range("2017-05-20", periods=20, freq="D"),
            "price": np.random.uniform(50.0, 200.0, 20),
            "freight_value": np.random.uniform(10.0, 30.0, 20),
            "product_length_cm": np.full(20, 20.0),
            "product_height_cm": np.full(20, 10.0),
            "product_width_cm": np.full(20, 15.0),
            "product_weight_g": np.full(20, 1000.0),
            "customer_state": np.full(20, "RS"),
            "seller_state": np.full(20, "SP"),
            "category": np.full(20, "danh_muc_hoan_toan_moi"),
            "sales": np.random.uniform(30.0, 200.0, 20)
        })

    def test_sales_feature_engineer_and_slices(self):
        """Kiểm tra SalesFeatureEngineer và cấu trúc feature_slices_."""
        fe = SalesFeatureEngineer(use_log_target=True)
        res_train = fe.fit_transform(self.df_train)

        self.assertTrue(fe.is_fitted_)
        self.assertGreater(len(fe.feature_slices_), 0)
        
        # Kiểm tra các nhóm ranh giới
        for group in ["pricing_dynamics", "physical_logistics", "lag_momentum", "calendar_cyclical", "rfm_marketing", "category_bayesian"]:
            self.assertIn(group, fe.feature_slices_)
            start, end = fe.get_feature_slice(group)
            self.assertLess(start, end)

        self.assertEqual(len(fe.list_feature_groups()), 6)
        self.assertIn("log_sales", res_train.columns)

    def test_data_leakage_protection_unseen_category(self):
        """Kiểm tra cơ chế chống rò rỉ dữ liệu khi gặp danh mục mới."""
        fe = SalesFeatureEngineer(use_log_target=True)
        fe.fit(self.df_train)

        res_test = fe.transform(self.df_test)
        self.assertIn("price_zscore_in_cat", res_test.columns)
        self.assertFalse(res_test["price_zscore_in_cat"].isna().any())

        expected_zscore = np.round((self.df_test["price"] - fe.global_price_mean_) / (fe.global_price_std_ + 1e-8), 3)
        np.testing.assert_allclose(res_test["price_zscore_in_cat"].values, expected_zscore.values, rtol=1e-3)

    def test_audit_feature_target_correlation(self):
        """Kiểm tra hàm kiểm toán tương quan Pearson và Spearman với biến mục tiêu."""
        fe = SalesFeatureEngineer(use_log_target=True)
        df_feat = fe.fit_transform(self.df_train)

        audit_df = audit_feature_target_correlation(df_feat, target_col="log_sales")
        self.assertIsInstance(audit_df, pd.DataFrame)
        self.assertGreater(len(audit_df), 10)
        self.assertIn("Tương quan Pearson (r)", audit_df.columns)
        self.assertIn("Tương quan Spearman (rho)", audit_df.columns)
        self.assertIn("Mức độ ảnh hưởng", audit_df.columns)
        
        # Giá trị |r| phải nằm trong [0, 1]
        self.assertTrue((audit_df["Độ lớn (|r|)"] >= 0.0).all())
        self.assertTrue((audit_df["Độ lớn (|r|)"] <= 1.0).all())

    def test_detect_multicollinear_pairs(self):
        """Kiểm tra thuật toán phát hiện các cặp đặc trưng bị đa cộng tuyến cao."""
        df_test_multi = pd.DataFrame({
            "var_a": np.linspace(1, 100, 50),
            "var_b": np.linspace(1, 100, 50) + np.random.normal(0, 0.01, 50),  # r ~ 1.0
            "var_c": np.random.uniform(0, 1, 50),
            "log_sales": np.linspace(2, 200, 50)
        })

        multi_df = detect_multicollinear_pairs(df_test_multi, threshold=0.85, target_col="log_sales")
        self.assertIsInstance(multi_df, pd.DataFrame)
        self.assertGreaterEqual(len(multi_df), 1)
        self.assertEqual(multi_df.iloc[0]["Đặc trưng 1 (Feature 1)"], "var_a")
        self.assertEqual(multi_df.iloc[0]["Đặc trưng 2 (Feature 2)"], "var_b")
        self.assertIn("Khuyến nghị giữ lại", multi_df.columns)

    def test_sales_feature_selector_scratch(self):
        """Kiểm tra toàn diện class SalesFeatureSelectorScratch (Variance, Importance, Top-K Sweep)."""
        X = np.random.randn(80, 20)
        # Cột hằng số để test variance threshold
        X[:, 0] = 5.0
        y = X[:, 1] * 2.5 + X[:, 2] * 1.5 + np.random.randn(80) * 0.1

        selector = SalesFeatureSelectorScratch(variance_threshold=1e-4, top_k=5)
        mask, variances = selector.fit_variance_threshold(X)
        self.assertFalse(mask[0])  # Cột hằng số bị loại

        # Test compute combined importance
        scores, ranked = selector.compute_combined_importance(X, y)
        self.assertEqual(len(scores), 20)
        self.assertEqual(len(ranked), 20)
        # Các cột 1 và 2 tạo nên y phải nằm trong top xếp hạng
        self.assertTrue(1 in ranked[:5] or 2 in ranked[:5])

        # Test sweep top k
        X_val = np.random.randn(20, 20)
        y_val = X_val[:, 1] * 2.5 + X_val[:, 2] * 1.5 + np.random.randn(20) * 0.1

        best_row, df_sweep = selector.sweep_top_k(X, y, X_val, y_val, candidate_k_list=[3, 5, 10, 15, 20], r2_target=0.80)
        self.assertIsInstance(df_sweep, pd.DataFrame)
        self.assertEqual(len(df_sweep), 5)
        self.assertIn("Balance Score", df_sweep.columns[8])

        # Test fit & transform
        X_sel = selector.fit_transform(X, y, top_k=5)
        self.assertEqual(X_sel.shape[1], 5)

    def test_custom_data_pipeline_with_selector(self):
        """Kiểm tra tích hợp SalesFeatureSelectorScratch trực tiếp vào CustomDataPipeline."""
        fe = SalesFeatureEngineer(use_log_target=True)
        selector = SalesFeatureSelectorScratch(top_k=8)
        pipeline = CustomDataPipeline(feature_engineer=fe, feature_selector=selector, use_log_target=True)

        X_train, y_train = pipeline.fit_transform(self.df_train)
        self.assertEqual(X_train.shape[1], 8)
        self.assertEqual(len(pipeline.feature_names_), 8)

        X_test, y_test = pipeline.transform(self.df_test)
        self.assertEqual(X_test.shape[1], 8)


if __name__ == "__main__":
    unittest.main()
