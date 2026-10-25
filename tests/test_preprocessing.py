"""
Unit tests for preprocessing module (Lab 2: Product Sales Predictor)
Kiểm thử toàn diện các module tiền xử lý dữ liệu thuần NumPy/Pandas:
- clean_sales_data
- audit_outliers_table
- CustomStandardScaler
- CustomOneHotEncoder
- CustomImputer
- CustomOutlierCapper
- temporal_split_data
- random_split_data
- compare_splits_temporal_vs_random
- CustomDataPipeline (OOP pipeline fit, transform, save, load)
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.preprocessing import (
    clean_sales_data,
    audit_outliers_table,
    CustomStandardScaler,
    CustomOneHotEncoder,
    CustomImputer,
    CustomOutlierCapper,
    temporal_split_data,
    random_split_data,
    compare_splits_temporal_vs_random,
    CustomDataPipeline,
)


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        """Khởi tạo tập dữ liệu kiểm thử giả lập."""
        np.random.seed(42)
        dates = pd.date_range("2017-01-01", periods=100, freq="D")
        self.df = pd.DataFrame({
            "order_purchase_timestamp": dates,
            "price": np.concatenate([np.random.uniform(10, 200, 95), [-10, 0, 5000, 6000, 7000]]),
            "freight_value": np.random.uniform(5, 50, 100),
            "product_weight_g": np.random.uniform(100, 5000, 100),
            "product_photos_qty": np.random.randint(1, 10, 100).astype(float),
            "customer_state": np.random.choice(["SP", "RJ", "MG", "RS"], 100),
            "product_category_name": np.random.choice(["beleza_saude", "relogios_presentes", "telefonia"], 100),
            "sales": np.concatenate([np.random.uniform(15, 250, 95), [-5, 0, 8000, 9000, 10000]]),
            "log_sales": np.log1p(np.clip(np.concatenate([np.random.uniform(15, 250, 95), [1, 1, 8000, 9000, 10000]]), 0, None))
        })
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_clean_sales_data(self):
        """Kiểm tra việc làm sạch loại bỏ bản ghi có price/sales <= 0."""
        cleaned_df, audit_df = clean_sales_data(self.df)
        self.assertTrue((cleaned_df["price"] > 0).all())
        self.assertTrue((cleaned_df["sales"] > 0).all())
        self.assertLess(len(cleaned_df), len(self.df))
        self.assertIsInstance(audit_df, pd.DataFrame)
        self.assertGreater(len(audit_df), 0)

    def test_audit_outliers_table(self):
        """Kiểm tra hàm tạo bảng kiểm toán ngoại lai trả về DataFrame chuẩn."""
        cleaned_df, _ = clean_sales_data(self.df)
        audit_table = audit_outliers_table(cleaned_df, num_cols=["price", "sales"])
        self.assertIsInstance(audit_table, pd.DataFrame)
        self.assertEqual(len(audit_table), 2)

    def test_custom_standard_scaler(self):
        """Kiểm tra CustomStandardScaler thuần NumPy."""
        data = np.array([[10.0, 20.0], [20.0, 40.0], [30.0, 60.0]])
        scaler = CustomStandardScaler()
        scaled = scaler.fit_transform(data)
        self.assertIsNotNone(scaler.mean_)
        self.assertIsNotNone(scaler.std_)
        self.assertAlmostEqual(float(scaled.mean()), 0.0, places=5)

    def test_custom_one_hot_encoder(self):
        """Kiểm tra CustomOneHotEncoder thuần Pandas/NumPy."""
        encoder = CustomOneHotEncoder()
        encoded_arr = encoder.fit_transform(self.df, columns=["customer_state"])
        self.assertIsInstance(encoded_arr, np.ndarray)
        self.assertGreater(len(encoder.categories_), 0)
        self.assertEqual(encoded_arr.shape[1], len(encoder.encoded_feature_names_))

    def test_custom_imputer(self):
        """Kiểm tra CustomImputer điền khuyết số bằng median, phân loại bằng mode."""
        df_missing = self.df.copy()
        df_missing.loc[0, "freight_value"] = np.nan
        df_missing.loc[1, "customer_state"] = np.nan
        
        imputer = CustomImputer()
        imputed_df = imputer.fit_transform(df_missing, num_cols=["freight_value"], cat_cols=["customer_state"])
        self.assertFalse(imputed_df["freight_value"].isnull().any())
        self.assertFalse(imputed_df["customer_state"].isnull().any())

    def test_custom_outlier_capper(self):
        """Kiểm tra CustomOutlierCapper theo phương pháp IQR."""
        cleaned_df, _ = clean_sales_data(self.df)
        capper = CustomOutlierCapper(factor=1.5)
        capped_df = capper.fit_transform(cleaned_df, columns=["price", "sales"])
        for col in ["price", "sales"]:
            lower = capper.lower_bounds_[col]
            upper = capper.upper_bounds_[col]
            self.assertTrue((capped_df[col] >= lower - 1e-5).all())
            self.assertTrue((capped_df[col] <= upper + 1e-5).all())

    def test_temporal_split_data(self):
        """Kiểm tra phân chia chuỗi thời gian không xáo trộn."""
        cleaned_df, _ = clean_sales_data(self.df)
        train_df, val_df, test_df = temporal_split_data(
            cleaned_df,
            timestamp_col="order_purchase_timestamp",
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(cleaned_df))
        max_train_time = train_df["order_purchase_timestamp"].max()
        min_val_time = val_df["order_purchase_timestamp"].min()
        max_val_time = val_df["order_purchase_timestamp"].max()
        min_test_time = test_df["order_purchase_timestamp"].min()

        self.assertLessEqual(max_train_time, min_val_time)
        self.assertLessEqual(max_val_time, min_test_time)

    def test_random_split_data(self):
        """Kiểm tra phân chia ngẫu nhiên thuần NumPy."""
        cleaned_df, _ = clean_sales_data(self.df)
        train_df, val_df, test_df = random_split_data(
            cleaned_df,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_state=42
        )
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(cleaned_df))

    def test_compare_splits_temporal_vs_random(self):
        """Kiểm tra bảng đối sánh chiến lược phân chia dữ liệu."""
        cleaned_df, _ = clean_sales_data(self.df)
        comp_df = compare_splits_temporal_vs_random(cleaned_df)
        self.assertIsInstance(comp_df, pd.DataFrame)
        self.assertEqual(len(comp_df), 2)

    def test_custom_data_pipeline_oop(self):
        """Kiểm thử toàn diện class CustomDataPipeline (OOP Fit, Transform, Save, Load)."""
        cleaned_df, _ = clean_sales_data(self.df)
        train_df, val_df, test_df = temporal_split_data(
            cleaned_df, timestamp_col="order_purchase_timestamp", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )

        pipeline = CustomDataPipeline(
            num_cols=["freight_value", "product_weight_g"],
            cat_cols=["customer_state"],
            outlier_cols=["freight_value"],
            use_log_target=True
        )

        X_train, y_train = pipeline.fit_transform(train_df)
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(y_train)

        X_val, y_val = pipeline.transform(val_df)
        X_test, y_test = pipeline.transform(test_df)
        self.assertEqual(X_train.shape[1], X_val.shape[1])
        self.assertEqual(X_train.shape[1], X_test.shape[1])

        # Test Save & Load
        save_path = Path(self.temp_dir) / "pipeline.pkl"
        pipeline.save(save_path)
        self.assertTrue(save_path.exists())

        loaded_pipeline = CustomDataPipeline.load(save_path)
        X_val_loaded, y_val_loaded = loaded_pipeline.transform(val_df)
        np.testing.assert_allclose(X_val, X_val_loaded, rtol=1e-5)
        np.testing.assert_allclose(y_val, y_val_loaded, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
