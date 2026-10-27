"""
Unit tests for models module (Lab 2: Product Sales Predictor)
=============================================================
Kiểm thử toàn diện 4 thuật toán hồi quy From-Scratch thuần NumPy:
- compute_regression_metrics (MSE, MAE, RMSE, R²)
- LinearRegressionFromScratch (Normal Equation OLS & Gradient Descent & Sparse matrix)
- DecisionTreeRegressorFromScratch (Variance Reduction & Feature Subsampling)
- RandomForestRegressorFromScratch (Ensemble Bagging & Feature Importances)
- GradientBoostingRegressorFromScratch (Ensemble Boosting & Residual Convergence)
- get_models & train_and_evaluate
- grid_search_from_scratch & tune_*
- persistence: save_artifacts, load_artifacts, save_pipeline, load_pipeline
"""

import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import scipy.sparse as sp

from src.models.regressors import (
    compute_mse,
    compute_mae,
    compute_rmse,
    compute_r2,
    compute_regression_metrics,
    LinearRegressionFromScratch,
    DecisionTreeRegressorFromScratch,
    RandomForestRegressorFromScratch,
    GradientBoostingRegressorFromScratch,
    get_models,
    train_and_evaluate,
)
from src.models.tuning import (
    grid_search_from_scratch,
    tune_decision_tree,
    tune_random_forest,
    tune_gradient_boosting,
    tune_all,
)
from src.models.persistence import (
    save_artifacts,
    load_artifacts,
    save_model,
    load_model,
    save_pipeline,
    load_pipeline,
)


class TestModelsFromScratch(unittest.TestCase):
    def setUp(self):
        """Khởi tạo dữ liệu giả lập cho huấn luyện và kiểm thử"""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 6

        # Dữ liệu tuyến tính nhân tạo: y = 3*x0 - 2*x1 + 1.5*x2 + 4 + nhiễu
        self.X_lin = np.random.randn(self.n_samples, self.n_features)
        self.y_lin = (
            3.0 * self.X_lin[:, 0]
            - 2.0 * self.X_lin[:, 1]
            + 1.5 * self.X_lin[:, 2]
            + 4.0
            + np.random.randn(self.n_samples) * 0.1
        )

        # Dữ liệu phi tuyến tính nhân tạo (hàm bậc hai + sin)
        self.X_nonlin = np.random.uniform(-3, 3, size=(self.n_samples, self.n_features))
        self.y_nonlin = (
            self.X_nonlin[:, 0] ** 2
            + 2.0 * np.sin(self.X_nonlin[:, 1])
            + np.abs(self.X_nonlin[:, 2])
            + np.random.randn(self.n_samples) * 0.05
        )

        # Phân chia Train / Val / Test tỷ lệ 8:1:1
        train_end = int(self.n_samples * 0.8)
        val_end = int(self.n_samples * 0.9)

        self.X_train = self.X_nonlin[:train_end]
        self.y_train = self.y_nonlin[:train_end]
        self.X_val = self.X_nonlin[train_end:val_end]
        self.y_val = self.y_nonlin[train_end:val_end]
        self.X_test = self.X_nonlin[val_end:]
        self.y_test = self.y_nonlin[val_end:]

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # -------------------------------------------------------------
    # 1. KIỂM THỬ BỘ CHỈ SỐ HỒI QUY (METRICS)
    # -------------------------------------------------------------
    def test_regression_metrics(self):
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])

        mse = compute_mse(y_true, y_pred)
        mae = compute_mae(y_true, y_pred)
        rmse = compute_rmse(y_true, y_pred)
        r2 = compute_r2(y_true, y_pred)

        # Sai số lý thuyết:
        # e = [-0.5, 0.5, 0.0, 1.0] => e^2 = [0.25, 0.25, 0, 1.0] => sum=1.5 => mean=0.375
        self.assertAlmostEqual(mse, 0.375, places=5)
        # |e| = [0.5, 0.5, 0, 1.0] => sum=2.0 => mean=0.5
        self.assertAlmostEqual(mae, 0.5, places=5)
        self.assertAlmostEqual(rmse, np.sqrt(0.375), places=5)
        self.assertGreater(r2, 0.9)

        all_m = compute_regression_metrics(y_true, y_pred)
        self.assertIn("MSE", all_m)
        self.assertIn("MAE", all_m)
        self.assertIn("RMSE", all_m)
        self.assertIn("R2", all_m)

    # -------------------------------------------------------------
    # 2. KIỂM THỬ LINEAR REGRESSION FROM SCRATCH
    # -------------------------------------------------------------
    def test_linear_regression_normal_equation(self):
        model = LinearRegressionFromScratch(fit_intercept=True, solver="normal_equation")
        model.fit(self.X_lin, self.y_lin)
        preds = model.predict(self.X_lin)

        # Mô hình OLS phải khôi phục được quan hệ tuyến tính với R² rất cao
        r2 = compute_r2(self.y_lin, preds)
        self.assertGreater(r2, 0.98)
        self.assertAlmostEqual(model.intercept_, 4.0, delta=0.5)
        self.assertAlmostEqual(model.coef_[0], 3.0, delta=0.5)
        self.assertAlmostEqual(model.coef_[1], -2.0, delta=0.5)

    def test_linear_regression_gradient_descent(self):
        model = LinearRegressionFromScratch(
            fit_intercept=True,
            solver="gradient_descent",
            learning_rate=0.05,
            n_iterations=500
        )
        model.fit(self.X_lin, self.y_lin)
        preds = model.predict(self.X_lin)

        r2 = compute_r2(self.y_lin, preds)
        self.assertGreater(r2, 0.95)
        self.assertEqual(len(model.cost_history_), 500)
        self.assertLess(model.cost_history_[-1], model.cost_history_[0])

    def test_linear_regression_sparse_matrix(self):
        X_sp = sp.csr_matrix(self.X_lin)
        model = LinearRegressionFromScratch(fit_intercept=True, solver="normal_equation")
        model.fit(X_sp, self.y_lin)
        preds = model.predict(X_sp)

        r2 = compute_r2(self.y_lin, preds)
        self.assertGreater(r2, 0.98)

    def test_ridge_solvers_agree_with_analytic_solution(self):
        """Both solvers use SSE + alpha*||w||², without penalizing bias."""
        for n in (20, 100):
            for intercept in (False, True):
                for sparse in (False, True):
                    with self.subTest(n=n, intercept=intercept, sparse=sparse):
                        X = np.tile([[-1.0], [1.0]], (n // 2, 1))
                        y = 2.0 * X[:, 0] + (3.0 if intercept else 0.0)
                        inputs = sp.csr_matrix(X) if sparse else X
                        for solver in ('normal_equation', 'gradient_descent'):
                            model = LinearRegressionFromScratch(
                                fit_intercept=intercept, solver=solver,
                                alpha_ridge=1.0, learning_rate=0.05,
                                n_iterations=1000, clip_predictions=False,
                            ).fit(inputs, y)
                            np.testing.assert_allclose(model.coef_, [2.0 * n / (n + 1)], atol=1e-8)
                            self.assertAlmostEqual(model.intercept_, 3.0 if intercept else 0.0, places=8)

    # -------------------------------------------------------------
    # 3. KIỂM THỬ DECISION TREE REGRESSOR FROM SCRATCH
    # -------------------------------------------------------------
    def test_decision_tree_variance_reduction(self):
        tree = DecisionTreeRegressorFromScratch(max_depth=5, min_samples_split=5, min_samples_leaf=2)
        tree.fit(self.X_train, self.y_train)
        preds = tree.predict(self.X_test)

        self.assertEqual(preds.shape[0], self.y_test.shape[0])
        # Sai số của cây phải nhỏ hơn mô hình dummy đoán giá trị trung bình
        dummy_rmse = compute_rmse(self.y_test, np.full_like(self.y_test, np.mean(self.y_train)))
        tree_rmse = compute_rmse(self.y_test, preds)
        self.assertLess(tree_rmse, dummy_rmse)

    def test_decision_tree_feature_subsampling_and_importances(self):
        tree = DecisionTreeRegressorFromScratch(max_depth=4, max_features="sqrt", random_state=42)
        tree.fit(self.X_train, self.y_train)

        self.assertIsNotNone(tree.feature_importances_)
        self.assertEqual(len(tree.feature_importances_), self.n_features)
        self.assertAlmostEqual(float(np.sum(tree.feature_importances_)), 1.0, places=4)

    # -------------------------------------------------------------
    # 4. KIỂM THỬ RANDOM FOREST REGRESSOR FROM SCRATCH
    # -------------------------------------------------------------
    def test_random_forest_bagging_and_importances(self):
        rf = RandomForestRegressorFromScratch(
            n_estimators=10,
            max_depth=5,
            min_samples_split=5,
            max_features="sqrt",
            random_state=42
        )
        rf.fit(self.X_train, self.y_train)
        preds = rf.predict(self.X_test)

        self.assertEqual(len(rf.trees_), 10)
        self.assertEqual(preds.shape[0], self.y_test.shape[0])

        # Đánh giá Feature Importances
        self.assertIsNotNone(rf.feature_importances_)
        self.assertEqual(len(rf.feature_importances_), self.n_features)
        self.assertAlmostEqual(float(np.sum(rf.feature_importances_)), 1.0, places=3)

    # -------------------------------------------------------------
    # 5. KIỂM THỬ GRADIENT BOOSTING REGRESSOR FROM SCRATCH
    # -------------------------------------------------------------
    def test_gradient_boosting_residuals_convergence(self):
        gb = GradientBoostingRegressorFromScratch(
            n_estimators=15,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        gb.fit(self.X_train, self.y_train)
        preds = gb.predict(self.X_test)

        self.assertEqual(len(gb.trees_), 15)
        self.assertEqual(len(gb.train_loss_history_), 15)
        # Hàm loss trên tập train phải có xu hướng giảm dần qua các lượt boosting
        self.assertLess(gb.train_loss_history_[-1], gb.train_loss_history_[0])

        r2 = compute_r2(self.y_test, preds)
        self.assertGreater(r2, 0.5)

    # -------------------------------------------------------------
    # 6. KIỂM THỬ API FACTORY & TRAIN_AND_EVALUATE
    # -------------------------------------------------------------
    def test_get_models_and_train_and_evaluate(self):
        models = get_models()
        self.assertIn("linear_regression", models)
        self.assertIn("decision_tree", models)
        self.assertIn("random_forest", models)
        self.assertIn("gradient_boosting", models)

        # Chạy thử train_and_evaluate với số cây nhỏ để test nhanh
        fast_models = {
            "lr": LinearRegressionFromScratch(solver="normal_equation"),
            "dt": DecisionTreeRegressorFromScratch(max_depth=3),
        }
        results = train_and_evaluate(fast_models, self.X_train, self.X_test, self.y_train, self.y_test)

        self.assertIn("lr", results)
        self.assertIn("dt", results)
        for name, res in results.items():
            self.assertIn("model", res)
            self.assertIn("metrics", res)
            self.assertIn("y_pred", res)
            self.assertIn("RMSE", res["metrics"])
            self.assertEqual(res["y_pred"].shape[0], self.y_test.shape[0])

    # -------------------------------------------------------------
    # 7. KIỂM THỬ GRID SEARCH & TUNING FROM SCRATCH
    # -------------------------------------------------------------
    def test_grid_search_from_scratch(self):
        param_grid = {
            "max_depth": [2, 4],
            "min_samples_split": [5, 10]
        }
        res = grid_search_from_scratch(
            model_class=DecisionTreeRegressorFromScratch,
            param_grid=param_grid,
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            scoring="rmse"
        )

        self.assertIn("best_estimator", res)
        self.assertIn("best_params", res)
        self.assertIn("best_score", res)
        self.assertEqual(len(res["cv_results"]), 4)
        self.assertIn(res["best_params"]["max_depth"], [2, 4])

    def test_tune_wrappers(self):
        dt_res = tune_decision_tree(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            param_grid={"max_depth": [2, 3]}
        )
        self.assertIn("best_estimator", dt_res)

        all_res = tune_all(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val
        )
        self.assertIn("decision_tree", all_res)
        self.assertIn("random_forest", all_res)
        self.assertIn("gradient_boosting", all_res)

    # -------------------------------------------------------------
    # 8. KIỂM THỬ LƯU VÀ NẠP MODEL / PIPELINE (PERSISTENCE)
    # -------------------------------------------------------------
    def test_save_and_load_artifacts(self):
        model = LinearRegressionFromScratch(solver="normal_equation")
        model.fit(self.X_train, self.y_train)
        orig_preds = model.predict(self.X_test)

        dummy_pipeline = {"status": "fitted_pipeline", "num_cols": ["c1", "c2"]}
        artifact_path = Path(self.temp_dir) / "test_artifact.pkl"

        save_artifacts(dummy_pipeline, model, artifact_path, metadata={"author": "Team"})
        self.assertTrue(artifact_path.exists())

        loaded_bundle = load_artifacts(artifact_path)
        self.assertIn("pipeline", loaded_bundle)
        self.assertIn("model", loaded_bundle)
        self.assertEqual(loaded_bundle["model_type"], "LinearRegressionFromScratch")

        loaded_model = loaded_bundle["model"]
        loaded_preds = loaded_model.predict(self.X_test)

        np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5, atol=1e-5)

    def test_save_and_load_model_and_pipeline_separate(self):
        model = DecisionTreeRegressorFromScratch(max_depth=3)
        model.fit(self.X_train, self.y_train)

        m_path = Path(self.temp_dir) / "sub_model.pkl"
        save_model(model, scaler=None, filepath=m_path)
        m_loaded, _ = load_model(m_path)
        self.assertIsInstance(m_loaded, DecisionTreeRegressorFromScratch)

        p_path = Path(self.temp_dir) / "sub_pipeline.pkl"
        save_pipeline({"pipeline_id": 999}, filepath=p_path)
        p_loaded = load_pipeline(p_path)
        self.assertEqual(p_loaded["pipeline_id"], 999)


if __name__ == "__main__":
    unittest.main()
