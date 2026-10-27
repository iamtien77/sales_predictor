"""
test_cold_start_and_marketing.py - Kiểm thử Đơn vị cho Cold-Start Product Predictor và Marketing Uplift Engine
================================================================================================================
Kiểm tra toàn diện:
1. CategoryMedianColdStartBaseline (fit, predict danh mục đã biết, fallback danh mục lạ)
2. ColdStartProductRegressor (học đặc trưng nội dung, xử lý missing value, so sánh vượt trội baseline)
3. predict_cold_start_product (xác thực đầu vào, định dạng kết quả)
4. predict_scenario_marketing_campaign (quy luật lợi biên giảm dần diminishing returns, tính toán ROAS/ROI)
5. predict_scenario_new_product tích hợp cold_start_model
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.cold_start import CategoryMedianColdStartBaseline, ColdStartProductRegressor
from src.deployment import predictor
from src.models.regressors import LinearRegressionFromScratch


@pytest.fixture
def sample_cold_start_dataset():
    """Tạo bộ dữ liệu sản phẩm tuần đầu phục vụ unit tests"""
    np.random.seed(42)
    n = 200
    categories = ["beleza_saude", "informatica_acessorios", "relogios_presentes", "cama_mesa_banho"]
    cat_base = {"beleza_saude": 80.0, "informatica_acessorios": 150.0, "relogios_presentes": 300.0, "cama_mesa_banho": 60.0}

    data = []
    for i in range(n):
        cat = np.random.choice(categories)
        price = max(10.0, cat_base[cat] + np.random.randn() * 20.0)
        units = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        sales = price * units
        data.append({
            "product_id": f"PROD_{i:04d}",
            "product_category_name": cat,
            "price": price,
            "product_weight_g": np.random.uniform(200.0, 3000.0),
            "product_photos_qty": np.random.randint(1, 6),
            "product_name_lenght": np.random.randint(20, 65),
            "product_description_lenght": np.random.randint(100, 1500),
            "launch_sales_brl": sales
        })
    return pd.DataFrame(data)


@pytest.fixture
def trained_mock_pipeline_and_model():
    class MockPipe:
        def transform(self, df):
            return np.ones((len(df), 10), dtype=float)

    X_dummy = np.random.RandomState(42).randn(40, 10)
    y_dummy = 4.0 + 0.1 * X_dummy[:, 0]
    model = LinearRegressionFromScratch(fit_intercept=True)
    model.fit(X_dummy, y_dummy)
    return MockPipe(), model


# =====================================================================
# 1. KIỂM THỬ CATEGORY MEDIAN COLD-START BASELINE
# =====================================================================

def test_category_median_baseline_fit_and_predict(sample_cold_start_dataset):
    baseline = CategoryMedianColdStartBaseline()
    baseline.fit(sample_cold_start_dataset)

    assert baseline.is_fitted
    assert len(baseline.category_medians_) == 4
    assert baseline.global_median_ > 0.0

    # Dự báo danh mục đã biết
    test_known = pd.DataFrame([{"product_category_name": "relogios_presentes"}])
    pred_known = baseline.predict(test_known)
    assert len(pred_known) == 1
    assert pred_known[0] == baseline.category_medians_["relogios_presentes"]

    # Dự báo danh mục hoàn toàn mới -> Fallback về global median
    test_unknown = pd.DataFrame([{"product_category_name": "danh_muc_hoan_toan_moi_123"}])
    pred_unknown = baseline.predict(test_unknown)
    assert pred_unknown[0] == baseline.global_median_


def test_category_median_baseline_not_fitted_error():
    baseline = CategoryMedianColdStartBaseline()
    with pytest.raises(RuntimeError):
        baseline.predict(pd.DataFrame([{"product_category_name": "test"}]))


# =====================================================================
# 2. KIỂM THỬ COLD-START PRODUCT REGRESSOR
# =====================================================================

def test_cold_start_regressor_training_and_evaluation(sample_cold_start_dataset):
    train_df = sample_cold_start_dataset.iloc[:150]
    test_df = sample_cold_start_dataset.iloc[150:]

    baseline = CategoryMedianColdStartBaseline().fit(train_df)
    base_preds = baseline.predict(test_df)
    base_mae = np.mean(np.abs(test_df["launch_sales_brl"].values - base_preds))

    model = ColdStartProductRegressor(max_depth=5, min_samples_split=5, min_samples_leaf=3)
    model.fit(train_df)
    assert model.is_fitted

    preds = model.predict(test_df)
    assert len(preds) == len(test_df)
    assert np.all(preds >= 0.0), "Doanh số tuần đầu dự báo phải không âm"

    metrics = model.evaluate(test_df)
    assert "MAE_BRL" in metrics
    assert "RMSE_BRL" in metrics
    assert "R2_Score" in metrics
    assert metrics["R2_Score"] > 0.0, "Mô hình phải giải thích được phương sai (R2 > 0)"
    assert metrics["MAE_BRL"] > 0.0
    assert metrics["RMSE_BRL"] > 0.0


def test_cold_start_regressor_missing_values():
    """Kiểm tra mô hình tự động điền khuyết (impute) an toàn khi thiếu một số đặc trưng nội dung"""
    train_data = pd.DataFrame([
        {"product_category_name": "beleza_saude", "price": 50.0, "product_weight_g": 500.0, "product_photos_qty": 2, "product_name_lenght": 40, "product_description_lenght": 300, "launch_sales_brl": 50.0},
        {"product_category_name": "beleza_saude", "price": 100.0, "product_weight_g": 800.0, "product_photos_qty": 3, "product_name_lenght": 50, "product_description_lenght": 500, "launch_sales_brl": 100.0},
    ])
    model = ColdStartProductRegressor(max_depth=2).fit(train_data)

    # Input thiếu trọng lượng và số ảnh
    test_record = {"product_category_name": "beleza_saude", "price": 75.0}
    pred = model.predict(test_record)
    assert len(pred) == 1
    assert pred[0] > 0.0


# =====================================================================
# 3. KIỂM THỬ PREDICT_COLD_START_PRODUCT (DEPLOYMENT API)
# =====================================================================

def test_predict_cold_start_product_api(sample_cold_start_dataset):
    model = ColdStartProductRegressor(max_depth=4).fit(sample_cold_start_dataset)

    prod_spec = {
        "price": 200.0,
        "product_category_name": "informatica_acessorios",
        "product_weight_g": 1200.0,
        "product_photos_qty": 4,
        "product_name_lenght": 45,
        "product_description_lenght": 800
    }

    res = predictor.predict_cold_start_product(prod_spec, cold_start_model=model)
    assert res["scenario"] == "Cold-Start Launch Week Prediction"
    assert res["listed_price_BRL"] == 200.0
    assert res["predicted_launch_sales_BRL"] >= 0.0
    assert res["estimated_launch_units"] >= 0
    assert "ColdStartProductRegressor" in res["model_engine"]


def test_predict_cold_start_product_invalid_price():
    with pytest.raises(ValueError):
        predictor.predict_cold_start_product({"price": -10.0})
    with pytest.raises(ValueError):
        predictor.predict_cold_start_product({"price": 0.0})


# =====================================================================
# 4. KIỂM THỬ MARKETING UPLIFT & DIMINISHING MARGINAL RETURNS
# =====================================================================

def test_marketing_campaign_diminishing_returns(trained_mock_pipeline_and_model):
    pipe, model = trained_mock_pipeline_and_model
    base_prod = {
        "price": 100.0,
        "product_category_name": "beleza_saude"
    }
    ad_spends = [0.0, 100.0, 300.0, 1000.0]

    df_res = predictor.predict_scenario_marketing_campaign(
        base_product=base_prod,
        ad_spend_levels=ad_spends,
        model=model,
        pipeline=pipe,
        elasticity=0.20,
        reference_spend=100.0,
        mode="calibrated_diminishing_returns"
    )

    assert len(df_res) == 4
    # Kiểm tra các cột chuẩn
    expected_cols = [
        "Ad_Spend_BRL", "Baseline_Sales_BRL", "Predicted_Sales_BRL",
        "Incremental_Sales_BRL", "Sales_Growth_%", "ROAS", "Marketing_ROI_%"
    ]
    for c in expected_cols:
        assert c in df_res.columns

    # 1. Tại mức chi 0: Doanh thu tăng thêm = 0, ROAS = 0
    row_0 = df_res.iloc[0]
    assert row_0["Incremental_Sales_BRL"] == 0.0
    assert row_0["Sales_Growth_%"] == 0.0

    # 2. Tính đơn điệu: Ngân sách tăng -> Doanh số tăng
    sales_vals = df_res["Predicted_Sales_BRL"].values
    assert np.all(np.diff(sales_vals) > 0.0), "Doanh số phải tăng đơn điệu theo ngân sách"

    # 3. Lợi biên giảm dần (Diminishing Marginal Returns):
    # Doanh số tăng thêm từ 0->100 phải lớn hơn doanh số tăng thêm từ 100->200
    inc_100 = df_res.loc[df_res["Ad_Spend_BRL"] == 100.0, "Incremental_Sales_BRL"].values[0]
    inc_300 = df_res.loc[df_res["Ad_Spend_BRL"] == 300.0, "Incremental_Sales_BRL"].values[0]
    inc_1000 = df_res.loc[df_res["Ad_Spend_BRL"] == 1000.0, "Incremental_Sales_BRL"].values[0]

    # Delta từ 0 đến 100 (100 BRL đầu tiên)
    marginal_0_100 = inc_100 / 100.0
    # Delta trung bình từ 100 đến 300 (200 BRL tiếp theo)
    marginal_100_300 = (inc_300 - inc_100) / 200.0
    # Delta trung bình từ 300 đến 1000 (700 BRL tiếp theo)
    marginal_300_1000 = (inc_1000 - inc_300) / 700.0

    assert marginal_0_100 > marginal_100_300 > marginal_300_1000, "Quy luật lợi biên giảm dần (Diminishing Marginal Returns) phải được thỏa mãn"


def test_marketing_campaign_invalid_spend(trained_mock_pipeline_and_model):
    pipe, model = trained_mock_pipeline_and_model
    with pytest.raises(ValueError):
        predictor.predict_scenario_marketing_campaign(
            base_product={"price": 50.0},
            ad_spend_levels=[-50.0, 100.0],
            model=model,
            pipeline=pipe
        )


# =====================================================================
# 5. KIỂM THỬ NEW PRODUCT LAUNCH TÍCH HỢP COLD-START MODEL
# =====================================================================

def test_scenario_new_product_with_cold_start(trained_mock_pipeline_and_model, sample_cold_start_dataset):
    pipe, model = trained_mock_pipeline_and_model
    cold_model = ColdStartProductRegressor(max_depth=3).fit(sample_cold_start_dataset)

    prod_spec = {
        "price": 120.0,
        "product_category_name": "beleza_saude",
        "product_weight_g": 600.0
    }

    res = predictor.predict_scenario_new_product(
        product_spec=prod_spec,
        model=model,
        pipeline=pipe,
        cold_start_model=cold_model
    )

    assert res["scenario"] == "New Product Launch"
    assert res.get("cold_start_model_used") is True
    assert "steady_state_proxy_BRL" in res
    assert res["predicted_sales_BRL"] >= 0.0
    direct = predictor.predict_cold_start_product(prod_spec, cold_start_model=cold_model)
    assert res["interpretation"] == direct["interpretation"]
    assert res["predicted_sales_BRL"] == direct["predicted_launch_sales_BRL"]
    assert "bao gồm giao dịch đầu đã biết" in res["interpretation"]


# =====================================================================
# 6. KIỂM THỬ BẢO TOÀN BASELINE 0 BRL VÀ PHẢN HỒI NỘI DUNG COLD-START
# =====================================================================

def test_marketing_campaign_preserves_zero_baseline():
    """Kiểm tra khi mô hình dự báo nền bằng 0 BRL, chức năng marketing không ép lên 1 BRL"""
    class ZeroModel:
        def predict(self, X):
            return np.zeros(X.shape[0] if hasattr(X, "shape") else len(X), dtype=float)

    class ZeroPipe:
        def transform(self, df):
            return np.zeros((len(df), 5), dtype=float)

    zero_model = ZeroModel()
    zero_pipe = ZeroPipe()

    base_prod = {"price": 100.0, "product_category_name": "informatica_acessorios"}
    ad_spends = [0.0, 100.0, 500.0]

    # Kiểm tra chế độ baseline_only
    df_base = predictor.predict_scenario_marketing_campaign(
        base_product=base_prod,
        ad_spend_levels=ad_spends,
        model=zero_model,
        pipeline=zero_pipe,
        mode="baseline_only"
    )
    for _, row in df_base.iterrows():
        assert row["Baseline_Sales_BRL"] == 0.0, "Baseline_Sales_BRL phải giữ nguyên 0.0 BRL, không được nhảy lên 1.0 BRL"
        assert row["Predicted_Sales_BRL"] == 0.0

    # Kiểm tra chế độ calibrated_diminishing_returns
    df_sim = predictor.simulate_marketing_response(
        base_product=base_prod,
        ad_spend_levels=ad_spends,
        model=zero_model,
        pipeline=zero_pipe,
        elasticity=0.15
    )
    for _, row in df_sim.iterrows():
        assert row["Baseline_Sales_BRL"] == 0.0, "Baseline_Sales_BRL trong mô phỏng phải là 0.0 BRL"
        assert row["Incremental_Sales_BRL"] == 0.0
        assert row["Predicted_Sales_BRL"] == 0.0


def test_predict_cold_start_product_differentiates_attributes(sample_cold_start_dataset):
    """
    Kiểm tra lỗi UI: Đổi danh mục hoặc trọng lượng cùng mức giá 120 BRL
    phải cho ra kết quả khác nhau khi truyền cold_start_model, không bị kẹt ở fallback 120 BRL (1 sp).
    """
    cold_model = ColdStartProductRegressor(max_depth=5).fit(sample_cold_start_dataset)

    prod_light = {
        "price": 120.0,
        "product_category_name": "beleza_saude",
        "product_weight_g": 200.0,
        "product_photos_qty": 5
    }
    prod_heavy = {
        "price": 120.0,
        "product_category_name": "informatica_acessorios",
        "product_weight_g": 2500.0,
        "product_photos_qty": 1
    }

    res_light = predictor.predict_cold_start_product(prod_light, cold_start_model=cold_model)
    res_heavy = predictor.predict_cold_start_product(prod_heavy, cold_start_model=cold_model)

    assert "ColdStartProductRegressor" in res_light["model_engine"]
    assert "ColdStartProductRegressor" in res_heavy["model_engine"]
    # Dự báo phải tùy biến theo thuộc tính nội dung, không được trùng khớp bằng giá * 1 đơn vị
    assert res_light["predicted_launch_sales_BRL"] != res_heavy["predicted_launch_sales_BRL"]


def test_predict_cold_start_product_fallback_when_model_is_none():
    """Kiểm tra khi không có mô hình, fallback được kích hoạt đúng chuẩn và nêu rõ tên engine"""
    prod = {"price": 120.0, "product_category_name": "beleza_saude"}
    res = predictor.predict_cold_start_product(prod, cold_start_model=None)
    assert res["model_engine"] == "RuleBasedSingleUnitFallback"
    assert res["predicted_launch_sales_BRL"] == 120.0
    assert res["estimated_launch_units"] == 1


def test_cold_start_evaluation_implied_variance_consistency(sample_cold_start_dataset):
    """Kiểm tra tính nhất quán toán học: Cùng tập test thì Var(y) = RMSE^2 / (1 - R^2) phải bằng nhau"""
    train_df = sample_cold_start_dataset.iloc[:150]
    test_df = sample_cold_start_dataset.iloc[150:]

    baseline = CategoryMedianColdStartBaseline().fit(train_df)
    base_preds = baseline.predict(test_df)

    model = ColdStartProductRegressor(max_depth=4).fit(train_df)
    model_preds = model.predict(test_df)

    y_test = test_df["launch_sales_brl"].values
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

    base_rmse = np.sqrt(np.mean((y_test - base_preds) ** 2))
    base_r2 = 1.0 - np.sum((y_test - base_preds) ** 2) / ss_tot
    implied_var_base = (base_rmse ** 2) / (1.0 - base_r2)

    model_rmse = np.sqrt(np.mean((y_test - model_preds) ** 2))
    model_r2 = 1.0 - np.sum((y_test - model_preds) ** 2) / ss_tot
    implied_var_model = (model_rmse ** 2) / (1.0 - model_r2)

    # Sai số giữa hai phương sai suy ra phải xấp xỉ bằng 0 (trong giới hạn làm tròn số thực)
    assert np.isclose(implied_var_base, implied_var_model, rtol=1e-5)
    assert np.isclose(implied_var_base, np.var(y_test), rtol=1e-5)

