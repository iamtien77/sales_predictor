"""
test_deployment.py - Bộ Kiểm thử Đơn vị cho Module Triển khai Dự đoán và Kịch bản Kinh doanh
=============================================================================================
Kiểm thử toàn diện:
1. test_predict_sales_with_matrix(): Dự đoán với ma trận số, chuyển đổi ngược expm1 hợp lệ
2. test_demo_prediction_structure(): Kiểm tra đối chiếu tập Test, cấu trúc cột DataFrame
3. test_scenario_new_product(): Kịch bản ra mắt sản phẩm mới hoàn toàn
4. test_scenario_marketing_campaign(): Kịch bản biến thiên ngân sách quảng cáo
5. test_scenario_seasonal_promo(): Kịch bản chiết khấu khuyến mãi mùa lễ hội
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.deployment import predictor
from src.models.regressors import LinearRegressionFromScratch


class MockPipeline:
    """Mock Pipeline tiền xử lý nhanh phục vụ unit tests"""
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n_samples = len(df)
        # Giả lập ma trận 10 đặc trưng số
        return np.ones((n_samples, 10), dtype=float)


@pytest.fixture
def trained_mock_model():
    """Tạo mô hình hồi quy From-Scratch tuyến tính đơn giản đã fit"""
    X_dummy = np.random.RandomState(42).randn(50, 10)
    # Log sales dao động trong khoảng [2, 6]
    y_dummy = 3.5 + 0.2 * X_dummy[:, 0] + np.random.RandomState(42).randn(50) * 0.1
    model = LinearRegressionFromScratch(fit_intercept=True)
    model.fit(X_dummy, y_dummy)
    return model


@pytest.fixture
def sample_product_dict():
    return {
        "order_id": "TEST_ORD_001",
        "customer_id": "TEST_CUST_001",
        "product_id": "TEST_PROD_001",
        "seller_id": "TEST_SELLER_001",
        "order_purchase_timestamp": "2018-05-15 10:00:00",
        "price": 120.0,
        "freight_value": 15.0,
        "product_category_name": "beleza_saude",
        "product_weight_g": 500.0,
        "product_length_cm": 20.0,
        "product_height_cm": 10.0,
        "product_width_cm": 15.0,
        "product_photos_qty": 2,
        "product_name_lenght": 45,
        "product_description_lenght": 400,
        "payment_type": "credit_card",
        "payment_installments": 2,
        "customer_state": "SP",
        "seller_state": "SP",
        "review_score": 4.5,
        "num_reviews": 12,
        "order_item_id": 1,
        "payment_value": 135.0,
        "ad_spend": 200.0,
        "discount_rate": 0.0
    }


def test_predict_sales_with_matrix(trained_mock_model):
    """Kiểm tra hàm predict_sales nhận ma trận số và xuất ra doanh số không âm"""
    X_test = np.random.randn(5, 10)
    sales_pred = predictor.predict_sales(model=trained_mock_model, pipeline=None, input_data=X_test)

    assert isinstance(sales_pred, np.ndarray)
    assert len(sales_pred) == 5
    assert np.all(sales_pred >= 0.0), "Doanh số thực tế dự đoán phải không âm (>= 0)"
    assert sales_pred.dtype in [np.float32, np.float64]


def test_demo_prediction_structure(trained_mock_model):
    """Kiểm tra cấu trúc DataFrame của demo_prediction"""
    X_test = np.random.randn(20, 10)
    y_test_log = np.random.uniform(2.0, 5.0, size=20)

    demo_df = predictor.demo_prediction(
        model=trained_mock_model,
        pipeline=None,
        X_test=X_test,
        y_test=y_test_log,
        n_samples=5,
        random_state=42
    )

    assert isinstance(demo_df, pd.DataFrame)
    assert len(demo_df) == 5
    expected_cols = [
        "Index", "Actual_Sales_BRL", "Predicted_Sales_BRL",
        "Absolute_Diff_BRL", "Error_Percentage_%"
    ]
    for col in expected_cols:
        assert col in demo_df.columns
    assert np.all(demo_df["Predicted_Sales_BRL"] >= 0.0)


def test_scenario_new_product(trained_mock_model):
    """Kiểm tra kịch bản sản phẩm mới ra mắt"""
    pipe = MockPipeline()
    product_spec = {
        "price": 250.0,
        "product_category_name": "relogios_presentes",
        "ad_spend": 500.0
    }

    result = predictor.predict_scenario_new_product(
        product_spec=product_spec,
        model=trained_mock_model,
        pipeline=pipe
    )

    assert isinstance(result, dict)
    assert result["scenario"] == "New Product Launch"
    assert result["category"] == "relogios_presentes"
    assert result["listed_price_BRL"] == 250.0
    assert result["predicted_sales_BRL"] >= 0.0
    assert result["estimated_units_sold"] == max(0, round(result["predicted_sales_BRL"] / 250.0))


def test_scenario_marketing_campaign(trained_mock_model, sample_product_dict):
    """Kiểm tra kịch bản biến thiên ngân sách marketing"""
    pipe = MockPipeline()
    ad_levels = [100.0, 300.0, 1000.0]

    df_campaign = predictor.predict_scenario_marketing_campaign(
        base_product=sample_product_dict,
        ad_spend_levels=ad_levels,
        model=trained_mock_model,
        pipeline=pipe
    )

    assert isinstance(df_campaign, pd.DataFrame)
    assert len(df_campaign) == 3
    assert "Ad_Spend_BRL" in df_campaign.columns
    assert "Predicted_Sales_BRL" in df_campaign.columns
    assert "Sales_Growth_%" in df_campaign.columns
    assert np.all(df_campaign["Predicted_Sales_BRL"] >= 0.0)


def test_scenario_seasonal_promo(trained_mock_model, sample_product_dict):
    """Kiểm tra kịch bản khuyến mãi ngày lễ"""
    pipe = MockPipeline()
    discounts = [0.0, 0.1, 0.2, 0.4]

    df_promo = predictor.predict_scenario_seasonal_promo(
        base_product=sample_product_dict,
        discount_rates=discounts,
        is_holiday=True,
        model=trained_mock_model,
        pipeline=pipe
    )

    assert isinstance(df_promo, pd.DataFrame)
    assert len(df_promo) == 4
    assert "Discount_Rate_%" in df_promo.columns
    assert "Effective_Price_BRL" in df_promo.columns
    assert "Predicted_Sales_BRL" in df_promo.columns
    assert "Estimated_Units" in df_promo.columns
    assert np.all(df_promo["Effective_Price_BRL"] <= sample_product_dict["price"])


def test_calculate_discount_scenario_values():
    """Kiểm tra ví dụ chuẩn trong tài liệu: giá 100 BRL, giảm 20%, bán 10 sp -> 800 BRL, cần 13 sp giữ 1.000 BRL"""
    df = predictor.calculate_discount_scenario(
        original_price=100.0,
        discount_rates=[0.0, 0.1, 0.2],
        assumed_units=10,
        reference_revenue=1000.0
    )
    assert len(df) == 3
    # Mức giảm 20%
    row_20 = df[df["Discount_Rate_%"] == 20].iloc[0]
    assert row_20["Discounted_Price_BRL"] == 80.0
    assert row_20["Scenario_Revenue_BRL"] == 800.0
    assert row_20["Required_Units_For_Ref_Revenue"] == 13


def test_calculate_discount_scenario_zero_price():
    """Giảm 100% (giá = 0) không gây lỗi chia cho 0 và trả về Required_Units_For_Ref_Revenue = None"""
    df = predictor.calculate_discount_scenario(
        original_price=100.0,
        discount_rates=[1.0],
        assumed_units=10,
        reference_revenue=1000.0
    )
    assert df.iloc[0]["Discounted_Price_BRL"] == 0.0
    assert df.iloc[0]["Scenario_Revenue_BRL"] == 0.0
    assert df.iloc[0]["Required_Units_For_Ref_Revenue"] is None


def test_calculate_discount_scenario_invalid_inputs():
    """Từ chối rõ ràng các đầu vào không hợp lệ"""
    import pytest
    with pytest.raises(ValueError):
        predictor.calculate_discount_scenario(original_price=-50.0, discount_rates=[0.1], assumed_units=5)
    with pytest.raises(ValueError):
        predictor.calculate_discount_scenario(original_price=100.0, discount_rates=[-0.2], assumed_units=5)
    with pytest.raises(ValueError):
        predictor.calculate_discount_scenario(original_price=100.0, discount_rates=[1.5], assumed_units=5)
    with pytest.raises(ValueError):
        predictor.calculate_discount_scenario(original_price=100.0, discount_rates=[0.1], assumed_units=-2)


if __name__ == "__main__":
    pytest.main(["-v", __file__])

