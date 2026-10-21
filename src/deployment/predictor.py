"""
predictor.py - Module Triển khai Dịch vụ Dự đoán Doanh số và Kịch bản Kinh doanh
==================================================================================
- predict_sales(model, pipeline, input_data): Dự đoán doanh số quy đổi về thang đo thực tế (BRL)
- demo_prediction(model, pipeline, X_test, y_test, n_samples): Thử nghiệm trên mẫu tập Test
- predict_scenario_new_product(product_spec, model, pipeline): Kịch bản ra mắt sản phẩm mới
- predict_scenario_marketing_campaign(base_product, ad_spend_levels, model, pipeline): Kịch bản tăng ngân sách marketing
- predict_scenario_seasonal_promo(base_product, discount_rates, is_holiday, model, pipeline): Kịch bản khuyến mãi ngày lễ
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.data import feature_engineering


def predict_sales(
    model: Any,
    pipeline: Optional[Any] = None,
    input_data: Union[pd.DataFrame, dict, np.ndarray, sp.spmatrix] = None
) -> np.ndarray:
    """
    Dự đoán doanh số thực tế (đơn vị: BRL - Đồng Real Brazil).
    Tự động áp dụng hàm nghịch đảo Log: sales = max(0, exp(y_log) - 1).
    
    Parameters:
    -----------
    model: Mô hình hồi quy đã huấn luyện From-Scratch
    pipeline: CustomDataPipeline đã fit (cần thiết nếu input là DataFrame hoặc Dict)
    input_data: DataFrame, Dict, hoặc Ma trận đặc trưng số đã qua tiền xử lý
    
    Returns:
    --------
    np.ndarray: Vector doanh số thực tế dự đoán (luôn >= 0)
    """
    if input_data is None:
        raise ValueError("input_data không được để trống!")

    # Trường hợp 1: input_data đã là ma trận số (NumPy hoặc SciPy CSR)
    if isinstance(input_data, (np.ndarray, sp.spmatrix)):
        X_matrix = input_data
    else:
        # Trường hợp 2: input_data là Dict hoặc DataFrame thô
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError(f"Kiểu dữ liệu input_data không được hỗ trợ: {type(input_data)}")

        if pipeline is None:
            raise ValueError("Cần cung cấp đối tượng pipeline để tiền xử lý DataFrame/Dict đầu vào!")
        feature_engineering.validate_sales_prices(df)

        # Kỹ nghệ các đặc trưng phái sinh nếu chưa có đủ
        if getattr(pipeline, "forecast_granularity_", None) == "product_week":
            if config.TIMESTAMP_COLUMN not in df:
                df[config.TIMESTAMP_COLUMN] = pipeline.default_forecast_week_
            # Thiếu lịch sử được hiểu là kịch bản sản phẩm mới với lịch sử 0,
            # không thay bằng lịch sử trung vị của sản phẩm trong tập Train.
            for col in ["past_review_count", "past_review_rating", "customer_past_spend", "customer_past_orders", "customer_state_count", "last_week_sales", "sales_lag_2_weeks", "sales_lag_4_weeks",
                        "sales_mean_4_weeks", "sales_std_4_weeks", "last_week_units", "weeks_since_first_sale"]:
                if col not in df:
                    df[col] = 0.0
        trans_res = pipeline.transform(df)
        X_matrix = trans_res[0] if isinstance(trans_res, tuple) else trans_res

    # Dự đoán trên không gian Log1p
    y_pred_log = model.predict(X_matrix)
    y_pred_log = np.asarray(y_pred_log, dtype=float).ravel()

    # Chuyển đổi ngược về thang đo doanh số thực tế: expm1(y) = exp(y) - 1
    # Kẹp giá trị không âm: max(0.0, sales)
    sales_pred = np.maximum(0.0, np.expm1(y_pred_log))
    return sales_pred


def demo_prediction(
    model: Any,
    pipeline: Optional[Any],
    X_test: Union[np.ndarray, sp.spmatrix],
    y_test: np.ndarray,
    n_samples: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Lấy ngẫu nhiên n_samples từ tập Test để đối chiếu trực tiếp giữa Thực tế và Dự đoán.
    
    Returns:
    --------
    pd.DataFrame: Bảng kết quả đối sánh mẫu thực tế
    """
    rng = np.random.RandomState(random_state)
    total_samples = X_test.shape[0]
    indices = rng.choice(total_samples, size=min(n_samples, total_samples), replace=False)

    X_sample = X_test[indices]
    y_true_log = np.asarray(y_test[indices], dtype=float).ravel()

    # Dự đoán
    y_pred_sales = predict_sales(model=model, pipeline=pipeline, input_data=X_sample)
    y_true_sales = np.maximum(0.0, np.expm1(y_true_log))

    abs_error = np.abs(y_true_sales - y_pred_sales)
    # Sai số phần trăm không xác định khi doanh thu thực bằng 0, thường gặp ở dữ liệu tuần.
    pct_error = np.divide(abs_error, y_true_sales,
                          out=np.full_like(abs_error, np.nan), where=y_true_sales > 0) * 100.0

    demo_df = pd.DataFrame({
        "Index": indices,
        "Actual_Sales_BRL": np.round(y_true_sales, 2),
        "Predicted_Sales_BRL": np.round(y_pred_sales, 2),
        "Absolute_Diff_BRL": np.round(abs_error, 2),
        "Error_Percentage_%": np.round(pct_error, 2)
    })
    return demo_df


def predict_scenario_new_product(
    product_spec: Dict[str, Any],
    model: Any,
    pipeline: Any,
    cold_start_model: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Kịch bản 1: Ra mắt Sản phẩm Mới hoàn toàn (New Product Launch).
    Điền các giá trị kỳ vọng ban đầu cho sản phẩm chưa có lịch sử mua hàng hoặc đánh giá.
    Nếu truyền cold_start_model, sẽ sử dụng mô hình nội dung chuyên biệt cho tuần mở bán.
    """
    # Khởi tạo bản ghi hoàn chỉnh với các giá trị mặc định hợp lý
    base_record = {
        "order_id": "SIM_NEW_001",
        "customer_id": "SIM_CUST_001",
        "product_id": "SIM_PROD_NEW",
        "seller_id": "SIM_SELLER_001",
        "order_purchase_timestamp": product_spec.get(config.TIMESTAMP_COLUMN, getattr(pipeline, "default_forecast_week_", "2018-09-03")),
        "price": product_spec.get("price", 150.0),
        "freight_value": product_spec.get("freight_value", 20.0),
        "product_category_name": product_spec.get("product_category_name", "informatica_acessorios"),
        "product_weight_g": product_spec.get("product_weight_g", 800.0),
        "product_length_cm": product_spec.get("product_length_cm", 25.0),
        "product_height_cm": product_spec.get("product_height_cm", 15.0),
        "product_width_cm": product_spec.get("product_width_cm", 20.0),
        "product_photos_qty": product_spec.get("product_photos_qty", 3),
        "product_name_lenght": product_spec.get("product_name_lenght", 50),
        "product_description_lenght": product_spec.get("product_description_lenght", 600),
        "payment_type": product_spec.get("payment_type", "credit_card"),
        "payment_installments": product_spec.get("payment_installments", 3),
        "customer_state": product_spec.get("customer_state", "SP"),
        "seller_state": product_spec.get("seller_state", "SP"),
        # Sản phẩm mới chưa có lịch sử giao dịch:
        "review_score": product_spec.get("review_score", 4.0),
        "num_reviews": product_spec.get("num_reviews", 0),
        "order_item_id": 1,
        "payment_value": product_spec.get("price", 150.0) + product_spec.get("freight_value", 20.0),
        "ad_spend": product_spec.get("ad_spend", 300.0),
        "discount_rate": product_spec.get("discount_rate", 0.0)
    }

    df_scenario = pd.DataFrame([base_record])
    sales_pred = predict_sales(model=model, pipeline=pipeline, input_data=df_scenario)
    predicted_val = float(sales_pred[0])

    # Nếu có mô hình Cold-Start chuyên biệt cho tuần ra mắt, tính thêm launch sales
    launch_sales_est = None
    if cold_start_model is not None:
        try:
            launch_sales_est = float(cold_start_model.predict(product_spec)[0])
        except Exception:
            launch_sales_est = None

    result = {
        "scenario": "New Product Launch",
        "interpretation": "Dự báo doanh số tuần đầu tiên. Đã tích hợp mô hình Cold-Start nội dung chuyên biệt." if launch_sales_est is not None else "Ước lượng doanh thu một tuần (zero-history proxy); chưa qua kiểm chứng cold-start riêng.",
        "category": base_record["product_category_name"],
        "listed_price_BRL": base_record["price"],
        "ad_spend_BRL": base_record["ad_spend"],
        "predicted_sales_BRL": round(launch_sales_est if launch_sales_est is not None else predicted_val, 2),
        "estimated_units_sold": max(0, int(round((launch_sales_est if launch_sales_est is not None else predicted_val) / max(base_record["price"], 1.0))))
    }
    if launch_sales_est is not None:
        result["cold_start_model_used"] = True
        result["steady_state_proxy_BRL"] = round(predicted_val, 2)
    return result


def predict_cold_start_product(
    product_spec: Dict[str, Any],
    cold_start_model: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Dự báo doanh số tuần mở bán đầu tiên (Launch Week Sales) cho sản phẩm mới ra mắt (Cold-Start).
    Sử dụng đặc trưng nội dung: Giá niêm yết, danh mục, kích thước, khối lượng, số lượng ảnh.
    
    Parameters:
    -----------
    product_spec: Dict chứa thông tin sản phẩm mới (price, product_category_name, product_weight_g, ...)
    cold_start_model: Đối tượng ColdStartProductRegressor (hoặc None để dùng baseline quy luật giá)
    
    Returns:
    --------
    Dict chứa doanh thu dự báo tuần đầu (BRL), sản lượng ước tính và diễn giải nghiệp vụ.
    """
    price = float(product_spec.get("price", 100.0))
    if price <= 0 or not np.isfinite(price):
        raise ValueError(f"Giá sản phẩm phải là số dương hợp lệ: {price}")

    category = str(product_spec.get("product_category_name", "beleza_saude"))
    
    if cold_start_model is not None:
        pred_val = float(cold_start_model.predict(product_spec)[0])
        model_name = type(cold_start_model).__name__
    else:
        # Fallback ước lượng thận trọng: 1 sản phẩm bán trong tuần ra mắt ở mức giá niêm yết
        pred_val = price * 1.0
        model_name = "RuleBasedSingleUnitFallback"

    pred_val = max(0.0, pred_val)
    units_sold = max(0, int(round(pred_val / price)))

    return {
        "scenario": "Cold-Start Launch Week Prediction",
        "model_engine": model_name,
        "category": category,
        "listed_price_BRL": round(price, 2),
        "predicted_launch_sales_BRL": round(pred_val, 2),
        "estimated_launch_units": units_sold,
        "interpretation": (
            "Proxy doanh thu 7 ngày kể từ giao dịch đầu (bao gồm giao dịch đầu đã biết); "
            "giá phải là giá giao dịch đầu. Chưa kiểm chứng dự báo trước ngày niêm yết."
            if cold_start_model is not None else
            "Giả định bán một sản phẩm theo giá nhập; chưa có mô hình đã huấn luyện."
        )
    }


def predict_scenario_marketing_campaign(
    base_product: Dict[str, Any],
    ad_spend_levels: List[float],
    model: Any,
    pipeline: Any,
    elasticity: float = 0.15,
    reference_spend: float = 100.0,
    mode: str = "baseline_only",
    gross_margin: float = 1.0
) -> pd.DataFrame:
    """
    Kịch bản 2: Mô phỏng Phản hồi Ngân sách Tiếp thị (Marketing Ad Spend Sensitivity Simulation).
    
    LƯU Ý KHOA HỌC & NGHIỆP VỤ:
    - Đây là CÔNG CỤ MÔ PHỎNG GIẢ ĐỊNH theo hệ số co giãn kinh tế lượng (AED - Advertising Elasticity
      of Demand), không phải mô hình dự báo tác động marketing nhân quả đã hiệu chỉnh trên dữ liệu thực tế
      (do tập dữ liệu Olist không có thông tin chi tiêu marketing lịch sử).
    - Chỉ số Marketing_ROI_% mặc định là Revenue-based ROI: ((Incremental_Sales * gross_margin - Spend) / Spend) * 100%.
    
    Hỗ trợ 2 chế độ:
      - "baseline_only" (mặc định): Dự báo doanh số cơ sở tự nhiên, giữ NaN cho Incremental Sales
        để trung thực khoa học khi chưa có dữ liệu can thiệp thực tế.
      - "calibrated_diminishing_returns": Áp dụng hàm phản hồi bão hòa phi tuyến (Diminishing Marginal Returns):
            Incremental_Sales(S) = Base_Sales * elasticity * ln(1 + S / S_ref)
            Total_Sales(S) = Base_Sales + Incremental_Sales(S)
    """
    if mode not in {"baseline_only", "calibrated_diminishing_returns"}:
        raise ValueError(f"Chế độ marketing không hợp lệ: {mode}")
    if not np.isfinite(elasticity) or elasticity < 0:
        raise ValueError("elasticity phải hữu hạn và không âm")
    if not np.isfinite(reference_spend) or reference_spend <= 0:
        raise ValueError("reference_spend phải hữu hạn và dương")
    if not np.isfinite(gross_margin) or not 0 <= gross_margin <= 1:
        raise ValueError("gross_margin phải trong [0, 1]")
    for s in ad_spend_levels:
        if s < 0 or not np.isfinite(s):
            raise ValueError(f"Ngân sách quảng cáo không được âm hoặc vô hạn: {s}")

    # Dự báo doanh thu cơ sở tự nhiên (Organic Baseline Sales khi ad_spend = 0)
    rec_zero = dict(base_product)
    rec_zero["ad_spend"] = 0.0
    df_zero = pd.DataFrame([rec_zero])
    base_sales_arr = predict_sales(model=model, pipeline=pipeline, input_data=df_zero)
    base_sales = float(base_sales_arr[0]) if len(base_sales_arr) > 0 else 0.0
    base_sales = max(base_sales, 0.0)  # Bảo toàn dự báo 0 BRL khi không có doanh thu cơ sở

    results = []
    for spend in ad_spend_levels:
        spend_val = float(spend)
        if mode == "baseline_only":
            pred_val = base_sales
            inc_sales = np.nan
            growth_pct = np.nan
            roas = np.nan
            roi_pct = np.nan
            interpretation = "Mô phỏng cơ sở: Doanh thu hiển thị là dự báo cơ sở (chưa có dữ liệu chiến dịch để ước lượng tác động)"
        else:
            # Mô hình kinh tế lượng phản hồi bão hòa phi tuyến
            if spend_val <= 0:
                inc_sales = 0.0
                pred_val = base_sales
                growth_pct = 0.0
                roas = 0.0
                roi_pct = 0.0
            else:
                lift_ratio = elasticity * np.log1p(spend_val / reference_spend)
                inc_sales = base_sales * lift_ratio
                pred_val = base_sales + inc_sales
                growth_pct = lift_ratio * 100.0 if base_sales > 0 else 0.0
                roas = inc_sales / spend_val
                # ROI tính theo biên lợi nhuận gộp (gross margin)
                gross_profit_incremental = inc_sales * gross_margin
                roi_pct = ((gross_profit_incremental - spend_val) / spend_val) * 100.0

            interpretation = (
                f"Mô phỏng giả định phản hồi bão hòa phi tuyến (AED = {elasticity:.2f}); "
                f"Lợi biên giảm dần khi ngân sách tăng."
            )

        results.append({
            "Interpretation": interpretation,
            "Ad_Spend_BRL": round(spend_val, 2),
            "Baseline_Sales_BRL": round(base_sales, 2),
            "Predicted_Sales_BRL": round(pred_val, 2),
            "Incremental_Sales_BRL": round(inc_sales, 2) if not np.isnan(inc_sales) else np.nan,
            "Sales_Growth_%": round(growth_pct, 2) if not np.isnan(growth_pct) else np.nan,
            "ROAS": round(roas, 2) if not np.isnan(roas) else np.nan,
            "Marketing_ROI_%": round(roi_pct, 2) if not np.isnan(roi_pct) else np.nan
        })

    return pd.DataFrame(results)


def simulate_marketing_response(
    base_product: Dict[str, Any],
    ad_spend_levels: List[float],
    model: Any,
    pipeline: Any,
    elasticity: float = 0.15,
    reference_spend: float = 100.0,
    gross_margin: float = 1.0
) -> pd.DataFrame:
    """
    Tiện ích mô phỏng phản hồi tiếp thị kinh tế lượng (Marketing Ad Spend Sensitivity Simulation).
    Tự động áp dụng chế độ calibrated_diminishing_returns.
    """
    return predict_scenario_marketing_campaign(
        base_product=base_product,
        ad_spend_levels=ad_spend_levels,
        model=model,
        pipeline=pipeline,
        elasticity=elasticity,
        reference_spend=reference_spend,
        mode="calibrated_diminishing_returns",
        gross_margin=gross_margin
    )


def calculate_discount_scenario(
    original_price: float,
    discount_rates: List[float],
    assumed_units: int,
    reference_revenue: Optional[float] = None
) -> pd.DataFrame:
    """
    Bảng tính doanh thu theo giả định sản lượng (Assumed Revenue Calculation).
    LƯU Ý NGHIỆP VỤ: ĐÂY LÀ PHÉP TÍNH GIẢ ĐỊNH SỐ HỌC, KHÔNG PHẢI DỰ BÁO TÁC ĐỘNG KHUYẾN MÃI CỦA MÔ HÌNH ML.
    
    Parameters:
    -----------
    original_price: float
        Giá bán gốc hợp lệ (> 0, hữu hạn).
    discount_rates: List[float]
        Danh sách tỷ lệ giảm giá (mỗi giá trị trong khoảng [0.0, 1.0]).
    assumed_units: int
        Sản lượng giả định bán được (số nguyên không âm).
    reference_revenue: Optional[float]
        Doanh thu tham chiếu cần duy trì (mặc định = original_price * assumed_units).
        
    Returns:
    --------
    pd.DataFrame: Bảng kết quả giả định sản lượng và doanh thu theo từng mức chiết khấu.
    """
    import math

    if not np.isfinite(original_price) or original_price <= 0:
        raise ValueError("Giá gốc (original_price) phải là số dương hữu hạn lớn hơn 0!")
    if not isinstance(assumed_units, (int, np.integer)) or assumed_units < 0:
        raise ValueError("Sản lượng giả định (assumed_units) phải là số nguyên không âm!")

    if reference_revenue is None:
        reference_revenue = float(original_price * assumed_units)
    elif not np.isfinite(reference_revenue) or reference_revenue < 0:
        raise ValueError("Doanh thu tham chiếu (reference_revenue) phải là số không âm hữu hạn!")

    records = []
    for rate in discount_rates:
        if not np.isfinite(rate) or not (0.0 <= rate <= 1.0):
            raise ValueError(f"Tỷ lệ giảm giá không hợp lệ: {rate}. Phải nằm trong khoảng [0.0, 1.0]!")

        discounted_price = original_price * (1.0 - rate)
        scenario_revenue = discounted_price * assumed_units

        if reference_revenue == 0:
            required_units = 0
        elif discounted_price == 0:
            required_units = None  # Không thể đạt doanh thu dương với giá bằng 0 BRL
        else:
            required_units = math.ceil(reference_revenue / discounted_price)

        diff_vs_ref = scenario_revenue - reference_revenue

        records.append({
            "Discount_Rate_%": int(round(rate * 100)),
            "Discounted_Price_BRL": round(discounted_price, 2),
            "Assumed_Units": int(assumed_units),
            "Scenario_Revenue_BRL": round(scenario_revenue, 2),
            "Required_Units_For_Ref_Revenue": required_units,
            "Revenue_Diff_vs_Ref_BRL": round(diff_vs_ref, 2)
        })

    return pd.DataFrame(records)


def predict_scenario_seasonal_promo(
    base_product: Dict[str, Any],
    discount_rates: List[float],
    is_holiday: bool = True,
    model: Any = None,
    pipeline: Any = None
) -> pd.DataFrame:
    """
    Kịch bản Khuyến mãi & Giảm giá theo Mùa Lễ hội (Holiday & Promotional Scenario).
    Đã được chuẩn hóa thành Bảng tính doanh thu theo giả định sản lượng.
    Mô hình hiện dự báo doanh thu cơ sở từ lịch sử bán hàng; bảng tính này đánh giá
    doanh thu giả định dựa trên sản lượng tham chiếu, không ước lượng quan hệ nhân quả.
    """
    original_price = float(base_product.get("price", 100.0))
    assumed_units = int(base_product.get("assumed_units", 10))
    ref_revenue = original_price * assumed_units

    calc_df = calculate_discount_scenario(
        original_price=original_price,
        discount_rates=discount_rates,
        assumed_units=assumed_units,
        reference_revenue=ref_revenue
    )

    # Bổ sung các cột định danh để tương thích và hiển thị trực quan
    calc_df["Interpretation"] = "Bảng tính theo giả định sản lượng; chưa xác định tác động can thiệp giảm giá hoặc ngày lễ"
    calc_df["Effective_Price_BRL"] = calc_df["Discounted_Price_BRL"]
    calc_df["Predicted_Sales_BRL"] = calc_df["Scenario_Revenue_BRL"]
    calc_df["Estimated_Units"] = calc_df["Assumed_Units"]
    calc_df["Scenario_Revenue_BRL"] = calc_df["Scenario_Revenue_BRL"]
    calc_df["Required_Units"] = calc_df["Required_Units_For_Ref_Revenue"]

    return calc_df

