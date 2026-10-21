"""Giao diện dự báo dùng chung artifact và bộ kiểm tra của pipeline."""
import pandas as pd
import streamlit as st
from datetime import date, timedelta
from src import config
from src.models.persistence import load_artifacts
from src.deployment.predictor import predict_sales, calculate_discount_scenario

st.set_page_config(page_title="Dự báo doanh thu sản phẩm", layout="wide")
st.title("Dự báo doanh thu sản phẩm theo tuần")
st.caption("Đơn vị BRL. Dữ liệu lịch sử phải có trước tuần dự báo.")

try:
    bundle = load_artifacts()
except FileNotFoundError:
    st.error("Chưa có mô hình. Hãy chạy python main.py để huấn luyện trước.")
    st.stop()
pipeline, model = bundle['pipeline'], bundle['model']
cold_start_model = bundle.get('cold_start_model')
model_name = bundle.get('metadata', {}).get('best_params', {}).get('selected_model', 'Two-Stage Hurdle')
cs_status = "Đã tích hợp mô hình ColdStartProductRegressor" if cold_start_model is not None else "Đang dùng fallback đơn vị cơ sở"
st.info(f"Mô hình tuần: **{model_name}**. Proxy sau giao dịch đầu: **{cs_status}**.")
with st.expander("Thông tin mô hình, kiến trúc và giới hạn kỹ thuật"):
    st.json(bundle.get('metadata', {}))
    st.markdown("""
    - **Cấp độ triển khai:** Local Interactive Demo & Verification Prototype phục vụ thẩm định học thuật và kiểm chứng cục bộ.
    - **Bản chất Hurdle:** Nhân xác suất bán với dự báo dương nghịch đảo từ log; không mặc nhiên là kỳ vọng số học của doanh thu.
    - **Giới hạn dữ liệu:** Dữ liệu Olist lịch sử (2016–2018). Các tuần không bán chưa phân biệt được giữa hết hàng tồn kho (*Stockout*) và không có nhu cầu (*Zero Demand*).
    """)

single, batch, promo_tab, cold_start_tab, marketing_tab = st.tabs([
    "Một sản phẩm",
    "Nhiều sản phẩm từ CSV",
    "Mô phỏng giả định giảm giá",
    "7 ngày sau lần bán đầu",
    "Mô phỏng Tiếp thị (Marketing Uplift)"
])
with single:
    with st.form('prediction'):
        today = date.today()
        next_monday = today + timedelta(days=(7 - today.weekday()) % 7)
        week = st.date_input("Ngày đầu tuần dự báo (thứ Hai)", value=next_monday)
        category = st.text_input("Danh mục sản phẩm", "informatica_acessorios")
        price = st.number_input("Giá quan sát gần nhất (BRL)", min_value=0.01, value=100.0)
        row = {config.TIMESTAMP_COLUMN: str(week), 'category': category, 'price': price}
        labels = {'last_week_sales': 'Doanh thu tuần trước', 'sales_lag_2_weeks': 'Doanh thu cách 2 tuần',
                  'sales_lag_4_weeks': 'Doanh thu cách 4 tuần', 'sales_mean_4_weeks': 'Doanh thu trung bình 4 tuần',
                  'sales_std_4_weeks': 'Độ lệch chuẩn doanh thu 4 tuần', 'last_week_units': 'Số sản phẩm bán tuần trước',
                  'weeks_since_first_sale': 'Số tuần từ lần bán đầu', 'past_review_count': 'Số đánh giá đã nhận',
                  'past_review_rating': 'Điểm đánh giá trung bình đã nhận', 'customer_past_spend': 'Chi tiêu quá khứ trung bình của nhóm khách mua gần nhất',
                  'customer_past_orders': 'Số đơn quá khứ trung bình của nhóm khách mua gần nhất',
                  'customer_state_count': 'Số bang của nhóm khách mua gần nhất'}
        for key, label in labels.items():
            row[key] = st.number_input(label, min_value=0.0, value=0.0, key=key)
        submitted = st.form_submit_button("Dự báo")
    if submitted:
        try:
            st.metric("Doanh thu dự báo", f"{predict_sales(model, pipeline, row)[0]:,.2f} BRL")
        except (ValueError, TypeError) as exc:
            st.error(f"Đầu vào không hợp lệ: {exc}")
with batch:
    st.write("CSV UTF-8 cần có category, price, order_purchase_timestamp (thứ Hai). Các cột lịch sử giống biểu mẫu; thiếu lịch sử được hiểu là 0.")
    st.download_button("Tải mẫu CSV", pd.DataFrame([row]).to_csv(index=False).encode('utf-8-sig'), 'mau_du_bao.csv', 'text/csv')
    upload = st.file_uploader("Chọn CSV", type=['csv'])
    if upload is not None and st.button("Dự báo hàng loạt"):
        try:
            frame = pd.read_csv(upload)
            required = {'category', 'price', config.TIMESTAMP_COLUMN}
            if not required.issubset(frame):
                raise ValueError(f"Thiếu cột: {sorted(required - set(frame.columns))}")
            if frame.empty or len(frame) > 10000:
                raise ValueError("Mỗi tệp cần từ 1 đến 10.000 dòng.")
            frame['predicted_sales_brl'] = predict_sales(model, pipeline, frame)
            st.dataframe(frame)
            st.download_button("Tải kết quả", frame.to_csv(index=False).encode('utf-8-sig'), 'du_bao.csv', 'text/csv')
        except (ValueError, TypeError) as exc:
            st.error(f"Không xử lý được dữ liệu: {exc}")

with promo_tab:
    st.markdown("### Bảng tính Doanh thu theo Giả định Sản lượng")
    st.caption("Lưu ý: Bảng này thực hiện phép tính số học thuần túy theo sản lượng giả định; mô hình ML không dự báo tác động can thiệp của khuyến mãi.")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        p_price = st.number_input("Giá gốc sản phẩm (BRL)", min_value=1.0, value=100.0, step=10.0)
    with col_p2:
        p_units = st.number_input("Sản lượng giả định bán được", min_value=1, value=10, step=1)
    with col_p3:
        p_ref = st.number_input("Doanh thu tham chiếu cần giữ (BRL)", min_value=0.0, value=float(p_price * p_units), step=50.0)

    if st.button("Tính toán bảng kịch bản giả định"):
        calc_result = calculate_discount_scenario(
            original_price=p_price,
            discount_rates=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
            assumed_units=int(p_units),
            reference_revenue=p_ref
        )
        st.dataframe(calc_result, use_container_width=True)

with cold_start_tab:
    st.markdown("### Doanh thu 7 ngày kể từ giao dịch đầu tiên")
    st.caption("Proxy sau lần bán đầu, bao gồm giao dịch đầu đã biết; chưa kiểm chứng trước ngày niêm yết hoặc cho sản phẩm chưa bán. Mô hình cây là ứng viên thực nghiệm, cần đối chiếu baseline một sản phẩm.")
    from src.deployment.predictor import predict_cold_start_product
    from src.models.cold_start import ColdStartProductRegressor

    col_cs1, col_cs2 = st.columns(2)
    with col_cs1:
        cs_cat = st.text_input("Danh mục sản phẩm mới", "beleza_saude", key="cs_cat")
        cs_price = st.number_input("Giá tại giao dịch đầu tiên (BRL)", min_value=1.0, value=120.0, step=10.0, key="cs_price")
        cs_weight = st.number_input("Khối lượng đóng gói (gram)", min_value=10.0, value=650.0, step=50.0, key="cs_weight")
    with col_cs2:
        cs_photos = st.number_input("Số lượng ảnh đại diện sản phẩm", min_value=1, value=3, step=1, key="cs_photos")
        cs_name_len = st.number_input("Độ dài tiêu đề (ký tự)", min_value=5, value=45, step=5, key="cs_name_len")
        cs_desc_len = st.number_input("Độ dài bài mô tả (ký tự)", min_value=20, value=550, step=50, key="cs_desc_len")

    if st.button("Ước lượng doanh thu cửa sổ 7 ngày"):
        cs_spec = {
            "product_category_name": cs_cat,
            "price": cs_price,
            "product_weight_g": cs_weight,
            "product_photos_qty": cs_photos,
            "product_name_lenght": cs_name_len,
            "product_description_lenght": cs_desc_len
        }
        res_cs = predict_cold_start_product(cs_spec, cold_start_model=cold_start_model)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Doanh thu 7 ngày dự báo", f"{res_cs['predicted_launch_sales_BRL']:,.2f} BRL")
        with col_m2:
            st.metric("Sản lượng 7 ngày ước tính", f"{res_cs['estimated_launch_units']} sản phẩm")
        st.caption(f"Baseline một sản phẩm: {cs_price:,.2f} BRL. Không mặc định mô hình tốt hơn baseline.")
        st.info(f"Động cơ tính toán: **{res_cs['model_engine']}**")
        st.success(res_cs["interpretation"])

with marketing_tab:
    st.markdown("### 📈 Mô Phỏng Độ Nhạy Ngân Sách Tiếp Thị (Marketing Ad Spend Sensitivity)")
    st.caption("Công cụ mô phỏng giả định phản hồi bão hòa phi tuyến theo hệ số co giãn kinh tế lượng (AED - Diminishing Marginal Returns Log-Response). Lưu ý: Đây là công cụ mô phỏng giả định theo kịch bản độ nhạy, không phải mô hình dự báo tác động marketing nhân quả đã hiệu chỉnh trên dữ liệu thực tế do Olist không có số liệu chi phí quảng cáo.")
    from src.deployment.predictor import simulate_marketing_response

    col_mk1, col_mk2 = st.columns(2)
    with col_mk1:
        mk_cat = st.text_input("Danh mục chạy chiến dịch", "informatica_acessorios", key="mk_cat")
        mk_price = st.number_input("Giá bán trung bình (BRL)", min_value=1.0, value=150.0, step=10.0, key="mk_price")
        mk_margin = st.slider("Biên lợi nhuận gộp giả định (%)", min_value=10, max_value=100, value=100, step=5,
                              help="Mặc định 100% tính Marketing ROI trên toàn bộ doanh thu tăng thêm. Giảm xuống biên gộp thực tế (ví dụ 30%) để tính ROI sau giá vốn.")
    with col_mk2:
        mk_elasticity = st.slider("Hệ số co giãn quảng cáo (AED)", min_value=0.05, max_value=0.30, value=0.15, step=0.01,
                                 help="Hệ số giả định để phân tích độ nhạy; chưa hiệu chỉnh từ dữ liệu chiến dịch.")
        mk_ref_spend = st.number_input("Ngân sách chuẩn hóa tham chiếu (BRL)", min_value=10.0, value=100.0, step=50.0)

    if st.button("Chạy mô phỏng hiệu quả ngân sách quảng cáo"):
        test_levels = [0.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0]
        mk_base_prod = {
            "price": mk_price,
            "product_category_name": mk_cat,
            config.TIMESTAMP_COLUMN: "2018-09-03"
        }
        df_mk = simulate_marketing_response(
            base_product=mk_base_prod,
            ad_spend_levels=test_levels,
            model=model,
            pipeline=pipeline,
            elasticity=mk_elasticity,
            reference_spend=mk_ref_spend,
            gross_margin=mk_margin / 100.0
        )
        st.dataframe(df_mk, use_container_width=True)
        st.line_chart(df_mk.set_index("Ad_Spend_BRL")[["Predicted_Sales_BRL", "Baseline_Sales_BRL"]])
        st.info("Nhận xét kinh tế học: Doanh thu tăng mạnh ở những đồng ngân sách đầu tiên, sau đó tốc độ tăng giảm dần theo định luật lợi biên giảm dần (Diminishing Marginal Returns).")


