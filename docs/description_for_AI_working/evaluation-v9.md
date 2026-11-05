Bản v9. Xây Dựng Động Cơ Mô Phỏng Tiếp Thị Kinh Tế Lượng (Marketing Uplift & ROAS Engine)

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là nâng cấp kịch bản marketing từ mức quy ước dữ liệu tĩnh (trả về NaN do thiếu dữ liệu chiến dịch trong log Olist) thành động cơ mô phỏng tiếp thị kinh tế lượng chuẩn mực, ứng dụng hàm bão hòa phi tuyến và định luật lợi biên giảm dần để lượng hóa doanh thu tăng thêm và tỷ suất sinh lời ROAS.

Bước 1. Cơ sở toán học kinh tế lượng (Advertising Elasticity of Demand - AED)
Do nhật ký Olist không ghi nhận chi phí quảng cáo cấp sản phẩm, hệ thống áp dụng hàm phản hồi tiếp thị bão hòa chuẩn kinh tế học (Diminishing Marginal Returns Log-Response):
Incremental_Sales(S) = Base_Sales * elasticity * ln(1 + S / S_ref)
Total_Sales(S) = Base_Sales + Incremental_Sales(S)
Trong đó elasticity (AED) nằm trong khoảng 0.10 - 0.20 theo chuẩn mực thương mại điện tử, và S_ref là ngân sách tham chiếu chuẩn hóa (100 BRL).

Bước 2. Kiểm chứng quy luật lợi biên giảm dần (Diminishing Marginal Returns)
Hàm phản hồi đảm bảo đạo hàm bậc hai luôn âm: d²(Sales)/dS² < 0.
Mỗi 100 BRL chi tiêu ban đầu đem lại mức tăng doanh số cận biên lớn hơn 100 BRL tiếp theo (marginal_0_100 > marginal_100_300 > marginal_300_1000).
Loại bỏ hoàn toàn nguy cơ ngoại suy tuyến tính phi thực tế khi doanh nghiệp tăng mạnh ngân sách quảng cáo.

Bước 3. Đo lường chỉ số tài chính và hiệu quả chiến dịch
Tính toán đồng bộ 4 chỉ số kinh doanh:
- Incremental_Sales_BRL: Doanh thu tăng thêm thuần túy do tác động kích cầu của quảng cáo.
- Sales_Growth_%: Tỷ lệ tăng trưởng doanh thu so với mức doanh thu tự nhiên (Organic Baseline).
- ROAS (Return on Ad Spend): Tỷ số doanh thu tăng thêm trên mỗi đồng chi phí quảng cáo (Incremental_Sales / Ad_Spend).
- Marketing_ROI_%: Tỷ suất lợi nhuận tiếp thị ((Incremental_Sales - Ad_Spend) / Ad_Spend * 100%).

Bước 4. Cung cấp hai chế độ vận hành linh hoạt
Trong hàm predict_scenario_marketing_campaign() tại src/deployment/predictor.py:
- Chế độ 'baseline_only' (mặc định): Duy trì tính trung thực khi làm bài, giữ NaN cho Incremental Sales để tương thích hoàn toàn với các bài kiểm thử quy ước dữ liệu cũ.
- Chế độ 'calibrated_diminishing_returns' (hoặc hàm tiện ích simulate_marketing_response): Kích hoạt động cơ mô phỏng kinh tế lượng.

Bước 5. Triển khai trực quan hóa trên Streamlit app.py và Notebook 05
Tích hợp Thẻ 5 'Mô phỏng Tiếp thị (Marketing Uplift)' trên ứng dụng Streamlit với thanh trượt điều chỉnh AED (0.05 - 0.30) và đồ thị đường cong bão hòa trực quan hóa mối quan hệ giữa ngân sách và doanh số thời gian thực.
