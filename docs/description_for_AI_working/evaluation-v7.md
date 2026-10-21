Bản v7. Triển Khai 3 Kịch Bản Kinh Doanh & Vận Hành Streamlit App
Mục tiêu của bản này là đưa mô hình vào ứng dụng thực tiễn thông qua việc mô phỏng 3 kịch bản kinh doanh và xây dựng giao diện tương tác trực quan.

Bước 1. Triển khai Kịch bản 1: Ra mắt Sản phẩm Mới (New Product Launch)
Hàm predict_scenario_new_product() trong src/deployment/predictor.py:
Mô phỏng sản phẩm mới chưa có lịch sử mua hàng (last_week_sales = 0, units = 0).
Tự động gán điểm đánh giá kỳ vọng 4.0 sao và số review = 0.
Mô hình ước lượng doanh thu tuần đầu tiên dựa trên danh mục ngành hàng, giá niêm yết và chu kỳ thời gian ra mắt.

Bước 2. Triển khai Kịch bản 2: Tăng Tốc Ngân Sách Quảng Cáo (Marketing Campaign)
Hàm predict_scenario_marketing_campaign() trong src/deployment/predictor.py:
Mô phỏng doanh số dự kiến khi biến thiên ngân sách marketing từ 100 đến 1.000 BRL.
Hỗ trợ nhà quản trị đánh giá độ nhạy và ước lượng điểm hoàn vốn quảng cáo.

Bước 3. Triển khai Kịch bản 3: Khuyến Mãi Mùa Lễ Hội Black Friday (Seasonal Promo)
Hàm predict_scenario_seasonal_promo() trong src/deployment/predictor.py:
Mô phỏng tác động kết hợp giữa việc giảm giá bán (chiết khấu 10%, 20%, 30%) và kích hoạt biến mùa mua sắm cao điểm is_peak_shopping_season = 1.
Ước lượng doanh thu và sản lượng xuất kho dự kiến trong tuần Black Friday.

Bước 4. Xây dựng ứng dụng Web tương tác Streamlit app.py
Khởi chạy ứng dụng: streamlit run app.py.
Thiết kế giao diện 2 thẻ chức năng:
- Thẻ 'Một sản phẩm': Nhập ngày thứ Hai, danh mục, giá bán và lịch sử để xem ngay doanh thu dự báo tức thời.
- Thẻ 'Nhiều sản phẩm từ CSV': Tải file CSV lên đến 10.000 dòng, hiển thị bảng kết quả và tải file dự báo về máy.

Bước 5. Nghiệm thu hoàn tất toàn bộ đồ án
Toàn bộ quy trình 10 bước vận hành trơn tru khép kín từ dữ liệu thô đến sản phẩm phần mềm ứng dụng thực tế.
