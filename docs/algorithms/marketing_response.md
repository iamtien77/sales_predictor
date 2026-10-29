# 📢 Mô Hình Phản Hồi Tiếp Thị Phi Tuyến (Marketing Diminishing Returns Response Model)

> **Vai trò:** Mô phỏng độ nhạy doanh thu theo giả định ngân sách; Olist không có dữ liệu chiến dịch để ước lượng tác động nhân quả.
> **Implementation:** [simulate_marketing_response và predict_scenario_marketing_campaign](../../src/deployment/predictor.py).
> **Giới hạn:** Tên API `calibrated_diminishing_returns` được giữ để tương thích; tên này không chứng minh hệ số đã được hiệu chuẩn.

## 1. Purpose (Mục đích & Động lực Kinh tế Lượng)



### 1.1. Thách thức Dữ liệu Thực tế

Đề bài yêu cầu các kịch bản marketing, nhưng Olist không có lịch sử chi tiêu, impressions hoặc clicks theo sản phẩm. Vì vậy, không thể xác nhận mức tăng doanh thu do quảng cáo từ bộ dữ liệu hiện tại. Mô phỏng này phục vụ khảo sát giả định, chưa đủ để khuyến nghị ngân sách thực tế.

### 1.2. Giải pháp: Mô hình Phản hồi Bão hòa Ngân sách Chuẩn tắc

Có hai chế độ:
- `baseline_only` (mặc định): giữ doanh thu cơ sở; Incremental Sales, Growth, ROAS và ROI là `NaN`.
- `calibrated_diminishing_returns`: dùng hàm log với `elasticity=0.15` và `reference_spend=100.0` mặc định. Đây là tham số giả định, không phải hệ số đã fit hoặc benchmark ngành.

Hàm có lợi biên giảm dần, nhưng không có trần bão hòa hữu hạn. Các tiêu đề dùng từ “bão hòa” được hiểu theo nghĩa giảm lợi biên trong tài liệu này.

## 2. Mathematical Formulation (Cơ sở Toán học & Công thức)



### 2.1. Hàm Tăng Trưởng Phản Hồi Tiếp Thị (Log-Response Uplift)

Với ngân sách $A$, mức chi tham chiếu $R$ và hệ số giả định $\beta$:

$$
u(A)=1+\beta\ln(1+A/R),\quad A\ge0,\ R>0,\ \beta\ge0.
$$

`R = reference_spend` cùng đơn vị BRL với `A`, **không phải giá sản phẩm**. Tham số tên `elasticity` điều khiển cường độ phản hồi; không phải elasticity doanh thu cố định. Khi $S_0>0$, elasticity theo ngân sách là $\frac{\beta A}{(R+A)[1+\beta\ln(1+A/R)]}$, phụ thuộc $A$.

### 2.2. Doanh Thu Dự Báo Theo Kịch Bản

$$
\hat S(A)=S_0[1+\beta\ln(1+A/R)].
$$

Code sao chép `base_product`, gán `ad_spend=0`, rồi gọi `predict_sales` để có $S_0\ge0$. Vì mô hình tuần không học tác động quảng cáo, thao tác này không xác lập một counterfactual “không quảng cáo” đã được kiểm chứng. Baseline và ngân sách cần cùng khoảng thời gian; API không tự kiểm tra sự tương thích đó.

### 2.3. Các Chỉ Số Đo Lường Hiệu Quả Tài Chính

Với $A>0$, $S_0>0$ và biên lợi nhuận giả định $m$ (biên lợi nhuận gộp `gross_margin`)$:

1. **Doanh thu gia tăng:** $\Delta S=S_0\beta\ln(1+A/R)$.
2. **Tăng trưởng:** $100\Delta S/S_0$.
3. **`ROAS`:** $\Delta S/A$; đây là incremental revenue/spend trong mô phỏng.
4. **`Marketing_ROI_%`:** $100(m\Delta S-A)/A$. Khi $m=1$ mặc định, đây là proxy dựa trên doanh thu, không phải lợi nhuận thực đo sau mọi chi phí.

Tại $A=0$, tỷ số ROAS/ROI không xác định về toán học; implementation trả **0** theo quy ước hiển thị. Khi $S_0=0$, code giữ doanh thu, incremental và growth bằng 0; với $A>0$, ROAS bằng 0 và ROI bằng -100%.

## 3. Input / Output (Đặc tả Giao diện)



### 3.1. Tham số Đầu vào

Hàm `simulate_marketing_response` nhận:
- `base_product`, `ad_spend_levels`, `model`, `pipeline`.
- `elasticity=0.15`: hữu hạn, không âm.
- `reference_spend=100.0`: hữu hạn, dương.
- `gross_margin=1.0`: hữu hạn, trong [0,1].

Mọi mức chi phải hữu hạn, không âm; vi phạm gây `ValueError`. `predict_scenario_marketing_campaign` nhận thêm `mode`; mode ngoài hai giá trị được hỗ trợ cũng gây `ValueError`. Không có tham số `baseline_sales`: baseline được suy ra qua model/pipeline. Đầu vào sản phẩm phải đáp ứng hợp đồng của pipeline đã fit.

### 3.2. Cấu trúc Bảng Kết quả Trả về

Bảng dưới là **ví dụ tính theo công thức**, giả định $S_0=100$ BRL, $R=100$ BRL, $\beta=0.15$, $m=0.4$; không phải kết quả đánh giá Olist. Các cột số được code làm tròn hai chữ số thập phân:

| Ad_Spend_BRL | Baseline_Sales_BRL | Predicted_Sales_BRL | Incremental_Sales_BRL | Sales_Growth_% | ROAS | Marketing_ROI_% |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100.00 | 100.00 | 110.40 | 10.40 | 10.40 | 0.10 | -95.84 |
| 500.00 | 100.00 | 126.88 | 26.88 | 26.88 | 0.05 | -97.85 |

Output còn có cột `Interpretation` mô tả chế độ và giả định. Làm tròn có thể che mất ROAS rất nhỏ; không tính lại ROI từ ROAS đã làm tròn. Với danh sách chi rỗng, code hiện trả DataFrame rỗng không có schema cột cố định.

## 4. Economic Properties & Boundary Conditions (Kiểm chứng Ranh giới)

Với $A\ge0$, $R>0$, $S_0\ge0$, $\beta\ge0$:

1. **Không chi:** $u(0)=1$, $\Delta S(0)=0$.
2. **Lợi biên:**
$$
\frac{\partial\Delta S}{\partial A}=\frac{S_0\beta}{R+A}\ge0,\qquad \frac{\partial^2\Delta S}{\partial A^2}=-\frac{S_0\beta}{(R+A)^2}\le0
$$
   Bất đẳng thức nghiêm chỉ khi $S_0\beta>0$; nếu baseline hoặc hệ số bằng 0 thì đường đáp ứng phẳng.
3. **Không có trần hữu hạn:** Khi $S_0\beta>0$, $\Delta S\to\infty$ khi $A\to\infty$, dù lợi biên tiến về 0. Chưa mô hình hóa quy mô thị trường, giới hạn tồn kho hay bão hòa hữu hạn.
4. **Hiệu quả theo giả định:** Với $A>0$, ROI âm khi $m\Delta S<A$. Điều kiện này không tự xác định ngân sách tối ưu; code chỉ tính trên các mức chi được cung cấp.
5. **Giới hạn số học:** Kiểm tra hữu hạn ở đầu vào không bảo đảm mọi tỷ số trung gian đều hữu hạn với giá trị cực đoan. Chưa có guard đầy đủ cho overflow trong $A/R$ hoặc baseline không hữu hạn.

## 5. Status & Integration (Trạng thái Triển khai)

- **Triển khai:** Hàm mô phỏng nằm trong [predictor.py](../../src/deployment/predictor.py), được gọi ở [main.py](../../main.py) và [app.py](../../app.py).
- **Kiểm chứng:** [tests/test_cold_start_and_marketing.py](../../tests/test_cold_start_and_marketing.py) chứa kiểm tra hành vi phần mềm, gồm lợi biên và bảo toàn zero baseline. Test tổng hợp không xác nhận hệ số kinh tế hay hiệu quả chiến dịch ngoài thực tế.
- **Đánh giá tiếp theo:** Muốn hiệu chuẩn cần dữ liệu ngân sách/doanh thu cùng horizon và thiết kế nhận diện tác động; trước đó chỉ nên báo cáo sensitivity theo các giả định. Lượt cập nhật tài liệu không tạo kết quả marketing thực nghiệm mới.
