# Kế Hoạch Kiểm Tra & Đánh Giá Mô Hình

Tài liệu này mô tả cách nhóm kiểm tra tính ổn định của các mô hình qua nhiều mốc thời gian khác nhau (cài đặt tại [`src/evaluation/experiments.py`](../../src/evaluation/experiments.py), chạy bằng `.venv/Scripts/python.exe -m src.evaluation.experiments --weekly`) và phân tích các trường hợp mô hình có thể dự báo sai, giúp hiểu rõ hơn hoạt động của từng thuật toán trong bài lab.

---

## 1. Thiết kế Đánh giá Dự báo Tuần qua Nhiều Cửa sổ Thời gian (Rolling Temporal Validation)

Để kiểm chứng xem sự vượt trội của mô hình (đặc biệt là kiến trúc Two-Stage Hurdle) có tính ổn định theo thời gian hay chỉ là hiện tượng ngẫu nhiên trên một lát cắt dữ liệu, nhóm thiết kế giao thức kiểm tra qua 3 cửa sổ chuỗi thời gian liên tiếp:

### 1.1. Cấu trúc các cửa sổ kiểm tra (Expanding Window Split)
- **Nguyên tắc cửa sổ mở rộng:** Tập Train được mở rộng dần theo thời gian quá khứ, trong khi các cửa sổ Test là các khoảng thời gian kế tiếp không giao nhau.
- **Độc lập thời gian:** Mỗi fold thực hiện fit lại toàn bộ các bước tiền xử lý (Imputer, Scaler, Encoder) CHỈ trên tập Train tương ứng của fold đó, sau đó transform sang tập Test để tránh nguy cơ rò rỉ dữ liệu.
- **Quy tắc tính toán đặc trưng trễ:** Các đặc trưng trễ (`last_week_sales`, `sales_lag_2_weeks`, `sales_lag_4_weeks`) và thống kê cuộn 4 tuần trong tập Test chỉ được sử dụng thông tin phát sinh trước thời điểm thứ Hai đầu tuần của từng tuần đích, mô phỏng đúng bài toán dự báo một tuần tiếp theo (Rolling One-week-ahead).

### 1.2. Mục tiêu so sánh đối chuẩn & Kết quả 3 cửa sổ (`weekly_rolling.csv`)
- So sánh sai số RMSE và MAE trên đơn vị BRL giữa mô hình Hai giai đoạn Hurdle với `Zero Baseline` và `Previous Week Baseline` qua từng cửa sổ.
- Trên cả 3 cửa sổ liên tiếp, `Two-Stage Hurdle` đều đạt RMSE (BRL) thấp hơn `Zero Baseline` và vượt xa `Previous Week Baseline`:

| Fold | Khoảng thời gian Test | Train / Test | Zero Baseline (MAE / RMSE) | Prev Week (MAE / RMSE) | Two-Stage Hurdle (MAE / RMSE / $R^2$) |
| --- | --- | --- | --- | --- | --- |
| Fold 1 | `2017-11-20` – `2018-01-29` | `285.189` / `164.091` | `8.29` / `73.40` | `15.67` / `94.98` | `13.01` / **`67.41`** / `0.1457` |
| Fold 2 | `2018-01-29` – `2018-04-09` | `449.280` / `205.786` | `7.05` / `69.35` | `12.53` / `83.00` | `11.30` / **`60.20`** / `0.2385` |
| Fold 3 | `2018-04-09` – `2018-08-21` | `655.066` / `543.116` | `4.91` / `57.87` | `9.91` / `78.45` | `9.08` / **`52.45`** / `0.1725` |

---

## 2. Thiết kế Kế hoạch Kiểm chứng Cold-Start Proxy

### 2.1. Giao thức đánh giá qua nhiều giai đoạn & Kết quả (`cold_rolling.csv`)
- Bài toán proxy ước lượng doanh thu 7 ngày kể từ giao dịch đầu tiên được kiểm chứng qua 3 cửa sổ thời gian khởi đầu độc lập.
- Các ứng viên tham gia đối chứng gồm:
  1. **Baseline Giá bán đơn vị (Price Baseline):** $\hat{s} = \text{first\_price}$.
  2. **Baseline Trung vị Danh mục (Category Median Baseline):** $\hat{s} = \text{median}(\text{category})$.
  3. **Cây Hồi quy Học trên Không gian Log (LogTree):** Cây quyết định học trên $\log(1 + \text{sales})$ từ thuộc tính sản phẩm và giá bán đầu tiên, sau đó chuyển ngược về BRL.
  4. **Cây Hồi quy Học Trực tiếp trên Doanh thu Gốc (RawRevenueTree):** Cây quyết định học trực tiếp trên thang đo tiền tệ BRL nhằm kiểm tra giả thuyết liệu việc biến đổi log có phải là nguyên nhân làm sai lệch phân phối hay không.

| Fold | Train / Test | Price Baseline (MAE / RMSE) | Cat Median (MAE / RMSE) | LogTree (MAE / RMSE) | RawRevenueTree (MAE / RMSE) |
| --- | --- | --- | --- | --- | --- |
| Fold 1 | `16.433` / `3.413` | **`24.39` / `122.07`** | `110.66` / `258.24` | `37.86` / `124.42` | `44.56` / `124.31` |
| Fold 2 | `19.734` / `3.780` | **`27.02`** / `150.67` | `123.79` / `294.68` | `41.72` / **`148.59`** | `51.38` / `179.97` |
| Fold 3 | `23.428` / `8.110` | **`23.01` / `120.91`** | `126.65` / `313.96` | `39.79` / `135.47` | `46.79` / `152.67` |

### 2.2. Phương pháp Phân tích Lỗi: Phân khúc Một đơn vị vs Nhiều đơn vị (`cold_subgroups.csv`)
Nhóm thiết kế kỹ thuật phân tích lỗi bóc tách (Error Decomposition) bằng cách chia tập dữ liệu thành hai phân khúc:
- **Phân khúc Một đơn vị (`one_unit`, chiếm lần lượt `2.921` / `3.198` / `6.932` mẫu qua 3 fold):** Các sản phẩm chỉ bán được đúng một đơn vị duy nhất trong vòng 7 ngày đầu. Trên nhóm này, `Price Baseline` đạt `MAE 0.0 / RMSE 0.0` (do nhãn mục tiêu đã tính cả giao dịch đầu tiên), trong khi `LogTree` chịu sai số do dự báo trung bình theo vùng lá (`MAE 18.06 / 20.98 / 22.01 BRL`).
- **Phân khúc Nhiều đơn vị (`multiple_units`, gồm `492` / `582` / `1.178` mẫu qua 3 fold):** Các sản phẩm phát sinh thêm giao dịch trong vòng 7 ngày đầu. Trên nhóm này, `LogTree` vượt trội hơn hẳn `Price Baseline` ở cả 3 fold (`MAE 155.42 / 155.70 / 144.42` so với `169.20 / 175.47 / 158.44 BRL`; `RMSE 309.71 / 343.88 / 299.79` so với `321.52 / 383.99 / 317.26 BRL`).

**Kết luận chẩn đoán:** Việc `Price Baseline` thắng trên toàn tập chủ yếu do khoảng `85%` sản phẩm chỉ bán 1 đơn vị trong 7 ngày đầu, chứ không phải vì mô hình cây mất khả năng học.

---

## 3. Thiết kế Thí nghiệm Chẩn đoán Hiện tượng Sai số Lớn khi Nghịch đảo Log của Linear Regression

### 3.1. Hiện tượng cần lưu ý khi làm bài
Mô hình hồi quy tuyến tính trong không gian logarit khi chuyển đổi ngược về thang tiền tệ BRL bằng hàm mũ ($\text{expm1}$) có nguy cơ phát sinh sai số rất lớn do hiệu ứng đòn bẩy hàm mũ ở phần đuôi phân phối.

### 3.2. Thiết kế Thí nghiệm Đối chứng
Nhóm thử nghiệm với các yếu tố:
1. **Hệ số điều chuẩn Ridge ($\alpha_{\text{ridge}}$):** Khảo sát các mức $\alpha \in \{10^{-5}, 1.0, 100.0\}$ để kiểm tra xem liệu việc tăng độ phạt trọng số có hạn chế được giá trị log ngoại lai hay không.
2. **Quy mô tập huấn luyện:** Khảo sát trên tập mẫu con (`fast20k`, `20.000` dòng) và toàn bộ tập Train (`full`, `655.066` dòng) để đánh giá ảnh hưởng của kích thước ma trận dữ liệu.
3. **Cơ chế Kẹp trần Thích ứng (Adaptive Clamping):** Thử nghiệm so sánh cơ chế kẹp trần an toàn giới hạn dự báo log không vượt quá miền giá trị cực đại quan sát được trên tập Train:

$$
\hat{y}_{\text{clipped}} = \min(\hat{y}, y_{\max,\text{train}})
$$

### 3.3. Tiêu chí phân tích & Kết quả chẩn đoán (`linear_diagnostics.csv`)
- **Kết quả đo lường thực tế:**
  - Trên `fast20k` (`20.000` Train, số điều kiện Gram `1.17e+20`, `MaxTrainLog = 8.7642`, `MaxPredLog = 13.1463` $\approx 512.114,20$ BRL): tại $\alpha = 10^{-5}$, `LogRMSE = 0.7297` nhưng Test `RMSE = 5.400,88 BRL` và `Top1PctSSEShare = 0.999997`; tăng $\alpha$ lên `1.0` hay `100.0` cho Test RMSE `5.401,59` và `5.367,36 BRL`. Nếu kẹp tại `TrainMax`, Test RMSE giảm xuống `137,51 BRL`.
  - Trên `full` (`655.066` Train, số điều kiện Gram giảm xuống `57.498,28`, `MaxTrainLog = 9.3498`, `MaxPredLog = 14.0248` $\approx 1.232.754,07$ BRL): tại $\alpha = 10^{-5}$, Val/Test `LogRMSE = 0.7694 / 0.7140`, nhưng Val/Test `RMSE = 11.211,01 / 5.511,67 BRL` và `Top1PctSSEShare = 0.999999 / 0.999998`; tăng $\alpha$ lên `1.0` hay `100.0` vẫn cho Test RMSE `5.511,85` và `5.520,40 BRL`. Khi kẹp chặt ở `TrainMax` (`9.3498`), Val/Test RMSE giảm còn `191,89 / 175,47 BRL`.
- **Bài học rút ra:** Sai số lớn trên thang BRL của `Linear Regression` không đến từ suy biến ma trận trên tập lớn hay thiếu điều chuẩn Ridge, mà do siêu phẳng không bị chặn trên đẩy khoảng `1%` điểm ngoại suy chạm ngưỡng kẹp thích ứng (`+0.5*span` $\to 14.0248$). Cần thiết lập biên chặn chặt chẽ hoặc dùng mô hình cây có miền lá bị chặn tự nhiên khi nghịch đảo hàm mũ.
