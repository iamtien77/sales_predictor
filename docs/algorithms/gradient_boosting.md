# ⚡ Tăng Cường Độ Dốc Hồi Quy (Gradient Boosting Regressor From-Scratch)

> **Vị trí trong bài lab:** Phương pháp học kết hợp tuần tự (Ensemble Boosting), được nhóm thiết kế nhằm huấn luyện chuỗi các cây quyết định nông bù trừ vào phần dư sai số.
> **Mục tiêu trong bài lab:** Thử nghiệm cơ chế học sửa sai từng bước kết hợp tốc độ học (learning rate) để xem có cải thiện dần sai số so với cây đơn lẻ hay không.

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Trong khi Bagging (như Random Forest) tập trung vào việc giảm phương sai bằng cách huấn luyện các cây độc lập, **Gradient Boosting** hướng tới mục tiêu tối ưu hóa mức độ khớp hàm bằng cách học tuần tự trên sai số còn tồn đọng của các thế hệ cây trước.

Mục đích của thuật toán Gradient Boosting trong thiết kế của nhóm:
- Học xấp xỉ hàm phi tuyến tính phức tạp bằng chuỗi cộng gộp các cây quyết định nông (*weak learners* / *shallow trees*).
- Bù trừ sai số bằng cách fit cây con vào phần dư (Residuals) âm của mô hình tích lũy tại mỗi bước lặp.
- Ứng dụng hệ số học co hẹp (*Learning Rate Shrinkage*) để kiểm soát tốc độ hội tụ, giảm thiểu nguy cơ quá khớp trên dữ liệu chuỗi thời gian bán lẻ có độ nhiễu cao.

### 1.2. Cơ sở toán học & Thuật toán huấn luyện tuần tự

#### A. Hàm mất mát tối ưu (Loss Function)
Mô hình hoạt động trong không gian log tự nhiên với hàm mất mát bình phương (Squared Error Loss):

$$
L(y, F(x)) = \frac{1}{2} \left( y - F(x) \right)^2
$$

Đạo hàm riêng (Gradient) âm của hàm mất mát theo giá trị dự báo hiện tại chính là phần dư (Residual):

$$
r_i = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F = F_{m-1}} = y_i - F_{m-1}(x_i)
$$

#### B. Quy trình thuật toán tuần tự $M$ vòng lặp
1. **Khởi tạo giá trị dự báo cơ sở $F_0(x)$:** Giá trị tối ưu là trung bình cộng của nhãn huấn luyện trên tập Train:

$$
   F_0(x) = \arg\min_{\gamma} \sum_{i=1}^{N} L(y_i, \gamma) = \frac{1}{N} \sum_{i=1}^{N} y_i = \bar{y}_{\text{train}}
$$

2. **Vòng lặp $m = 1, 2, \dots, M$ (với $M$ là số lượng cây `n_estimators`):**
   - **Bước 2.1 (Tính phần dư):** Với mỗi mẫu $i \in \{1, \dots, N\}$, tính phần dư bằng negative gradient:

$$
     r_{im} = y_i - F_{m-1}(x_i)
$$

   - **Bước 2.2 (Huấn luyện cây con trên phần dư):** Khớp một cây quyết định hồi quy nông $h_m(x)$ (thiết kế với `max_depth=3`) trên tập dữ liệu phần dư $\{(x_i, r_{im})\}_{i=1}^N$.
   - **Bước 2.3 (Cập nhật mô hình tích lũy với hệ số co hẹp $\eta$):**

$$
     F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)
$$

     Trong đó $\eta \in (0, 1]$ là `learning_rate` (hệ số co hẹp), giúp kiểm soát đóng góp của từng cây con và chống phân kỳ.

3. **Dự báo cuối cùng:**

$$
   \hat{y}(x) = F_M(x) = F_0(x) + \eta \sum_{m=1}^{M} h_m(x)
$$

---

## 2. Input / Output (Đầu vào & Đầu ra)

### 2.1. Định dạng dữ liệu đầu vào (Input)
- **Ma trận đặc trưng $X$:** Mảng ma trận 2D số thực kích thước $(N, D)$ gồm 16 đặc trưng số học đã chuẩn hóa Z-Score và các cột danh mục sản phẩm đã mã hóa One-Hot.
- **Vector nhãn mục tiêu $y$:** Vector số thực 1D kích thước $(N,)$ trong không gian log tự nhiên: $y = \log(1 + \text{sales})$.

### 2.2. Định dạng dữ liệu đầu ra (Output)
- **Dự báo trong không gian log:** Trả về vector dự báo liên tục $\hat{y} = F_M(X)$.
- **Hàm chuyển đổi tiền tệ thực tế BRL:**

$$
\hat{y}_{\text{sales}} = \max\left(0.0, \exp(\hat{y}) - 1\right) \text{ BRL}
$$

---

## 3. Data Used (Dữ liệu sử dụng & Ràng buộc thời gian)

- **Phân chia Temporal Split 80/10/10:**
  - Tập Huấn luyện (Train Set): 80% số tuần đầu (`78` tuần, `655.066` dòng), dùng để học tuần tự chuỗi cây trên phần dư.
  - Tập Kiểm định (Validation Set): 10% số tuần kế tiếp (`10` tuần, `249.399` dòng), dùng để chọn số lượng cây $M$ và độ sâu cây.
  - Tập Kiểm thử Độc lập (Test Set): 10% số tuần cuối (`10` tuần, `293.717` dòng), dùng để đánh giá cuối cùng.
- **Bảo toàn tuần doanh thu bằng 0:** Giữ nguyên các tuần `sales = 0.0` BRL để phản ánh đúng cấu trúc dữ liệu thương mại điện tử thực tế.
- **Nguyên tắc chống rò rỉ:** Preprocessing chỉ fit trên Train; toàn bộ đặc trưng trễ/cuộn chỉ lấy từ thông tin đã xảy ra trước thời điểm dự báo.

---

## 4. Evaluation (Phương pháp đánh giá)

- **Hàm mất mát tối ưu:** L2 Loss (Mean Squared Error) trên không gian log tự nhiên: $\mathcal{L} = \frac{1}{N} \sum (y_i - \hat{y}_i)^2$.
- **Lựa chọn siêu tham số:** Đánh giá trên tập Validation độc lập bằng thang đo RMSE tiền tệ BRL.
- **Bộ chỉ số đo lường trên thang đo BRL:**
  1. Mean Squared Error (MSE): $\text{BRL}^2$.
  2. Mean Absolute Error (MAE): $\text{BRL}$.
  3. Root Mean Squared Error (RMSE): $\text{BRL}$ (tiêu chí chính).
  4. Hệ số xác định ($R^2$ Score).
  5. Weighted Absolute Percentage Error (WAPE).
  6. Cost_Per_Item_BRL: Độ đo chi phí bất đối xứng giữa dự báo thiếu và dự báo thừa.

---

## 5. Status (Trạng thái cài đặt & Tối ưu kỹ thuật)

- **Cài đặt From-Scratch ([`src/models/ensemble.py`](../../src/models/ensemble.py)):** Lớp `GradientBoostingRegressorFromScratch` dùng `DecisionTreeRegressorFromScratch` làm weak learner. Trong `main.py` Step 6, cấu hình mặc định chạy với `n_estimators=15, learning_rate=0.1, max_depth=3, min_samples_split=10, min_samples_leaf=5` (fit `235,84s`), đạt Validation RMSE `62,8054 BRL` và Test RMSE `52,0406 BRL` (`MAE 4,4176 BRL`, $R^2$ `0,0109`, WAPE `1,0323`, Cost `42,4902`).
- **Kiến trúc cây nông (`max_depth=3`):** Độ sâu nông cho từng cây con giúp mỗi cây chỉ đóng vai trò "người học yếu" (*weak learner*), hạn chế quá khớp sớm.
- **Theo dõi hội tụ hàm mất mát & Độ quan trọng đặc trưng:** Lưu giá trị MSE phần dư qua từng vòng lặp $m$ và tính trung bình cộng lượng giảm phương sai của $M$ cây con:

$$
\text{Importance}(j) = \frac{1}{M} \sum_{m=1}^{M} \text{Importance}_{h_m}(j)
$$

---

## 6. Kế Hoạch & Kết Quả Tinh Chỉnh Siêu Tham Số

### 6.1. Không gian tinh chỉnh siêu tham số trong `main.py`
Trong bước tinh chỉnh (`main.py` Step 7), nhóm khảo sát lưới 4 cấu hình trên tập Validation:
- Số lượng cây (`n_estimators`): $\{15, 30\}$.
- Độ sâu cây con (`max_depth`): $\{2, 3\}$.
- Tốc độ học (`learning_rate` $\eta$): cố định `[0.1]` trong lưới chạy thực tế của `main.py` (mức `0.05` chỉ nằm trong `PARAM_GRIDS` cấu hình dự phòng ở [`src/config.py`](../../src/config.py)).

### 6.2. Tiêu chí lựa chọn & kết quả thực nghiệm (`tuning_history.csv`)
- Toàn bộ cấu hình được đánh giá trên tập Validation độc lập theo **Validation RMSE (BRL)**, không dùng tập Test để chọn siêu tham số:

| `n_estimators` | `max_depth` | `learning_rate` | Validation RMSE (BRL) | Ghi chú |
| --- | --- | --- | --- | --- |
| `15` | `2` | `0.1` | `63.1879` | |
| `15` | `3` | `0.1` | `62.8054` | Trùng cấu hình Step 6 |
| `30` | `2` | `0.1` | `62.4007` | |
| `30` | `3` | `0.1` | **`61.2637`** | Tốt nhất trong họ Gradient Boosting |

- Dù tốt hơn mốc baseline tốt nhất (`Revenue Mean Baseline`: `63,3996 BRL`), cấu hình tốt nhất của Gradient Boosting (`61,2637 BRL`) vẫn kém `Decision Tree` sau tuning (`59,2288 BRL`) và thời gian huấn luyện chậm hơn đáng kể do phải xây cây tuần tự.
