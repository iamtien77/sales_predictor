# Proxy Doanh thu Bảy Ngày Kể Từ Giao Dịch Đầu Tiên Quan Sát Được

Tài liệu này trình bày thiết kế và kết quả thực nghiệm cho bài toán thử nghiệm ước lượng doanh thu trong 7 ngày đầu của sản phẩm mới (được cài đặt tại [`src/data/builder.py`](../../src/data/builder.py), [`src/models/cold_start.py`](../../src/models/cold_start.py) và [`src/evaluation/cold_start.py`](../../src/evaluation/cold_start.py)). Nhóm định vị rõ ràng đây là **thí nghiệm proxy ngoại tuyến sau lần bán đầu tiên**, không đại diện cho bài toán dự báo trước khi sản phẩm được niêm yết trên sàn.

## 1. Quy ước bài toán dự báo

- **Mốc bắt đầu (Origin):** Thời điểm giao dịch đầu tiên quan sát được của sản phẩm trong tập dữ liệu.
- **Giá đầu vào:** Giá bán của giao dịch đầu tiên đã biết, tuyệt đối không dùng giá trung vị của các giao dịch trong tương lai.
- **Biến mục tiêu (Target):** Tổng doanh thu các mặt hàng của sản phẩm trong khoảng thời gian $[t_{\text{origin}}, t_{\text{origin}} + 7 \text{ ngày})$, bao gồm cả giao dịch đầu tiên đã biết.
- **Điều kiện hợp lệ của mẫu:** Loại bỏ các sản phẩm không có đủ 7 ngày quan sát sau mốc bắt đầu để tránh biến mục tiêu bị cắt ngắn.
- **Giới hạn dữ liệu:** Dữ liệu Olist không chứa các sản phẩm niêm yết nhưng chưa từng bán được hàng. Do đó, nhóm xác định thử nghiệm này chỉ đánh giá khả năng ước lượng quán tính ban đầu của các sản phẩm đã phát sinh giao dịch, chưa thể kết luận về các sản phẩm ra mắt không có người mua.

## 2. Thiết kế Huấn luyện và Đánh giá

Nhóm thiết kế quy trình phân chia dữ liệu theo trục thời gian mốc bắt đầu:
- **Tập Kiểm thử (Test Set):** Dành 20% số ngày bắt đầu phân biệt cuối cùng làm tập Test độc lập.
- **Cơ chế Purging chống rò rỉ nhãn:** Các mẫu trong tập Train có cửa sổ nhãn 7 ngày vượt qua ranh giới thời gian của tập Test đều bị loại bỏ hoàn toàn (purge) để tránh rò rỉ dữ liệu.
- **Tiền xử lý độc lập:** Bộ Target Encoding cho danh mục (`smoothing=10.0`) và 5 đặc trưng số học biến đổi `log1p` CHỈ được fit trên tập Train, sau đó áp dụng (transform) sang tập Test.

Các ứng viên mô hình được cài đặt để đánh giá so sánh:
1. **`CategoryMedianColdStartBaseline` (Baseline Trung vị Danh mục):** Dự báo bằng trung vị doanh thu của danh mục sản phẩm tương ứng, học trên tập Train.
2. **`SingleUnitPriceBaseline` (Baseline Một đơn vị):** Dự báo doanh thu bằng đúng giá bán của giao dịch đầu tiên đã biết ($\hat{s} = \text{first\_price}$).
3. **`ColdStartProductRegressor` (Cây Hồi quy thuộc tính sản phẩm):** Dùng `DecisionTreeRegressorFromScratch` (`max_depth=6, min_samples_split=20, min_samples_leaf=10`) trên 5 đặc trưng số `log1p` và danh mục đã Target Encoding (`smoothing=10.0`), học trên biến mục tiêu logarit rồi nghịch đảo về BRL.

### Vì sao baseline một đơn vị là đối chứng quan trọng?
- Theo thiết kế bài toán, biến mục tiêu 7 ngày đã bao hàm cả giao dịch đầu tiên.
- Đối với các sản phẩm có sức mua chậm (chỉ bán được đúng 1 đơn vị duy nhất trong vòng 7 ngày đầu — thực tế chiếm tới `85,47%` tập Test hiện tại), biến mục tiêu thực tế sẽ bằng đúng giá bán của giao dịch đó. Do đó, Baseline Một đơn vị sẽ đạt sai số bằng 0 trên phân khúc sản phẩm này.
- Mô hình cây quyết định dự báo trung bình theo vùng lá trong không gian logarit, nên không tự nhiên gán $\hat{s} = \text{first\_price}$. Việc so sánh với Baseline Một đơn vị giúp nhóm đánh giá xem liệu các thuộc tính sản phẩm có thực sự bổ sung thông tin hữu ích hay mô hình đang bị nhiễu so với một quy tắc suy luận cơ bản.

## 3. Thiết kế Kiểm chứng, Chống rò rỉ & Kết quả Thực nghiệm

Quy trình thực nghiệm được nhóm thiết kế với các quy tắc rõ ràng và chạy kiểm chứng trên bộ Olist (`cutoff = 2018-04-24T00:00:00`):
- **Lưu vết thông tin thử nghiệm:** Ghi nhận `23.428` mẫu Train hợp lệ, `483` mẫu Train bị loại qua purging, `8.110` mẫu Test và tỷ lệ bán đúng 1 đơn vị `85,47%`.
- **Đánh giá trên thang tiền tệ BRL (Kết quả Test `11/11/2026`):**

| Mô hình Cold-Start | Test MAE (BRL) | Test RMSE (BRL) | Test $R^2$ | Ghi chú |
| --- | --- | --- | --- | --- |
| `SingleUnitPriceBaseline` | **`23.0145`** | **`120.9134`** | **`0.8520`** | Cao nhất do `85,47%` sản phẩm chỉ bán 1 đơn vị |
| `ColdStartProductRegressor` | `39.7939` | `135.4682` | `0.8142` | Thuộc tính sản phẩm hữu ích khi bán nhiều đơn vị |
| `CategoryMedianColdStartBaseline` | `126.6543` | `313.9562` | `0.0022` | Không dùng giá bán đầu tiên |

- **Bảo toàn tập Test:** Kết quả trên tập Test chỉ dùng để so sánh đánh giá lần cuối, tuyệt đối không dùng để điều chỉnh lại độ sâu hay siêu tham số của cây.

## 4. Cơ chế Dự phòng và Giới hạn Áp dụng

- **Cơ chế dự phòng:** Trong trường hợp gặp sản phẩm mới thiếu thông tin đặc trưng hoặc danh mục chưa từng xuất hiện, mô hình quy về thống kê học từ Train (`smoothing=10.0`) hoặc dùng giá bán ban đầu làm mốc ước lượng tối thiểu.
- **Giới hạn bài toán:** Nhóm lưu ý rằng kết quả proxy 7 ngày sau lần bán đầu không đại diện cho bài toán dự báo trước khi sản phẩm ra mắt. Để giải quyết bài toán trước niêm yết, cần có dữ liệu catalog thực tế như quy định trong [Quy chuẩn Dữ liệu Niêm yết](listing_data_contract.md).
