# Kiểm chứng thực nghiệm và chẩn đoán mô hình

Lệnh tái lập: `.venv/Scripts/python.exe -m src.evaluation.experiments --weekly`.
Nguồn: `results/metrics/experiments/`, có hash từng CSV trong manifest.
Đây là phân tích thăm dò sau khi đã xem Test ở các vòng trước, **không phải tập
kiểm định mù mới**. Không chọn tham số trên Test, không thay artifact triển khai.
Ba cửa sổ Test không giao nhau; các cửa sổ Train mở rộng và chỉ chứa nhãn đã hoàn tất.

## Dự báo tuần qua ba mốc

| Fold | Start | EndExclusive | TrainCount | TestCount | Zero_RMSE | Hurdle_RMSE | RMSE_improvement_pct | Zero_MAE | Hurdle_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2017-11-20 00:00:00 | 2018-01-29 00:00:00 | 285189 | 164091 | 73.4037 | 67.4105 | 8.1646 | 8.2928 | 13.0057 |
| 2 | 2018-01-29 00:00:00 | 2018-04-09 00:00:00 | 449280 | 205786 | 69.3522 | 60.2049 | 13.1896 | 7.0495 | 11.2984 |
| 3 | 2018-04-09 00:00:00 | 2018-08-21 00:00:00 | 655066 | 543116 | 57.8666 | 52.4506 | 9.3595 | 4.9107 | 9.0776 |

Hurdle giảm RMSE so với Zero ở cả ba cửa sổ nhưng MAE cao hơn ở cả ba.
Đây là sự ổn định của đánh đổi quan sát được, không phải bảo đảm tương lai.
Mỗi fold fit lại preprocessing trên Train và huấn luyện Hurdle trên toàn Train,
giữ tham số mặc định cố định. Lag trong Test dùng lịch sử đã xảy ra trước từng
tuần đích: đánh giá one-week-ahead cập nhật quan sát, không phải dự báo đa bước
không có quan sát mới. Không dùng các bảng này để đổi artifact đang triển khai.

## Cold-start proxy: ba cửa sổ

| Fold | Model | TrainCount | TestCount | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Price | 16433 | 3413 | 24.3905 | 122.0739 | 0.7645 |
| 1 | CategoryMedian | 16433 | 3413 | 110.6553 | 258.2364 | -0.0540 |
| 1 | LogTree | 16433 | 3413 | 37.8606 | 124.4177 | 0.7553 |
| 1 | RawRevenueTree | 16433 | 3413 | 44.5552 | 124.3070 | 0.7558 |
| 2 | Price | 19734 | 3780 | 27.0166 | 150.6722 | 0.7526 |
| 2 | CategoryMedian | 19734 | 3780 | 123.7925 | 294.6785 | 0.0536 |
| 2 | LogTree | 19734 | 3780 | 41.7226 | 148.5883 | 0.7594 |
| 2 | RawRevenueTree | 19734 | 3780 | 51.3779 | 179.9708 | 0.6470 |
| 3 | Price | 23428 | 8110 | 23.0145 | 120.9134 | 0.8520 |
| 3 | CategoryMedian | 23428 | 8110 | 126.6543 | 313.9562 | 0.0022 |
| 3 | LogTree | 23428 | 8110 | 39.7939 | 135.4682 | 0.8142 |
| 3 | RawRevenueTree | 23428 | 8110 | 46.7903 | 152.6722 | 0.7640 |

### Vì sao cây thua baseline giá?

| Fold | Model | Group | Count | MAE | RMSE |
| --- | --- | --- | --- | --- | --- |
| 1 | Price | one_unit | 2921 | 0.0000 | 0.0000 |
| 1 | Price | multiple_units | 492 | 169.1965 | 321.5204 |
| 1 | LogTree | one_unit | 2921 | 18.0598 | 43.9440 |
| 1 | LogTree | multiple_units | 492 | 155.4180 | 309.7067 |
| 2 | Price | one_unit | 3198 | 0.0000 | 0.0000 |
| 2 | Price | multiple_units | 582 | 175.4688 | 383.9878 |
| 2 | LogTree | one_unit | 3198 | 20.9807 | 67.6426 |
| 2 | LogTree | multiple_units | 582 | 155.6964 | 343.8815 |
| 3 | Price | one_unit | 6932 | 0.0000 | 0.0000 |
| 3 | Price | multiple_units | 1178 | 158.4444 | 317.2577 |
| 3 | LogTree | one_unit | 6932 | 22.0148 | 78.7251 |
| 3 | LogTree | multiple_units | 1178 | 144.4163 | 299.7873 |

Nhóm một đơn vị chiếm đa số. Với target bao gồm giao dịch đầu, baseline giá đúng
hoàn toàn ở nhóm này theo cấu trúc bài toán. Cây dự báo trung bình theo vùng đặc
trưng nên sai ở nhóm một đơn vị. Cây giảm lỗi ở nhóm nhiều đơn vị nhưng lợi ích
không bù được MAE phát sinh trên nhóm lớn. Phân nhóm dựa vào nhãn thật chỉ dùng
phân tích sau đánh giá, không được dùng để định tuyến dự báo lúc triển khai.

`RawRevenueTree` giữ cùng dữ liệu, phép biến đổi đặc trưng và tham số cây, chỉ đổi
nhãn từ log sang BRL. Kết quả không cải thiện nhất quán: không thể quy mọi lỗi
cho log-transform hoặc khẳng định cây trên BRL luôn tốt hơn. Cold-start này vẫn
là proxy **sau lần bán đầu**, không phải trước niêm yết.

## Linear Regression: tái hiện và thí nghiệm đối chứng

| Mode | Alpha | Split | LogRMSE | RMSE | MaxPredictedLog | MaxTrainLog | Top1PctSSEShare | TrainMaxCapRMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fast20k | 0.0000 | Validation | 0.7802 | 6557.3543 | 13.1463 | 8.7642 | 1.0000 | 140.1342 |
| fast20k | 0.0000 | Test | 0.7297 | 5400.8772 | 13.1463 | 8.7642 | 1.0000 | 137.5077 |
| fast20k | 1.0000 | Validation | 0.7801 | 6557.3042 | 13.1463 | 8.7642 | 1.0000 | 140.1280 |
| fast20k | 1.0000 | Test | 0.7295 | 5401.5922 | 13.1463 | 8.7642 | 1.0000 | 137.5025 |
| fast20k | 100.0000 | Validation | 0.7784 | 6533.1311 | 13.1463 | 8.7642 | 1.0000 | 139.8398 |
| fast20k | 100.0000 | Test | 0.7274 | 5367.3550 | 13.1463 | 8.7642 | 1.0000 | 137.1950 |
| full | 0.0000 | Validation | 0.7694 | 11211.0106 | 14.0248 | 9.3498 | 1.0000 | 191.8852 |
| full | 0.0000 | Test | 0.7140 | 5511.6709 | 14.0248 | 9.3498 | 1.0000 | 175.4652 |
| full | 1.0000 | Validation | 0.7694 | 11211.1544 | 14.0248 | 9.3498 | 1.0000 | 191.8858 |
| full | 1.0000 | Test | 0.7140 | 5511.8488 | 14.0248 | 9.3498 | 1.0000 | 175.4671 |
| full | 100.0000 | Validation | 0.7694 | 11215.3307 | 14.0248 | 9.3498 | 1.0000 | 191.9016 |
| full | 100.0000 | Test | 0.7139 | 5520.3969 | 14.0248 | 9.3498 | 1.0000 | 175.5541 |

Hai chế độ tái hiện dùng đúng ma trận cache và seed 42: 20.000 dòng ngẫu nhiên
hoặc toàn bộ Train. Cột `Top1PctSSEShare` đo tỷ trọng SSE của 1% mẫu lỗi cao nhất.
Sai số log nhỏ về trung bình không bảo đảm sai số BRL nhỏ: expm1 khuếch đại phần
đuôi dương; giới hạn hiện tại vẫn cho phép dự báo log vượt cực đại Train 50% span.
Các mẫu cực trị và dự báo tiền tệ được xuất ở `linear_extremes.csv`.

Ridge 1 và 100 hầu như không cải thiện RMSE so với 1e-5; đây không phải bằng
chứng lỗi chỉ do thiếu regularization hay do dữ liệu chắc chắn phi tuyến.
`TrainMaxCapRMSE` là can thiệp chẩn đoán: giữ nguyên trọng số, chỉ chặn log tại
cực đại nhãn Train. RMSE giảm mạnh ngay cả trên Validation, xác nhận phần đuôi
nghịch đảo log là tác nhân quan trọng. Tuy nhiên mức lỗi vẫn cao; không tuyên bố
clipping đã giải quyết dự báo, cũng không âm thầm thay thuật toán bằng kết quả Test.
Gram condition và norm hệ số được lưu để kiểm tra số học, không tự coi condition
lớn là nguyên nhân duy nhất (one-hot và intercept có thể phụ thuộc tuyến tính).

## Cold-start trước lần bán đầu

Chưa có catalog thực để kiểm chứng. Luồng mới `src/data/launch.py` giữ cả sản phẩm
không bán, dùng giá catalog và mốc niêm yết; `src/evaluation/listing_launch.py`
huấn luyện, đánh giá ba fold và xuất artifact riêng khi có dữ liệu đó.
Xem [hợp đồng dữ liệu](listing_data_contract.md). Test tổng hợp chỉ xác minh luồng
và chống rò rỉ thời gian, không dùng làm metric thị trường.
