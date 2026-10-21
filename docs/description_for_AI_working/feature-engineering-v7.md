Bản v7. Kỹ Nghệ Biến Mùa Vụ, Ngày Lễ & Tuần Hoàn Lượng Giác
Mục tiêu của bản này là mô hình hóa các biến động theo chu kỳ lịch, tuần lễ hội và mùa cao điểm mua sắm bán lẻ.

Bước 1. Kỹ nghệ biến chỉ báo mùa cao điểm mua sắm (is_peak_shopping_season)
Xác định các tuần có lưu lượng giao dịch đột biến trong năm:
- Tuần lễ hội mua sắm Black Friday (tuần cuối cùng của tháng 11).
- Mùa mua sắm chuẩn bị cho Giáng sinh và Năm mới (các tuần trong tháng 12).
Gán giá trị nhị phân 1 nếu tuần target thuộc giai đoạn cao điểm này, ngược lại gán 0.

Bước 2. Kỹ nghệ biến chỉ báo tuần có ngày lễ quốc gia (is_holiday_week)
Tra cứu lịch lễ hội quốc gia Brazil: Carnival, Ngày Quốc khánh (7/9), Ngày Lao động (1/5), Lễ Tạ ơn...
Gán giá trị nhị phân 1 nếu trong tuần target có ít nhất một ngày lễ quốc gia.

Bước 3. Kỹ nghệ biến tuần hoàn lượng giác của tháng (month_sin, month_cos)
Trích xuất tháng trong năm của ngày thứ Hai đầu tuần (tháng 1 đến tháng 12).
Áp dụng phép biến đổi lượng giác:
- month_sin = sin(2 * pi * month / 12)
- month_cos = cos(2 * pi * month / 12)
Biến đổi này bảo đảm rằng khoảng cách giữa tháng 12 và tháng 1 năm sau là liền mạch trên đường tròn lượng giác.

Bước 4. Bổ sung các biến chu kỳ phụ trợ
Trích xuất week_of_year (tuần thứ 1 đến 52 trong năm) và quarter (quý 1 đến quý 4).
Hỗ trợ các mô hình cây học các ngưỡng cắt theo quý và theo tuần mùa vụ.

Bước 5. Kiểm tra tính liên tục và phạm vi giá trị
Xác nhận month_sin và month_cos luôn nằm trong đoạn [-1.0, 1.0], không bị lỗi số học.
