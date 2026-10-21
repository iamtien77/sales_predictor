Bản v4. Dò Tìm Siêu Tham Số Cho Gradient Boosting
Mục tiêu của bản này là khảo sát số vòng lặp boosting và độ sâu cây con để tối ưu hóa quá trình học phần dư.

Bước 1. Thiết lập không gian lưới tham số cho Gradient Boosting
Cấu hình param_grid:
- n_estimators: [15, 30] cây nông.
- max_depth: [2, 3].
- learning_rate: [0.1].
Tổng cộng 4 cấu hình thử nghiệm.

Bước 2. Thực hiện quét lưới trên Validation Set
Chạy grid search trên 249.399 dòng Validation:
- n_estimators = 15, max_depth = 2: Val RMSE = 61.45 BRL.
- n_estimators = 30, max_depth = 3: Val RMSE = 60.12 BRL.
Ghi nhận hàm mất mát train_loss_history_ giảm đều đặn qua các vòng lặp, chứng minh thuật toán hội tụ ổn định.

Bước 3. Đánh giá tốc độ và tài nguyên
Do thuật toán Gradient Boosting học tuần tự (cây sau phụ thuộc vào phần dư của cây trước), không thể song song hóa hoàn toàn như Random Forest.
Thời gian quét lưới lâu hơn so với Decision Tree đơn lẻ.

Bước 4. So sánh với các ứng viên khác
Gradient Boosting đạt Val RMSE 60.12 BRL, tốt hơn Random Forest (60.52 BRL) nhưng vẫn xếp sau Tuned Decision Tree (59.23 BRL).

Bước 5. Đóng gói kết quả thử nghiệm
Lưu lại toàn bộ các thông số vào lịch sử tìm kiếm siêu tham số chung.
