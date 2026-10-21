Bản v5. Xây Dựng Pipeline Tiền Xử Lý Một Chiều (CustomDataPipeline)
Mục tiêu của bản này là đóng gói toàn bộ quy trình tiền xử lý thành pipeline một chiều tuân thủ nguyên tắc đóng băng thống kê, chỉ fit trên tập Train.

Bước 1. Thiết lập bộ điền khuyết thiếu CustomSimpleImputer
Cài đặt bộ điền khuyết thiếu thuần túy From-Scratch:
- Với các biến số học: tính giá trị trung vị (Median) trên tập Train và lưu lại.
- Với các biến danh mục: tính giá trị yếu vị (Mode) trên tập Train và lưu lại.
- Khi transform trên Val và Test, chỉ áp dụng giá trị đã lưu, không tính toán lại.

Bước 2. Thiết lập bộ mã hóa One-Hot CustomOneHotEncoder
Xác định danh mục ngành hàng xuất hiện trên tập Train (73 nhóm ngành hàng Olist).
Mã hóa thành 73 cột nhị phân (0 hoặc 1).
Cơ chế an toàn: nếu trên tập Val hoặc Test xuất hiện danh mục mới chưa từng thấy ở Train, bộ mã hóa tự động gán toàn bộ 73 cột bằng 0, không làm thay đổi số chiều ma trận.

Bước 3. Thiết lập bộ chuẩn hóa thang đo CustomStandardScaler
Tính toán kỳ vọng trung bình (mean) và độ lệch chuẩn (std) của 16 đặc trưng số học trên tập Train: z = (x - mean) / std.
Bảo toàn mean và std này để chuẩn hóa cho Val và Test.

Bước 4. Quy định nguyên tắc gọi fit() duy nhất trên Train
Tuyệt đối chỉ gọi pipeline.fit(train_df) một lần duy nhất.
Tập Val và Test chỉ được gọi pipeline.transform(val_df) và pipeline.transform(test_df).
Bảo đảm không có bất kỳ thông tin thống kê nào của Val và Test bị lộ vào quy trình tiền xử lý.

Bước 5. Biến đổi nhãn log1p và lưu trữ ma trận đệm
Tạo vector nhãn: y = log(1 + max(0, sales)).
Chuyển ma trận đặc trưng sang định dạng SciPy CSR và lưu cache vào data/processed/: X_train.npz, y_train.npy, X_val.npz, y_val.npy, X_test.npz, y_test.npy.
Lưu đối tượng pipeline vào cache_pipeline.pkl phục vụ tái lập 100%.
