"""
test_experiments_and_launch.py - Kiểm thử quy trình thực nghiệm và khởi chạy hệ thống
======================================================================================
- test_full_pipeline_run_fast_mode(): Kiểm tra thực thi trọn vẹn 10 bước của pipeline ở chế độ --fast-mode hoàn tất thành công
- test_artifacts_generation(): Kiểm tra tạo và xuất đầy đủ các tệp kết quả (metrics CSV, figures PNG, saved_models PKL)
- test_model_artifact_load_and_predict(): Kiểm tra tải lại tệp best_sales_model.pkl và thực hiện dự báo khớp với kết quả nghiệm thu
- test_cli_argument_parsing(): Kiểm tra tính đúng đắn của bộ phân tích cú pháp dòng lệnh CLI (các cờ --skip-eda, --skip-tuning, --fast-mode)
- test_demo_app_launch_readiness(): Kiểm tra tính sẵn sàng của service dự báo phục vụ ứng dụng Streamlit (app.py)
"""
