"""
test_requirement_updates.py - Kiểm thử cập nhật yêu cầu đề bài: Chống rò rỉ review, Thang đo BRL & Tuning 3 họ cây
==================================================================================================================
- test_review_leakage_prevention(): Kiểm tra review_score chỉ tính từ các đánh giá trước tuần dự báo, không rò rỉ review đồng thời/tương lai
- test_currency_scale_consistency_brl(): Kiểm tra tính nhất quán của đơn vị tiền tệ Brazil Real (BRL) trên toàn bộ các trường price, freight, revenue
- test_tree_hyperparameter_grids(): Kiểm tra không gian siêu tham số cấu hình cho 3 họ mô hình cây (Decision Tree, Random Forest, Gradient Boosting)
- test_grid_search_tree_optimization(): Kiểm tra quá trình GridSearchCV trên tập Validation 10% chọn ra bộ tham số tối ưu và không bị overfitting
- test_target_scale_and_evaluation(): Kiểm tra biến mục tiêu doanh số bán hàng được tính toán đúng thang đo và đánh giá độc lập trên tập Test
"""
