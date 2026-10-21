"""
tree.py - Cây quyết định hồi quy From-Scratch (Decision Tree Regressor)
=======================================================================
Hiện thực cây quyết định hồi quy thuần NumPy:

Classes:
- TreeNode: Cấu trúc nút lưu trữ ngưỡng cắt (split_threshold), đặc trưng cắt (feature_idx), và giá trị dự đoán lá (leaf_value)
- DecisionTreeRegressorScratch:
  * Tiêu chuẩn phân nhánh dựa trên giảm phương sai (Variance Reduction / Mean Squared Error)
  * Kiểm soát dừng phân nhánh thông qua max_depth, min_samples_split, min_samples_leaf
  * Hỗ trợ tìm kiếm vết cắt tối ưu trên ma trận đặc trưng
"""
