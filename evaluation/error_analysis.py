"""
error_analysis.py - Phân tích lỗi dự đoán
===========================================
- analyze_residuals(X_test, y_test, y_pred, feature_names): Phân tích residuals theo features
- find_worst_predictions(X_test, y_test, y_pred, n): Tìm n mẫu có sai số lớn nhất
- plot_error_distribution(y_test, y_pred): Biểu đồ phân bố sai số (histogram)
- compare_error_patterns(results_dict, X_test, y_test): So sánh pattern lỗi giữa các mô hình
"""
