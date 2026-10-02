"""
visualization.py - Trực quan hóa kết quả
==========================================
- plot_eda(df): EDA (phân bố sales, correlation heatmap, boxplot outliers)
- plot_actual_vs_predicted(results, y_test): Scatter actual vs predicted cho từng mô hình
- plot_residuals(results, y_test): Biểu đồ residuals
- plot_feature_importance(model, feature_names): Feature Importance (RF / GB)
- plot_model_comparison(summary_df): Biểu đồ cột so sánh MSE/MAE/RMSE/R²
- plot_seasonal_trends(df): Biểu đồ xu hướng sales theo mùa
"""
