"""
persistence.py - Lưu và tải model đã huấn luyện
=================================================
- save_model(model, scaler, filepath): Lưu model + scaler bằng joblib
- load_model(filepath): Tải model đã lưu từ file
- save_pipeline(pipeline, filepath): Lưu toàn bộ pipeline (scaler + encoder + model)
- load_pipeline(filepath): Tải pipeline đã lưu
"""
