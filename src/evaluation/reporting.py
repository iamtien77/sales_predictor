"""Render README results from CSVs and the deployed bundle, not copied numbers."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.models.persistence import load_artifacts


def markdown_table(frame):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f'{value:.4f}'
        return str(value).replace('|', '/')
    rows = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
            '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    rows.extend('| ' + ' | '.join(cell(v) for v in row) + ' |'
                for row in frame.itertuples(index=False, name=None))
    return '\n'.join(rows)


def sync_readme():
    root = Path(__file__).resolve().parents[2]
    weekly = pd.read_csv(config.METRICS_DIR / 'model_comparison.csv')
    validation = pd.read_csv(config.METRICS_DIR / 'validation_comparison.csv')
    cold = pd.read_csv(config.METRICS_DIR / 'cold_start_evaluation.csv')
    manifest = json.loads((config.METRICS_DIR / 'cold_start_manifest.json').read_text(encoding='utf-8'))
    bundle = load_artifacts()
    provenance = bundle['metadata'].get('cold_start_evaluation', {})
    for filename, key in [('cold_start_evaluation.csv', 'evaluation_sha256'),
                          ('cold_start_manifest.json', 'manifest_sha256')]:
        actual = hashlib.sha256((config.METRICS_DIR / filename).read_bytes()).hexdigest()
        if actual != provenance.get(key):
            raise ValueError(f'Cold-start artifact and {filename} do not match')
    params = bundle['metadata']['best_params']
    selected = params['selected_model']
    val = validation.loc[validation.Model.eq(selected)].iloc[0]
    if not bool(val.Selected) or not np.isclose(val.Validation_RMSE_BRL, params['validation_rmse_brl']):
        raise ValueError('Artifact and validation CSV do not match; refusing to publish mixed results')
    mode = params.get('training_mode', 'Không được lưu trong lần chạy cũ; không suy đoán full/fast')
    text = f'''## 🏆 4. Kết quả thực nghiệm hiện tại

<!-- GENERATED RESULTS: python -m src.evaluation.reporting -->
Các bảng dưới được sinh trực tiếp từ CSV trong `results/metrics` và artifact triển khai.
Chạy `.venv/Scripts/python.exe -m src.evaluation.reporting` để đồng bộ sau thực nghiệm.
Mô hình tuần được chọn bằng Validation: **{selected}**, Validation RMSE **{float(val.Validation_RMSE_BRL):.4f} BRL**.
Chế độ huấn luyện tuần: **{mode}**. Thời điểm huấn luyện tuần được ghi trong metadata: `{bundle['metadata'].get('timestamp', 'unknown')}`.
Đây là snapshot của lần chạy đã lưu, không trộn số liệu từ lần chạy khác.

### 4.1. Dự báo sản phẩm–tuần (Test, BRL)

{markdown_table(weekly)}

RMSE và MAE phản ánh các mục tiêu khác nhau; không suy ra lợi ích tồn kho chỉ từ R².
Chi phí là điểm phạt giả định trên sai lệch doanh thu, không phải chi phí vận hành được đo thực tế.
Chưa có đánh giá riêng precision/recall cho phát hiện tuần nhu cầu đột biến.

### 4.2. Proxy bảy ngày từ giao dịch đầu tiên

Nguồn: `cold_start_evaluation.csv`, quy trình: `cold_start_manifest.json`.
Train: **{manifest['train_count']}** sản phẩm; Test: **{manifest['test_count']}**;
loại **{manifest['purged_count']}** mẫu Train có cửa sổ nhãn vượt mốc **{manifest['cutoff']}**.
Giá là giá giao dịch đầu tiên. Mọi cửa sổ nhãn phải có đủ bảy ngày quan sát.

{markdown_table(cold[['Model', 'MAE_BRL', 'RMSE_BRL', 'R2_Score', 'Test_Count']])}

So sánh cả baseline danh mục và baseline một sản phẩm (`sales = first_price`).
Không chọn mô hình bằng Test: cây với tham số cố định là ứng viên thực nghiệm được lưu,
không được gọi là tốt nhất nếu baseline thắng. Tỷ lệ Test chỉ bán một đơn vị: **{manifest['test_single_unit_fraction']:.2%}**.

Giới hạn: mốc bắt đầu là lần bán đầu, không phải ngày niêm yết; target bao gồm giao dịch đầu đã biết.
Không có sản phẩm chưa bán; metadata sản phẩm và trạng thái delivered là snapshot cuối,
chưa có lịch sử để xác minh chúng sẵn có tại thời điểm dự báo. Chưa chứng minh cold-start trước bán.
<!-- END GENERATED RESULTS -->

'''
    readme = root / 'README.md'
    content = readme.read_text(encoding='utf-8')
    start = content.index('## 🏆 4.')
    end = content.index('## 📁 5.', start)
    readme.write_text(content[:start] + text + content[end:], encoding='utf-8')


if __name__ == '__main__':
    sync_readme()
