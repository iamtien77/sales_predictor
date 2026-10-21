"""Auditable research runs; never overwrite deployed models or select on Test.

Run: python -m src.evaluation.experiments [--weekly]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import config
from src.data.builder import build_cold_start_dataset
from src.models.cold_start import CategoryMedianColdStartBaseline, ColdStartProductRegressor
from src.models.linear import LinearRegressionFromScratch
from src.models.tree import DecisionTreeRegressorFromScratch


def scores(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    mse = np.mean((y - pred) ** 2)
    return {'MAE': float(np.mean(np.abs(y - pred))), 'RMSE': float(np.sqrt(mse)),
            'R2': float(1 - mse / np.var(y)) if np.var(y) else np.nan}


def rolling_cohorts(df, origin, label_end):
    """Expanding Train and disjoint Test periods: 60-70%, 70-80%, 80-100% dates."""
    days = pd.DatetimeIndex(df[origin].dt.normalize().unique()).sort_values()
    if len(days) < 10:
        raise ValueError('Need at least ten origin dates')
    edges = [days[int(len(days) * f)] for f in [.6, .7, .8]] + [days[-1] + pd.Timedelta(days=1)]
    for fold, (start, stop) in enumerate(zip(edges[:-1], edges[1:]), 1):
        train = df.loc[(df[origin] < start) & (df[label_end] <= start)].copy()
        test = df.loc[(df[origin] >= start) & (df[origin] < stop)].copy()
        if train.empty or test.empty:
            raise ValueError('Empty fold after label purge')
        yield fold, train, test, start, stop


def cold_experiment(out):
    df = build_cold_start_dataset()
    rows, groups = [], []
    for fold, train, test, start, stop in rolling_cohorts(df, 'first_purchase_time', 'target_window_end'):
        model = ColdStartProductRegressor().fit(train)
        base = CategoryMedianColdStartBaseline().fit(train)
        # Controlled ablation: same content transform, same tree settings; change only target scale.
        raw_tree = DecisionTreeRegressorFromScratch(max_depth=6, min_samples_split=20, min_samples_leaf=10)
        raw_tree.fit(model._prepare_features(train), train.launch_sales_brl.to_numpy())
        predictions = {'Price': test.price.to_numpy(), 'CategoryMedian': base.predict(test),
                       'LogTree': model.predict(test),
                       'RawRevenueTree': np.maximum(0, raw_tree.predict(model._prepare_features(test)))}
        y = test.launch_sales_brl.to_numpy()
        for name, pred in predictions.items():
            rows.append({'Fold': fold, 'Model': name, 'Start': str(start), 'EndExclusive': str(stop),
                         'TrainCount': len(train), 'TestCount': len(test),
                         'TrainLastLabelEnd': str(train.target_window_end.max()), **scores(y, pred)})
            for group, mask in [('one_unit', test.launch_units.to_numpy() == 1), ('multiple_units', test.launch_units.to_numpy() > 1)]:
                if mask.any():
                    groups.append({'Fold': fold, 'Model': name, 'Group': group, 'Count': int(mask.sum()), **scores(y[mask], pred[mask])})
        print(f'cold fold {fold}: train={len(train)}, test={len(test)}', flush=True)
    pd.DataFrame(rows).to_csv(out / 'cold_rolling.csv', index=False)
    pd.DataFrame(groups).to_csv(out / 'cold_subgroups.csv', index=False)


def linear_experiment(out):
    """Reproduce fast/full fits. Ridge ablation on Validation; Test is diagnostic only."""
    X = sp.load_npz(config.CACHE_X_TRAIN)
    y = np.load(config.CACHE_Y_TRAIN)
    V, T = sp.load_npz(config.CACHE_X_VAL), sp.load_npz(config.CACHE_X_TEST)
    v, t = np.load(config.CACHE_Y_VAL), np.load(config.CACHE_Y_TEST)
    idx = np.random.default_rng(config.RANDOM_STATE).choice(len(y), min(20000, len(y)), replace=False)
    rows, extremes = [], []
    for mode, indices in [('fast20k', idx), ('full', np.arange(len(y)))]:
        A, b = X[indices], y[indices]
        gram = (A.T @ A).toarray()
        condition = float(np.linalg.cond(gram))
        for alpha in [1e-5, 1., 100.]:
            model = LinearRegressionFromScratch(alpha_ridge=alpha).fit(A, b)
            for split, B, target in [('Validation', V, v), ('Test', T, t)]:
                logs = model.predict(B)
                revenue = np.maximum(0, np.expm1(logs))
                truth = np.expm1(target)
                errors = (revenue - truth) ** 2
                top = np.argsort(errors)[-max(1, len(errors) // 100):]
                train_cap = np.maximum(0, np.expm1(np.clip(logs, 0, np.max(b))))
                rows.append({'Mode': mode, 'Alpha': alpha, 'Split': split, 'TrainCount': len(b),
                             'GramCondition': condition, 'CoefficientL2': float(np.linalg.norm(model.coef_)),
                             'LogRMSE': float(np.sqrt(np.mean((logs - target) ** 2))),
                             'MaxPredictedLog': float(logs.max()), 'MaxTrainLog': float(b.max()),
                             'MaxPredictedBRL': float(revenue.max()),
                             'Top1PctSSEShare': float(errors[top].sum() / errors.sum()),
                             'TrainMaxCapRMSE': scores(truth, train_cap)['RMSE'], **scores(truth, revenue)})
                if alpha == 1e-5 and split == 'Test':
                    for i in np.argsort(errors)[-10:]:
                        extremes.append({'Mode': mode, 'TestRow': int(i), 'ActualBRL': float(truth[i]),
                                         'PredictedBRL': float(revenue[i]), 'PredictedLog': float(logs[i]),
                                         'ActualLog': float(target[i])})
            print(f'linear {mode}, alpha={alpha}', flush=True)
    pd.DataFrame(rows).to_csv(out / 'linear_diagnostics.csv', index=False)
    pd.DataFrame(extremes).to_csv(out / 'linear_extremes.csv', index=False)


def weekly_experiment(out):
    from src.data.loader import load_raw_sales_data
    from src.data.builder import build_weekly_sales_dataset
    from src.data.preprocessing import CustomDataPipeline
    from src.data.feature_engineering import WeeklySalesFeatureEngineer
    from src.models.hurdle import TwoStageHurdleRegressorFromScratch
    df = build_weekly_sales_dataset(load_raw_sales_data())
    df['label_end'] = df[config.TIMESTAMP_COLUMN] + pd.Timedelta(days=7)
    rows = []
    for fold, train, test, start, stop in rolling_cohorts(df, config.TIMESTAMP_COLUMN, 'label_end'):
        pipe = CustomDataPipeline(num_cols=config.WEEKLY_NUMERICAL_COLS, cat_cols=config.WEEKLY_CATEGORICAL_COLS,
                                  outlier_cols=[], use_log_target=True, feature_engineer=WeeklySalesFeatureEngineer())
        X, y = pipe.fit_transform(train)
        Xt, _ = pipe.transform(test)
        truth = test.sales.to_numpy()
        model = TwoStageHurdleRegressorFromScratch().fit(X, y)
        preds = {'Zero': np.zeros(len(test)), 'PreviousWeek': test.last_week_sales.to_numpy(),
                 'Hurdle': np.maximum(0, np.expm1(model.predict(Xt)))}
        for name, pred in preds.items():
            rows.append({'Fold': fold, 'Model': name, 'Start': str(start), 'EndExclusive': str(stop),
                         'TrainCount': len(train), 'TestCount': len(test),
                         'TrainLastLabelEnd': str(train.label_end.max()), **scores(truth, pred)})
        print(f'weekly fold {fold} completed', flush=True)
    pd.DataFrame(rows).to_csv(out / 'weekly_rolling.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weekly', action='store_true')
    args = parser.parse_args()
    out = config.METRICS_DIR / 'experiments'
    out.mkdir(parents=True, exist_ok=True)
    cold_experiment(out)
    linear_experiment(out)
    if args.weekly:
        weekly_experiment(out)
    manifest = {'purpose': 'Exploratory diagnostics after prior Test inspection; not new blind evidence or model selection.',
                'cold': 'first-sale proxy, not listing-date cold start', 'weekly': args.weekly,
                'weekly_hyperparameters': 'TwoStageHurdleRegressorFromScratch defaults, full Train each fold',
                'linear': 'fixed cached train/validation/test; both fast20k seed42 and full fits',
                'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    render_report(out)


def render_report(out):
    """Read measured results; preserve individual folds rather than cherry-pick."""
    from src.evaluation.reporting import markdown_table
    manifest = json.loads((out / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Experiment result changed: {name}')
    cold = pd.read_csv(out / 'cold_rolling.csv')
    groups = pd.read_csv(out / 'cold_subgroups.csv')
    linear = pd.read_csv(out / 'linear_diagnostics.csv')
    weekly_section = ''
    if manifest['weekly']:
        weekly = pd.read_csv(out / 'weekly_rolling.csv')
        comparisons = []
        for fold, rows in weekly.groupby('Fold'):
            zero = rows.set_index('Model').loc['Zero']
            hurdle = rows.set_index('Model').loc['Hurdle']
            comparisons.append({'Fold': fold, 'Start': hurdle.Start, 'EndExclusive': hurdle.EndExclusive,
                                'TrainCount': hurdle.TrainCount, 'TestCount': hurdle.TestCount,
                                'Zero_RMSE': zero.RMSE, 'Hurdle_RMSE': hurdle.RMSE,
                                'RMSE_improvement_pct': 100 * (1 - hurdle.RMSE / zero.RMSE),
                                'Zero_MAE': zero.MAE, 'Hurdle_MAE': hurdle.MAE})
        weekly_section = '## Dự báo tuần qua ba mốc\n\n' + markdown_table(pd.DataFrame(comparisons)) + '''

Hurdle giảm RMSE so với Zero ở cả ba cửa sổ nhưng MAE cao hơn ở cả ba.
Đây là sự ổn định của đánh đổi quan sát được, không phải bảo đảm tương lai.
Mỗi fold fit lại preprocessing trên Train và huấn luyện Hurdle trên toàn Train,
giữ tham số mặc định cố định. Lag trong Test dùng lịch sử đã xảy ra trước từng
tuần đích: đánh giá one-week-ahead cập nhật quan sát, không phải dự báo đa bước
không có quan sát mới. Không dùng các bảng này để đổi artifact đang triển khai.
'''
    content = '''# Kiểm chứng thực nghiệm và chẩn đoán mô hình

Lệnh tái lập: `.venv/Scripts/python.exe -m src.evaluation.experiments --weekly`.
Nguồn: `results/metrics/experiments/`, có hash từng CSV trong manifest.
Đây là phân tích thăm dò sau khi đã xem Test ở các vòng trước, **không phải tập
kiểm định mù mới**. Không chọn tham số trên Test, không thay artifact triển khai.
Ba cửa sổ Test không giao nhau; các cửa sổ Train mở rộng và chỉ chứa nhãn đã hoàn tất.

''' + weekly_section + '\n## Cold-start proxy: ba cửa sổ\n\n' + markdown_table(
        cold[['Fold', 'Model', 'TrainCount', 'TestCount', 'MAE', 'RMSE', 'R2']]) + '''

### Vì sao cây thua baseline giá?

''' + markdown_table(groups.loc[groups.Model.isin(['Price', 'LogTree']),
                                ['Fold', 'Model', 'Group', 'Count', 'MAE', 'RMSE']]) + '''

Nhóm một đơn vị chiếm đa số. Với target bao gồm giao dịch đầu, baseline giá đúng
hoàn toàn ở nhóm này theo cấu trúc bài toán. Cây dự báo trung bình theo vùng đặc
trưng nên sai ở nhóm một đơn vị. Cây giảm lỗi ở nhóm nhiều đơn vị nhưng lợi ích
không bù được MAE phát sinh trên nhóm lớn. Phân nhóm dựa vào nhãn thật chỉ dùng
phân tích sau đánh giá, không được dùng để định tuyến dự báo lúc triển khai.

`RawRevenueTree` giữ cùng dữ liệu, phép biến đổi đặc trưng và tham số cây, chỉ đổi
nhãn từ log sang BRL. Kết quả không cải thiện nhất quán: không thể quy mọi lỗi
cho log-transform hoặc khẳng định cây trên BRL luôn tốt hơn. Cold-start này vẫn
là proxy **sau lần bán đầu**, không phải trước niêm yết.

## Linear Regression: tái hiện và thí nghiệm đối chứng

''' + markdown_table(linear[['Mode', 'Alpha', 'Split', 'LogRMSE', 'RMSE',
                            'MaxPredictedLog', 'MaxTrainLog', 'Top1PctSSEShare', 'TrainMaxCapRMSE']]) + '''

Hai chế độ tái hiện dùng đúng ma trận cache và seed 42: 20.000 dòng ngẫu nhiên
hoặc toàn bộ Train. Cột `Top1PctSSEShare` đo tỷ trọng SSE của 1% mẫu lỗi cao nhất.
Sai số log nhỏ về trung bình không bảo đảm sai số BRL nhỏ: expm1 khuếch đại phần
đuôi dương; giới hạn hiện tại vẫn cho phép dự báo log vượt cực đại Train 50% span.
Các mẫu cực trị và dự báo tiền tệ được xuất ở `linear_extremes.csv`.

Ridge 1 và 100 hầu như không cải thiện RMSE so với 1e-5; đây không phải bằng
chứng lỗi chỉ do thiếu regularization hay do dữ liệu chắc chắn phi tuyến.
`TrainMaxCapRMSE` là can thiệp chẩn đoán: giữ nguyên trọng số, chỉ chặn log tại
cực đại nhãn Train. RMSE giảm mạnh ngay cả trên Validation, xác nhận phần đuôi
nghịch đảo log là tác nhân quan trọng. Tuy nhiên mức lỗi vẫn cao; không tuyên bố
clipping đã giải quyết dự báo, cũng không âm thầm thay thuật toán bằng kết quả Test.
Gram condition và norm hệ số được lưu để kiểm tra số học, không tự coi condition
lớn là nguyên nhân duy nhất (one-hot và intercept có thể phụ thuộc tuyến tính).

## Cold-start trước lần bán đầu

Chưa có catalog thực để kiểm chứng. Luồng mới `src/data/launch.py` giữ cả sản phẩm
không bán, dùng giá catalog và mốc niêm yết; `src/evaluation/listing_launch.py`
huấn luyện, đánh giá ba fold và xuất artifact riêng khi có dữ liệu đó.
Xem [hợp đồng dữ liệu](listing_data_contract.md). Test tổng hợp chỉ xác minh luồng
và chống rò rỉ thời gian, không dùng làm metric thị trường.
'''
    path = config.BASE_DIR / 'docs' / 'algorithms' / 'experimental_validation.md'
    path.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
