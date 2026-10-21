"""Reproducible first-observed-sale proxy evaluation, with purged temporal holdout.

Run ``python -m src.evaluation.cold_start`` to refresh only this experiment and
the cold-start member of the existing deployment bundle.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data.builder import build_cold_start_dataset
from src.models.cold_start import CategoryMedianColdStartBaseline, ColdStartProductRegressor


def temporal_split(df, train_fraction=0.8):
    """Hold out the last origin dates; purge training labels crossing cutoff."""
    if not 0 < train_fraction < 1:
        raise ValueError('train_fraction must be between zero and one')
    days = pd.DatetimeIndex(df.first_purchase_time.dt.normalize().unique()).sort_values()
    if len(days) < 2:
        raise ValueError('Need at least two distinct origin dates')
    cutoff = days[min(len(days) - 1, max(1, int(len(days) * train_fraction)))]
    train = df.loc[(df.first_purchase_time < cutoff) & (df.target_window_end <= cutoff)].copy()
    test = df.loc[df.first_purchase_time >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError('Empty temporal partition after purging seven-day labels')
    return train, test, cutoff


def train_and_evaluate(raw_dir=None, output_dir=None):
    """Fit fixed hyperparameters on Train only; report all baselines on same Test.

    No model selection on Test. The saved tree remains the pre-specified candidate,
    even when a baseline wins. Product metadata is an untimed Olist snapshot.
    """
    df = build_cold_start_dataset(raw_dir)
    train, test, cutoff = temporal_split(df)
    baseline = CategoryMedianColdStartBaseline().fit(train)
    model = ColdStartProductRegressor(max_depth=6, min_samples_split=20, min_samples_leaf=10).fit(train)
    predictions = {
        'CategoryMedianColdStartBaseline': baseline.predict(test),
        'SingleUnitPriceBaseline': test.price.to_numpy(dtype=float),
        'ColdStartProductRegressor': model.predict(test),
    }
    y = test.launch_sales_brl.to_numpy(dtype=float)
    variance = float(np.var(y))
    rows = []
    for name, pred in predictions.items():
        mse = float(np.mean((y - pred) ** 2))
        rows.append({'Model': name, 'MAE_BRL': float(np.mean(np.abs(y - pred))),
                     'RMSE_BRL': float(np.sqrt(mse)),
                     'R2_Score': float(1 - mse / variance) if variance > 0 else None,
                     'Target_Variance': variance, 'Test_Count': len(test)})
    summary = pd.DataFrame(rows)
    manifest = {
        'target': 'seven_day_revenue_from_first_observed_sale_including_first_sale',
        'price_source': 'first_observed_transaction_not_target_window_median',
        'split': 'temporal_holdout_80_percent_origin_dates_with_label_purge',
        'cutoff': cutoff.isoformat(), 'train_count': len(train), 'test_count': len(test),
        'purged_count': len(df) - len(train) - len(test),
        'train_last_label_end': train.target_window_end.max().isoformat(),
        'test_first_origin': test.first_purchase_time.min().isoformat(),
        'test_last_origin': test.first_purchase_time.max().isoformat(),
        'test_single_unit_fraction': float(test.launch_units.eq(1).mean()),
        'dataset_sha256': hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest(),
        'hyperparameters': {'max_depth': 6, 'min_samples_split': 20, 'min_samples_leaf': 10},
        'selection': 'pre_specified_tree_no_test_selection',
        'limitations': ['Not listing-date cold start; products without sales are absent.',
                       'Includes the known first transaction in the target.',
                       'Delivered status and product metadata are retrospective snapshots; availability is assumed.',
                       'No causal marketing interpretation.'],
    }
    out = Path(output_dir) if output_dir is not None else config.METRICS_DIR
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / 'cold_start_evaluation.csv', index=False)
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    (out / 'cold_start_manifest.json').write_text(manifest_text, encoding='utf-8')
    manifest['evaluation_sha256'] = hashlib.sha256((out / 'cold_start_evaluation.csv').read_bytes()).hexdigest()
    manifest['manifest_sha256'] = hashlib.sha256((out / 'cold_start_manifest.json').read_bytes()).hexdigest()
    return model, summary, manifest


def main():
    from src.models.persistence import load_artifacts, save_artifacts
    bundle = load_artifacts()  # Fail before training if weekly artifact is missing.
    model, summary, manifest = train_and_evaluate()
    metadata = dict(bundle.get('metadata', {}))
    metadata.update(has_cold_start_model=True, cold_start_evaluation=manifest)
    save_artifacts(bundle['pipeline'], bundle['model'], metadata=metadata, cold_start_model=model)
    from src.evaluation.reporting import sync_readme
    sync_readme()
    print(summary.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
