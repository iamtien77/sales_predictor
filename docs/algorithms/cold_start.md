# First-observed-sale seven-day revenue proxy

The historical class names `ColdStartProductRegressor` and `launch_sales_brl`
remain for compatibility. They do **not** establish listing-date cold-start accuracy.

## Forecast contract

- Origin: first observed delivered purchase of the product.
- Price: the first transaction price, never the median of future transactions.
- Target: total item revenue in [origin, origin + 7 days), including the known first transaction.
- Products with less than seven days of observable follow-up are excluded.
- Products without sales are absent. No claim is made about zero-sale launches.
- Delivered status and product metadata are retrospective Olist snapshots; their
  historical availability cannot be verified. This is an offline proxy experiment.

## Training and evaluation

`src/evaluation/cold_start.py` holds out the last 20% of distinct origin dates.
Training samples whose seven-day labels cross the Test cutoff are purged.
The category encoding and numeric imputation are fitted on Train only.
The pre-specified tree uses depth 6, minimum split 20 and minimum leaf 10;
Test is not used to choose a model or its hyperparameters.

All candidates share exactly the same Test rows:

1. Category median revenue, fitted on Train.
2. Single-unit price baseline: predict the known first transaction price.
3. Content tree: five numeric features plus smoothed category target encoding,
   fit to log1p revenue and inverse-transformed for scoring.

The inverse log prediction is not generally an arithmetic conditional expectation
or a conditional median. Compare both MAE and RMSE instead of assuming superiority.

## Reproduce

```powershell
.venv/Scripts/python.exe -m src.evaluation.cold_start
```

This rebuilds the experiment, exports `results/metrics/cold_start_evaluation.csv`
and `cold_start_manifest.json`, replaces only the cold-start member in the existing
weekly deployment bundle, and regenerates README results. It requires the raw Olist
files and an existing weekly bundle. Full `main.py` uses the same evaluation function.

The manifest records dates, counts, purged samples, data fingerprint and assumptions.
Artifact metadata includes hashes linking the evaluation files to the fitted model.
Read the generated README section 4 for current values rather than copied numbers.
The deployed tree is an experimental candidate, not a claim that it beats the price baseline.

## UI and fallback

The UI loads the fitted model from the bundle and displays the single-unit baseline.
Without that model, the helper returns a clearly labelled one-unit rule, not an ML forecast.
Marketing is a separate assumption-based simulation, not a learned causal effect.
