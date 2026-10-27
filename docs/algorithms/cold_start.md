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

### Why the single-unit baseline is a strong comparator

The target includes the first transaction, whose price is already an input. When
only one item is sold during the seven-day window, the target equals that price,
so the single-unit baseline has zero error on that sample. The saved experiment's
manifest reports 85.47% single-unit Test products; its evaluation CSV shows lower
MAE and RMSE for this baseline than for the content tree. This is evidence about
that saved holdout, not a guarantee that the baseline wins on other datasets.

Tree leaves pool products and predict a mean in log space, so they need not
preserve the identity `sales = first_price` even for a single-unit product.
Inverse-transforming that mean also does not directly minimize squared error in
BRL. These mechanisms may contribute to the observed gap; attributing the gap
to either one requires a controlled comparison, not just the aggregate scores.
The single-unit baseline can still underpredict products with additional sales.

The retrospective `delivered` filter also affects the weekly forecasting data:
orders are grouped by purchase time using their eventual delivery outcome.
Train-only preprocessing and temporal splits do not make that future outcome
available at an earlier forecast origin. Weekly results therefore remain an
offline snapshot backtest until order-state availability is reconstructed and
the booked-versus-realized revenue target is explicitly resolved.

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
