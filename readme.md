# PreFlash Model

A conservative daily **sales forecasting** engine for PepsiCo financial series. The
script ingests historical daily sales (`Venta Pesos`), builds a clean daily time
series, trains two models (XGBoost + Prophet), and blends them with a robust
statistical baseline to produce a short‑horizon forecast. Predictions are
disaggregated back to the `Canal de Venta` × `Unidad Negocio` segment level and
written to a timestamped CSV.

The model places special emphasis on **weekend and Sunday behavior**, which are the
most volatile parts of the series, and lets the operator flag each forecasted
Sunday as *working* (stores open) or *non‑working* (stores closed).

- **Entry point:** `Main_PreFlash.py`
- **Core class:** `PepsicoFinancialForecast`
- **Cadence:** run monthly.
- **Language of the codebase:** Python 3.

---

## Operating cadence

The model runs on a **monthly** basis. Before each run, refresh the file(s) in the
`Input/` folder so that:

- Every **closed month** is present in full.
- The **month being predicted** contains data up to the last available day.

The forecast horizon starts on the day *after* the last date found in the data.

---

## Requirements

| Package  | Pin          | Purpose                                                                 |
|----------|--------------|-------------------------------------------------------------------------|
| pandas   | `==1.5.3`    | Tabular data manipulation (DataFrame / Series).                         |
| numpy    | `>=1.21.0`   | Numeric operations, arrays, statistical helpers.                        |
| xgboost  | `>=1.5.0`    | Gradient‑boosted trees for the machine‑learning forecast.               |
| prophet  | `>=1.1.0`    | Additive time‑series model (trend + seasonality + regressors).          |

Install with:

```bash
pip install -r requirements.txt
```

Both `xgboost` and `prophet` are imported defensively. If a package is missing, the
corresponding model is skipped and the pipeline falls back to the remaining
components (the statistical baseline always runs).

---

## Input data

CSV files are read from the `Input/` directory (all `*.csv` files are concatenated),
or a single file can be passed with `--input`. Each file must contain the following
columns; the run aborts with a clear error if any are missing:

| Column           | Type            | Notes                                             |
|------------------|-----------------|---------------------------------------------------|
| `Dia Fecha`      | date            | Parsed with format `%m/%d/%Y`.                    |
| `Canal de Venta` | text            | Sales channel; used for segmentation.             |
| `Unidad Negocio` | text            | Business unit; used for segmentation.             |
| `Venta Pesos`    | numeric (MXN)   | Daily sales amount; the forecast target.          |

---

## Output

Results are written to the `Output/` directory as
`forecast_weekend_opt_<timestamp>[suffix].csv`, with one row per forecast day and
segment:

| Column           | Description                                  |
|------------------|----------------------------------------------|
| `fecha`          | Forecast date (`YYYY-MM-DD`).                |
| `Canal de Venta` | Sales channel.                               |
| `Unidad Negocio` | Business unit.                               |
| `prediccion`     | Forecasted sales for that segment and day.   |

---

## How to run

```bash
# Interactive (default): prompts for the number of days (1–15)
python Main_PreFlash.py

# Forecast a fixed number of days
python Main_PreFlash.py --days 7

# Use one specific input file instead of the whole Input/ folder
python Main_PreFlash.py --input path/to/data.csv

# Auto mode (sets a default horizon of 7 when --days is omitted)
python Main_PreFlash.py --auto
```

**Command‑line arguments**

| Argument   | Default        | Description                                             |
|------------|----------------|---------------------------------------------------------|
| `--days`   | interactive    | Number of days to forecast.                             |
| `--input`  | `Input/*.csv`  | A single CSV file to use instead of the folder.         |
| `--auto`   | off            | Skips the initial day prompt (defaults to 7 days).      |

If any forecasted day falls on a **Sunday**, the run asks the operator to classify
each of those Sundays as *working* (`1`) or *non‑working* (`2`) before producing the
forecast.

---

## Pipeline overview

The `run()` method orchestrates the following stages.

### 1. Load — `load_all_data()`
Reads and concatenates every CSV in `Input/`, logging the row count of each file and
the combined total.

### 2. Preprocess — `preprocess_data()`
- Validates the four required columns.
- Coerces types: dates via `to_datetime` (`%m/%d/%Y`), `Venta Pesos` via
  `to_numeric` (invalid values become `0`), and trims whitespace on the text columns.
- Filters out rows with unparseable dates, negative sales, or empty/`nan` channel or
  business‑unit labels.
- Aggregates to a **total daily series** (`daily_total`) by summing `Venta Pesos`
  per date.
- Runs day‑of‑week outlier handling (see below).
- Linearly interpolates any remaining missing values.
- Builds a **segment‑level daily table** (`daily_segments`) grouped by
  date × channel × business unit.

### 3. Outlier handling (per day of week)
Outliers are treated **separately for each day of the week**, because weekday and
weekend distributions differ. For each day of week with more than 5 observations the
script computes the interquartile range (IQR = Q3 − Q1) and flags values outside:

- **Weekdays (Mon–Fri):** `Q1 − 1.5·IQR` to `Q3 + 1.5·IQR` (standard Tukey fences).
- **Weekends (Sat–Sun):** `Q1 − 2.0·IQR` to `Q3 + 2.0·IQR` (wider fences, because
  weekend sales are naturally more volatile).

Flagged values are **replaced with the median of the same day of the week** (winsor‑
style correction rather than deletion), and the counts of corrected weekday vs.
weekend points are logged.

### 4. Weekend pattern analysis — `analyze_weekend_patterns()`
Over the most recent 90 days, computes mean, median, standard deviation and
coefficient of variation (CV) per day of week, then runs a **statistical sanity check
on Sundays**:

- Sample‑size warnings when there are fewer than 5 or fewer than 8 Sundays.
- Volatility warnings when the Sunday CV exceeds 0.3 (high) or 0.5 (extreme).

These checks are informational (they surface confidence caveats in the logs).

### 5. Model training — `train_models()`

**XGBoost** (`XGBRegressor`) — trained when the package is available and there are at
least 60 days of history (and at least 20 usable samples). For each day from index 21
onward the script builds a feature row:

- *Calendar:* day of week, `is_weekend`, `is_saturday`, `is_sunday`, `is_weekday`,
  day of month, month, `is_month_end` (day ≥ 28).
- *Reference factors:* the monthly seasonal factor, the day‑of‑week factor and the
  day‑of‑week volatility (looked up from the tables below).
- *Lags:* values 1, 7 and 14 days back.
- *Rolling means:* trailing 7‑ and 14‑day averages.
- *Weekend features:* the recent weekend‑to‑weekday ratio and the recent weekend
  average.

Key hyper‑parameters: `objective=reg:squarederror`, `max_depth=4`,
`learning_rate=0.05`, `n_estimators=200`, `subsample=0.8`, `colsample_bytree=0.8`,
`reg_alpha=0.1`, `reg_lambda=0.1`, `random_state=42`.

**Prophet** — trained when the package is available. It uses linear growth with
additive seasonality and tight priors (`changepoint_prior_scale=0.002`,
`seasonality_prior_scale=0.03`, `interval_width=0.80`). The default weekly
seasonality is disabled and replaced with:

- A **custom weekly seasonality** (`period=7`, `fourier_order=3`).
- Four **extra regressors**: `is_weekend`, `is_saturday`, `is_sunday` (given the
  largest prior, reflecting Sunday volatility) and `is_month_end`.

### 6. Forecast generation

The horizon is `--days` calendar days starting the day after the last observed date.
There are two prediction paths:

**a) Full ensemble — `predict_with_weekend_calibration()`**
Used when the horizon contains **no Sundays**. For each day it combines three
estimates:

- **XGBoost** prediction.
- **Prophet** prediction.
- **Statistical baseline:** the 40th percentile of the last 6 values for the same day
  of week (a deliberately conservative central estimate), with a factor‑based
  fallback when there are too few observations.

These are blended with **day‑type weights** that lean on the robust statistical
baseline, most heavily on volatile days:

| Day type   | Statistical | XGBoost | Prophet |
|------------|-------------|---------|---------|
| Weekday    | 0.50        | 0.35    | 0.15    |
| Saturday   | 0.70        | 0.25    | 0.05    |
| Sunday     | 0.80        | 0.15    | 0.05    |

The blended value is then multiplied by the day‑of‑week factor, adjusted by the
Sunday scenario factor and confidence band when applicable, and finally **clipped to
volatility‑aware bounds** (wide for Sundays, moderate for Saturdays, tight for
weekdays) and floored at zero.

**b) Custom‑Sunday path — `predict_with_custom_sundays()` / `_predict_single_day()`**
Used when the horizon **contains one or more Sundays**. The operator classifies each
Sunday as working/non‑working, and the forecast is produced from the conservative
statistical baseline plus the appropriate Sunday factor and day bounds.

### 7. Segment allocation — `generate_segments()`
Converts the daily totals into segment‑level forecasts:

- Computes each segment's historical **share** over the last 60 days (falling back to
  the most recent 2,000 records if the recent window is sparse).
- Applies a small share floor (`0.0001`) and re‑normalizes so shares sum to 1.
- Multiplies each daily total by each segment share, with a per‑segment floor of 50.

### 8. Save — `save_forecast()`
Writes the segment‑level forecast to `Output/` with a timestamped filename and logs
the grand total.

---

## Calibration tables

These reference factors are defined in `PepsicoFinancialForecast.__init__` and drive
both the features and the post‑model calibration.

**Day‑of‑week factors** (multiplier vs. a normal day)

| Mon  | Tue  | Wed  | Thu  | Fri  | Sat  | Sun  |
|------|------|------|------|------|------|------|
| 1.05 | 1.03 | 1.08 | 1.02 | 0.98 | 0.75 | 0.30 |

**Sunday scenario factors** (share of a normal day)

| Scenario     | Factor | Confidence band |
|--------------|--------|-----------------|
| Working      | 0.45   | 0.35 – 0.55     |
| Non‑working  | 0.15   | 0.10 – 0.20     |

**Day‑of‑week volatility**

| Mon   | Tue   | Wed   | Thu   | Fri   | Sat   | Sun   |
|-------|-------|-------|-------|-------|-------|-------|
| 0.081 | 0.073 | 0.067 | 0.053 | 0.062 | 0.052 | 0.627 |

**Monthly seasonal factors**

| Jan  | Feb  | Mar  | Apr  | May  | Jun  | Jul  | Aug  | Sep  | Oct  | Nov  | Dec  |
|------|------|------|------|------|------|------|------|------|------|------|------|
| 0.98 | 0.95 | 1.02 | 1.01 | 1.04 | 1.02 | 1.06 | 1.04 | 1.00 | 1.08 | 1.12 | 1.15 |

---

## Project layout

```
.
├── Main_PreFlash.py     # Forecasting engine (this script)
├── requirements.txt     # Python dependencies
├── readme.md            # This document
├── Input/               # Monthly source CSVs (created/maintained by the operator)
└── Output/              # Timestamped forecast CSVs (created at run time)
```

## Design rationale (models)

- **XGBoost** captures non‑linear interactions between calendar signals, recent lags
  and rolling averages.
- **Prophet** contributes a smooth trend, custom weekly seasonality and month‑end
  effects, and is robust to gaps in the series.
- **Statistical baseline** (same‑weekday 40th percentile) provides a stable,
  conservative anchor that dominates the blend on the most volatile days.

Combining the three in a weighted ensemble aims to keep the forecast stable and
conservative while still benefiting from the pattern‑learning of the two models.
