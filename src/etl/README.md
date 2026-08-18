# ETL — Extract, Transform, Load

Self-contained pipeline that pulls macroeconomic time-series from
[FRED](https://fred.stlouisfed.org/) and produces analysis-ready CSVs.

## Layout

```
src/etl/
├── paths.py       # Repo-relative path helpers (REPO_ROOT, resolve())
├── loader.py      # Extract: pandas-datareader → data/raw/*.csv
├── processor.py   # Transform: dataset-specific feature engineering
├── pipeline.py    # Orchestrator: download → transform
└── README.md
```

## Configuration

Datasets are declared in two YAML files under `configs/`:

| file                                          | purpose                                                              |
|-----------------------------------------------|----------------------------------------------------------------------|
| [`paper.yaml`](../../configs/paper.yaml)      | `lotka_volterra` — the only data used by the estimates (`src/experiment/`) |
| [`eda.yaml`](../../configs/eda.yaml)          | `corporate_profits`, `global_growth`, `capital_cycle` — Tapia replication figures (`src/eda/`) |

Each dataset entry specifies:

| field         | meaning                                         |
|---------------|-------------------------------------------------|
| `source`      | `"fred"` (only backend currently supported)     |
| `start_date`  | inclusive ISO date                              |
| `frequency`   | pandas resample rule (`QE`, `A`, …)             |
| `save_path`   | output CSV, relative to repo root               |
| `indicators`  | mapping `column_name → FRED_series_id`          |

## Running

```bash
# normal: skip already-downloaded raw files
uv run python -m src.etl.pipeline

# only the paper's data (configs/paper.yaml)
uv run python -m src.etl.pipeline --config paper

# re-download everything from FRED
uv run python -m src.etl.pipeline --force
```

Outputs:

- `data/raw/<dataset>.csv`       — verbatim FRED response, resampled
- `data/processed/<dataset>.csv` — engineered features (see below)

## Transforms

Each dataset has a pure function in `processor.py` — easy to test, easy to extend.

| dataset             | transform                                                             |
|---------------------|-----------------------------------------------------------------------|
| `lotka_volterra`    | `PROFITS_YOY`, `INVEST_YOY` (`pct_change(4)*100`), `P_z`, `I_z` (z-score) |
| `global_growth`     | `GROWTH` (annual `pct_change`*100), `TREND_10Y` (centered rolling mean)  |
| `capital_cycle`     | `INV_RATIO` = `REAL_INVESTMENT / REAL_GDP * 100`                      |
| `corporate_profits` | pass-through (Tapia Fig 2.12 plots raw levels on log scale)           |

## Data validation

Tests live in [`tests/test_etl.py`](../../tests/test_etl.py) and check:

- raw CSVs have the expected columns
- processed CSVs include the engineered features
- z-score columns have mean ≈ 0 and std ≈ 1
- no NaN in critical columns of processed data
- the date range covers the configured `start_date` onwards

Run with:

```bash
uv run pytest
```

## Adding a new dataset

1. Append an entry to `configs/paper.yaml` (if the estimates use it) or `configs/eda.yaml`.
2. Add a `transform_<name>` function to `processor.py`, register it in `TRANSFORMS`.
3. Add a row to `tests/test_etl.py` describing expected columns and invariants.
