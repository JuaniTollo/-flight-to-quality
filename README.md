# Flight to Quality — Profit-Investment Cycles in the US Economy

Empirical study of US business cycles using José A. Tapia's reformulation of
Goodwin (1967): a Lotka-Volterra predator-prey system with **corporate profits
as prey** and **private investment as predator**, calibrated against quarterly
FRED data from 1948 onwards.

This repository implements the data pipeline and the EDA — there is currently
**no calibration code**; that is future work.

## Theoretical setup

Following Tapia (*Six Crises of the World Economy*, Palgrave Macmillan, 2023):

| Role         | Variable             | Rationale                                                         |
|--------------|----------------------|-------------------------------------------------------------------|
| Prey  (P)    | Corporate profits    | Self-expanding, but eventually suppressed by capital over-accumulation |
| Predator (I) | Private investment   | Grows with profitability, then destroys the conditions for its own reproduction |

The continuous-time system

$$\frac{dP}{dt} = \alpha P - \beta P I, \qquad \frac{dI}{dt} = \delta P I - \gamma I$$

predicts anticlockwise orbits in (P, I) phase space — the property the EDA in
this repo tests visually and via cross-correlation.

## Data

All series are pulled from [FRED](https://fred.stlouisfed.org/) by
`pandas-datareader`:

| Variable          | Ticker                | Description                                                |
|-------------------|-----------------------|------------------------------------------------------------|
| Profits (prey)    | `A053RC1Q027SBEA`     | Corporate profits w/ IVA & CCAdj (NIPA, quarterly)        |
| Investment (pred.)| `GPDI`                | Gross private domestic investment (quarterly, billions $) |
| Recession band    | `USREC`               | NBER recession indicator                                   |

…plus three companion datasets used for the Tapia replication figures
(`A446RC1Q027SBEA`, `A448RC1Q027SBEA`, `NYGDPPCAPKDWLD`, `GDPC1`, `GPDIC1`,
`CBIC1`). All declared in
[`configs/benchmarks/finance.yaml`](configs/benchmarks/finance.yaml).

## Repository layout

```
.
├── configs/benchmarks/finance.yaml   # FRED tickers, dates, save paths
├── src/
│   ├── etl/                          # Extract / Transform / Load (see src/etl/README.md)
│   │   ├── paths.py                  #   repo-rooted path helpers
│   │   ├── loader.py                 #   FRED downloader, idempotent
│   │   ├── processor.py              #   per-dataset feature engineering
│   │   └── pipeline.py               #   orchestrator
│   └── eda/                          # Plot scripts (one per concern)
│       ├── eda_00_tapia_figures.py
│       ├── eda_01_series_temporales.py
│       ├── eda_02_rezagos_causalidad.py
│       ├── eda_03_espacio_fases.py
│       └── run_all.py
├── tests/                            # pytest data-validation tests
├── report/                           # Committed: EDA write-up + plots
│   ├── README.md                     #   Read this for the analysis
│   └── plots/*.png                   #   Generated figures
├── data/                             # Gitignored: raw & processed CSVs
└── pyproject.toml
```

## Quickstart

```bash
uv sync                                   # install deps

uv run pytest                             # 13 tests, no FRED calls (synthetic data)

uv run python -m src.etl.pipeline         # download + transform (skips cached files)
uv run python -m src.etl.pipeline --force # force re-download

uv run python -m src.eda.run_all          # regenerate report/plots/*.png
```

## Read the EDA

The narrative analysis with embedded plots lives in
**[`report/README.md`](report/README.md)**. It walks through:

1. Tapia replication figures (corporate profits, world growth, capital cycle)
2. Time-series view: profits leading investment, with per-cycle zooms
3. Cross-correlation: lead-lag at the aggregate and by NBER phase
4. Phase space: z-score normalization, full orbit, per-cycle small-multiples

## Tests

```bash
uv run pytest -q
```

The test suite uses synthetic raw data (deterministic, seeded) so it doesn't
hit FRED. It verifies:

- raw → processed transforms add the expected columns
- z-score features have mean 0 / std 1
- no NaN in the engineered columns of `lotka_volterra` processed CSV
- the date range matches the configured `start_date`
- repo-rooted path resolution is stable

## Status & roadmap

What's here:

- ETL pipeline (idempotent, configurable, tested)
- Four EDA modules covering replication, lead-lag and phase-space topology
- A committed report rendering the plots above

What's not (and is not promised to be):

- ODE-constrained calibration of $(\alpha, \beta, \delta, \gamma)$
- Any inference, Bayesian or otherwise

If/when calibration is added, it will live in a new module (e.g.
`src/calibration/`) with its own README and its own tests, and this section
will be updated.
