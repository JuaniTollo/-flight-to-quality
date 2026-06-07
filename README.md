# Flight to Quality — Profit-Investment Cycles in the US Economy

Empirical study of US business cycles using José A. Tapia's reformulation of
Goodwin (1967): a Lotka-Volterra predator-prey system with **corporate profits
as prey** and **private investment as predator**, calibrated against quarterly
FRED data from 1948 onwards.

The repository implements the data pipeline, the EDA, and a sequence of modelling
experiments (`src/experiment/`, pieces P1–P11) that test the Lotka-Volterra
formulation and develop an alternative. **Main finding:** the fixed-phase
predator-prey oscillator does not capture the cycle; the profit→investment
overaccumulation feedback is a **distributed maturation delay centred at ~1 year**,
modelled with a physical maturation-chain ODE calibrated by an inverse PINN.

- Orientation, what was tried, and how to continue: [`notes/STATUS.md`](notes/STATUS.md).
- Quantitative findings (P1–P11): [`EXPERIMENT.md`](EXPERIMENT.md).
- Identifiability analysis of the distributed-lag kernel: [`notes/P10_CONSOLIDADO.md`](notes/P10_CONSOLIDADO.md).

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
│   ├── eda/                          # Plot scripts (one per concern)
│   │   └── …                         #   replication, lead-lag, phase space
│   └── experiment/                   # Modelling experiments P1–P11 (Python + Julia)
│       ├── common.py                 #   data, NBER dating, oscillators, fits
│       ├── p7_overaccumulation.py    #   CCF (P7)
│       ├── p8_event_study.py         #   crisis event study (P8)
│       ├── p9_*.py                   #   crisis-regime, ablation, placebo, pooled (P9)
│       ├── p10_*.py                  #   distributed-lag / stochastic-delay bench (P10)
│       ├── p11_physical_delay.py     #   maturation model + identifiability (P11)
│       ├── models.jl                 #   oscillators: solver + inverse PINN
│       └── p11_pinn_maturation.jl    #   inverse PINN of the maturation model
├── julia/                            # Pinned Julia env (Project.toml + Manifest.toml)
├── notes/                            # STATUS.md, P10_CONSOLIDADO.md, method notes
├── results/                          # Committed numeric artefacts (e.g. PINN estimates)
├── tests/                            # pytest data-validation tests
├── report/                           # Committed: EDA write-up + plots
├── data/                             # Versioned: raw & processed CSVs (FRED snapshot)
├── EXPERIMENT.md                     # Quantitative findings P1–P11
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

- ETL pipeline (idempotent, configurable, tested) + EDA modules + committed report.
- Modelling experiments P1–P11 (`src/experiment/`): oscillator fits (solver + inverse
  PINN), dynamical evaluation, cross-correlation, crisis event study, distributed-lag /
  stochastic-delay bench, and a physical maturation-delay model.
- Pinned environments for exact reproduction: `uv.lock` (Python) and `julia/` (Julia).

Current direction and open work (full detail in [`notes/STATUS.md`](notes/STATUS.md)):

- **Affirmative result:** distributed maturation delay (~1 year) in the profit→investment
  feedback; physical maturation-chain model captures it; fixed-phase oscillators do not.
- **Open:** finish the inverse-PINN calibration of the maturation model; map the
  identifiability frontier in synthetic; differentiable DDE + hierarchical pooling.

For where the project has been, what was tried, what to reuse and what to avoid re-trying,
read **[`notes/STATUS.md`](notes/STATUS.md)**.
