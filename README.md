# Flight to Quality — Profit-Investment Cycles in the US Economy

Empirical study of the US profit→investment cycle, starting from José A. Tapia's
reformulation of Goodwin (1967) — a Lotka-Volterra predator-prey system with
**corporate profits as prey** and **private investment as predator** — and testing
it against quarterly FRED data from 1948 onwards.

**Main finding.** The fixed-phase predator-prey oscillator does **not** capture the
cycle. The profit→investment overaccumulation feedback is a **distributed maturation
delay centred at ~1 year**: more accumulated investment depresses profits about four
quarters later. The result is one of **identifiability** — the delay *centre* and the
sign of the effect are robust and converge across methods, but the *shape* of the
delay distribution is not estimable with this data (n≈9 crises, SNR≈3%).

This repository holds exactly what backs the paper:

- The write-up: [`paper/main_es.tex`](paper/main_es.tex) (compiled `paper/main_es.pdf`).
- Citation provenance for every claim in the paper: [`notes/FUENTES_CITAS.md`](notes/FUENTES_CITAS.md).

## Theoretical setup

Following Tapia (*Six Crises of the World Economy*, Palgrave Macmillan, 2023):

| Role         | Variable             | Rationale                                                         |
|--------------|----------------------|-------------------------------------------------------------------|
| Prey  (P)    | Corporate profits    | Self-expanding, but eventually suppressed by capital over-accumulation |
| Predator (I) | Private investment   | Grows with profitability, then destroys the conditions for its own reproduction |

The continuous-time system

$$\frac{dP}{dt} = \alpha P - \beta P I, \qquad \frac{dI}{dt} = \delta P I - \gamma I$$

predicts anticlockwise orbits in (P, I) phase space — the property the EDA in this
repo tests visually and via cross-correlation, and that the paper ultimately rejects
in favour of the distributed-delay model.

## Data

All series are pulled from [FRED](https://fred.stlouisfed.org/) by
`pandas-datareader`:

| Variable          | Ticker                | Description                                                |
|-------------------|-----------------------|------------------------------------------------------------|
| Profits (prey)    | `A053RC1Q027SBEA`     | Corporate profits w/ IVA & CCAdj (NIPA, quarterly)        |
| Investment (pred.)| `GPDI`                | Gross private domestic investment (quarterly, billions $) |
| Recession band    | `USREC`               | NBER recession indicator                                   |

…plus the companion datasets used for the Tapia replication figures
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
│   ├── eda/                          # Plot scripts (replication, lead-lag, phase space)
│   └── experiment/                   # The experiments behind the paper (Python + Julia)
│       ├── common.py                 #   data, NBER dating, shared fits
│       ├── p15_crisis_series.py      #   Fig. 1: profit/investment cycle + recessions
│       ├── p7b_ccf_surrogate.py      #   CCF + AR-surrogate null (Slutsky-Yule shield)
│       ├── p8_event_study.py         #   superposed-epoch study around NBER recessions
│       ├── p10_distlag.py            #   distributed-lag regression (delay ~4 quarters)
│       ├── p11_physical_delay.py     #   physical maturation-chain model + identifiability
│       ├── p12_*.py                  #   k regression, placebo, lag stability, VAR-Lyapunov, transform robustness
│       └── p17_bayes_lag.jl          #   Bayesian inverse (Turing/NUTS): posterior over mu and k
├── julia/                            # Pinned Julia env (Project.toml + Manifest.toml)
├── notes/FUENTES_CITAS.md            # Citation provenance for the paper
├── paper/                            # main_es.tex + compiled PDF
├── report/                           # Committed: EDA write-up + plots
├── results/                          # Committed numeric artefacts
├── data/                             # Versioned raw & processed CSVs (FRED snapshot)
├── tests/                            # pytest data-validation tests
└── pyproject.toml
```

## Quickstart

```bash
uv sync                                   # install deps

uv run pytest                             # data-validation tests, no FRED calls (synthetic data)

uv run python -m src.etl.pipeline         # download + transform (skips cached files)
uv run python -m src.etl.pipeline --force # force re-download

uv run python -m src.eda.run_all          # regenerate report/plots/*.png
```

## Run the paper's experiments

Each experiment is a standalone module under `src/experiment/`. Prepare both
environments first:

```bash
uv sync                                                  # Python deps (uv.lock)
julia --project=julia -e 'using Pkg; Pkg.instantiate()'  # Julia deps (Manifest.toml)
```

Figures:

```bash
uv run python -m src.experiment.p15_crisis_series        # Fig. 1: profit/investment cycle + recessions
uv run python -m src.experiment.p7b_ccf_surrogate        # CCF + AR-surrogate null (p~0.006)
uv run python -m src.experiment.p11_physical_delay       # maturation-chain model + identifiability (Fig. model)
julia --project=julia src/experiment/p17_bayes_lag.jl    # Bayesian inverse: joint posterior over mu and k (Fig. bayes)
```

Delay estimate & overaccumulation effect:

```bash
uv run python -m src.experiment.p10_distlag              # distributed-lag regression (~4 quarters)
uv run python -m src.experiment.p12_overaccum_decomp     # overaccumulation regression (k, t-stats, circular-shift placebo)
```

Identifiability controls & robustness:

```bash
uv run python -m src.experiment.p8_event_study           # event study around NBER recessions
uv run python -m src.experiment.p12_lag_stability        # leave-one-crisis-out
uv run python -m src.experiment.p12_kernel_placebo       # placebo: the negative hump is not crisis-specific (p~0.35)
uv run python -m src.experiment.p12_var_lyapunov         # VAR-Lyapunov observational-equivalence check
uv run python -m src.experiment.p12_transform_robustness # QoQ vs YoY (Slutsky-Yule)
```

Figures are written to `output/experiment/` (gitignored, regenerable). The FRED
snapshot in `data/` is versioned, so every experiment reproduces without
re-downloading.

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

The test suite uses synthetic raw data (deterministic, seeded) so it doesn't hit
FRED. It verifies the raw → processed transforms, z-score features (mean 0 / std 1),
absence of NaN in engineered columns, the configured date range, and repo-rooted
path resolution.
