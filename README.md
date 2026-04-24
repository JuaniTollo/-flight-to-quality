# Modeling Economic Cycles with Lotka-Volterra

Empirical implementation of José A. Tapia's endogenous profit-investment cycle model using predator-prey differential equations on US macroeconomic data (1947–present).

## Theoretical Basis

Based on Tapia (*Six Crises of the World Economy*, Palgrave Macmillan, 2023), who replaces Goodwin's Employment/Wages variables with:

| Role | Variable | Rationale |
|------|----------|-----------|
| **Prey (P)** | Corporate Profits | Self-expanding but suppressed by capital overaccumulation |
| **Predator (I)** | Private Investment | Grows with profitability but destroys the conditions for its own reproduction |

> *"Movements in profits are followed some quarters later by movements in investment in the same direction, and movements in investment are followed by movements in profits in the opposite direction."* — Tapia (2023, pp. 198–199)

### Differential System

$$\frac{dP}{dt} = \alpha P - \beta P I \qquad \frac{dI}{dt} = \delta P I - \gamma I$$

Parameters $(\alpha, \beta, \delta, \gamma)$ calibrated empirically via `scipy.optimize.minimize` on Z-Score preprocessed data.

## Data

All series downloaded from FRED (Federal Reserve Economic Data):

| Variable | Ticker | Description |
|----------|--------|-------------|
| Profits (Prey) | `A053RC1Q027SBEA` | Corporate Profits w/ IVA & Capital Consumption Adjustments (NIPA, quarterly) |
| Investment (Predator) | `GPDI` | Gross Private Domestic Investment (quarterly, billions USD) |
| Reference cycles | `USREC` | NBER Recession Indicators |

Period: **1947-Q1 – present** (300+ quarters, 10+ complete cycles).

## Preprocessing

```python
# Remove long-run trend, expose cyclical dynamics
df['P'] = df['PROFITS'].pct_change(4) * 100
df['I'] = df['INVESTMENT'].pct_change(4) * 100

# Z-Score normalization (required before optimizer — profits are ~2.5x more volatile than investment)
df['P_z'] = (df['P'] - df['P'].mean()) / df['P'].std()
df['I_z'] = (df['I'] - df['I'].mean()) / df['I'].std()
```

## Quickstart

```bash
uv sync

# Download and process all FRED data
uv run python -m src.benchmarks.finance.pipeline

# Then run notebooks in order
```

## EDA Notebooks

| Notebook | Content |
|----------|---------|
| `00_tapia_figures.ipynb` | Replication of Tapia (2023) figures: corporate profits, world GDP growth, capital cycle |
| `01_eda_series_temporales.ipynb` | YoY time series with NBER recession bands. Verifies profits peak and fall before investment |
| `02_eda_rezagos_causalidad.ipynb` | Cross-correlation functions: Corr(P_t, I_{t−k}) and Corr(I_t, P_{t−k}). Profits lead investment by ~2 quarters; overaccumulation depresses profits ~6 quarters later |
| `03_eda_espacio_fases.ipynb` | Z-Score phase space: anticlockwise orbits confirm Lotka-Volterra topology. Includes interactive Plotly animation (`output/notebooks/03_animacion_espacio_fases.html`) |

## Repository Structure

```
├── configs/benchmarks/finance.yaml   # FRED tickers, dates, paths
├── notebooks/                        # EDA notebooks (executed, outputs embedded)
├── src/benchmarks/finance/
│   ├── loader.py                     # FRED API wrapper
│   ├── processor.py                  # ETL transformations
│   └── pipeline.py                   # Automated orchestrator (no args, runs all)
├── pyproject.toml                    # Dependencies (uv)
└── output/notebooks/                 # Generated figures and HTML (gitignored)
```

## Next Steps

1. **Calibration**: fit $(\alpha, \beta, \delta, \gamma)$ with `scipy.integrate.odeint` + `scipy.optimize.minimize` against Z-Score data
2. **Validation**: compare theoretical orbit vs empirical data per individual cycle
3. **Structural extension**: add interest rate as exogenous shock variable for monetary policy regime analysis
