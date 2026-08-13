"""Shared fixtures: synthetic raw data so tests don't depend on FRED/network."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a fake `data/raw/` tree with realistic schemas for each dataset.

    The shapes match what `pandas-datareader` returns for the FRED tickers
    declared in `configs/datasets.yaml`, so the processor transforms
    can run on it end-to-end.
    """
    root = tmp_path_factory.mktemp("repo")
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)

    rng = np.random.default_rng(seed=42)

    # --- lotka_volterra: quarterly 1947-Q1 → 2020-Q4 ---
    qidx = pd.date_range("1947-03-31", "2020-12-31", freq="QE")
    n_q = len(qidx)
    profits = pd.Series(
        100 * np.exp(0.02 * np.arange(n_q) / 4) * (1 + 0.05 * np.sin(np.arange(n_q) / 6)),
        index=qidx,
    )
    investment = pd.Series(
        80 * np.exp(0.018 * np.arange(n_q) / 4) * (1 + 0.07 * np.sin((np.arange(n_q) - 2) / 6)),
        index=qidx,
    )
    recession = pd.Series((rng.random(n_q) < 0.1).astype(float), index=qidx)
    pd.DataFrame({"PROFITS": profits, "INVESTMENT": investment, "RECESSION": recession}).to_csv(
        raw / "lotka_volterra.csv"
    )

    # --- corporate_profits: quarterly ---
    pbt = profits * 1.0
    pat = profits * 0.7
    pd.DataFrame({"PROFITS_BEFORE_TAX": pbt, "PROFITS_AFTER_TAX": pat, "RECESSION": recession}).to_csv(
        raw / "corporate_profits.csv"
    )

    # --- global_growth: annual 1961 → 2020 ---
    aidx = pd.date_range("1961-12-31", "2020-12-31", freq="A")
    n_a = len(aidx)
    gdp_pc = pd.Series(5000 * np.exp(0.02 * np.arange(n_a)), index=aidx)
    pd.DataFrame({"WGDP_PC_LEVEL": gdp_pc}).to_csv(raw / "global_growth.csv")

    # --- capital_cycle: quarterly ---
    real_gdp = profits * 10
    real_inv = investment * 1.5
    inv_change = pd.Series(rng.normal(0, 50, n_q), index=qidx)
    pd.DataFrame(
        {
            "REAL_GDP": real_gdp,
            "REAL_INVESTMENT": real_inv,
            "REAL_INVENTORY_CHANGE": inv_change,
            "RECESSION": recession,
        }
    ).to_csv(raw / "capital_cycle.csv")

    return root


@pytest.fixture(scope="session")
def synthetic_config_path(synthetic_data_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A YAML config pointing save_paths to the synthetic raw directory."""
    cfg_dir = synthetic_data_root / "configs"
    cfg_dir.mkdir(parents=True)
    yaml_text = f"""\
datasets:
  corporate_profits:
    source: "fred"
    start_date: "1947-01-01"
    frequency: "QE"
    save_path: "{synthetic_data_root}/data/raw/corporate_profits.csv"
    indicators:
      PROFITS_BEFORE_TAX: "A446RC1Q027SBEA"
      PROFITS_AFTER_TAX: "A448RC1Q027SBEA"
      RECESSION: "USREC"
  global_growth:
    source: "fred"
    start_date: "1961-01-01"
    frequency: "A"
    save_path: "{synthetic_data_root}/data/raw/global_growth.csv"
    indicators:
      WGDP_PC_LEVEL: "NYGDPPCAPKDWLD"
  capital_cycle:
    source: "fred"
    start_date: "1947-01-01"
    frequency: "QE"
    save_path: "{synthetic_data_root}/data/raw/capital_cycle.csv"
    indicators:
      REAL_GDP: "GDPC1"
      REAL_INVESTMENT: "GPDIC1"
      REAL_INVENTORY_CHANGE: "CBIC1"
      RECESSION: "USREC"
  lotka_volterra:
    source: "fred"
    start_date: "1947-01-01"
    frequency: "QE"
    save_path: "{synthetic_data_root}/data/raw/lotka_volterra.csv"
    indicators:
      PROFITS: "A053RC1Q027SBEA"
      INVESTMENT: "GPDI"
      RECESSION: "USREC"
"""
    cfg = cfg_dir / "datasets.yaml"
    cfg.write_text(yaml_text)
    return cfg
