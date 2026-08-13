"""End-to-end tests for the ETL: synthetic raw CSVs → transforms → invariants."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.etl.processor import (
    DataProcessor,
    transform_capital_cycle,
    transform_corporate_profits,
    transform_global_growth,
    transform_lotka_volterra,
)


# === Pure transform tests (no IO) ===

def _quarterly_profits_investment(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("1970-03-31", periods=n, freq="QE")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "PROFITS": 100 * np.exp(0.005 * np.arange(n)) + rng.normal(0, 1, n),
            "INVESTMENT": 80 * np.exp(0.004 * np.arange(n)) + rng.normal(0, 1, n),
            "RECESSION": (rng.random(n) < 0.1).astype(float),
        },
        index=idx,
    )


def test_lotka_volterra_transform_adds_yoy_and_zscore_columns() -> None:
    df = transform_lotka_volterra(_quarterly_profits_investment())

    for col in ("PROFITS_YOY", "INVEST_YOY", "P_z", "I_z"):
        assert col in df.columns, f"missing {col}"

    assert df[["PROFITS_YOY", "INVEST_YOY", "P_z", "I_z"]].notna().all().all()


def test_lotka_volterra_zscore_is_standardized() -> None:
    df = transform_lotka_volterra(_quarterly_profits_investment())
    assert abs(df["P_z"].mean()) < 1e-9
    assert abs(df["I_z"].mean()) < 1e-9
    assert df["P_z"].std() == pytest.approx(1.0, abs=1e-9)
    assert df["I_z"].std() == pytest.approx(1.0, abs=1e-9)


def test_lotka_volterra_drops_first_year_for_yoy() -> None:
    raw = _quarterly_profits_investment(n=200)
    out = transform_lotka_volterra(raw)
    # pct_change(4) loses 4 quarters at the head
    assert len(out) == len(raw) - 4


def test_global_growth_transform() -> None:
    idx = pd.date_range("1961-12-31", periods=60, freq="A")
    raw = pd.DataFrame({"WGDP_PC_LEVEL": np.linspace(5000, 12000, 60)}, index=idx)

    out = transform_global_growth(raw)

    assert "GROWTH" in out.columns
    assert "TREND_10Y" in out.columns
    assert out["GROWTH"].notna().all()
    # rising series → positive growth
    assert (out["GROWTH"] > 0).all()


def test_capital_cycle_adds_inv_ratio() -> None:
    idx = pd.date_range("1970-03-31", periods=40, freq="QE")
    raw = pd.DataFrame(
        {
            "REAL_GDP": np.linspace(1000, 2000, 40),
            "REAL_INVESTMENT": np.linspace(150, 320, 40),
            "REAL_INVENTORY_CHANGE": np.linspace(-10, 10, 40),
            "RECESSION": np.zeros(40),
        },
        index=idx,
    )
    out = transform_capital_cycle(raw)
    assert "INV_RATIO" in out.columns
    expected = raw["REAL_INVESTMENT"] / raw["REAL_GDP"] * 100
    pd.testing.assert_series_equal(out["INV_RATIO"], expected, check_names=False)


def test_corporate_profits_passthrough() -> None:
    idx = pd.date_range("1970-03-31", periods=10, freq="QE")
    raw = pd.DataFrame(
        {
            "PROFITS_BEFORE_TAX": np.arange(10, dtype=float),
            "PROFITS_AFTER_TAX": np.arange(10, dtype=float) * 0.7,
            "RECESSION": np.zeros(10),
        },
        index=idx,
    )
    out = transform_corporate_profits(raw)
    pd.testing.assert_frame_equal(out, raw)


# === Pipeline integration test (writes processed CSVs to a temp dir) ===

EXPECTED_PROCESSED_COLUMNS = {
    "lotka_volterra": {"PROFITS", "INVESTMENT", "RECESSION", "PROFITS_YOY", "INVEST_YOY", "P_z", "I_z"},
    "global_growth": {"WGDP_PC_LEVEL", "GROWTH", "TREND_10Y"},
    "capital_cycle": {"REAL_GDP", "REAL_INVESTMENT", "REAL_INVENTORY_CHANGE", "RECESSION", "INV_RATIO"},
    "corporate_profits": {"PROFITS_BEFORE_TAX", "PROFITS_AFTER_TAX", "RECESSION"},
}


def test_processor_produces_expected_processed_csvs(synthetic_config_path: Path) -> None:
    DataProcessor(synthetic_config_path).run()

    repo_root = synthetic_config_path.parents[1]
    for name, expected_cols in EXPECTED_PROCESSED_COLUMNS.items():
        out = repo_root / "data" / "processed" / f"{name}.csv"
        assert out.exists(), f"{name}: processed CSV not written"
        df = pd.read_csv(out, index_col=0, parse_dates=True)
        missing = expected_cols - set(df.columns)
        assert not missing, f"{name}: missing columns {missing}"
        assert len(df) > 0, f"{name}: empty dataframe"


def test_processed_lotka_volterra_has_no_nan_in_features(synthetic_config_path: Path) -> None:
    DataProcessor(synthetic_config_path).run()
    repo_root = synthetic_config_path.parents[1]
    df = pd.read_csv(repo_root / "data" / "processed" / "lotka_volterra.csv", index_col=0, parse_dates=True)
    for col in ("PROFITS_YOY", "INVEST_YOY", "P_z", "I_z"):
        assert df[col].notna().all(), f"{col} has NaN"


def test_processed_zscore_invariants(synthetic_config_path: Path) -> None:
    DataProcessor(synthetic_config_path).run()
    repo_root = synthetic_config_path.parents[1]
    df = pd.read_csv(repo_root / "data" / "processed" / "lotka_volterra.csv", index_col=0, parse_dates=True)
    assert abs(df["P_z"].mean()) < 1e-6
    assert abs(df["I_z"].mean()) < 1e-6
    assert df["P_z"].std() == pytest.approx(1.0, abs=1e-6)
    assert df["I_z"].std() == pytest.approx(1.0, abs=1e-6)


def test_processed_date_coverage(synthetic_config_path: Path) -> None:
    DataProcessor(synthetic_config_path).run()
    repo_root = synthetic_config_path.parents[1]
    df = pd.read_csv(repo_root / "data" / "processed" / "lotka_volterra.csv", index_col=0, parse_dates=True)
    # config starts 1947-01-01 but YoY drops the first 4 quarters → first row in 1948
    assert df.index.min().year == 1948
    assert df.index.max().year >= 2020
