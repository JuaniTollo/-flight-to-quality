from pathlib import Path

import pandas as pd
import yaml

from src.etl.paths import resolve


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def transform_lotka_volterra(df: pd.DataFrame) -> pd.DataFrame:
    """Quarterly profits/investment → YoY % change + z-score features."""
    df = df.copy()
    df["PROFITS_YOY"] = df["PROFITS"].pct_change(4) * 100
    df["INVEST_YOY"] = df["INVESTMENT"].pct_change(4) * 100
    df = df.dropna(subset=["PROFITS_YOY", "INVEST_YOY"])
    df["P_z"] = _zscore(df["PROFITS_YOY"])
    df["I_z"] = _zscore(df["INVEST_YOY"])
    return df


def transform_global_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Annual GDP/capita level → growth % + 10-year structural trend."""
    df = df.copy()
    df["GROWTH"] = df["WGDP_PC_LEVEL"].pct_change() * 100
    df = df.dropna(subset=["GROWTH"])
    df["TREND_10Y"] = df["GROWTH"].rolling(window=10, center=True).mean()
    return df


def transform_capital_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Real GDP / Investment / Inventory change → investment-to-GDP ratio."""
    df = df.copy()
    if {"REAL_INVESTMENT", "REAL_GDP"}.issubset(df.columns):
        df["INV_RATIO"] = (df["REAL_INVESTMENT"] / df["REAL_GDP"]) * 100
    return df


def transform_corporate_profits(df: pd.DataFrame) -> pd.DataFrame:
    """Pass-through for now — Tapia Fig 2.12 uses raw levels on a log scale."""
    return df.copy()


TRANSFORMS = {
    "lotka_volterra": transform_lotka_volterra,
    "global_growth": transform_global_growth,
    "capital_cycle": transform_capital_cycle,
    "corporate_profits": transform_corporate_profits,
}


class DataProcessor:
    """
    Reads raw FRED CSVs, applies dataset-specific transforms, writes processed CSVs.

    Mapping of save_path raw → processed swaps the leaf directory name `raw`
    with `processed`. Transforms are pure, deterministic, and tested.
    """

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def load_raw(self, dataset_name: str) -> pd.DataFrame:
        raw_path = resolve(self.config["datasets"][dataset_name]["save_path"])
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file missing: {raw_path}. Run loader first.")
        return pd.read_csv(raw_path, index_col=0, parse_dates=True)

    def processed_path(self, dataset_name: str) -> Path:
        raw_path = resolve(self.config["datasets"][dataset_name]["save_path"])
        # data/raw/foo.csv → data/processed/foo.csv
        parts = list(raw_path.parts)
        parts[parts.index("raw")] = "processed"
        return Path(*parts)

    def run(self) -> None:
        print("[PROCESS] Starting transforms")
        for name in self.config["datasets"]:
            transform = TRANSFORMS.get(name)
            if transform is None:
                print(f"[WARN] No transform registered for '{name}', skipping")
                continue
            df_raw = self.load_raw(name)
            df = transform(df_raw)
            out = self.processed_path(name)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out)
            print(f"[SAVE]  {out.relative_to(out.parents[2])}  ({len(df)} rows, {len(df.columns)} cols)")
