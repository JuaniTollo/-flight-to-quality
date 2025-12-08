import pandas as pd
import yaml
from pathlib import Path
from typing import Optional

class DataProcessor:
    """
    Preprocessing pipeline for Economic Benchmarks.
    Transforms Nominal Raw Data -> Real Structural Features.
    Calculates Drawdown metrics for Crisis Classification (Ground Truth).
    """
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        if not self.config_path.exists():
             raise FileNotFoundError(f"Config not found at: {self.config_path}")
             
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def load_raw_data(self, dataset_name: str) -> pd.DataFrame:
        if dataset_name not in self.config['datasets']:
            raise ValueError(f"Dataset '{dataset_name}' not found in config.")
            
        raw_path = Path(self.config['datasets'][dataset_name]['save_path'])
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file not found: {raw_path}. Run loader step first.")
            
        print(f"   [IO] Loading raw data: {raw_path}")
        return pd.read_csv(raw_path, index_col=0, parse_dates=True)

    def deflate_nominal_series(self, df: pd.DataFrame, cpi_col: str = 'CPI_ADJUSTMENT') -> pd.DataFrame:
        """Adjusts nominal prices to real values using CPI."""
        if cpi_col not in df.columns:
            print("   [WARN] CPI column not found. Skipping inflation adjustment.")
            return df
            
        cpi_base = df[cpi_col].iloc[-1]
        print(f"   [MATH] Deflating series to current USD (Base CPI: {cpi_base:.2f})")
        
        exclude_cols = [cpi_col, 'RISK_FREE_RATE', 'CAPACITY_UTIL', 'RECESSION']
        
        df_real = df.copy()
        for col in df.columns:
            if col not in exclude_cols and not col.endswith('_REAL'):
                new_col_name = f"{col}_REAL"
                df_real[new_col_name] = df[col] * (cpi_base / df[cpi_col])
        
        return df_real

    def calculate_drawdown(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Calculates percentage decline from historical peak (Structural Drawdown)."""
        if target_col not in df.columns:
            return df
        
        rolling_peak = df[target_col].cummax()
        df['DRAWDOWN'] = (df[target_col] - rolling_peak) / rolling_peak
        
        print(f"   [MATH] Calculated Structural Drawdown for {target_col}")
        return df

    def run(self):
        print("Starting Data Processing Pipeline...")
        
        for name, params in self.config['datasets'].items():
            print(f"\nProcessing Dataset: {name}")
            
            # 1. Load
            df = self.load_raw_data(name)
            
            # 2. Deflate
            df = self.deflate_nominal_series(df, cpi_col='CPI_ADJUSTMENT')
            
            # 3. Calculate Crisis Metrics
            target = 'MARKET_VALUATION_REAL' if 'MARKET_VALUATION_REAL' in df.columns else None
            if target:
                df = self.calculate_drawdown(df, target_col=target)
            
            # 4. Save
            processed_path = Path(params['save_path'].replace('raw', 'processed'))
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(processed_path)
            print(f"   [SUCCESS] Saved processed data to: {processed_path}")