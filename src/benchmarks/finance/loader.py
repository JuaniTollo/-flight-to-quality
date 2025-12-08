import pandas_datareader.data as web
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

def fetch_dataset(name: str, params: Dict[str, Any]) -> None:
    """
    Fetches economic time-series data from FRED (Federal Reserve Economic Data).
    
    Args:
        name: Name of the dataset (for logging).
        params: Dictionary containing 'indicators', 'start_date', 'frequency', and 'save_path'.
    """
    print(f"\n⬇️  [EXTRACT] Fetching dataset: '{name}'")
    indicators = params['indicators']
    
    try:
        # 1. Download RAW Data (Nominal)
        df = web.DataReader(list(indicators.values()), 'fred', params['start_date'], datetime.now())
        
        # 2. Rename Columns for Readability
        df.rename(columns={v: k for k, v in indicators.items()}, inplace=True)
        
        # 3. Basic Resampling (Mean aggregation + Forward Fill for missing days)
        df = df.resample(params['frequency']).mean().ffill()
        
        # 4. Save to Disk
        # Resolve path relative to where the script is run (or absolute if provided)
        raw_path = Path(params['save_path'])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(raw_path)
        print(f"✅ Saved RAW data to: {raw_path}")
        
    except Exception as e:
        print(f"❌ Error fetching '{name}': {e}")