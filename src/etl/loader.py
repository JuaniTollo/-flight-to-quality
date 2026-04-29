from datetime import datetime
from typing import Any, Dict

import pandas_datareader.data as web

from src.etl.paths import resolve


def fetch_dataset(name: str, params: Dict[str, Any], force: bool = False) -> None:
    """
    Fetch an economic time-series from FRED and write it to ``params['save_path']``.

    Idempotent unless ``force=True``: if the target file already exists, the
    download is skipped.
    """
    raw_path = resolve(params["save_path"])

    if raw_path.exists() and not force:
        print(f"[SKIP] {name}: {raw_path.relative_to(raw_path.parents[2])} already present")
        return

    print(f"[FETCH] {name}")
    indicators = params["indicators"]

    df = web.DataReader(
        list(indicators.values()), "fred", params["start_date"], datetime.now()
    )
    df.rename(columns={v: k for k, v in indicators.items()}, inplace=True)
    df = df.resample(params["frequency"]).mean().ffill()

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path)
    print(f"[SAVE]  {raw_path.relative_to(raw_path.parents[2])}  ({len(df)} rows)")
