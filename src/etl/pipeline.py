"""Run the full ETL: download FRED data, then apply transforms.

Usage:  uv run python -m src.etl.pipeline
        uv run python -m src.etl.pipeline --force   # re-download even if cached
"""
import argparse

import yaml

from src.etl import loader
from src.etl.paths import CONFIG_PATH
from src.etl.processor import DataProcessor


def run(force: bool = False) -> None:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    print(f"[ETL] config: {CONFIG_PATH.name}")

    print("\n=== [1/2] DOWNLOAD ===")
    for name, params in config["datasets"].items():
        loader.fetch_dataset(name, params, force=force)

    print("\n=== [2/2] PROCESS ===")
    DataProcessor(CONFIG_PATH).run()

    print("\n[ETL] complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()
    run(force=args.force)
