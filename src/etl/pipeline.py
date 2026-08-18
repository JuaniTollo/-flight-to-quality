"""Run the full ETL: download FRED data, then apply transforms.

Usage:  uv run python -m src.etl.pipeline                 # paper + eda configs
        uv run python -m src.etl.pipeline --config paper  # only what the paper needs
        uv run python -m src.etl.pipeline --force         # re-download even if cached
"""
import argparse
from pathlib import Path

import yaml

from src.etl import loader
from src.etl.paths import CONFIG_PATHS
from src.etl.processor import DataProcessor


def run_config(config_path: Path, force: bool = False) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"\n[ETL] config: {config_path.name}")

    print("=== [1/2] DOWNLOAD ===")
    for name, params in config["datasets"].items():
        loader.fetch_dataset(name, params, force=force)

    print("=== [2/2] PROCESS ===")
    DataProcessor(config_path).run()


def run(which: str = "all", force: bool = False) -> None:
    names = list(CONFIG_PATHS) if which == "all" else [which]
    for name in names:
        run_config(CONFIG_PATHS[name], force=force)
    print("\n[ETL] complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["all", *CONFIG_PATHS], default="all",
                        help="Which config to run (default: all)")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()
    run(which=args.config, force=args.force)
