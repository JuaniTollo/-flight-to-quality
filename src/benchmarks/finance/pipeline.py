import yaml
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(BASE_DIR))

from src.benchmarks.finance import loader
from src.benchmarks.finance.processor import DataProcessor

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run():
    config_path = BASE_DIR / "configs/benchmarks/finance.yaml"

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return

    config = load_config(config_path)
    print(f"Loaded config: {config_path.name}")

    print("\n=== [1] DATA INGESTION ===")
    for name, params in config['datasets'].items():
        loader.fetch_dataset(name, params)

    print("\n=== [2] DATA PROCESSING ===")
    processor = DataProcessor(config_path)
    processor.run()

    print("\nETL complete. Data available in data/processed/")

if __name__ == "__main__":
    run()
