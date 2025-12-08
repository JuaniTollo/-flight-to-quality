import argparse
import yaml
import sys
from pathlib import Path

# --- PATH SETUP ---
# Ensure we can import from src root
BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(BASE_DIR))

# --- MODULE IMPORTS ---
from src.benchmarks.finance import loader
from src.benchmarks.finance.processor import DataProcessor
from src.benchmarks.finance.visualization import main as visualize_dataset

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_pipeline():
    parser = argparse.ArgumentParser(description="NeurIPS Finance Benchmark Pipeline")
    parser.add_argument("--step", choices=["all", "load", "process", "viz"], default="all")
    args = parser.parse_args()

    # Centralized Config Path
    config_path = BASE_DIR / "configs/benchmarks/finance.yaml"
    
    if not config_path.exists():
        print(f"❌ Configuration file not found at: {config_path}")
        return

    config = load_config(config_path)
    print(f"📜 Loaded configuration: {config_path.name}")

    # --- STEP 1: DATA INGESTION (Loader) ---
    if args.step in ["all", "load"]:
        print("\n=== [STEP 1] DATA INGESTION ===")
        for name, params in config['datasets'].items():
            loader.fetch_dataset(name, params)

    # --- STEP 2: DATA PROCESSING (Processor) ---
    if args.step in ["all", "process"]:
        print("\n=== [STEP 2] DATA PROCESSING ===")
        # Declarative instantiation
        processor = DataProcessor(config_path)
        processor.run()

    # --- STEP 3: VISUALIZATION (Analysis) ---
    if args.step in ["all", "viz"]:
        print("\n=== [STEP 3] VISUALIZATION ===")
        for ds_name in config['datasets'].keys():
            print(f">> Plotting {ds_name}...")
            try:
                visualize_dataset(ds_name)
            except Exception as e:
                print(f"⚠️  Could not plot {ds_name}. Check if processed data exists. Error: {e}")

if __name__ == "__main__":
    run_pipeline()