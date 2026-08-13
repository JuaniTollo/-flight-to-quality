from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "datasets.yaml"
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"


def resolve(p: str | Path) -> Path:
    """Resolve a path against REPO_ROOT if it's not absolute."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
