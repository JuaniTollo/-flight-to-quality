from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
# paper.yaml: datos de las estimaciones (src/experiment); eda.yaml: figuras auxiliares (src/eda).
CONFIG_PATHS = {
    "paper": CONFIG_DIR / "paper.yaml",
    "eda": CONFIG_DIR / "eda.yaml",
}
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"


def resolve(p: str | Path) -> Path:
    """Resolve a path against REPO_ROOT if it's not absolute."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
