"""Path helper invariants — make sure repo-rooted resolution is stable."""
from pathlib import Path

from src.etl.paths import REPO_ROOT, resolve


def test_repo_root_contains_pyproject() -> None:
    assert (REPO_ROOT / "pyproject.toml").exists()


def test_resolve_relative_path_is_anchored_to_repo_root() -> None:
    out = resolve("data/raw/foo.csv")
    assert out == REPO_ROOT / "data" / "raw" / "foo.csv"


def test_resolve_absolute_path_passes_through(tmp_path: Path) -> None:
    p = tmp_path / "abs.csv"
    assert resolve(p) == p
