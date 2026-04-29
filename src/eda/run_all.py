"""Convenience entrypoint: runs every EDA script in order.

Run:  uv run python -m src.eda.run_all
"""
from src.eda import (
    eda_00_tapia_figures,
    eda_01_series_temporales,
    eda_02_rezagos_causalidad,
    eda_03_espacio_fases,
)


def main() -> None:
    eda_00_tapia_figures.main()
    eda_01_series_temporales.main()
    eda_02_rezagos_causalidad.main()
    eda_03_espacio_fases.main()


if __name__ == "__main__":
    main()
