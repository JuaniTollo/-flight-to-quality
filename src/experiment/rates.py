"""Preparación de datos compartida: tasas de variación log QoQ y fechado NBER.

Los análisis descriptivos (CCF, estudio de evento, series por crisis) deben partir
de QuarterlyRates para garantizar que todos usan exactamente la misma
transformación: 100·Δln trimestre a trimestre sobre las series en niveles.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.experiment import common as C

COL_PROFITS = "PROFITS_QOQ"
COL_INVEST = "INVEST_QOQ"


@dataclass(frozen=True)
class QuarterlyRates:
    """Tasas log QoQ (100·Δln) de ganancias e inversión, indexadas por trimestre."""

    df: pd.DataFrame

    @classmethod
    def load(cls, path: Path = C.DATA) -> "QuarterlyRates":
        raw = pd.read_csv(path, parse_dates=["DATE"]).set_index("DATE")
        df = pd.DataFrame({
            COL_PROFITS: np.log(raw["PROFITS"]).diff() * 100.0,
            COL_INVEST: np.log(raw["INVESTMENT"]).diff() * 100.0,
        }).dropna()
        return cls(df)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.df.index

    @property
    def p(self) -> np.ndarray:
        return self.df[COL_PROFITS].to_numpy()

    @property
    def i(self) -> np.ndarray:
        return self.df[COL_INVEST].to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def crisis_anchors(self, pick: Callable[[np.ndarray], int] | None = None,
                       ) -> list[tuple[str, int, pd.Timestamp]]:
        """Un ancla por recesión NBER, buscada en la ventana pico−2 … valle+4 trim.

        pick recibe las posiciones de la ventana y devuelve la posición del ancla;
        por defecto, el trimestre de inversión más deprimida.
        """
        idx, i = self.index, self.i
        if pick is None:
            pick = lambda pos: pos[np.argmin(i[pos])]
        out = []
        for lbl, pk, tr in C.RECESSIONS:
            pk, tr = pd.Timestamp(pk), pd.Timestamp(tr)
            win = (idx >= pk - pd.offsets.QuarterEnd(2)) & (idx <= tr + pd.offsets.QuarterEnd(4))
            pos = np.where(win)[0]
            if len(pos) == 0:
                continue
            a = pick(pos)
            out.append((lbl, a, idx[a]))
        return out
