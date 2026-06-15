"""Pieza 15 — Gráfico para público general: ganancias e inversión a lo largo del tiempo,
con las recesiones de EE.UU. sombreadas.

Objetivo divulgativo: que se VEA, sin tecnicismos, el ciclo ganancias–inversión —ambas
crecen juntas en los auges y se desploman en cada recesión— y el patrón de sobreacumulación
(la inversión sigue alta cuando la ganancia ya gira hacia abajo, antes de la recesión).

Series: crecimiento log trimestre a trimestre (consistente con el resto del trabajo),
suavizado a 1 año (media móvil 4 trim) y z-scoreado para que sean comparables.

Run:  uv run python -m src.experiment.p15_crisis_series
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment import common as C


def main():
    raw = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    g = pd.DataFrame({
        "Ganancias": np.log(raw["PROFITS"]).diff() * 100,
        "Inversión": np.log(raw["INVESTMENT"]).diff() * 100,
    }).dropna()
    # suavizado a 1 año y z-score para comparar en la misma escala
    s = g.rolling(4, center=True, min_periods=1).mean()
    z = (s - s.mean()) / s.std()

    fig, ax = plt.subplots(figsize=(11, 4.8))
    # recesiones NBER sombreadas (pico → valle)
    for lbl, pk, tr in C.RECESSIONS:
        ax.axvspan(pd.Timestamp(pk), pd.Timestamp(tr), color="0.80", alpha=0.6, lw=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.plot(z.index, z["Ganancias"], color="tab:blue", lw=1.7, label="Ganancias (crecimiento)")
    ax.plot(z.index, z["Inversión"], color="tab:orange", lw=1.7, label="Inversión (crecimiento)")

    ax.set_title("Ganancias e inversión en EE.UU. (1948–2026): suben juntas en los auges "
                 "y caen en cada recesión\n(crecimiento suavizado a 1 año; franjas grises = "
                 "recesiones; la inversión sigue alta cuando la ganancia ya gira a la baja)",
                 fontsize=10)
    ax.set_ylabel("crecimiento (desvíos estándar)")
    ax.set_xlabel("año")
    ax.legend(loc="lower left", fontsize=9, ncol=2, framealpha=0.9)
    ax.margins(x=0.01)
    fig.tight_layout(); fig.savefig(C.OUTDIR / "p15_crisis_series.pdf"); fig.savefig(C.OUTDIR / "p15_crisis_series.png", dpi=200); plt.close(fig)
    print(f"✓ figura: {C.OUTDIR}/p15_crisis_series.png")
    # correlación de co-movimiento (informativa para el pie)
    r = z["Ganancias"].corr(z["Inversión"])
    print(f"correlación contemporánea (suavizada) ganancias↔inversión = {r:+.2f}")


if __name__ == "__main__":
    main()
