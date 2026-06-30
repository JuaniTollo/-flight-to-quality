"""Pieza 15 — Gráfico para público general: ganancias e inversión a lo largo del tiempo,
con las recesiones de EE.UU. sombreadas.

Objetivo divulgativo: que se VEA, sin tecnicismos, el ciclo ganancias–inversión —ambas
crecen juntas en los auges y se desploman en cada recesión— y el patrón de sobreacumulación
(la inversión sigue alta cuando la ganancia ya gira hacia abajo, antes de la recesión).

Series: crecimiento log trimestre a trimestre (la MISMA que usa el análisis, SIN suavizar) y
z-scoreado para que sean comparables. El eje temporal se parte en dos filas (1948–1987 /
1987–2026) para ganar resolución horizontal en una serie larga y ruidosa, y el eje y se recorta
al grueso de los datos anotando el outlier de la GFC (ganancias −7,5σ en 2008 Q4), que de otro
modo aplastaría la escala de todo el resto.

Run:  uv run python -m src.experiment.p15_crisis_series
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment import common as C

SPLIT = pd.Timestamp("1987-01-01")   # corte del eje temporal entre las dos filas
YLIM = (-4.2, 4.7)                    # recorte del eje y: cubre el grueso; deja afuera solo 2008


def main():
    raw = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    g = pd.DataFrame({
        "Ganancias": np.log(raw["PROFITS"]).diff() * 100,
        "Inversión": np.log(raw["INVESTMENT"]).diff() * 100,
    }).dropna()
    # crecimiento log-trim SIN suavizar, solo z-score para comparar en la misma escala
    z = (g - g.mean()) / g.std()

    # outlier de ganancias (GFC) que se recorta y se anota
    omin_date = z["Ganancias"].idxmin()
    omin_val = z["Ganancias"].min()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.0))
    rows = [(ax1, z.index.min(), SPLIT), (ax2, SPLIT, z.index.max())]
    for ax, lo, hi in rows:
        for lbl, pk, tr in C.RECESSIONS:
            ax.axvspan(pd.Timestamp(pk), pd.Timestamp(tr), color="0.80", alpha=0.6, lw=0)
        ax.axhline(0, color="k", lw=0.6)
        ax.plot(z.index, z["Ganancias"], color="tab:blue", lw=1.3, label="Ganancias (crecimiento)")
        ax.plot(z.index, z["Inversión"], color="tab:orange", lw=1.3, label="Inversión (crecimiento)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(*YLIM)
        ax.set_ylabel("crecimiento\n(desvíos estándar)")

    # anotación del outlier recortado (cae en la fila de abajo)
    ax2.annotate(f"Ganancias {omin_val:.1f}σ\n(2008 Q4, GFC)",
                 xy=(omin_date, YLIM[0]), xytext=(omin_date, YLIM[0] + 1.5),
                 ha="center", va="bottom", fontsize=8, color="tab:blue",
                 arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=1.2))

    ax1.set_title("Ganancias e inversión en EE.UU. (1948–2026): suben juntas en los auges y caen "
                  "en cada recesión\n(crecimiento log-trimestral sin suavizar, en desvíos estándar; "
                  "eje temporal partido; franjas grises = recesiones)", fontsize=10)
    ax2.set_xlabel("año")
    ax1.legend(loc="lower left", fontsize=9, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(C.OUTDIR / "p15_crisis_series.pdf")
    fig.savefig(C.OUTDIR / "p15_crisis_series.png", dpi=200)
    plt.close(fig)
    print(f"✓ figura: {C.OUTDIR}/p15_crisis_series.png")

    # correlaciones informativas para el pie
    r0 = z["Ganancias"].corr(z["Inversión"])
    r_lead = z["Ganancias"].corr(z["Inversión"].shift(-1))   # inversión 1 trim DESPUÉS
    r_lag = z["Ganancias"].corr(z["Inversión"].shift(1))     # inversión 1 trim ANTES
    print(f"corr contemporánea (log-trim crudo) = {r0:+.2f}")
    print(f"corr con inversión +1 trim = {r_lead:+.2f}   |   −1 trim = {r_lag:+.2f}")


if __name__ == "__main__":
    main()
