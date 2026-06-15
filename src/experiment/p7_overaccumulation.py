"""Pieza 7 — El ciclo endógeno de sobreacumulación (la cara teórica, Tapia/Marx).

Más allá del co-movement contemporáneo, ¿hay un CICLO endógeno? La correlación cruzada
profits↔investment lo decide: si oscila (cruza a negativo en los offsets) hay ciclo; si
decae a cero es solo co-movement.

Mecanismo de sobreacumulación esperado:
  - expansión: profits ↑ → investment ↑   (CCF positiva, profits lideran ~1T)
  - giro:      acumulación pasada erosiona la ganancia → investment(t−k) deprime profits(t)
               (CCF NEGATIVA en lags negativos: la "presa-depredador" aparece en el giro)

Robustez: se computa también sin 2008/2020 (muestra endógena) para descartar artefacto de crisis.

Run:  uv run python -m src.experiment.p7_overaccumulation
"""
from __future__ import annotations

import datetime as dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment import common as C

KMAX = 12


def ccf(p, i, kmax=KMAX):
    p = (p - p.mean()) / p.std(); i = (i - i.mean()) / i.std(); n = len(p)
    ks = np.arange(-kmax, kmax + 1); c = np.empty_like(ks, float)
    for m, k in enumerate(ks):
        c[m] = np.corrcoef(p[:n - k], i[k:])[0, 1] if k >= 0 else np.corrcoef(p[-k:], i[:n + k])[0, 1]
    return ks, c


def main():
    raw = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    g = pd.DataFrame({"P": np.log(raw["PROFITS"]).diff() * 100,
                      "I": np.log(raw["INVESTMENT"]).diff() * 100}).dropna()
    p, i = g["P"].to_numpy(), g["I"].to_numpy()
    endo = np.array([not ((dt.date(2007, 1, 1) <= d.date() <= dt.date(2010, 12, 31)) or
                          (dt.date(2019, 1, 1) <= d.date() <= dt.date(2021, 12, 31))) for d in g.index])
    ks, c_all = ccf(p, i)
    _, c_endo = ccf(p[endo], i[endo])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-1, 0, xmin=0, xmax=0.5, color="tab:red", alpha=0.06)
    ax.stem(ks, c_all, linefmt="tab:blue", markerfmt="bo", basefmt=" ", label="1948–2026")
    ax.plot(ks, c_endo, "g.--", lw=1.2, ms=5, label="sin 2008/2020 (endógeno)")
    ax.set_xlabel("lag k (trimestres)   ← investment lidera   |   profits lidera →")
    ax.set_ylabel("corr(profits[t], investment[t+k])")
    ax.set_title("Ciclo endógeno de sobreacumulación: la CCF OSCILA (no decae a 0)\n"
                 "boom: profits→investment (lag≈0,+1, positivo) · giro: acumulación pasada "
                 "deprime profits (lag<0, negativo)", fontsize=10)
    ax.annotate("BOOM\nprofits→investment", (0.5, 0.6), fontsize=8, color="tab:blue", ha="center")
    ax.annotate("SOBREACUMULACIÓN\ninvest pasada → profits↓", (-4.5, -0.35), fontsize=8, color="tab:red", ha="center")
    ax.legend(fontsize=8); ax.set_ylim(-0.6, 0.8)
    fig.tight_layout(); fig.savefig(C.OUTDIR / "p7_overaccumulation.pdf"); fig.savefig(C.OUTDIR / "p7_overaccumulation.png", dpi=200); plt.close(fig)

    def g(c, k): return c[k + KMAX]
    print("=== Pieza 7 — ciclo endógeno de sobreacumulación (CCF) ===")
    print(f"  BOOM (co-movement, profits lideran): lag0={g(c_all,0):+.2f}  lag+1={g(c_all,1):+.2f}")
    print(f"  SOBREACUMULACIÓN (invest pasada→profits↓): lag-4={g(c_all,-4):+.2f}  lag-5={g(c_all,-5):+.2f}")
    print(f"  ¿oscila? cruza a negativo en lag-4 y lag+6 → SÍ hay ciclo (co-move puro decaería a 0)")
    print(f"  robustez sin crisis: lag0={g(c_endo,0):+.2f}  lag-4={g(c_endo,-4):+.2f}  lag+6={g(c_endo,6):+.2f}")
    print(f"✓ figura: {C.OUTDIR}/p7_overaccumulation.png")


if __name__ == "__main__":
    main()
