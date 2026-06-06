"""Pieza 2 — La frontera de amplitud (el hallazgo central).

Para cada recesión NBER, ¿la excursión de la crisis cae DENTRO del rango de amplitud
que abarcaron los ciclos previos, o lo excede (OUTLIER)? Walk-forward y anti-leakage:
la referencia de cada crisis usa SOLO datos anteriores a ella.

Métrica (por serie): desviación máxima respecto de la media previa, dentro de la ventana
de crisis, dividida por la desviación máxima alcanzada en toda la historia previa.
  ratio > 1  ⇒  la crisis llegó más lejos que nada visto antes = fuera de régimen.

Lectura económica (Astarita/Tapia): NO es "shock exógeno". Es la misma dinámica de
acumulación en una fase descendente de amplitud que el régimen ordinario no alcanza —
el modelo de amplitud fija no la abarca.

Run:  uv run python -m src.experiment.p2_amplitude_boundary
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment import common as C

OUTLIER_RATIO = 1.0   # ratio > 1 ⇒ excede todo lo previo


def analyze():
    df = C.load()
    rows = []
    for lbl, peak, trough in C.RECESSIONS:
        peak, trough = pd.Timestamp(peak), pd.Timestamp(trough)
        win = (df.index >= peak - pd.DateOffset(months=6)) & (df.index <= trough + pd.DateOffset(months=12))
        prior = df.index < peak - pd.DateOffset(months=6)
        if prior.sum() < 12:
            continue                                   # historia previa insuficiente
        rec = {"crisis": lbl}
        for c, nm in zip(C.COLS, ["prof", "inv"]):
            pm = df.loc[prior, c].mean()
            prior_ext = (df.loc[prior, c] - pm).abs().max()
            win_ext = (df.loc[win, c] - pm).abs().max()
            rec[f"{nm}_ratio"] = win_ext / prior_ext
            rec[f"{nm}_min"] = df.loc[win, c].min()
            rec[f"{nm}_max"] = df.loc[win, c].max()
            rec[f"{nm}_priorlo"] = df.loc[prior, c].min()
            rec[f"{nm}_priorhi"] = df.loc[prior, c].max()
        rec["outlier"] = (rec["prof_ratio"] > OUTLIER_RATIO) or (rec["inv_ratio"] > OUTLIER_RATIO)
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    res = analyze()
    res.to_csv(C.OUTDIR / "p2_amplitude_boundary.csv", index=False)

    print("=== Pieza 2 — frontera de amplitud (ratio = excursión crisis / máx previo; >1 ⇒ outlier) ===")
    print(f"{'crisis':8s} {'prof_ratio':>10s} {'inv_ratio':>10s}  {'régimen':>8s}")
    for _, r in res.iterrows():
        tag = "OUTLIER" if r["outlier"] else "dentro"
        print(f"{r['crisis']:8s} {r['prof_ratio']:10.2f} {r['inv_ratio']:10.2f}  {tag:>8s}")

    # ---- figura: ratio por crisis y serie ----
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(res)); w = 0.38
    cp = ["tab:red" if v > OUTLIER_RATIO else "tab:blue" for v in res["prof_ratio"]]
    ci = ["tab:red" if v > OUTLIER_RATIO else "tab:green" for v in res["inv_ratio"]]
    ax.bar(x - w / 2, res["prof_ratio"], w, color=cp, label="Profits")
    ax.bar(x + w / 2, res["inv_ratio"], w, color=ci, alpha=0.6, label="Investment")
    ax.axhline(OUTLIER_RATIO, color="black", ls="--", lw=1, label="frontera (=1): excede todo lo previo")
    ax.set_xticks(x); ax.set_xticklabels(res["crisis"], rotation=45)
    ax.set_ylabel("ratio de excursión (crisis / máx. histórico previo)")
    ax.set_title("Frontera de amplitud: 2008 y 2020 exceden el rango de los ciclos previos\n"
                 "(rojo = fuera de régimen)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.OUTDIR / "p2_amplitude_boundary.png", dpi=130); plt.close(fig)
    print(f"✓ tabla y figura: {C.OUTDIR}/p2_amplitude_boundary.csv , p2_amplitude_boundary.png")


if __name__ == "__main__":
    main()
