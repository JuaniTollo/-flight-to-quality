"""Pieza 9 — Test de placebo / identificabilidad de la ventana de crisis.

TEST DE FALSIFICACIÓN. La hipótesis bajo test es: "el oscilador es un modelo de RÉGIMEN
DE CRISIS y ajusta bien las vecindades de crisis (±6 trim del fondo)". Pero una vecindad de crisis tiene ~13 trimestres = aproximadamente UNA sola
oscilación (un bajón + una recuperación). Un oscilador no lineal de 2–4 parámetros puede
ajustar UNA sola oscilación casi siempre. Entonces "LV/FN ajusta bien las crisis" podría
ser VACÍO: ajustaría igual de bien CUALQUIER ventana de 13 trim de la serie.

DISEÑO.
  1. Ventanas de crisis: mismo anclaje que p8 (trimestre de inversión mínima en la
     vecindad NBER) ± 6 trim = 13 trim.
  2. Ventanas PLACEBO: barrido SISTEMÁTICO de TODAS las posiciones posibles de ventanas
     de 13 trim sobre la serie completa, excluyendo las que solapan cualquier ventana de
     crisis. (Determinista, sin RNG. Se reporta también un sub-muestreo sembrado para el
     histograma si hiciera falta.)
  3. Ajuste FN y LV con common.fit (scipy single-shooting) a TODAS las ventanas.
     Métricas de bondad de ajuste por ventana:
        - nRMSE: RMSE de la trayectoria z-scoreada (RMSE / desvío de la ventana). Menor=mejor.
        - R² de fase (gradient matching, à la p4): fracción de la velocidad du/dt
          explicada por el campo f(u,p). Mayor=mejor.
  4. TEST: ¿caen las crisis en la COLA buena de la distribución placebo? Percentil de
     cada crisis y p-valor (mediana de crisis vs mediana placebo; Mann–Whitney one-sided).
  5. Figura: histograma de la métrica placebo con las crisis marcadas.

Run:  uv run python -m src.experiment.p9_placebo
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.experiment import common as C

H = 6                       # ±6 trimestres -> ventana de 2H+1 = 13 trimestres
W = 2 * H + 1


# --------------------------------------------------------------------------- anclas
def crisis_anchors(df, p, i):
    """Ancla de cada recesion = trimestre de inversion minima en su vecindad NBER (= p8)."""
    idx = df.index
    out = []
    for lbl, pk, tr in C.RECESSIONS:
        pk, tr = pd.Timestamp(pk), pd.Timestamp(tr)
        win = (idx >= pk - pd.offsets.QuarterEnd(2)) & (idx <= tr + pd.offsets.QuarterEnd(4))
        pos = np.where(win)[0]
        if len(pos) == 0:
            continue
        a = pos[np.argmin(i[pos])]
        out.append((lbl, int(a)))
    return out


def windows_from_anchors(anchors, n):
    """Ventanas de crisis [a-H, a+H] que caben enteras en la serie."""
    return [(lbl, a) for lbl, a in anchors if a - H >= 0 and a + H < n]


def placebo_centers(crisis_anchor_pos, n, sep):
    """Barrido sistematico: TODA posicion central c con ventana entera tal que el fondo
    de NINGUNA crisis quede a <= `sep` trim del centro placebo. Dos modos:
      sep=2H -> NO solapa ninguna ventana de crisis (estricto, pocas ventanas).
      sep=H  -> el fondo de la crisis no cae DENTRO de la ventana placebo (laxo, ~muchas).
    Ambos son definiciones legitimas de 'ventana fuera de crisis'."""
    return [c for c in range(H, n - H)
            if all(abs(c - a) > sep for a in crisis_anchor_pos)]


# --------------------------------------------------------------------------- metricas
def derivatives(t, Z):
    """du/dt por diferencias finitas con suavizado leve (igual que p4)."""
    Zs = pd.DataFrame(Z).rolling(3, center=True, min_periods=1).mean().to_numpy()
    return np.gradient(Zs, t, axis=0)


def fit_window(model, Z):
    """Ajusta el oscilador a una ventana z-scoreada (2,W -> trayectoria) via single-shooting.
    Devuelve (nRMSE, theta). nRMSE = sqrt(SSE / (2W)) / std(Z)  (RMSE normalizado)."""
    V, R = Z[:, 0], Z[:, 1]
    t = np.arange(len(V), dtype=float)            # paso unitario; escala absorbida en theta
    theta, sse = C.fit(model, t, V, R)
    rmse = np.sqrt(sse / Z.size)
    return rmse / Z.std(), theta


def field_r2_overall(model, Z, theta, t):
    """R^2 del campo de fase agregando ambas ecuaciones (1 - SSres/SStot conjunto)."""
    dU = derivatives(t, Z)
    F = np.array([model.rhs(u, theta) for u in Z])
    ss_res = np.sum((dU - F) ** 2)
    ss_tot = np.sum((dU - dU.mean(0)) ** 2)
    return 1 - ss_res / ss_tot


def zwindow(df, a):
    """Ventana [a-H, a+H] z-scoreada con SUS PROPIOS momentos (offset + si el modelo
    lo necesita se aplica afuera). Devuelve raw window (W,2)."""
    raw = df[C.COLS].to_numpy()[a - H:a + H + 1]
    return raw


def zscore_local(raw, needs_offset):
    mu, sd = raw.mean(0), raw.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    z = (raw - mu) / sd
    if needs_offset:
        z = z - z.min() + 1.0                     # cuadrante + para LV
    return z


# --------------------------------------------------------------------------- evaluacion
def eval_all(df, model, sep):
    p = df["PROFITS_YOY"].to_numpy()
    i = df["INVEST_YOY"].to_numpy()
    n = len(df)
    anchors = crisis_anchors(df, p, i)
    crisis = windows_from_anchors(anchors, n)
    anchor_pos = [a for _, a in anchors]
    centers = placebo_centers(anchor_pos, n, sep)

    t = np.arange(W, dtype=float)

    def metrics(a):
        z = zscore_local(zwindow(df, a), model.needs_offset)
        nrmse, theta = fit_window(model, z)
        r2 = field_r2_overall(model, z, theta, t)
        return nrmse, r2

    crisis_m = {lbl: metrics(a) for lbl, a in crisis}
    placebo_m = np.array([metrics(c) for c in centers])      # (Nplac, 2): [nrmse, r2]
    return crisis, crisis_m, centers, placebo_m


def report(name, crisis, crisis_m, placebo_m):
    plac_rmse = placebo_m[:, 0]
    plac_r2 = placebo_m[:, 1]
    Np = len(plac_rmse)

    print(f"\n================= {name} =================")
    print(f"ventanas de crisis: {len(crisis)} | ventanas placebo (barrido): {Np}")
    print(f"PLACEBO nRMSE  : mediana={np.median(plac_rmse):.3f}  "
          f"[p10={np.percentile(plac_rmse,10):.3f}, p90={np.percentile(plac_rmse,90):.3f}]")
    print(f"PLACEBO R2 fase: mediana={np.median(plac_r2):.3f}  "
          f"[p10={np.percentile(plac_r2,10):.3f}, p90={np.percentile(plac_r2,90):.3f}]")

    # percentil de cada crisis dentro del placebo.
    # nRMSE: MENOR es mejor -> percentil = % de placebos PEORES (mayor rmse). cola buena = alto.
    # R2  : MAYOR es mejor -> percentil = % de placebos peores (menor r2). cola buena = alto.
    print("\n  crisis |  nRMSE  pctil(buena cola)  |   R2fase  pctil(buena cola)")
    cr_rmse, cr_r2 = [], []
    for lbl, _ in crisis:
        nr, r2 = crisis_m[lbl]
        cr_rmse.append(nr); cr_r2.append(r2)
        p_rmse = 100.0 * np.mean(plac_rmse > nr)     # % placebos con ajuste PEOR
        p_r2 = 100.0 * np.mean(plac_r2 < r2)         # % placebos con R2 PEOR
        print(f"  {lbl:6s} | {nr:6.3f}   {p_rmse:5.1f}%          | {r2:+7.3f}   {p_r2:5.1f}%")
    cr_rmse, cr_r2 = np.array(cr_rmse), np.array(cr_r2)

    # p-valores one-sided: crisis ajustan MEJOR (rmse menor / r2 mayor) que placebo?
    u1, p1 = mannwhitneyu(cr_rmse, plac_rmse, alternative="less")
    u2, p2 = mannwhitneyu(cr_r2, plac_r2, alternative="greater")
    print(f"\n  Mann-Whitney one-sided:")
    print(f"    nRMSE crisis < placebo : p = {p1:.3f}   "
          f"(mediana crisis {np.median(cr_rmse):.3f} vs placebo {np.median(plac_rmse):.3f})")
    print(f"    R2fase crisis > placebo: p = {p2:.3f}   "
          f"(mediana crisis {np.median(cr_r2):+.3f} vs placebo {np.median(plac_r2):+.3f})")
    return dict(cr_rmse=cr_rmse, cr_r2=cr_r2, plac_rmse=plac_rmse, plac_r2=plac_r2,
                labels=[l for l, _ in crisis], p_rmse=p1, p_r2=p2)


def figure(res_by_model):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, (name, r) in enumerate(res_by_model.items()):
        # fila 0: nRMSE (menor=mejor, cola buena a la izquierda)
        ax = axes[0, col]
        ax.hist(r["plac_rmse"], bins=30, color="tab:gray", alpha=0.6, label="placebo (13 trim random)")
        for lbl, x in zip(r["labels"], r["cr_rmse"]):
            ax.axvline(x, color="tab:red", lw=1.2, alpha=0.8)
        ax.axvline(np.median(r["plac_rmse"]), color="black", ls="--", lw=1, label="mediana placebo")
        ax.set_title(f"{name} — nRMSE de ajuste  (menor = mejor)\n"
                     f"crisis (rojo) vs placebo · MW p={r['p_rmse']:.3f}", fontsize=10)
        ax.set_xlabel("nRMSE (RMSE z / std ventana)"); ax.set_ylabel("# ventanas placebo")
        if col == 0:
            ax.legend(fontsize=7)
        # fila 1: R2 de fase (mayor=mejor). Clip a [p5,p95] placebo: la cola de blowups
        # (R2 ~ -1e6..-1e13 en ventanas degeneradas) hace ilegible el eje y no aporta.
        ax = axes[1, col]
        lo, hi = np.percentile(r["plac_r2"], [5, 95])
        clip = lambda v: np.clip(v, lo, hi)
        ax.hist(clip(r["plac_r2"]), bins=30, color="tab:gray", alpha=0.6, label="placebo")
        for x in r["cr_r2"]:
            ax.axvline(clip(x), color="tab:red", lw=1.2, alpha=0.8)
        ax.axvline(clip(np.median(r["plac_r2"])), color="black", ls="--", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_title(f"{name} — R2 del campo de fase  (mayor = mejor)\n"
                     f"crisis (rojo) vs placebo · MW p={r['p_r2']:.3f}", fontsize=10)
        ax.set_xlabel("R2 de fase (gradient matching)"); ax.set_ylabel("# ventanas placebo")
    fig.suptitle("P9 — Test de placebo: ¿el ajuste del oscilador a las crisis es ESPECIAL "
                 "o TRIVIAL?\n(barras rojas = 11-12 crisis; gris = ~270 ventanas de 13 trim "
                 "fuera de crisis)", fontsize=12)
    fig.tight_layout()
    out = C.OUTDIR / "p9_placebo.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def main():
    df = C.load()
    print("=== Pieza 9 — placebo / identificabilidad de la ventana de crisis ===")
    print(f"serie: {len(df)} trimestres | ventana = {W} trim (±{H}) ≈ una oscilacion")

    # Dos definiciones de placebo: estricta (no solapa ninguna crisis) y laxa
    # (el fondo de crisis no cae dentro de la ventana placebo -> mas muestras).
    for sep, tag in [(2 * H, "ESTRICTO: ventanas placebo sin solape con NINGUNA crisis"),
                     (H, "LAXO: ventanas placebo que no contienen el fondo de una crisis")]:
        print(f"\n################ PLACEBO {tag} (sep={sep}) ################")
        res_by_model = {}
        for model in (C.FN, C.LV):
            crisis, crisis_m, centers, placebo_m = eval_all(df, model, sep)
            res_by_model[model.name] = report(model.name, crisis, crisis_m, placebo_m)

        if sep == H:                                   # figura con el set grande
            out = figure(res_by_model)
            print(f"\n✓ figura: {out}")

        print("\n=== VEREDICTO ({}) ===".format("estricto" if sep == 2 * H else "laxo"))
        for name, r in res_by_model.items():
            med_pctile_rmse = np.mean([100.0 * np.mean(r["plac_rmse"] > x) for x in r["cr_rmse"]])
            print(f"  {name}: N_plac={len(r['plac_rmse'])} | crisis en pctil medio "
                  f"{med_pctile_rmse:.0f} (nRMSE) | p(rmse)={r['p_rmse']:.3f}, "
                  f"p(r2)={r['p_r2']:.3f}  -> "
                  + ("ajuste mejor que placebo (cola buena)" if r['p_rmse'] < 0.05
                     else "INDISTINGUIBLE del placebo"))


if __name__ == "__main__":
    main()
