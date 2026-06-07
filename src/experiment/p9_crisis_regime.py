"""Pieza 9 — ¿El oscilador es un modelo de RÉGIMEN DE CRISIS, no de la serie completa?

Hipótesis (derivada de P8): el acople presa-depredador (lag−4 profits→investment) está
CONCENTRADO en la vecindad de las crisis (−0.63 cerca vs −0.10 en expansión). Si eso es
así, un oscilador no lineal (LV/FN) debería describir la dinámica MUCHO mejor en ventanas
de crisis que en ventanas de expansión equivalentes — no es un modelo de los 78 años (donde
degenera), sino de la fase de crisis.

Diseño (escéptico, controlado):
  1. VENTANA DE CRISIS = ±6 trimestres del fondo (= trim. de inversión mínima en la vecindad
     de cada recesión NBER), 13 trimestres. Igual ancla que P8.
  2. VENTANA DE EXPANSIÓN de control = mismo largo (13 trim.), centrada en el PUNTO MEDIO
     entre dos anclas de crisis consecutivas (lo más lejos posible de toda crisis). Una por
     período inter-crisis.
  3. Ajuste FN, LV, LIN por single-shooting (common.fit) sobre cada ventana z-scoreada.
  4. Métricas por ajuste: R² del campo de fase (gradient matching de los parámetros
     single-shooting, lógica de P4) y RMSE de trayectoria normalizado (RMSE/σ de los datos).
     También período implícito y autovalores cuando es estable.
  5. Test pareado crisis vs su expansión vecina: medias, distribución, cuántas crisis ganan.
  6. Figura comparativa R² y RMSE crisis vs expansión por modelo.

Caveat estructural: 13 puntos ≈ una oscilación → riesgo de sobreajuste. Por eso reportamos
el R² del CAMPO (gradient matching, no ajustado a la trayectoria que minimiza el RMSE) como
la métrica principal de "describe la dinámica", y el RMSE como métrica de ajuste de
trayectoria. Si el campo NO mejora en crisis, la hipótesis NO se sostiene.

Run:  uv run python -m src.experiment.p9_crisis_regime
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from src.experiment import common as C
from src.experiment.p4_dynamical_eval import (
    derivatives, field_r2, grad_match, jacobian, classify,
)

HALF = 6                   # ±6 trimestres → ventana de 13 trimestres
WIN = 2 * HALF + 1
MODELS = (C.FN, C.LV, C.LIN)


# --------------------------------------------------------------------------- ventanas
def crisis_anchor_positions(df, p, i):
    """Para cada recesión NBER: posición del trimestre de inversión mínima en su vecindad."""
    idx = df.index
    out = []
    for lbl, pk, tr in C.RECESSIONS:
        pk, tr = pd.Timestamp(pk), pd.Timestamp(tr)
        win = (idx >= pk - pd.offsets.QuarterEnd(2)) & (idx <= tr + pd.offsets.QuarterEnd(4))
        pos = np.where(win)[0]
        if len(pos) == 0:
            continue
        out.append((lbl, pos[np.argmin(i[pos])]))
    return out


def make_windows(df, p, i):
    """Genera ventanas de crisis (±6 del fondo) y de expansión de control (medio inter-crisis).

    Devuelve dos listas de dicts {label, a (centro), lo, hi}. La ventana de expansión k está
    emparejada con la crisis k que la cierra (la que viene DESPUÉS del hueco), para el test
    pareado 'crisis vs su expansión inmediatamente previa'.
    """
    n = len(p)
    anchors = crisis_anchor_positions(df, p, i)
    crises = []
    for lbl, a in anchors:
        lo, hi = a - HALF, a + HALF + 1
        if lo < 0 or hi > n:
            continue                     # ventana incompleta (bordes de la muestra)
        crises.append(dict(label=lbl, a=a, lo=lo, hi=hi))

    # ventana de expansión = punto medio entre anclas consecutivas, ±6
    exps = []
    for (lbl0, a0), (lbl1, a1) in zip(anchors[:-1], anchors[1:]):
        mid = (a0 + a1) // 2
        lo, hi = mid - HALF, mid + HALF + 1
        # debe caber y NO solaparse con ninguna vecindad de crisis (±6 de cada ancla)
        if lo < 0 or hi > n:
            continue
        gap = a1 - a0
        if gap < 3 * HALF:               # no hay lugar para una ventana de expansión "limpia"
            continue
        exps.append(dict(label=lbl1, a=mid, lo=lo, hi=hi, prev=lbl0))
    return crises, exps


# --------------------------------------------------------------------------- ajuste/eval
def fit_eval(model, df, lo, hi):
    """Ajusta el modelo por single-shooting sobre la ventana z-scoreada y evalúa.

    R² del campo: gradient matching de los parámetros sobre la ventana (lógica P4) — mide si
    la VELOCIDAD observada sigue el campo del oscilador.
    RMSE norm.: RMSE de la trayectoria single-shooting integrada / σ de los datos (por serie,
    promediado). <1 = mejor que predecir la media.
    """
    sub = df.iloc[lo:hi]
    to_z, to_raw, off, mu, sd = C.zscorer(sub, model.needs_offset)
    Z = to_z(sub[C.COLS].to_numpy())                 # (n,2)
    V, R = Z[:, 0], Z[:, 1]
    t = (sub.index - sub.index[0]).days.to_numpy() / 365.25

    # --- single-shooting (ajuste de trayectoria) ---
    theta, sse = C.fit(model, t, V, R)
    y = C.integrate(model, theta, np.array([V[0], R[0]]), t)
    if y is None:
        rmse_norm = np.nan
    else:
        # σ de los datos por serie (z-scoreados → ≈1, pero LV tiene offset; calculamos directo)
        rmse_p = np.sqrt(np.mean((y[0] - V) ** 2)) / (V.std() + 1e-9)
        rmse_r = np.sqrt(np.mean((y[1] - R) ** 2)) / (R.std() + 1e-9)
        rmse_norm = 0.5 * (rmse_p + rmse_r)

    # --- R² del campo de fase (gradient matching, métrica principal) ---
    dZ = derivatives(t, Z)
    p_gm = grad_match(model, Z, dZ)
    r2 = field_r2(model, Z, dZ, p_gm)                 # [profits, investment]
    r2_mean = float(np.mean(r2))

    # --- período implícito / autovalores (si el punto fijo es razonable) ---
    period, re_dom = np.nan, np.nan
    try:
        fp, info, ier, _ = fsolve(lambda u: model.rhs(u, p_gm), Z.mean(0), full_output=True)
        if ier == 1:
            eig = np.linalg.eigvals(jacobian(model, p_gm, fp))
            period, re_dom, _ = classify(eig)
    except Exception:
        pass

    return dict(r2_mean=r2_mean, r2_p=float(r2[0]), r2_i=float(r2[1]),
                rmse_norm=float(rmse_norm) if np.isfinite(rmse_norm) else np.nan,
                period=period, re_dom=re_dom, sse=sse)


# --------------------------------------------------------------------------- main
def main():
    df = C.load()
    p, i = df["PROFITS_YOY"].to_numpy(), df["INVEST_YOY"].to_numpy()
    crises, exps = make_windows(df, p, i)

    print("=== Pieza 9 — oscilador como modelo de RÉGIMEN DE CRISIS ===")
    print(f"Ventana = ±{HALF} trim ({WIN} trimestres ≈ {WIN*0.25:.1f} años).")
    print(f"{len(crises)} ventanas de crisis (completas), {len(exps)} ventanas de expansión de control.\n")

    print("Ventanas de crisis:")
    for c in crises:
        print(f"  {c['label']:5s}: {df.index[c['lo']].date()} → {df.index[c['hi']-1].date()}")
    print("Ventanas de expansión (medio inter-crisis):")
    for e in exps:
        print(f"  {e['prev']}→{e['label']}: {df.index[e['lo']].date()} → {df.index[e['hi']-1].date()}")
    print()

    # --- ajustar y evaluar ---
    res = {m.name: {"crisis": {}, "exp": {}} for m in MODELS}
    for m in MODELS:
        for c in crises:
            res[m.name]["crisis"][c["label"]] = fit_eval(m, df, c["lo"], c["hi"])
        for e in exps:
            res[m.name]["exp"][e["label"]] = fit_eval(m, df, e["lo"], e["hi"])

    # --- tabla agregada R²/RMSE crisis vs expansión ---
    print("=== R² del campo de fase (gradient matching) — métrica principal ===")
    print(f"{'modelo':6s} | {'crisis μ(med)':>16s} | {'expansión μ(med)':>18s} | Δ(crisis−exp)")
    agg = {}
    for m in MODELS:
        rc = np.array([res[m.name]["crisis"][c["label"]]["r2_mean"] for c in crises])
        re_ = np.array([res[m.name]["exp"][e["label"]]["r2_mean"] for e in exps])
        agg[m.name] = dict(rc=rc, re=re_)
        print(f"{m.name:6s} | {np.nanmean(rc):+7.2f} ({np.nanmedian(rc):+.2f}) | "
              f"{np.nanmean(re_):+8.2f} ({np.nanmedian(re_):+.2f}) | {np.nanmean(rc)-np.nanmean(re_):+.2f}")

    print("\n=== RMSE de trayectoria normalizado (single-shooting, RMSE/σ; <1 mejor que la media) ===")
    print(f"{'modelo':6s} | {'crisis μ(med)':>16s} | {'expansión μ(med)':>18s} | Δ(crisis−exp)")
    for m in MODELS:
        rc = np.array([res[m.name]["crisis"][c["label"]]["rmse_norm"] for c in crises])
        re_ = np.array([res[m.name]["exp"][e["label"]]["rmse_norm"] for e in exps])
        agg[m.name]["rmse_c"], agg[m.name]["rmse_e"] = rc, re_
        print(f"{m.name:6s} | {np.nanmean(rc):7.2f} ({np.nanmedian(rc):.2f}) | "
              f"{np.nanmean(re_):8.2f} ({np.nanmedian(re_):.2f}) | {np.nanmean(rc)-np.nanmean(re_):+.2f}")

    # --- test pareado: crisis k vs su expansión PREVIA (mismo hueco inter-crisis) ---
    # cada ventana de expansión 'e' tiene label = crisis que la cierra → emparejar.
    print("\n=== TEST PAREADO (crisis k vs su expansión inmediatamente previa) ===")
    exp_labels = {e["label"] for e in exps}
    pairs = [c["label"] for c in crises if c["label"] in exp_labels]
    print(f"  pares válidos (crisis con expansión previa): {len(pairs)} → {pairs}")
    for metric, better in [("r2_mean", "mayor"), ("rmse_norm", "menor")]:
        print(f"\n  -- métrica: {metric} ({'mejor='+better}) --")
        for m in MODELS:
            dc = np.array([res[m.name]["crisis"][l][metric] for l in pairs])
            de = np.array([res[m.name]["exp"][l][metric] for l in pairs])
            d = dc - de
            ok = np.isfinite(d)
            d = d[ok]
            if metric == "r2_mean":
                wins = int(np.sum(d > 0))               # crisis mejor = R² mayor
            else:
                wins = int(np.sum(d < 0))               # crisis mejor = RMSE menor
            # test de signos (binomial exacto, two-sided) sin scipy.stats:
            n = len(d)
            from math import comb
            k = max(wins, n - wins)
            pval = min(1.0, 2 * sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n) if n else np.nan
            print(f"    {m.name:5s}: crisis mejor en {wins}/{n} pares | "
                  f"Δμ(crisis−exp)={np.mean(d):+.3f} | signo p={pval:.3f}")

    # --- detalle por crisis (R² campo) ---
    print("\n=== Detalle por crisis: R² del campo (crisis | expansión previa) ===")
    print(f"{'crisis':6s} | " + " | ".join(f"{m.name:^17s}" for m in MODELS))
    for l in pairs:
        cells = []
        for m in MODELS:
            rc = res[m.name]["crisis"][l]["r2_mean"]
            re_ = res[m.name]["exp"][l]["r2_mean"]
            cells.append(f"{rc:+.2f} | {re_:+.2f}")
        print(f"{l:6s} | " + " | ".join(f"{c:^17s}" for c in cells))

    # --- períodos implícitos (chequeo de plausibilidad) ---
    print("\n=== Período implícito medio (años) — campo gradient-matched ===")
    for m in MODELS:
        pc = np.array([res[m.name]["crisis"][c["label"]]["period"] for c in crises])
        pe = np.array([res[m.name]["exp"][e["label"]]["period"] for e in exps])
        pc = pc[np.isfinite(pc) & (pc < 200)]
        pe = pe[np.isfinite(pe) & (pe < 200)]
        mc = f"{np.median(pc):.1f}a" if len(pc) else "n/a"
        me = f"{np.median(pe):.1f}a" if len(pe) else "n/a"
        print(f"  {m.name:5s}: crisis med={mc} (n={len(pc)})  expansión med={me} (n={len(pe)})")

    # --- CSV ---
    rows = []
    for m in MODELS:
        for regime, key in [("crisis", "crisis"), ("expansion", "exp")]:
            for lbl, d in res[m.name][key].items():
                rows.append(dict(model=m.name, regime=regime, label=lbl, **d))
    out_csv = C.OUTDIR / "p9_crisis_regime.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n✓ tabla: {out_csv}")

    # --- figura comparativa ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    x = np.arange(len(MODELS))
    w = 0.35
    # panel 1: R² campo
    ax = axes[0]
    rc_m = [np.nanmean(agg[m.name]["rc"]) for m in MODELS]
    re_m = [np.nanmean(agg[m.name]["re"]) for m in MODELS]
    ax.bar(x - w/2, rc_m, w, label="crisis", color="tab:red", alpha=0.8)
    ax.bar(x + w/2, re_m, w, label="expansión", color="tab:blue", alpha=0.8)
    # puntos individuales
    for j, m in enumerate(MODELS):
        ax.scatter(np.full_like(agg[m.name]["rc"], j - w/2), agg[m.name]["rc"],
                   color="darkred", s=14, zorder=3, alpha=0.7)
        ax.scatter(np.full_like(agg[m.name]["re"], j + w/2), agg[m.name]["re"],
                   color="navy", s=14, zorder=3, alpha=0.7)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([m.name for m in MODELS])
    ax.set_ylabel("R² del campo de fase (gradient matching)")
    ax.set_title("R² del campo: crisis vs expansión\n(barras=media, puntos=ventanas)", fontsize=10)
    ax.legend(fontsize=9)
    # panel 2: RMSE norm
    ax = axes[1]
    rc_m = [np.nanmean(agg[m.name]["rmse_c"]) for m in MODELS]
    re_m = [np.nanmean(agg[m.name]["rmse_e"]) for m in MODELS]
    ax.bar(x - w/2, rc_m, w, label="crisis", color="tab:red", alpha=0.8)
    ax.bar(x + w/2, re_m, w, label="expansión", color="tab:blue", alpha=0.8)
    for j, m in enumerate(MODELS):
        ax.scatter(np.full_like(agg[m.name]["rmse_c"], j - w/2), agg[m.name]["rmse_c"],
                   color="darkred", s=14, zorder=3, alpha=0.7)
        ax.scatter(np.full_like(agg[m.name]["rmse_e"], j + w/2), agg[m.name]["rmse_e"],
                   color="navy", s=14, zorder=3, alpha=0.7)
    ax.axhline(1.0, color="grey", lw=0.6, ls="--", label="RMSE=σ (media)")
    ax.set_xticks(x); ax.set_xticklabels([m.name for m in MODELS])
    ax.set_ylabel("RMSE de trayectoria / σ (single-shooting)")
    ax.set_title("Ajuste de trayectoria: crisis vs expansión\n(menor = mejor)", fontsize=10)
    ax.legend(fontsize=9)
    fig.suptitle("Pieza 9 — ¿el oscilador describe mejor la crisis que la expansión?", fontsize=12)
    fig.tight_layout()
    out_fig = C.OUTDIR / "p9_crisis_regime.png"
    fig.savefig(out_fig, dpi=130); plt.close(fig)
    print(f"✓ figura: {out_fig}")


if __name__ == "__main__":
    main()
