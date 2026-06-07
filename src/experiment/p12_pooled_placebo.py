"""Pieza 12 — PLACEBO del pooling (P9): ¿la dinámica compartida de las crisis es ESPECIAL?

P9 (`p9_pooled.py`) mostró que UN único vector de parámetros presa-depredador θ_shared
describe las 11 crisis perdiendo solo +37 % de ajuste (LV) / +74 % (FN) respecto del ajuste
por-crisis (ratio SSE_pooled/SSE_indep = 1.37 LV / 1.74 FN) → "dinámica de crisis compartida
plausible". Pero ese resultado puede ser TRIVIAL: si CUALQUIER conjunto de ventanas de 13
trimestres (no solo las crisis) poolea igual de bien, entonces el bajo ratio no dice nada
sobre la crisis en particular — dice que un θ común describe cualquier conjunto de
oscilaciones cortas.

DISEÑO (placebo del pooling, no del ajuste por-ventana de p9_placebo).
  1. Réplica EXACTA de la maquinaria de p9_pooled (misma integración rápida, mismo ajuste
     INDEPENDIENTE con polish, mismo pooled warm-started, misma métrica ratio = SSE_pooled /
     SSE_indep). Lo único que cambia es el CONJUNTO de ventanas que se poolea.
  2. Ratio real de crisis: se computa con las K ventanas de crisis de p9 (ancla = fondo de
     inversión, ±6 trim, z-score por ventana). Es el observado a contrastar.
  3. Distribución nula: se muestrean MUCHOS conjuntos placebo de K ventanas cada uno,
     sorteadas de TODAS las posiciones de 13 trim que NO solapan ninguna crisis (expansiones /
     tramos no-crisis). Dentro de un conjunto las ventanas tampoco se solapan entre sí (para
     imitar la estructura de las ventanas de crisis). RNG sembrado determinista (reproducible).
     Para cada conjunto se computa su propio ratio pooled/indep con la MISMA maquinaria.
  4. TEST: ¿dónde cae el ratio real de crisis en la distribución de ratios placebo?
       - percentil del ratio real (cola BUENA = baja: poolea MEJOR que el placebo típico),
       - p-valor empírico one-sided p = (#{ratio_placebo <= ratio_real} + 1)/(N+1)
         = probabilidad de que un conjunto no-crisis poolee TAN bien o mejor que las crisis.
     p chico → el pooling de crisis es ESPECIAL (poolea inusualmente bien).
     p ~ 0.5 → es TRIVIAL (las crisis poolean como cualquier conjunto de oscilaciones).

  Series en transform QoQ-consistente con el resto del experimento (las columnas del CSV ya
  son las usadas por p9; el z-score por ventana las vuelve comparables en escala).

Caveats honestos: n chico por ventana (13 pts), identificabilidad limitada, ratio sensible a
inicialización (warm-start + polish lo estabilizan), u0 libre da grados de libertad al pooled.
El placebo controla TODOS esos efectos por construcción (la misma maquinaria se aplica a
crisis y a placebos), así que el contraste es de like-for-like.

Smoke test:  uv run python -m src.experiment.p12_pooled_placebo --smoke
Run full:    uv run python -m src.experiment.p12_pooled_placebo
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiment import common as C
from src.experiment.p9_pooled import (
    HW, WIN, crisis_windows, fit_independent, fit_pooled, _polish_independent,
)

# Ratios reales del pooling de crisis (de p9_pooled; se re-computan en runtime para coherencia,
# pero se dejan como referencia/control). LV=1.37, FN=1.74.
REF_RATIO = {"LV": 1.37, "FN": 1.74}


# --------------------------------------------------------------------- ventanas no-crisis
def noncrisis_centers(df, sep=None):
    """Centros c tales que la ventana [c-HW, c+HW] entra entera y su centro está a > sep
    trimestres del fondo de CUALQUIER crisis (sep por defecto = 2*HW = no solapa ninguna
    ventana de crisis). Estos son los tramos de expansión / no-crisis."""
    if sep is None:
        sep = 2 * HW
    p = df["PROFITS_YOY"].to_numpy()
    i = df["INVEST_YOY"].to_numpy()
    n = len(df)
    # anclas de crisis = fondo de inversión en la vecindad NBER (mismo criterio que p9)
    idx = df.index
    import pandas as pd
    anchor_pos = []
    for lbl, pk, tr in C.RECESSIONS:
        pk, tr = pd.Timestamp(pk), pd.Timestamp(tr)
        win = (idx >= pk - pd.offsets.QuarterEnd(2)) & (idx <= tr + pd.offsets.QuarterEnd(4))
        pos = np.where(win)[0]
        if len(pos):
            anchor_pos.append(int(pos[np.argmin(i[pos])]))
    centers = [c for c in range(HW, n - HW)
               if all(abs(c - a) > sep for a in anchor_pos)]
    return centers, p, i


def windows_from_centers(centers, p, i):
    """Construye ventanas (label, t, V_z, R_z) z-scoreadas por ventana, igual que p9."""
    out = []
    for c in centers:
        sl = slice(c - HW, c + HW + 1)
        V, R = p[sl].copy(), i[sl].copy()
        sV, sR = V.std(), R.std()
        if sV < 1e-9 or sR < 1e-9:
            continue
        V = (V - V.mean()) / sV
        R = (R - R.mean()) / sR
        t = np.arange(WIN, dtype=float)
        out.append((f"c{c}", t, V, R))
    return out


def sample_disjoint_set(rng, centers, k, min_gap=WIN):
    """Sortea k centros no-crisis cuyas ventanas NO se solapen entre sí (separación >= WIN),
    para imitar la estructura de las ventanas de crisis (disjuntas). Greedy con reintentos."""
    centers = list(centers)
    for _ in range(200):                      # reintentos del muestreo greedy
        rng.shuffle(centers)
        chosen = []
        for c in centers:
            if all(abs(c - x) >= min_gap for x in chosen):
                chosen.append(c)
            if len(chosen) == k:
                break
        if len(chosen) == k:
            return sorted(chosen)
    # si no caben k disjuntos (placebo pool chico), relajar a sin-repetición
    return sorted(rng.choice(centers, size=min(k, len(centers)), replace=False).tolist())


# --------------------------------------------------------------------- ratio de un conjunto
def pooled_ratio(model, windows, restarts, budget_s):
    """Ratio SSE_pooled/SSE_indep para UN conjunto de ventanas, con la maquinaria de p9."""
    nd = len(model.p0)
    sse_i, per_i = fit_independent(model, windows, restarts)
    sse_p, per_p, theta = fit_pooled(model, windows, restarts, per_i, budget_s=budget_s)
    # mismo polish que p9: garantiza que el independiente sea un piso honesto (ratio >= 1)
    per_i = _polish_independent(model, windows, per_i, per_p, theta, nd)
    sse_i = sum(per_i[w[0]]["sse"] for w in windows)
    ratio = sse_p / sse_i if sse_i > 0 else np.inf
    return ratio, sse_i, sse_p, theta


# --------------------------------------------------------------------- figura
def figure(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 4.2))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, r) in zip(axes, results.items()):
        plac = np.asarray(r["placebo_ratios"], float)
        plac = plac[np.isfinite(plac)]
        ax.hist(plac, bins=min(25, max(8, len(plac) // 3)), color="tab:gray",
                alpha=0.65, label=f"placebo (n={len(plac)} conjuntos)")
        ax.axvline(r["real_ratio"], color="tab:red", lw=2.0,
                   label=f"crisis real = {r['real_ratio']:.2f}")
        ax.axvline(np.median(plac), color="black", ls="--", lw=1.0,
                   label=f"mediana placebo = {np.median(plac):.2f}")
        ax.set_title(f"{name} — ratio pooled/indep\n"
                     f"percentil crisis = {r['percentile']:.0f} · p = {r['pval']:.3f}",
                     fontsize=10)
        ax.set_xlabel("ratio SSE pooled / indep (menor = poolea mejor)")
        ax.set_ylabel("# conjuntos placebo")
        ax.legend(fontsize=7)
    fig.suptitle("P12 — PLACEBO del pooling: ¿la dinámica de crisis compartida es ESPECIAL?\n"
                 "rojo = ratio del pooling de las crisis · gris = ratios de conjuntos "
                 "no-crisis del mismo tamaño (cola izquierda = poolea mejor)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = C.OUTDIR / "p12_pooled_placebo.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------- driver
def run(n_sets, budget_s, seed=20260606, restarts=(1.0, 1.2, 0.8), models=(C.LV, C.FN)):
    df = C.load()
    crisis_w = crisis_windows(df)
    k = len(crisis_w)
    centers, p, i = noncrisis_centers(df)
    print(f"serie: {len(df)} trim | ventana = {WIN} trim (±{HW})")
    print(f"ventanas de crisis (K) = {k} -> {[w[0] for w in crisis_w]}")
    print(f"centros no-crisis disponibles (sep=2HW) = {len(centers)}")
    print(f"conjuntos placebo a evaluar = {n_sets} (K={k} ventanas c/u, RNG seed={seed})")
    print(f"budget pooled por ajuste = {budget_s:.0f}s")

    results = {}
    for model in models:
        print(f"\n{'='*70}\n  MODELO {model.name}\n{'='*70}", flush=True)
        # --- ratio REAL de crisis (misma maquinaria) ---
        t0 = time.time()
        real_ratio, rsi, rsp, _ = pooled_ratio(model, crisis_w, restarts, budget_s)
        print(f"  ratio REAL crisis = {real_ratio:.3f}  "
              f"(SSE_indep={rsi:.1f}, SSE_pooled={rsp:.1f})  "
              f"[ref p9: {REF_RATIO.get(model.name, float('nan')):.2f}]  "
              f"[{time.time()-t0:.0f}s]", flush=True)

        # --- distribución placebo ---
        rng = np.random.default_rng(seed + hash(model.name) % 10_000)
        plac = []
        for s in range(n_sets):
            cs = sample_disjoint_set(rng, centers, k)
            w = windows_from_centers(cs, p, i)
            t0 = time.time()
            ratio, _, _, _ = pooled_ratio(model, w, restarts, budget_s)
            plac.append(ratio)
            print(f"    placebo {s+1:>3}/{n_sets}: ratio={ratio:6.3f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        plac = np.array(plac, float)
        valid = plac[np.isfinite(plac)]

        # percentil del ratio real (menor ratio = mejor pooling -> cola izquierda)
        pct = 100.0 * np.mean(valid > real_ratio)          # % de placebos que poolean PEOR
        # p-valor empírico one-sided: P(placebo poolea TAN bien o mejor que crisis)
        pval = (np.sum(valid <= real_ratio) + 1) / (len(valid) + 1)

        print(f"\n  >>> {model.name}: ratio crisis = {real_ratio:.3f}")
        print(f"      placebo: mediana={np.median(valid):.3f}  "
              f"[p10={np.percentile(valid,10):.3f}, p90={np.percentile(valid,90):.3f}]  "
              f"min={valid.min():.3f}")
        print(f"      percentil del ratio real (cola buena = bajo) = {pct:.0f}")
        print(f"      p-valor empírico (placebo poolea >= bien que crisis) = {pval:.3f}")
        verdict = ("ESPECIAL: las crisis poolean mejor que conjuntos no-crisis comparables"
                   if pval < 0.05 else
                   "TRIVIAL: las crisis poolean como cualquier conjunto de oscilaciones")
        print(f"      -> {verdict}")

        results[model.name] = dict(real_ratio=real_ratio, placebo_ratios=plac,
                                   percentile=pct, pval=pval,
                                   plac_median=float(np.median(valid)))

    out = figure(results)
    print(f"\nfigura -> {out}")

    print(f"\n{'='*70}\n  VEREDICTO P12\n{'='*70}")
    for name, r in results.items():
        tag = "ESPECIAL" if r["pval"] < 0.05 else "TRIVIAL"
        print(f"  {name}: ratio_crisis={r['real_ratio']:.2f} vs mediana_placebo="
              f"{r['plac_median']:.2f} | percentil={r['percentile']:.0f} "
              f"| p={r['pval']:.3f} -> {tag}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test: pocos conjuntos placebo y budget chico")
    ap.add_argument("--n-sets", type=int, default=None,
                    help="número de conjuntos placebo (default: 150 full / 4 smoke)")
    ap.add_argument("--budget", type=float, default=None,
                    help="budget en s del pooled por ajuste (default: 75 full / 15 smoke)")
    ap.add_argument("--seed", type=int, default=20260606)
    args = ap.parse_args()

    if args.smoke:
        n_sets = args.n_sets if args.n_sets is not None else 4
        budget = args.budget if args.budget is not None else 15.0
        print("### SMOKE TEST (no es el run completo) ###")
    else:
        n_sets = args.n_sets if args.n_sets is not None else 150
        budget = args.budget if args.budget is not None else 75.0

    run(n_sets=n_sets, budget_s=budget, seed=args.seed)


if __name__ == "__main__":
    main()
