"""Pieza 12 — PLACEBO formal del kernel / joroba de lag 4-5.

PREGUNTA. El feedback de sobreacumulación (inversión pasada -> deprime ganancia) aparece
como una JOROBA NEGATIVA del kernel de lag distribuido / de la CCF profits↔investment en
los rezagos 4-5 (~1 año), concentrada en la vecindad de las crisis (P7, P8, P10). Pero esa
joroba, ¿es PROPIA del régimen de crisis o es UBICUA? Si la misma joroba 4-5 aparece también
en ventanas de EXPANSIÓN o en ventanas ALEATORIAS no-crisis, entonces no es una firma del
giro del ciclo: es un artefacto de la dinámica de fondo de las series.

DISEÑO (test de falsificación, análogo a P9 pero sobre el ESTADÍSTICO DEL KERNEL, no sobre
el ajuste de un oscilador).

  Estadístico de la joroba por ventana (dos lecturas, ambas QoQ = pct_change(1) de niveles):
    * KERNEL: β_k de  profits_t ~ Σ_{k=0..KMAX} β_k · investment_{t-k}  (OLS por ventana,
      z-scoreada). La joroba = la masa negativa en k∈{4,5}; resumen escalar hump_kernel =
      (β4 + β5) / 2  (negativo => joroba de sobreacumulación).
    * CCF: corr(profits_t, investment_{t-k}) a lag k (investment LIDERA). Resumen escalar
      hump_ccf = (CCF[-4] + CCF[-5]) / 2  (la convención de P7: lag negativo = invest lidera).
    Para ambos, MÁS NEGATIVO = joroba más fuerte.

  Tres conjuntos de ventanas (mismo ancho, mismo z-score local, mismos lags):
    (a) CRISIS    — ancla = inversión QoQ mínima en la vecindad NBER (= P8/P9/P10).
    (b) EXPANSIÓN — centros a media distancia entre crisis consecutivas (lejos de todo fondo).
    (c) ALEATORIAS no-crisis — RNG sembrado; centros que no caen dentro de NINGUNA ventana de
        crisis (definición laxa: el fondo de crisis no entra en la ventana placebo).

  VEREDICTO. Se compara el valor de crisis (mediana de la joroba sobre las ventanas de crisis,
  y el POOLED de todas las filas de crisis) contra la distribución placebo:
    - percentil del valor de crisis en la distribución placebo (cola buena = MÁS NEGATIVO);
    - p-valor empírico one-sided  P(placebo <= crisis)  (¿la joroba de crisis es atípicamente
      negativa respecto del placebo?);
    - Mann-Whitney one-sided (medianas por-ventana: crisis < expansión y crisis < aleatorias).
  Si el percentil es bajo / p chico => la joroba 4-5 está CONCENTRADA en crisis (no ubicua).
  Si el percentil es ~medio / p grande => la joroba es UBICUA (no es firma del régimen de crisis).

SMOKE TEST: NPLAC_SMOKE ventanas aleatorias (pocas). main() corre el set completo.

Run (full):   uv run python -m src.experiment.p12_kernel_placebo
Run (smoke):  uv run python -m src.experiment.p12_kernel_placebo --smoke
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.experiment import common as C

KMAX = 8                    # rezagos 0..8 para el kernel (igual que p10_distlag)
HW = 6                      # media-ventana ±6 trim -> ventana objetivo de 2*HW+1 = 13 trim
LAGS_HUMP = (4, 5)         # la joroba de sobreacumulación
SEED = 12345
NPLAC_FULL = 4000          # ventanas aleatorias no-crisis (set completo)
NPLAC_SMOKE = 60           # smoke test: pocas


# --------------------------------------------------------------------------- datos QoQ
def load_qoq():
    """QoQ = pct_change(1) sobre NIVELES PROFITS, INVESTMENT (transform honesto, ver P10 §4)."""
    df = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    out = pd.DataFrame(index=df.index)
    out["PROF"] = df["PROFITS"].pct_change(1) * 100.0
    out["INV"] = df["INVESTMENT"].pct_change(1) * 100.0
    return out.dropna()


def crisis_anchors(df):
    """Ancla por recesión = inversión QoQ más deprimida en su vecindad NBER (= P8/P9/P10)."""
    idx = df.index
    inv = df["INV"].to_numpy()
    out = []
    for lbl, pk, tr in C.RECESSIONS:
        pk, tr = pd.Timestamp(pk), pd.Timestamp(tr)
        win = (idx >= pk - pd.offsets.QuarterEnd(2)) & (idx <= tr + pd.offsets.QuarterEnd(4))
        pos = np.where(win)[0]
        if len(pos) == 0:
            continue
        out.append((lbl, int(pos[np.argmin(inv[pos])])))
    return out


# --------------------------------------------------------------------------- estadístico por ventana
def window_arrays(prof, inv, center, standardize=True):
    """Serie local de la ventana: objetivo [center-HW, center+HW], extendida KMAX a la
    izquierda para los rezagos. z-score por ventana (sobre el tramo extendido). Devuelve
    (prof_z, inv_z, lo_ext, hi_t) o None si la ventana no cabe entera."""
    lo_t = center - HW
    hi_t = center + HW
    lo_ext = lo_t - KMAX
    if lo_ext < 0 or hi_t >= len(prof):
        return None
    pe = prof[lo_ext:hi_t + 1].copy()
    ie = inv[lo_ext:hi_t + 1].copy()
    if standardize:
        ps, is_ = pe.std(), ie.std()
        if ps < 1e-9 or is_ < 1e-9:
            return None
        pe = (pe - pe.mean()) / ps
        ie = (ie - ie.mean()) / is_
    return pe, ie, lo_ext, hi_t


def kernel_window(prof, inv, center):
    """OLS por ventana: profits_t ~ const + Σ_{k=0..KMAX} β_k inv_{t-k}, filas t∈[center±HW].
    Devuelve el kernel β (KMAX+1,) o None. Pseudo-inversa: la ventana es chica respecto de
    KMAX+1 coef -> mal condicionada por ventana (lo reportamos; el estadístico de interés es
    sólo la joroba 4-5, robusta a esa colinealidad porque se lee como bloque)."""
    arr = window_arrays(prof, inv, center)
    if arr is None:
        return None
    pe, ie, lo_ext, hi_t = arr
    off = -lo_ext                                  # índice local del t global = t_global - lo_ext
    rows_y, rows_X = [], []
    for t in range(center - HW, center + HW + 1):
        ti = t - lo_ext
        rows_y.append(pe[ti])
        rows_X.append([ie[ti - k] for k in range(0, KMAX + 1)])
    y = np.asarray(rows_y); X = np.asarray(rows_X)
    Xd = np.column_stack([X, np.ones(len(X))])
    beta = np.linalg.pinv(Xd.T @ Xd) @ Xd.T @ y
    return beta[:KMAX + 1]


def ccf_window(prof, inv, center):
    """CCF por ventana: corr(prof_t, inv_{t-k}) a los lags de la joroba. Usa la ventana
    extendida (más puntos para la correlación). Convención P7: lag negativo = invest lidera.
    Devuelve dict lag(negativo)->corr para -max(LAGS_HUMP)..-1, o None."""
    arr = window_arrays(prof, inv, center)
    if arr is None:
        return None
    pe, ie, _, _ = arr
    m = len(pe)
    out = {}
    for k in range(1, max(LAGS_HUMP) + 1):
        # invest lidera por k: corr(prof[k:], inv[:-k])
        a, b = pe[k:], ie[:m - k]
        if len(a) < 4 or a.std() < 1e-9 or b.std() < 1e-9:
            out[-k] = np.nan
        else:
            out[-k] = float(np.corrcoef(a, b)[0, 1])
    return out


def hump_kernel(beta):
    """Resumen escalar de la joroba del kernel: media de β en los lags 4-5 (negativo = joroba)."""
    return float(np.mean([beta[k] for k in LAGS_HUMP]))


def hump_ccf(ccf_dict):
    """Resumen escalar de la joroba de la CCF: media de la CCF en lag -4,-5 (negativo = joroba)."""
    vals = [ccf_dict[-k] for k in LAGS_HUMP]
    return float(np.nanmean(vals))


# --------------------------------------------------------------------------- conjuntos de ventanas
def expansion_centers(anchor_pos, n):
    """Centros de expansión: punto medio entre fondos de crisis consecutivos, si la ventana
    cabe entera y queda lejos (> HW) de todo fondo de crisis."""
    a = sorted(anchor_pos)
    cents = []
    for x, y in zip(a[:-1], a[1:]):
        c = (x + y) // 2
        if c - HW - KMAX >= 0 and c + HW < n and all(abs(c - p) > HW for p in anchor_pos):
            cents.append(c)
    return cents


def random_noncrisis_centers(anchor_pos, n, nplac, seed=SEED):
    """RNG sembrado: centros que NO caen dentro de ninguna ventana de crisis (laxo: el fondo
    de crisis no entra en la ventana placebo, |c - a| > HW) y con ventana entera (incluye la
    cola de KMAX rezagos a la izquierda). Muestrea SIN reemplazo del conjunto válido; si hay
    menos válidos que nplac, los devuelve todos."""
    rng = np.random.default_rng(seed)
    valid = [c for c in range(HW + KMAX, n - HW)
             if all(abs(c - a) > HW for a in anchor_pos)]
    if len(valid) <= nplac:
        return np.array(valid, int)
    return np.sort(rng.choice(valid, size=nplac, replace=False))


# --------------------------------------------------------------------------- evaluación de un set
def eval_centers(prof, inv, centers):
    """Calcula (hump_kernel, hump_ccf) por ventana para una lista de centros.
    Devuelve dict con arrays alineados (descarta ventanas inválidas)."""
    hk, hc = [], []
    for c in centers:
        beta = kernel_window(prof, inv, c)
        cd = ccf_window(prof, inv, c)
        if beta is None or cd is None:
            continue
        hk.append(hump_kernel(beta))
        hc.append(hump_ccf(cd))
    return dict(hump_kernel=np.array(hk), hump_ccf=np.array(hc))


def pooled_crisis_kernel(prof, inv, anchors):
    """Kernel POOLED de crisis: apila todas las filas de todas las crisis (z-score por crisis)
    y corre UN OLS. Devuelve (beta, hump). Es el estadístico de crisis 'duro' (más estable que
    la mediana de kernels por ventana, que son casi singulares individualmente)."""
    rows_y, rows_X = [], []
    for _, a in anchors:
        arr = window_arrays(prof, inv, a)
        if arr is None:
            continue
        pe, ie, lo_ext, hi_t = arr
        for t in range(a - HW, a + HW + 1):
            ti = t - lo_ext
            rows_y.append(pe[ti])
            rows_X.append([ie[ti - k] for k in range(0, KMAX + 1)])
    y = np.asarray(rows_y); X = np.asarray(rows_X)
    Xd = np.column_stack([X, np.ones(len(X))])
    beta = (np.linalg.pinv(Xd.T @ Xd) @ Xd.T @ y)[:KMAX + 1]
    return beta, hump_kernel(beta)


# --------------------------------------------------------------------------- reporte de un metric
def report_metric(name, crisis_vals, exp_vals, plac_vals, crisis_pooled, lower_is_hump=True):
    """Reporta percentil + p-valor del valor de crisis dentro de la distribución placebo.
    lower_is_hump=True: MÁS NEGATIVO = joroba más fuerte (cola buena = baja)."""
    crisis_med = float(np.median(crisis_vals))
    print(f"\n  --- {name} (joroba en lags {LAGS_HUMP}; más NEGATIVO = joroba más fuerte) ---")
    print(f"    CRISIS    : mediana por-ventana = {crisis_med:+.3f}  | POOLED = {crisis_pooled:+.3f}"
          f"  (n={len(crisis_vals)} ventanas)")
    if len(exp_vals):
        print(f"    EXPANSIÓN : mediana = {np.median(exp_vals):+.3f}  "
              f"[min {exp_vals.min():+.3f}, max {exp_vals.max():+.3f}]  (n={len(exp_vals)})")
    print(f"    ALEATORIO : mediana = {np.median(plac_vals):+.3f}  "
          f"[p10 {np.percentile(plac_vals,10):+.3f}, p50 {np.percentile(plac_vals,50):+.3f}, "
          f"p90 {np.percentile(plac_vals,90):+.3f}]  (n={len(plac_vals)})")

    # percentil del valor de crisis (mediana y pooled) dentro del placebo aleatorio.
    # cola buena = MÁS NEGATIVO -> percentil = % de placebos MÁS NEGATIVOS (<=) que crisis.
    def pctile_and_p(x):
        frac_le = float(np.mean(plac_vals <= x))          # P(placebo <= crisis)
        return 100.0 * frac_le, frac_le                   # percentil bajo = crisis en cola negativa
    pct_med, p_med = pctile_and_p(crisis_med)
    pct_pool, p_pool = pctile_and_p(crisis_pooled)
    print(f"    percentil de la joroba de crisis en el placebo aleatorio:")
    print(f"      mediana por-ventana: percentil {pct_med:4.1f}  -> p(placebo <= crisis) = {p_med:.3f}")
    print(f"      POOLED de crisis   : percentil {pct_pool:4.1f}  -> p(placebo <= crisis) = {p_pool:.3f}")

    # Mann-Whitney one-sided: crisis MÁS NEGATIVO (less) que expansión y que aleatorio.
    p_mw_exp = np.nan
    if len(exp_vals) >= 3:
        try:
            _, p_mw_exp = mannwhitneyu(crisis_vals, exp_vals, alternative="less")
        except ValueError:
            pass
    try:
        _, p_mw_plac = mannwhitneyu(crisis_vals, plac_vals, alternative="less")
    except ValueError:
        p_mw_plac = np.nan
    print(f"    Mann-Whitney (crisis < ...): vs expansión p={p_mw_exp:.3f} | vs aleatorio p={p_mw_plac:.3f}")

    concentrated = (p_pool < 0.05) or (p_med < 0.05) or (not np.isnan(p_mw_plac) and p_mw_plac < 0.05)
    print(f"    => joroba {'CONCENTRADA en crisis' if concentrated else 'UBICUA (no propia del régimen de crisis)'}")
    return dict(name=name, crisis_med=crisis_med, crisis_pooled=crisis_pooled,
                exp_vals=exp_vals, plac_vals=plac_vals, crisis_vals=crisis_vals,
                pct_med=pct_med, p_med=p_med, pct_pool=pct_pool, p_pool=p_pool,
                p_mw_exp=p_mw_exp, p_mw_plac=p_mw_plac, concentrated=concentrated)


# --------------------------------------------------------------------------- figura
def figure(results, tag):
    fig, axes = plt.subplots(1, len(results), figsize=(6.5 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        ax.hist(r["plac_vals"], bins=30, color="tab:gray", alpha=0.6,
                label=f"aleatorio no-crisis (n={len(r['plac_vals'])})")
        if len(r["exp_vals"]):
            for x in r["exp_vals"]:
                ax.axvline(x, color="tab:orange", lw=1.0, alpha=0.7)
            ax.axvline(r["exp_vals"][0], color="tab:orange", lw=1.0, alpha=0.7, label="expansión")
        ax.axvline(np.median(r["plac_vals"]), color="black", ls="--", lw=1, label="mediana placebo")
        ax.axvline(r["crisis_med"], color="tab:red", lw=2.0, label="crisis (mediana ventanas)")
        ax.axvline(r["crisis_pooled"], color="tab:red", lw=2.0, ls=":", label="crisis (POOLED)")
        ax.axvline(0, color="grey", lw=0.6)
        verdict = "concentrada en crisis" if r["concentrated"] else "UBICUA"
        ax.set_title(f"{r['name']}: joroba lag {LAGS_HUMP}\n"
                     f"crisis pctil {r['pct_pool']:.0f} (POOLED), p={r['p_pool']:.3f} -> {verdict}",
                     fontsize=10)
        ax.set_xlabel("joroba (más negativo = más fuerte)")
        ax.set_ylabel("# ventanas placebo")
        ax.legend(fontsize=7)
    fig.suptitle("P12 — Placebo formal del kernel: ¿la joroba de lag 4-5 es propia de las crisis "
                 "o ubicua?\n(QoQ; rojo = crisis, naranja = expansión, gris = aleatorias no-crisis)",
                 fontsize=12)
    fig.tight_layout()
    out = C.OUTDIR / f"p12_kernel_placebo{tag}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


# --------------------------------------------------------------------------- driver
def run(nplac, tag=""):
    df = load_qoq()
    prof, inv = df["PROF"].to_numpy(), df["INV"].to_numpy()
    n = len(df)
    anchors = crisis_anchors(df)
    # crisis usables (ventana entera con cola de rezagos)
    anchors = [(lbl, a) for lbl, a in anchors if a - HW - KMAX >= 0 and a + HW < n]
    anchor_pos = [a for _, a in anchors]

    print("=== Pieza 12 — placebo formal del kernel / joroba de lag 4-5 ===")
    print(f"serie QoQ: {n} trim, {df.index[0].date()}..{df.index[-1].date()}")
    print(f"ventana objetivo = {2*HW+1} trim (±{HW}) + {KMAX} de cola para rezagos | lags joroba = {LAGS_HUMP}")
    print(f"crisis usables: {len(anchors)} -> {[l for l,_ in anchors]}")

    exp_centers = expansion_centers(anchor_pos, n)
    plac_centers = random_noncrisis_centers(anchor_pos, n, nplac)
    print(f"ventanas: expansión={len(exp_centers)} | aleatorias no-crisis={len(plac_centers)} "
          f"(RNG sembrado seed={SEED})")

    crisis = eval_centers(prof, inv, anchor_pos)
    expans = eval_centers(prof, inv, exp_centers)
    plac = eval_centers(prof, inv, plac_centers)
    _, pooled_hump = pooled_crisis_kernel(prof, inv, anchors)

    # CCF pooled de crisis (mediana de las CCF por ventana de crisis como referencia POOLED)
    pooled_hump_ccf = float(np.median(crisis["hump_ccf"]))

    print("\n################ RESULTADOS ################")
    r_kernel = report_metric("KERNEL β_k", crisis["hump_kernel"], expans["hump_kernel"],
                             plac["hump_kernel"], pooled_hump, lower_is_hump=True)
    r_ccf = report_metric("CCF profits↔investment", crisis["hump_ccf"], expans["hump_ccf"],
                          plac["hump_ccf"], pooled_hump_ccf, lower_is_hump=True)

    out = figure([r_kernel, r_ccf], tag)
    print(f"\n✓ figura: {out}")

    print(f"\n{'='*70}\n  VEREDICTO\n{'='*70}")
    for r in (r_kernel, r_ccf):
        print(f"  {r['name']}: crisis joroba (POOLED) = {r['crisis_pooled']:+.3f}, "
              f"percentil {r['pct_pool']:.0f} en placebo -> "
              + ("CONCENTRADA en crisis (joroba atípicamente negativa)"
                 if r["concentrated"] else
                 "UBICUA (aparece igual fuera de crisis -> NO es firma del régimen de crisis)"))
    return r_kernel, r_ccf


def main():
    smoke = "--smoke" in sys.argv
    if smoke:
        print(">>> SMOKE TEST: pocas ventanas aleatorias <<<\n")
        run(NPLAC_SMOKE, tag="_smoke")
    else:
        run(NPLAC_FULL, tag="")


if __name__ == "__main__":
    main()
