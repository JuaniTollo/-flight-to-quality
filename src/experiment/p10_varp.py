"""Pieza 10 — VAR(p) como KERNEL DE LAG DISTRIBUIDO (profit-investment, QoQ).

Pregunta: el feedback de sobreacumulación (inversión
pasada -> deprime ganancia) NO es un retardo determinístico de τ fijo ni un oscilador
de fase fija, sino un KERNEL DE LAG DISTRIBUIDO (la demora τ es una variable aleatoria
que además fluctúa entre crisis). Un VAR(p) es, por construcción, un modelo de lag
distribuido: la ecuación de profits regresa profits_t sobre {profits_{t-k}, investment_{t-k}}
para k=1..p. Los coeficientes de investment_{t-k} SON un kernel de lag estimado libremente.

Este script evalúa SOLO la clase VAR(p), p=1..6:
  - Ajusta en la serie completa (QoQ) y en las ventanas de crisis (±6 trim del fondo,
    pooled apilando ventanas pero SIN cruzar bordes de ventana al armar los lags).
  - Reporta el KERNEL DE LAG implícito = coef. de investment_{t-k} en la ec. de profits.
  - Reporta la IRF (respuesta de profits a un shock en investment, orden Cholesky
    [investment, profits] -> investment puede mover profits contemporáneamente).
  - Reporta AIC/BIC por p (selección + alerta de sobreajuste con n chico).
  - Compara con el kernel empírico: β0=+0.59 (contemp.), joroba neg en k=4-5
    (β1=-0.17, β4=-0.26, β5=-0.24).

CAVEAT CENTRAL (honesto): un VAR REDUCIDO tiene lags k=1..p, NO un término de lag 0.
El co-movimiento contemporáneo (lag0=+0.62, el término que DOMINA el kernel empírico)
NO aparece como coeficiente: vive en la covarianza de residuos (correlación contemp.).
Por eso reportamos también esa correlación y la IRF con identificación Cholesky.

Run:  uv run python -m src.experiment.p10_varp
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.tsa.api import VAR

from src.experiment import common as C

HW = 6                       # media-ventana de crisis: ±6 trim del fondo (mismo ancla que p8/p9)
WIN = 2 * HW + 1
PMAX = 6
DROP = ("2008", "2020")      # crisis extremas: las saco para el bloque "cerca de crisis"

# kernel empírico (QoQ, sin 2008/2020) — referencia para comparar
EMP_KERNEL = {0: +0.59, 1: -0.17, 2: 0.0, 3: 0.0, 4: -0.26, 5: -0.24}
EMP_CCF = {0: +0.62, 4: -0.32, 5: -0.35}


# --------------------------------------------------------------------------- datos QoQ
def load_qoq():
    """QoQ = pct_change(1) sobre los NIVELES PROFITS, INVESTMENT (transform honesto)."""
    df = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    q = pd.DataFrame({
        "profits": df["PROFITS"].pct_change(1) * 100.0,
        "investment": df["INVESTMENT"].pct_change(1) * 100.0,
    }).dropna()
    return df, q


def crisis_anchors(df_levels):
    """Ancla por recesión = inversión (NIVEL) más deprimida en su vecindad NBER."""
    idx = df_levels.index
    inv = df_levels["INVESTMENT"].to_numpy()
    out = []
    for lbl, pk, tr in C.RECESSIONS:
        pk, tr = pd.Timestamp(pk), pd.Timestamp(tr)
        win = (idx >= pk - pd.offsets.QuarterEnd(2)) & (idx <= tr + pd.offsets.QuarterEnd(4))
        pos = np.where(win)[0]
        if len(pos) == 0:
            continue
        a = pos[np.argmin(inv[pos])]
        out.append((lbl, idx[a]))
    return out


def crisis_windows(df_levels, q, drop_labels=DROP):
    """Lista de DataFrames QoQ (profits, investment), uno por crisis, ±HW del fondo.
    El fondo se ubica en NIVEL de inversión; la ventana se recorta sobre la serie QoQ."""
    anchors = crisis_anchors(df_levels)
    qidx = q.index
    wins = []
    for lbl, anchor_date in anchors:
        if lbl in drop_labels:
            continue
        # posición del ancla dentro del índice QoQ
        if anchor_date not in qidx:
            # el primer trimestre se pierde por pct_change; busco el más cercano
            loc = qidx.get_indexer([anchor_date], method="nearest")[0]
        else:
            loc = qidx.get_loc(anchor_date)
        if loc - HW < 0 or loc + HW >= len(q):
            continue
        w = q.iloc[loc - HW: loc + HW + 1].copy()
        wins.append((lbl, w))
    return wins


# --------------------------------------------------------------------------- kernel de lag
def profit_eq_kernel(res, p):
    """Extrae el kernel de lag: coef. de investment_{t-k}, k=1..p, en la ec. de profits.
    statsmodels VARResults.coefs: shape (p, k, k) = coefs[lag-1][eq, var].
    eq=0 -> profits ; var=1 -> investment (orden de columnas = [profits, investment])."""
    coefs = res.coefs                       # (p, 2, 2)
    # también el self-lag de profits (coef de profits_{t-k} en la ec de profits)
    inv_on_prof = {k: float(coefs[k - 1, 0, 1]) for k in range(1, p + 1)}
    prof_on_prof = {k: float(coefs[k - 1, 0, 0]) for k in range(1, p + 1)}
    return inv_on_prof, prof_on_prof


def resid_contemp_corr(res):
    """Correlación contemporánea de los residuos = el 'lag0' que un VAR reducido NO
    pone como coeficiente. Es el análogo VAR del β0 empírico (+0.59 / CCF +0.62)."""
    sigma = res.sigma_u
    s = np.asarray(sigma)
    return float(s[0, 1] / np.sqrt(s[0, 0] * s[1, 1]))


def irf_profits_from_invest(res, horizon=12):
    """IRF ortogonalizada (Cholesky) de profits ante un shock de 1 sd en investment.
    Orden de Cholesky = [investment, profits]: investment puede impactar profits en h=0,
    NO al revés. Esto hace que h=0 capture el co-movimiento contemporáneo (β0 empírico).
    Devuelve array (horizon+1,)."""
    # reordeno columnas a [investment, profits] para el orden causal deseado
    irf = res.irf(horizon)
    # irf.orth_irfs: (horizon+1, k, k) = [h, response_var, impulse_var]
    # con columnas [profits, investment]: response profits=0, impulse investment=1
    orth = irf.orth_irfs
    return orth[:, 0, 1]


# --------------------------------------------------------------------------- ajuste full
def fit_full(q):
    """VAR(p) p=1..6 sobre la serie QoQ completa. Devuelve dict por p."""
    print("\n" + "=" * 78)
    print("  A. SERIE COMPLETA (QoQ, 1948–2026) — VAR(p), columnas [profits, investment]")
    print("=" * 78)
    print(f"  n obs QoQ = {len(q)}")
    model = VAR(q[["profits", "investment"]])
    out = {}
    for p in range(1, PMAX + 1):
        res = model.fit(p)
        inv_k, prof_k = profit_eq_kernel(res, p)
        rho0 = resid_contemp_corr(res)
        irf = irf_profits_from_invest(res, horizon=12)
        out[p] = dict(res=res, inv_k=inv_k, prof_k=prof_k, rho0=rho0, irf=irf,
                      aic=res.aic, bic=res.bic, nobs=res.nobs)
    return out


def fit_crisis_pooled(wins):
    """VAR(p) 'pooled' sobre las ventanas de crisis. Apila las ventanas pero el filtro
    de lags de statsmodels cruzaría bordes; para evitarlo, construyo a mano la matriz de
    regresión (Y, X) emparejando t y t-1..t-p SOLO dentro de cada ventana, y estimo por
    OLS por ecuación. Devuelve dict por p con el kernel y AIC/BIC pooled.

    Nota honesta: n efectivo por ventana = WIN - p; con p grande casi no quedan filas.
    """
    cols = ["profits", "investment"]
    print("\n" + "=" * 78)
    print(f"  B. VENTANAS DE CRISIS pooled (±{HW} trim del fondo, sin {DROP}) — OLS por ecuación")
    print("=" * 78)
    print(f"  ventanas usables = {len(wins)} -> {[l for l, _ in wins]}  ({WIN} pts c/u)")
    out = {}
    for p in range(1, PMAX + 1):
        Yrows, Xrows = [], []
        for lbl, w in wins:
            M = w[cols].to_numpy()          # (WIN, 2)
            n = len(M)
            for t in range(p, n):           # SOLO pares dentro de la ventana
                lags = M[t - 1:t - p - 1:-1] if p > 1 else M[t - 1:t]
                lags = M[t - p:t][::-1]      # orden lag1, lag2, ..., lagp
                Xrows.append(np.concatenate([[1.0], lags.ravel()]))
                Yrows.append(M[t])
        X = np.array(Xrows)                  # (N, 1 + 2p)
        Y = np.array(Yrows)                  # (N, 2)
        N = len(Y)
        if N <= X.shape[1]:
            out[p] = dict(insufficient=True, N=N, ncoef=X.shape[1])
            continue
        # OLS por ecuación: B = (X'X)^-1 X'Y
        XtX = X.T @ X
        B = np.linalg.solve(XtX, X.T @ Y)   # (1+2p, 2)
        resid = Y - X @ B
        # kernel: ec de profits = columna 0 de Y. Coef de investment_{t-k}:
        # X col layout: [const, lag1_prof, lag1_inv, lag2_prof, lag2_inv, ...]
        inv_k, prof_k = {}, {}
        for k in range(1, p + 1):
            prof_k[k] = float(B[1 + 2 * (k - 1) + 0, 0])
            inv_k[k] = float(B[1 + 2 * (k - 1) + 1, 0])
        # AIC/BIC pooled (gaussiano, det de cov residual)
        Sig = (resid.T @ resid) / N
        kparams = X.shape[1] * 2            # coefs totales (ambas ecuaciones)
        ll_det = np.log(np.linalg.det(Sig) + 1e-300)
        aic = ll_det + 2.0 * kparams / N
        bic = ll_det + np.log(N) * kparams / N
        rho0 = float(Sig[0, 1] / np.sqrt(Sig[0, 0] * Sig[1, 1]))
        out[p] = dict(inv_k=inv_k, prof_k=prof_k, rho0=rho0, aic=aic, bic=bic,
                      N=N, ncoef=X.shape[1])
    return out


# --------------------------------------------------------------------------- reporte
def print_kernel_table(out, label, has_irf):
    print(f"\n  --- {label}: kernel de lag (coef de investment_{{t-k}} en ec. de profits) ---")
    print(f"  {'p':>2} | {'AIC':>7} | {'BIC':>7} | {'ρ0(resid)':>9} | "
          f"coef invest_{{t-k}}  (k=1..p)")
    print("  " + "-" * 76)
    for p in range(1, PMAX + 1):
        d = out[p]
        if d.get("insufficient"):
            print(f"  {p:>2} | datos insuficientes (N={d['N']} <= ncoef={d['ncoef']})")
            continue
        ks = " ".join(f"k{k}={d['inv_k'][k]:+.2f}" for k in range(1, p + 1))
        print(f"  {p:>2} | {d['aic']:7.3f} | {d['bic']:7.3f} | {d['rho0']:+9.2f} | {ks}")


def compare_to_empirical(out_full, out_cri, irf_best_p):
    print("\n" + "=" * 78)
    print("  C. COMPARACIÓN CON EL KERNEL EMPÍRICO (QoQ, sin 2008/2020)")
    print("=" * 78)
    print("  Kernel empírico (β_k de profits_t ~ Σ investment_{t-k}, pooled):")
    print("    β0=+0.59 (CONTEMPORÁNEO, domina)  β1=-0.17  β4=-0.26  β5=-0.24  resto~0")
    print("    CCF cerca de crisis: lag0=+0.62, lag4=-0.32, lag5=-0.35")
    print()
    print("  PUNTO CLAVE: un VAR REDUCIDO no tiene coef de lag0. El co-movimiento")
    print("  contemporáneo (el término que DOMINA) vive en ρ(residuos), no en el kernel.")
    print()
    print("  ¿Capta la JOROBA NEGATIVA en lag 4-5?")
    for src, out in [("FULL", out_full), ("CRISIS", out_cri)]:
        for p in (5, 6):
            d = out.get(p, {})
            if not d or d.get("insufficient"):
                print(f"   {src} p={p}: n/d")
                continue
            k4 = d["inv_k"].get(4, np.nan)
            k5 = d["inv_k"].get(5, np.nan)
            sign4 = "NEG✓" if k4 < 0 else "pos✗"
            sign5 = "NEG✓" if k5 < 0 else "pos✗"
            print(f"   {src} p={p}: k4={k4:+.2f} ({sign4})  k5={k5:+.2f} ({sign5})  "
                  f"ρ0={d['rho0']:+.2f}")


def plot_kernels(out_full, out_cri):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (0,0) kernel full por p
    ax = axes[0, 0]
    ks_emp = sorted(EMP_KERNEL)
    ax.bar([k - 0.15 for k in ks_emp], [EMP_KERNEL[k] for k in ks_emp], width=0.3,
           color="black", alpha=0.6, label="empírico β_k")
    for p, c in [(5, "tab:blue"), (6, "tab:orange")]:
        d = out_full[p]
        ks = sorted(d["inv_k"])
        ax.plot(ks, [d["inv_k"][k] for k in ks], "-o", color=c, ms=5, label=f"VAR({p}) full")
    ax.axhline(0, color="grey", lw=0.6)
    ax.scatter([0], [out_full[5]["rho0"]], color="tab:blue", marker="*", s=120,
               zorder=5, label="ρ0 resid (VAR5)")
    ax.set_title("Kernel de lag — SERIE COMPLETA vs empírico\n"
                 "(VAR no tiene coef lag0; ρ0=contemp en residuos)", fontsize=10)
    ax.set_xlabel("lag k (trimestres)"); ax.set_ylabel("coef invest_{t-k} en ec. profits")
    ax.legend(fontsize=8)

    # (0,1) kernel crisis por p
    ax = axes[0, 1]
    ax.bar([k - 0.15 for k in ks_emp], [EMP_KERNEL[k] for k in ks_emp], width=0.3,
           color="black", alpha=0.6, label="empírico β_k")
    for p, c in [(5, "tab:green"), (6, "tab:red")]:
        d = out_cri.get(p, {})
        if d and not d.get("insufficient"):
            ks = sorted(d["inv_k"])
            ax.plot(ks, [d["inv_k"][k] for k in ks], "-o", color=c, ms=5,
                    label=f"VAR({p}) crisis")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title("Kernel de lag — VENTANAS DE CRISIS (pooled) vs empírico", fontsize=10)
    ax.set_xlabel("lag k"); ax.set_ylabel("coef invest_{t-k}")
    ax.legend(fontsize=8)

    # (1,0) IRF full
    ax = axes[1, 0]
    for p, c in [(1, "grey"), (2, "tab:purple"), (4, "tab:cyan"),
                 (5, "tab:blue"), (6, "tab:orange")]:
        irf = out_full[p]["irf"]
        ax.plot(range(len(irf)), irf, "-o", ms=3, color=c, label=f"VAR({p})")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title("IRF: respuesta de profits a shock en investment (Cholesky)\n"
                 "h=0 capta co-movimiento; ¿hay caída en h=4-5?", fontsize=10)
    ax.set_xlabel("horizonte h (trimestres)"); ax.set_ylabel("respuesta profits")
    ax.legend(fontsize=8)

    # (1,1) AIC/BIC full
    ax = axes[1, 1]
    ps = list(range(1, PMAX + 1))
    ax.plot(ps, [out_full[p]["aic"] for p in ps], "-o", label="AIC full")
    ax.plot(ps, [out_full[p]["bic"] for p in ps], "-s", label="BIC full")
    ax.set_title("Selección de orden (serie completa)\nBIC penaliza p grande", fontsize=10)
    ax.set_xlabel("p"); ax.set_ylabel("criterio (menor=mejor)")
    ax.legend(fontsize=8)

    fig.suptitle("VAR(p) como kernel de lag distribuido — profit-investment QoQ",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = C.OUTDIR / "p10_varp_kernel.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def dump_csv(out_full, out_cri):
    rows = []
    for src, out in [("full", out_full), ("crisis", out_cri)]:
        for p in range(1, PMAX + 1):
            d = out.get(p, {})
            if not d or d.get("insufficient"):
                continue
            row = dict(source=src, p=p, aic=d["aic"], bic=d["bic"], rho0=d["rho0"],
                       N=d.get("nobs", d.get("N")))
            for k in range(1, p + 1):
                row[f"invk{k}"] = d["inv_k"][k]
            rows.append(row)
    path = C.OUTDIR / "p10_varp_kernel.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main():
    df_levels, q = load_qoq()
    out_full = fit_full(q)
    print_kernel_table(out_full, "SERIE COMPLETA", has_irf=True)

    # IRF resumen full
    print("\n  --- IRF (respuesta de profits a shock invest, Cholesky [invest,profits]) ---")
    print(f"  {'p':>2} | " + " ".join(f"h{h:>2}" for h in range(0, 9)))
    for p in range(1, PMAX + 1):
        irf = out_full[p]["irf"]
        print(f"  {p:>2} | " + " ".join(f"{irf[h]:+.2f}" for h in range(0, 9)))
    print("  (h=0 = impacto contemporáneo identificado por Cholesky)")

    wins = crisis_windows(df_levels, q)
    out_cri = fit_crisis_pooled(wins)
    print_kernel_table(out_cri, "VENTANAS DE CRISIS (pooled)", has_irf=False)

    compare_to_empirical(out_full, out_cri, irf_best_p=5)

    fig = plot_kernels(out_full, out_cri)
    csv = dump_csv(out_full, out_cri)
    print(f"\n  figura -> {fig}")
    print(f"  csv    -> {csv}")

    # selección de orden
    bic_full = {p: out_full[p]["bic"] for p in range(1, PMAX + 1)}
    aic_full = {p: out_full[p]["aic"] for p in range(1, PMAX + 1)}
    print("\n" + "=" * 78)
    print("  SELECCIÓN DE ORDEN (serie completa)")
    print("=" * 78)
    print(f"  BIC mínimo en p={min(bic_full, key=bic_full.get)}  |  "
          f"AIC mínimo en p={min(aic_full, key=aic_full.get)}")


if __name__ == "__main__":
    main()
