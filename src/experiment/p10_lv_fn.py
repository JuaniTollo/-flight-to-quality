"""Pieza 10 — ¿Un oscilador de FASE FIJA (LV / FN) puede reproducir un KERNEL DE LAG
DISTRIBUIDO?

Hipótesis: el feedback de sobreacumulación (inversión pasada
deprime la ganancia) NO es un retardo determinístico de τ fijo NI un oscilador de fase
fija; es un DELAY ESTOCÁSTICO / KERNEL DE LAG DISTRIBUIDO. La demora τ es una variable
aleatoria con distribución propia que además FLUCTÚA entre crisis.

Patrón empírico (QoQ = pct_change(1), sin 2008/2020) que el modelo debe reproducir:
  - CCF profits<->investment: lag0 = +0.62 (co-movimiento que DOMINA), joroba NEGATIVA
    DISTRIBUIDA en lag4 = -0.32, lag5 = -0.35 (no un pico aislado).
  - Kernel β_k (profits_t ~ Σ investment_{t-k}, pooled QoQ): β0=+0.59, β1=-0.17,
    β4=-0.26, β5=-0.24, resto ~0.
  - Distribución del lag del trough por crisis: media 4.2, sd 0.9, rango [2,5] -> FLUCTÚA.

Este script:
  1. Reconstruye QoQ desde los NIVELES (PROFITS, INVESTMENT) con pct_change(1).
  2. Define ventanas de crisis ±6 trim del fondo de inversión (mismo ancla que p8/p9), QoQ.
  3. Ajusta LV y FN por single-shooting a cada ventana (z-score por ventana).
  4. Genera la trayectoria IMPLICADA por el oscilador ajustado y calcula SU CCF.
  5. Compara CCF/kernel del modelo vs empírico, por crisis y pooled.
  6. PREGUNTA CRÍTICA: ¿un oscilador de fase fija puede dar co-movimiento en lag0 Y joroba
     negativa distribuida en lag4-5 a la vez? ¿Qué lag implica (determinístico, no
     distribuido)? ¿Por qué NO puede representar un lag que fluctúa?

Run:  uv run python -m src.experiment.p10_lv_fn
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from src.experiment import common as C

HW = 6                       # media-ventana: ±6 trim del fondo
WIN = 2 * HW + 1             # 13 puntos
KMAX = 6                     # lags a reportar en la CCF (0..6, donde vive la joroba)
CAP = 1e3
DROP = ("2008", "2020")      # excluidas del análisis empírico honesto

# Patrón empírico de referencia (QoQ, sin 2008/2020)
EMP_CCF = {0: +0.62, 1: None, 2: None, 3: None, 4: -0.32, 5: -0.35, 6: None}
EMP_KERNEL = {0: +0.59, 1: -0.17, 2: 0.0, 3: 0.0, 4: -0.26, 5: -0.24, 6: 0.0}
EMP_TROUGH_LAGS = {"1954": -5, "1958": -4, "1961": -4, "1970": -5, "1975": -2,
                   "1980": -4, "1982": -5, "1991": -4, "2001": -5}


# --------------------------------------------------------------------------- integración
def integ(model, theta, u0, t):
    """RK45 guardado y rápido (mismo criterio que p9_pooled). Devuelve 2×n o None."""
    theta = np.asarray(theta, float)
    if not (np.all(np.isfinite(theta)) and model.guard(theta)):
        return None

    def f(tt, u):
        return model.rhs(u, theta)

    def blow(tt, u):
        return CAP - np.max(np.abs(u))
    blow.terminal = True
    blow.direction = -1

    try:
        sol = solve_ivp(f, (t[0], t[-1]), u0, t_eval=t, method="RK45",
                        rtol=1e-5, atol=1e-7, max_step=1.0, events=blow)
    except Exception:
        return None
    if not sol.success or sol.y.shape[1] != len(t) or not np.all(np.isfinite(sol.y)):
        return None
    return sol.y


# --------------------------------------------------------------------------- datos QoQ
def load_qoq():
    """Reconstruye QoQ = pct_change(1) desde los NIVELES (transform honesto)."""
    df = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    out = pd.DataFrame(index=df.index)
    out["P"] = df["PROFITS"].pct_change(1) * 100
    out["I"] = df["INVESTMENT"].pct_change(1) * 100
    return out.dropna()


def crisis_windows(df):
    """±HW del fondo de inversión por recesión, QoQ, z-score POR ventana."""
    idx = df.index
    p, i = df["P"].to_numpy(), df["I"].to_numpy()
    out = []
    for lbl, pk, tr in C.RECESSIONS:
        pk, tr = pd.Timestamp(pk), pd.Timestamp(tr)
        win = (idx >= pk - pd.offsets.QuarterEnd(2)) & (idx <= tr + pd.offsets.QuarterEnd(4))
        pos = np.where(win)[0]
        if len(pos) == 0:
            continue
        a = pos[np.argmin(i[pos])]
        if a - HW < 0 or a + HW >= len(p):
            continue
        sl = slice(a - HW, a + HW + 1)
        V, R = p[sl].copy(), i[sl].copy()
        V = (V - V.mean()) / V.std()
        R = (R - R.mean()) / R.std()
        out.append((lbl, V, R))
    return out


# --------------------------------------------------------------------------- CCF / kernel
def ccf(p, i, kmax=KMAX):
    """CCF a lags 0..kmax. lag k = corr(profits_t, investment_{t+k}); para la joroba de
    'inversión pasada deprime la ganancia' miramos lag NEGATIVO, i.e. corr(profits_t,
    investment_{t-k}). Reportamos corr(profits_t, investment_{t-k}) para k=0..kmax
    (que es el lag empírico 'lag4=-0.32' del proyecto: inv adelanta a prof por 4)."""
    p = (p - p.mean()) / (p.std() + 1e-12)
    i = (i - i.mean()) / (i.std() + 1e-12)
    m = len(p)
    out = {}
    for k in range(kmax + 1):
        if k == 0:
            out[k] = float(np.corrcoef(p, i)[0, 1])
        else:
            # corr(profits_t , investment_{t-k}): inversión PASADA vs ganancia presente
            out[k] = float(np.corrcoef(p[k:], i[:m - k])[0, 1])
    return out


def kernel_betas(P, I, kmax=KMAX):
    """OLS pooled: profits_t ~ Σ_{k=0..kmax} β_k investment_{t-k}, pares dentro de ventana.
    P, I: listas de arrays (una por ventana). Devuelve dict β_k."""
    X, y = [], []
    for p, i in zip(P, I):
        m = len(p)
        for t in range(kmax, m):
            X.append([i[t - k] for k in range(kmax + 1)])
            y.append(p[t])
    X = np.asarray(X); y = np.asarray(y)
    # estandarizar columnas para comparabilidad de β
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-12)
    yc = (y - y.mean()) / (y.std() + 1e-12)
    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    return {k: float(beta[k]) for k in range(kmax + 1)}


# --------------------------------------------------------------------------- ajuste
def fit_window(model, V, R, restarts=(1.0, 1.2, 0.8, 0.5, 1.5)):
    """Single-shooting a una ventana. Devuelve (theta, sse, y_implied)."""
    t = np.arange(len(V), dtype=float)
    u0 = np.array([V[0], R[0]])

    def loss(theta):
        y = integ(model, theta, u0, t)
        if y is None:
            return 1e8
        return float(np.sum((y[0] - V) ** 2 + (y[1] - R) ** 2))

    best, best_f = model.p0.copy(), np.inf
    for s in restarts:
        res = minimize(loss, model.p0 * s, method="Nelder-Mead",
                       options={"maxiter": 6000, "xatol": 1e-6, "fatol": 1e-8})
        if res.fun < best_f:
            best_f, best = res.fun, res.x
    y = integ(model, best, u0, t)
    return best, best_f, y


def implied_period(model, theta, u0, n=400):
    """Periodo aproximado de la oscilación implícita (trim) vía cruces por cero del estado
    centrado de la 1ra componente. Devuelve None si no hay >=2 cruces ascendentes."""
    t = np.linspace(0, 40, n)
    y = integ(model, theta, u0, t)
    if y is None:
        return None
    x = y[0] - y[0].mean()
    cr = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    if len(cr) < 2:
        return None
    return float(np.mean(np.diff(t[cr])))


# --------------------------------------------------------------------------- main
def main():
    df = load_qoq()
    print(f"QoQ reconstruido desde niveles: {len(df)} trim, {df.index[0].date()}..{df.index[-1].date()}")
    wins = crisis_windows(df)
    use = [(lbl, V, R) for (lbl, V, R) in wins if lbl not in DROP]
    print(f"Ventanas de crisis usables (sin 2008/2020): {[l for l,_,_ in use]} (n={len(use)})\n")

    # ----- 0. CCF EMPÍRICA reconstruida sobre las MISMAS ventanas (sanity) -----
    Pe = [V for _, V, _ in use]
    Ie = [R for _, _, R in use]
    # CCF empírica pooled (pares dentro de ventana)
    def pooled_ccf(P, I, kmax=KMAX):
        out = {}
        for k in range(kmax + 1):
            xs, ys = [], []
            for p, i in zip(P, I):
                m = len(p)
                if k == 0:
                    xs.append(p); ys.append(i)
                else:
                    xs.append(p[k:]); ys.append(i[:m - k])
            xs = np.concatenate(xs); ys = np.concatenate(ys)
            out[k] = float(np.corrcoef(xs, ys)[0, 1])
        return out
    emp_ccf = pooled_ccf(Pe, Ie)
    emp_ker = kernel_betas(Pe, Ie)
    print("=== 0. EMPÍRICO reconstruido (pooled, QoQ, sin 2008/2020, ventanas ±6) ===")
    print("  lag | CCF emp(recon) | CCF emp(ref proyecto) | kernel β(recon) | β(ref)")
    for k in range(KMAX + 1):
        ref_c = EMP_CCF.get(k); ref_b = EMP_KERNEL.get(k)
        print(f"  {k:+d}  |   {emp_ccf[k]:+.2f}        |  "
              f"{('  '+f'{ref_c:+.2f}') if ref_c is not None else '   .  '}              |  "
              f"{emp_ker[k]:+.2f}          | {ref_b:+.2f}")
    print()

    # ----- 1. AJUSTE de cada oscilador y CCF IMPLICADA -----
    rows = []
    fig, axes = plt.subplots(2, len(use), figsize=(2.4 * len(use), 5.2), sharex=True)
    if len(use) == 1:
        axes = axes.reshape(2, 1)
    for col, mdl in enumerate([C.LV, C.FN]):
        Pm, Im = [], []          # trayectorias implícitas (para kernel pooled del modelo)
        periods = []
        print(f"=== 1.{col+1} {mdl.name}: ajuste por crisis y CCF implicada ===")
        print("  crisis | sse  | CCF_mod lag0 | lag4 | lag5 | periodo(trim)")
        for j, (lbl, V, R) in enumerate(use):
            theta, sse, y = fit_window(mdl, V, R)
            if y is None:
                print(f"  {lbl}: ajuste falló")
                continue
            cm = ccf(y[0], y[1])
            per = implied_period(mdl, theta, np.array([V[0], R[0]]))
            if per is not None:
                periods.append(per)
            Pm.append(y[0]); Im.append(y[1])
            rows.append({"model": mdl.name, "crisis": lbl, "sse": sse,
                         **{f"ccf_mod_lag{k}": cm[k] for k in range(KMAX + 1)},
                         "period": per})
            print(f"  {lbl}   | {sse:5.2f}| {cm[0]:+.2f}        | {cm[4]:+.2f}| {cm[5]:+.2f}| "
                  f"{per:.1f}" if per is not None else
                  f"  {lbl}   | {sse:5.2f}| {cm[0]:+.2f}        | {cm[4]:+.2f}| {cm[5]:+.2f}| n/a")
            # plot observado vs implícito (fila por variable)
            ax = axes[0, j]
            ax.plot(V, "k-", lw=1, label="obs")
            ax.plot(y[0], "C0--", lw=1.2, label="mod")
            ax.set_title(lbl, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"profits (z)\n[fila {mdl.name}]" if col == 0 else "profits (z)", fontsize=7)
            ax2 = axes[1, j]
            ax2.plot(R, "k-", lw=1)
            ax2.plot(y[1], "C2--", lw=1.2)
            if j == 0:
                ax2.set_ylabel("investment (z)", fontsize=7)

        # CCF y kernel IMPLICADOS pooled del modelo
        mod_ccf = pooled_ccf(Pm, Im)
        mod_ker = kernel_betas(Pm, Im)
        per_arr = np.array(periods)
        print(f"\n  --- {mdl.name} POOLED implicado (sobre trayectorias ajustadas) ---")
        print("  lag | CCF_mod | CCF_emp | β_mod | β_emp")
        for k in range(KMAX + 1):
            print(f"  {k:+d}  | {mod_ccf[k]:+.2f}   | {emp_ccf[k]:+.2f}   | "
                  f"{mod_ker[k]:+.2f} | {emp_ker[k]:+.2f}")
        if len(per_arr):
            print(f"  periodo implícito: media={per_arr.mean():.1f} trim, sd={per_arr.std():.1f}, "
                  f"rango=[{per_arr.min():.1f},{per_arr.max():.1f}]")
        # firma de "lag determinístico": el lag del mínimo de la CCF del modelo
        argmin_lag = min(range(KMAX + 1), key=lambda k: mod_ccf[k])
        print(f"  lag del MÍNIMO de la CCF del modelo (su 'retardo' implícito): {argmin_lag}")
        rows.append({"model": mdl.name, "crisis": "POOLED", "sse": np.nan,
                     **{f"ccf_mod_lag{k}": mod_ccf[k] for k in range(KMAX + 1)},
                     "period": per_arr.mean() if len(per_arr) else np.nan})
        print()

    fig.suptitle("p10: observado (negro) vs oscilador ajustado (punteado) — ventanas de crisis QoQ",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(C.OUTDIR / "p10_lv_fn_fits.png", dpi=120)
    plt.close(fig)

    # ----- 2. FIGURA: CCF empírica vs implicada por LV y FN -----
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    ks = list(range(KMAX + 1))
    ax.plot(ks, [emp_ccf[k] for k in ks], "ko-", lw=2, label="empírico (QoQ)")
    for mdl_name, c in [("LV", "C0"), ("FN", "C3")]:
        mr = [r for r in rows if r["model"] == mdl_name and r["crisis"] == "POOLED"][0]
        ax.plot(ks, [mr[f"ccf_mod_lag{k}"] for k in ks], f"{c}s--", lw=1.5, label=f"{mdl_name} implicado")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvspan(3.5, 5.5, color="orange", alpha=0.12, label="joroba neg. empírica (lag4-5)")
    ax.set_xlabel("lag k = corr(profits_t, investment_{t-k})")
    ax.set_ylabel("correlación")
    ax.set_title("CCF empírica (lag0+ , joroba neg 4-5) vs implicada por osciladores de fase fija")
    ax.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(C.OUTDIR / "p10_ccf_compare.png", dpi=130)
    plt.close(fig2)

    out = pd.DataFrame(rows)
    out.to_csv(C.OUTDIR / "p10_lv_fn.csv", index=False)
    print(f"✓ figuras: {C.OUTDIR}/p10_lv_fn_fits.png , p10_ccf_compare.png")
    print(f"✓ csv: {C.OUTDIR}/p10_lv_fn.csv")

    # ----- 3. resumen crítico de identificabilidad / distribución de lag -----
    print("\n=== 3. DIAGNÓSTICO: ¿lag fijo vs distribuido? ===")
    print(f"  empírico: trough lag por crisis = {EMP_TROUGH_LAGS} -> media 4.2, sd 0.9, rango [2,5] (FLUCTÚA)")
    for mdl_name in ("LV", "FN"):
        lags = [r["ccf_mod_lag0"] for r in rows]  # placeholder; real argmin below
        mr_pool = [r for r in rows if r["model"] == mdl_name and r["crisis"] == "POOLED"][0]
        mins = []
        for r in rows:
            if r["model"] == mdl_name and r["crisis"] != "POOLED":
                argmin_lag = min(range(KMAX + 1), key=lambda k: r[f"ccf_mod_lag{k}"])
                mins.append(argmin_lag)
        if mins:
            print(f"  {mdl_name}: lag del mínimo de CCF por crisis = {mins} "
                  f"(media {np.mean(mins):.1f}, sd {np.std(mins):.1f})")


if __name__ == "__main__":
    main()
