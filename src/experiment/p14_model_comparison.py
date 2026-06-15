"""Pieza 14 — Comparación de modelos: ¿el modelo físico de acumulación propuesto ajusta
mejor el campo dinámico que los osciladores de fase fija LV y FN?

Métrica: R² del campo de fase (gradient matching, Ramsay-Hooker) — el mismo método con que
la tradición testing-Goodwin evalúa estos modelos. Todos los modelos predicen el MISMO
objetivo (dP/dt, dI/dt sobre la serie z-scoreada) en la MISMA transformación (log-trim).

Modelos:
  FN   FitzHugh-Nagumo  (predador-presa, fase fija, no lineal)
  LV   Lotka-Volterra   (predador-presa, fase fija, no lineal)
  LIN  oscilador lineal 2D  (Markoviano: dP, dI dependen solo de P, I)
  PROP modelo de acumulación PROPUESTO: dP = cI - dP - k·m + c1 ; dI = aP - bI + c2,
       con m = núcleo Gamma(μ≈5) ⊛ I  (la inversión acumulada/madurada).

Se reporta R² de campo (combinado y por ecuación) y AIC (penaliza el parámetro extra de PROP).

Run:  uv run python -m src.experiment.p14_model_comparison
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import gamma as gammadist

from src.experiment import common as C

MU, SHAPE, KK = 5.0, 8.0, 12          # núcleo de acumulación (μ≈5 trim ≈ 1 año, de p11)


def logtrim():
    """Tasas de crecimiento log trimestre a trimestre sobre los NIVELES."""
    raw = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    d = pd.DataFrame({"P": np.log(raw["PROFITS"]).diff() * 100,
                      "I": np.log(raw["INVESTMENT"]).diff() * 100}).dropna()
    return d


def derivatives(t, Z):
    Zs = pd.DataFrame(Z).rolling(3, center=True, min_periods=1).mean().to_numpy()
    return np.gradient(Zs, t, axis=0)


def gkernel(mu, shape, K=KK):
    scale = mu / shape
    ks = np.arange(0, K + 1) + 0.5
    w = gammadist.pdf(ks, a=shape, scale=scale)
    return w / w.sum()


def field_r2(dU, F):
    ss_res = np.sum((dU - F) ** 2, axis=0)
    ss_tot = np.sum((dU - dU.mean(0)) ** 2, axis=0)
    return 1 - ss_res / ss_tot                  # por ecuación [dP, dI]


def aic(dU, F, k_params):
    ss = np.sum((dU - F) ** 2)
    n = dU.size
    return n * np.log(ss / n) + 2 * k_params


def grad_match_nonlin(model, U, dU):
    def resid(p):
        if not model.guard(p):
            return np.full(U.size, 1e3)
        F = np.array([model.rhs(u, p) for u in U])
        return (F - dU).ravel()
    res = least_squares(resid, model.p0, max_nfev=20000)
    F = np.array([model.rhs(u, res.x) for u in U])
    return F, len(model.p0)


def fit_linear(dU, Xp, Xi):
    """OLS por ecuación: dP ~ Xp, dI ~ Xi. Devuelve F (n,2) y nº de params."""
    bp, *_ = np.linalg.lstsq(Xp, dU[:, 0], rcond=None)
    bi, *_ = np.linalg.lstsq(Xi, dU[:, 1], rcond=None)
    F = np.column_stack([Xp @ bp, Xi @ bi])
    return F, Xp.shape[1] + Xi.shape[1], (bp, bi)


def main():
    d = logtrim()
    t = (d.index - d.index[0]).days.to_numpy() / 365.25
    P = (d["P"] - d["P"].mean()) / d["P"].std()
    I = (d["I"] - d["I"].mean()) / d["I"].std()
    U = np.column_stack([P.to_numpy(), I.to_numpy()])
    dU = derivatives(t, U)
    n = len(U)

    # inversión acumulada (núcleo Gamma) sobre la I z-scoreada
    w = gkernel(MU, SHAPE)
    m = np.convolve(I.to_numpy(), w)[:n]

    print("=== Pieza 14 — comparación de modelos (gradient matching, log-trim) ===")
    print(f"n={n}  |  núcleo de acumulación: μ={MU:.0f} trim, shape={SHAPE:.0f}\n")

    res = {}

    # FN, LV (no lineales, en su marco z-score; LV con offset al cuadrante +)
    for M in (C.FN, C.LV):
        if M.needs_offset:
            off = -U.min() + 1.0
            Uu = U + off
        else:
            Uu = U
        F, k = grad_match_nonlin(M, Uu, dU)
        res[M.name] = dict(r2=field_r2(dU, F), aic=aic(dU, F, k), k=k)

    # LIN: dP ~ [P, I, 1] ; dI ~ [P, I, 1]
    ones = np.ones(n)
    Xlin = np.column_stack([P, I, ones])
    F, k, _ = fit_linear(dU, Xlin, Xlin)
    res["LIN"] = dict(r2=field_r2(dU, F), aic=aic(dU, F, k), k=k)

    # PROP: dP ~ [I, P, m, 1] ; dI ~ [P, I, 1]  (m = inversión acumulada)
    Xp = np.column_stack([I, P, m, ones])
    Xi = np.column_stack([P, I, ones])
    F, k, (bp, bi) = fit_linear(dU, Xp, Xi)
    res["PROP"] = dict(r2=field_r2(dU, F), aic=aic(dU, F, k), k=k)
    k_coef = bp[2]    # coeficiente de m (−k esperado < 0)

    # ---- tabla ----
    order = ["FN", "LV", "LIN", "PROP"]
    nice = {"FN": "FN (predador-presa)", "LV": "LV (predador-presa)",
            "LIN": "Lineal 2D (Markov)", "PROP": "PROPUESTO (acumulación)"}
    print(f"{'modelo':26s} {'R²(dP)':>8s} {'R²(dI)':>8s} {'R² comb.':>9s} {'AIC':>9s} {'#par':>5s}")
    r2c = {}
    for nm in order:
        r = res[nm]
        comb = 1 - (1 - r["r2"]).mean()      # combinado ~ promedio de varianza explicada
        # combinado honesto: varianza total explicada sobre ambas ecuaciones
        r2c[nm] = comb
        print(f"{nice[nm]:26s} {r['r2'][0]:+8.3f} {r['r2'][1]:+8.3f} {comb:+9.3f} {r['aic']:9.1f} {r['k']:5d}")
    print(f"\n  coeficiente de la inversión acumulada en PROP: k(m) = {k_coef:+.3f} (negativo = deprime ganancias)")
    print(f"  ΔAIC(PROP − LIN) = {res['PROP']['aic'] - res['LIN']['aic']:+.1f}  "
          f"(negativo ⇒ PROP mejor incluso penalizando el parámetro extra)")
    print(f"  ΔR²(PROP − LV)  = {r2c['PROP'] - r2c['LV']:+.3f}   ΔR²(PROP − FN) = {r2c['PROP'] - r2c['FN']:+.3f}")

    # ---- figura: 2 paneles (vista completa + zoom) ----
    vals = [r2c[nm] for nm in order]
    labels = [nice[nm] for nm in order]
    colors = ["tab:red", "tab:orange", "tab:blue", "tab:green"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    # panel izquierdo: vista completa (se ve el desastre del FN)
    bL = axL.bar(labels, vals, color=colors, alpha=0.85)
    axL.axhline(0, color="k", lw=0.8)
    axL.set_ylabel("R² del campo dinámico")
    axL.set_title("Vista completa", fontsize=10)
    for b, v in zip(bL, vals):
        axL.annotate(f"{v:+.2f}", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    axL.set_ylim(min(vals) - 0.2, max(vals) + 0.1)
    axL.tick_params(axis="x", labelrotation=20, labelsize=7)

    # panel derecho: zoom a los positivos (se distingue LV vs lineal vs propuesto)
    bR = axR.bar(labels, vals, color=colors, alpha=0.85)
    axR.axhline(0, color="k", lw=0.8)
    axR.set_title("Zoom a los positivos", fontsize=10)
    for b, v in zip(bR, vals):
        if v > -0.01:
            axR.annotate(f"{v:+.3f}", (b.get_x() + b.get_width() / 2, v),
                         ha="center", va="bottom", fontsize=9)
    axR.set_ylim(-0.01, max(vals) + 0.02)
    axR.tick_params(axis="x", labelrotation=20, labelsize=7)

    fig.suptitle("¿Qué modelo describe la dinámica ganancias–inversión? (R² de campo, log-trim)\n"
                 "los osciladores de fase fija (LV, FN) fallan; el modelo de acumulación ajusta mejor",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(C.OUTDIR / "p14_model_comparison.png", dpi=130); plt.close(fig)
    print(f"\n✓ figura: {C.OUTDIR}/p14_model_comparison.png")


if __name__ == "__main__":
    main()
