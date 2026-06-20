"""Pieza 18 — ¿La sobreacumulación es un TIPPING POINT (no-linealidad de umbral)?

Pregunta (la del usuario): el aumento de inversión que detectamos ~5 trimestres antes de la
caída de ganancias, ¿es solo un ADELANTO (lead/lag lineal) o un PUNTO DE BIFURCACIÓN —un umbral
de no-retorno donde, al pasar cierto nivel de acumulación, el efecto negativo se AMPLIFICA?

Una bifurcación formal (autovalores) necesita los parámetros estructurales de estabilidad, que
NO están identificados. Pero el ESPÍRITU —¿hay tipping?— sí es testeable como no-linealidad:

    P_t = β0 + c·I_t + e·P_{t-1} + k·m_t(μ) + γ·(m_t × S_t) + ε

donde m_t = inversión "madurada" (núcleo Gamma, μ≈5 trim) y S_t = estado de ACUMULACIÓN
(media móvil de la inversión, alto = sobreacumulado). Hipótesis de tipping: γ<0 (el efecto
negativo se intensifica cuando la inversión viene alta). Si γ≈0 → es adelanto lineal, no
bifurcación. Errores Newey-West (HAC) por la autocorrelación. Régimen alto vs bajo aparte.

Run:  uv run python -m src.experiment.p18_overaccum_threshold
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.experiment import common as C

SHAPE = 4.0
MU = 5.0          # retardo identificado robusto (~1 año)
KMAX = 10


def gamma_w(mu, K=KMAX):
    s = mu / SHAPE
    x = np.arange(0, K + 1) + 0.5
    w = x ** (SHAPE - 1) * np.exp(-x / s)
    return w / w.sum()


def zscore(v):
    return (v - np.nanmean(v)) / np.nanstd(v)


def main():
    raw = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    P = np.log(raw["PROFITS"]).diff() * 100
    I = np.log(raw["INVESTMENT"]).diff() * 100
    d = pd.DataFrame({"P": P, "I": I}).dropna()
    d = d[d.index.year >= 1948]
    Pv, Iv = zscore(d["P"].to_numpy()), zscore(d["I"].to_numpy())

    w = gamma_w(MU)
    m = np.convolve(Iv, w)[:len(Iv)]                       # inversión madurada (sobreacumulación)
    S = pd.Series(Iv).rolling(8, min_periods=8).mean().to_numpy()  # estado de acumulación (2 años)
    S = zscore(S)

    n = len(Pv)
    lo = KMAX + 1                                          # arranque válido (kernel + rolling)
    idx = np.arange(lo, n)
    y = Pv[idx]
    I0 = Iv[idx]
    Plag = Pv[idx - 1]
    mt = m[idx]
    St = S[idx]
    keep = ~np.isnan(St)
    y, I0, Plag, mt, St = y[keep], I0[keep], Plag[keep], mt[keep], St[keep]
    print(f"=== Pieza 18 — ¿tipping en la sobreacumulación? (log-trim, μ={MU:.0f} trim, n={len(y)}) ===\n")

    def ols_hac(X, names, title):
        Xc = sm.add_constant(X)
        res = sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        print(title)
        for nm, b, se, p in zip(["const"] + names, res.params, res.bse, res.pvalues):
            star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
            print(f"   {nm:<14} {b:+.4f}  (HAC se {se:.4f},  p={p:.3f}) {star}")
        print(f"   R²={res.rsquared:.3f}\n")
        return res

    # (1) lineal: recupera k<0 identificado
    ols_hac(np.column_stack([I0, Plag, mt]), ["I_t", "P_{t-1}", "m_t (k)"],
            "(1) LINEAL  P_t = c·I + e·P_{t-1} + k·m_t :")

    # (2) interacción continua: ¿γ<0?  (m_t × estado de acumulación)
    res2 = ols_hac(np.column_stack([I0, Plag, mt, mt * St]),
                   ["I_t", "P_{t-1}", "m_t (k)", "m_t×S (γ)"],
                   "(2) INTERACCIÓN continua  + γ·(m_t × S_t) :  [γ<0 ⇒ tipping]")
    g, gse, gp = res2.params[-1], res2.bse[-1], res2.pvalues[-1]

    # (3) régimen alto vs bajo (tercil superior de acumulación = sobreacumulado)
    thr = np.quantile(St, 2 / 3)
    hi = St >= thr
    print(f"(3) RÉGIMEN: k de m_t por separado (alto = tercil superior de acumulación, umbral S={thr:+.2f})")
    for lab, msk in [("inversión ALTA (sobreacumulada)", hi), ("inversión normal/baja", ~hi)]:
        Xc = sm.add_constant(np.column_stack([I0[msk], Plag[msk], mt[msk]]))
        r = sm.OLS(y[msk], Xc).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        k, kse, kp = r.params[-1], r.bse[-1], r.pvalues[-1]
        print(f"   {lab:<34} k={k:+.3f}  (HAC se {kse:.3f}, p={kp:.3f},  n={msk.sum()})")

    print("\nLECTURA:")
    if gp < 0.05 and g < 0:
        print(f"  γ={g:+.3f} (p={gp:.3f}) SIGNIFICATIVO y negativo ⇒ el efecto de sobreacumulación se "
              "AMPLIFICA cuando la inversión viene alta: evidencia de TIPPING (umbral).")
    elif g < 0:
        print(f"  γ={g:+.3f} (p={gp:.3f}) negativo pero NO significativo ⇒ la dirección es la del "
              "tipping, pero con n={} la señal débil no alcanza para afirmarlo.".format(len(y)))
    else:
        print(f"  γ={g:+.3f} (p={gp:.3f}) ⇒ sin evidencia de amplificación: el efecto es un ADELANTO "
              "lineal (lead/lag), no un punto de bifurcación.")


if __name__ == "__main__":
    main()
