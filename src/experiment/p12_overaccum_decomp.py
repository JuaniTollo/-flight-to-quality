"""Pieza 12 — DESCOMPOSICIÓN: ¿cuánto de la CAÍDA de ganancias explica la sobreacumulación?

El placebo (p12_kernel_placebo) mostró que la joroba de lag 4-5 es UBICUA, no firma de
crisis. Esta pieza responde la pregunta económica que SÍ importa: el término de
sobreacumulación +k·m con k<0 (inversión madurada ~1 año atrás deprime ganancias), ¿explica algo
de la caída de ganancias, por encima del co-movimiento contemporáneo? ¿O es despreciable?

Datos: idénticos al mejor experimento (p12_pinn_fourier) — niveles PROFITS/INVESTMENT,
QoQ = pct_change(1), ventana 1990–, z-score. dP/dt por diferencias centradas (como el PINN).

Modelo físico (mismo de p11/p12):  dP/dt = c·I − d·P + k·m,  m = Σ_j w_j(μ)·I_{t−j},
w = kernel Gamma media μ, shape=4. Lag del PINN Fourier: μ ≈ 3.9 trim.

Test (regresión jerárquica, OLS transparente y atribuible):
  M0 (contemporáneo):  dP ~ I, P              [solo demanda actual + decay]
  M1 (+ sobreacum.):   dP ~ I, P, m           [añade inversión madurada]
  ΔR² = aporte marginal de la sobreacumulación a EXPLICAR la dinámica de ganancias.
  signo y significancia de β_m (esperado < 0 = deprime), t y F parcial.
  Barrido del lag μ → ¿el aporte se concentra cerca de ~4 trim (la joroba)?
  En el peor decil de CAÍDAS (dP más negativo): contribución media de cada término.

Corre:  PYTHONPATH=. .venv/bin/python -m src.experiment.p12_overaccum_decomp
"""
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSVP = REPO / "data" / "processed" / "lotka_volterra.csv"
SHAPE, KMAX = 4.0, 10
MU_PINN = 3.904          # lag de maduración del p12_pinn_fourier (real, QoQ)


def gamma_kernel(mu, shape=SHAPE, kmax=KMAX):
    xs = np.arange(0, kmax + 1) + 0.5
    w = xs ** (shape - 1) * np.exp(-xs / (mu / shape))
    return w / w.sum()


def load_window(from_year=1990):
    df = pd.read_csv(CSVP)
    df = df.dropna(subset=["PROFITS", "INVESTMENT"]).reset_index(drop=True)
    P = 100 * np.log(df["PROFITS"]).diff()      # QoQ
    I = 100 * np.log(df["INVESTMENT"]).diff()
    yr = df["DATE"].str[:4].astype(int)
    keep = (yr >= from_year) & P.notna() & I.notna()
    z = lambda v: (v - v.mean()) / v.std()
    return z(P[keep]).to_numpy(), z(I[keep]).to_numpy()


def conv_m(I, mu):
    w = gamma_kernel(mu)
    n = len(I)
    return np.array([sum(w[j] * I[i - j] for j in range(0, min(KMAX, i) + 1)) for i in range(n)])


def ols(y, X):
    """OLS con intercepto. Devuelve coef, R², RSS, t-stats."""
    Xd = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    rss = resid @ resid
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - rss / tss
    n, k = Xd.shape
    sigma2 = rss / (n - k)
    cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    return beta, r2, rss, t, n, k


def main():
    P, I = load_window()
    n = len(P)
    # dP/dt por diferencias centradas (como nn_dudt del PINN); puntos interiores con historia
    dP = (P[2:] - P[:-2]) / 2.0
    lo, hi = KMAX, n - 1            # i interior con historia de kernel y derivada central
    idx = np.arange(lo, hi)
    dPi = dP[idx - 1]              # dP en el índice interior (dP indexado desde 1)
    Ii, Pi = I[idx], P[idx]

    print("=" * 72)
    print("DESCOMPOSICIÓN — aporte de la sobreacumulación (+k·m, k<0) a la dinámica de ganancias")
    print("=" * 72)
    print(f"ventana 1990– │ {n} trim QoQ z-score │ {len(idx)} puntos interiores │ lag PINN μ={MU_PINN:.2f}")

    m = conv_m(I, MU_PINN)[idx]
    # M0 contemporáneo vs M1 con sobreacumulación
    b0, r2_0, rss0, t0, n0, k0 = ols(dPi, np.column_stack([Ii, Pi]))
    b1, r2_1, rss1, t1, n1, k1 = ols(dPi, np.column_stack([Ii, Pi, m]))
    dR2 = r2_1 - r2_0
    F = (rss0 - rss1) / (rss1 / (n1 - k1))
    print(f"\n  M0  dP ~ I,P         : R² = {r2_0:.3f}")
    print(f"  M1  dP ~ I,P,m       : R² = {r2_1:.3f}   (ΔR² = {dR2:+.3f})")
    print(f"  β_m (sobreacum.)     = {b1[3]:+.3f}   t = {t1[3]:+.2f}   F-parcial = {F:.2f}")
    sig = "SIGNIFICATIVO" if abs(t1[3]) > 1.96 else "NO significativo (|t|<1.96)"
    sign = "deprime ganancias (signo correcto)" if b1[3] < 0 else "POSITIVO (signo CONTRARIO al esperado)"
    print(f"  → β_m {sign}; {sig}")

    # Barrido del lag μ: ¿dónde se maximiza el aporte?
    print(f"\n  Barrido del lag μ (ΔR² del término m sobre el contemporáneo):")
    mus = np.arange(1.0, 9.1, 0.5)
    dr2s = []
    for mu in mus:
        mm = conv_m(I, mu)[idx]
        _, r2m, _, _, _, _ = ols(dPi, np.column_stack([Ii, Pi, mm]))
        dr2s.append(r2m - r2_0)
    dr2s = np.array(dr2s)
    best = mus[np.argmax(dr2s)]
    for mu, d in zip(mus, dr2s):
        bar = "#" * int(max(0, d) * 400)
        mark = " ←máx" if mu == best else (" (PINN)" if abs(mu - MU_PINN) < 0.3 else "")
        print(f"    μ={mu:4.1f} trim │ ΔR²={d:+.4f} {bar}{mark}")
    print(f"  → aporte máximo en μ={best:.1f} trim (PINN Fourier dio {MU_PINN:.1f})")

    # Caídas de ganancias: peor decil de dP. Contribución media de cada término (coef M1).
    cont_I = b1[1] * Ii
    cont_P = b1[2] * Pi
    cont_m = b1[3] * m
    worst = dPi <= np.percentile(dPi, 10)
    print(f"\n  PEOR DECIL DE CAÍDAS de ganancias ({worst.sum()} trim, dP medio={dPi[worst].mean():+.3f}):")
    print(f"    contrib. media  demanda c·I = {cont_I[worst].mean():+.3f}")
    print(f"    contrib. media  decay −d·P  = {cont_P[worst].mean():+.3f}")
    print(f"    contrib. media  SOBREACUM m = {cont_m[worst].mean():+.3f}  "
          f"({100*cont_m[worst].mean()/dPi[worst].mean():.0f}% de la caída media)")
    # correlación del término de sobreacumulación con las caídas (todas)
    r = np.corrcoef(cont_m, dPi)[0, 1]
    print(f"\n  corr(término sobreacum., dP) en toda la ventana = {r:+.3f}")
    print("=" * 72)
    print("LECTURA: ΔR² = cuánta varianza EXTRA de la dinámica de ganancias explica la")
    print("sobreacumulación por encima del co-movimiento contemporáneo. β_m<0 y |t|>1.96")
    print("→ el canal existe y es significativo; ΔR² chico → es modesto en magnitud.")


if __name__ == "__main__":
    main()
