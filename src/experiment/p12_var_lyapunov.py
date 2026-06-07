"""Pieza 12 — CCF TEÓRICA de un VAR(p) vía Lyapunov: ¿"kernel de delay" o "lineal de
memoria larga"?

Pregunta abierta (P10, próximos pasos #2). La joroba negativa empírica en lag 4-5 de la
CCF profits↔investment (QoQ) se interpretó como un KERNEL DE LAG DISTRIBUIDO (delay de
maduración). Pero un VAR(p≥4) lineal y estacionario es, por construcción, un proceso
gaussiano de memoria finita cuya función de autocovarianza de 2º orden está totalmente
determinada por los coeficientes. La prueba decisiva:

    Ajustar VAR(p), derivar la CCF TEÓRICA implicada por los coeficientes
    (NO simulando, sino resolviendo la ecuación de Lyapunov para la covarianza
    estacionaria Γ(0) y propagando Γ(h)=A·Γ(h-1) en la forma de estados / companion),
    y ver si reproduce la joroba negativa empírica en lag 4-5.

Si la CCF de 2º orden de un VAR(p) lineal-estocástico ya genera la joroba, entonces
"kernel de delay" y "sistema lineal estocástico de memoria larga" son OBSERVACIONALMENTE
EQUIVALENTES a nivel de la CCF: la joroba no es evidencia de un mecanismo de delay por
encima de un VAR lineal. Es la versión "Lyapunov" del descargo de P10(f)#2.

Por qué Lyapunov y NO el VAR(1) de |λ| chico
---------------------------------------------
Un VAR(1) con autovalores pequeños tiene una CCF que decae geométricamente (half-life < 1
trim): no puede tener masa negativa aislada en lag 4-5 (es lo que refutó P10, modelo #6).
La pregunta es si un VAR(p) de orden ALTO (p≥4), que sí tiene memoria de p trimestres en
sus coeficientes, reproduce la joroba SIN postular un kernel de delay explícito. La forma
de estados (companion) convierte el VAR(p) en un VAR(1) de dimensión 2p; ahí la covarianza
estacionaria se obtiene exacta de la ecuación de Lyapunov discreta.

Matemática
----------
VAR(p):  y_t = Σ_{k=1..p} A_k y_{t-k} + ε_t ,  Cov(ε)=Σ_u  (y_t = [profits, investment]).
Companion (estado z_t = [y_t; y_{t-1}; ...; y_{t-p+1}], dim 2p):
    z_t = F z_{t-1} + η_t ,  Cov(η)=Q  (Σ_u en el bloque superior-izquierdo, 0 resto).
Covarianza estacionaria del estado:  Γ_z(0) = F Γ_z(0) Fᵀ + Q   (Lyapunov discreta).
Autocovarianza del estado a lag h≥0:  Γ_z(h) = Fʰ Γ_z(0).
La autocovarianza de y a lag h se lee en el bloque (0,0) de Γ_z(h):
    Γ_y(h) = E[y_t y_{t-h}ᵀ]  = bloque 2×2 superior-izquierdo de Γ_z(h).
CCF teórica (misma convención que P7/p10):  corr(profits[t], investment[t+k]):
    k≥0  (profits lidera):  Γ_y(k)[profits, investment] / (σ_p σ_i)
    k<0  (investment lidera): Γ_y(-k)[investment, profits] / (σ_p σ_i)
  donde σ_p²=Γ_y(0)[0,0], σ_i²=Γ_y(0)[1,1].

Smoke test:  uv run python -m src.experiment.p12_var_lyapunov --smoke   (un solo p)
Full:        uv run python -m src.experiment.p12_var_lyapunov           (p=2..6)
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.api import VAR

from src.experiment import common as C

KMAX = 8                         # rango de lags de la CCF a reportar
PMIN, PMAX = 2, 6
SMOKE_P = 5                      # un p representativo (p≥4, donde vive la joroba)

# Joroba empírica de referencia (QoQ, serie completa) — recomputada en load_qoq()/empirical_ccf
# para no hardcodear; estos son los valores esperados (ver encabezado y P10 §d):
#   lag0=+0.49  lag-4=-0.23  lag-5=-0.21   (investment lidera 4-5 trim -> deprime profits)


def load_qoq():
    """QoQ = pct_change(1)*100 sobre los NIVELES PROFITS, INVESTMENT (transform honesto, P10 §4).
    Índice con frecuencia trimestral explícita (evita el ValueWarning de statsmodels)."""
    df = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    q = pd.DataFrame({
        "profits": df["PROFITS"].pct_change(1) * 100.0,
        "investment": df["INVESTMENT"].pct_change(1) * 100.0,
    }).dropna()
    q.index = pd.PeriodIndex(q.index, freq="Q").to_timestamp(how="end")
    q.index.freq = q.index.inferred_freq
    return q


# --------------------------------------------------------------------------- CCF empírica
def empirical_ccf(q, kmax=KMAX):
    """CCF empírica corr(profits[t], investment[t+k]), misma convención que P7."""
    p = q["profits"].to_numpy(); i = q["investment"].to_numpy()
    p = (p - p.mean()) / p.std(); i = (i - i.mean()) / i.std()
    n = len(p)
    ks = np.arange(-kmax, kmax + 1)
    c = np.empty_like(ks, float)
    for m, k in enumerate(ks):
        if k >= 0:
            c[m] = np.corrcoef(p[:n - k], i[k:])[0, 1]
        else:
            c[m] = np.corrcoef(p[-k:], i[:n + k])[0, 1]
    return ks, c


# --------------------------------------------------------------------------- Lyapunov / companion
def companion_matrices(res):
    """Construye (F, Q) de la forma companion de un VAR(p) ajustado.
    res.coefs: (p, k, k) con coefs[lag-1][eq, var]; res.sigma_u: (k, k).
    Estado z_t = [y_t; y_{t-1}; ...; y_{t-p+1}] de dimensión k*p.
    F = [[A_1, A_2, ..., A_p],
         [ I , 0  , ..., 0  ],
         [ 0 , I  , ..., 0  ],
         ...                 ]
    Q tiene Σ_u en el bloque (0,0) y 0 en el resto.
    """
    A = res.coefs                         # (p, k, k)
    p, k, _ = A.shape
    dim = k * p
    F = np.zeros((dim, dim))
    F[:k, :] = np.hstack([A[j] for j in range(p)])      # fila superior = [A_1 ... A_p]
    if p > 1:
        F[k:, :k * (p - 1)] = np.eye(k * (p - 1))       # sub-diagonal identidad (shift)
    Q = np.zeros((dim, dim))
    Q[:k, :k] = np.asarray(res.sigma_u)
    return F, Q, k, p


def theoretical_acov(F, Q, k, hmax):
    """Autocovarianzas teóricas Γ_y(h), h=0..hmax, del bloque y (k×k) vía Lyapunov.
    Devuelve lista [Γ_y(0), Γ_y(1), ..., Γ_y(hmax)] (cada uno k×k = E[y_t y_{t-h}ᵀ])
    y un flag de estabilidad (módulo del mayor autovalor de F < 1)."""
    eig = np.max(np.abs(np.linalg.eigvals(F)))
    stable = eig < 1.0 - 1e-9
    # Γ_z(0) estacionaria: Γ = F Γ Fᵀ + Q.  scipy resuelve  X = A X Aᵀ + Q  con A=F.
    Gz0 = solve_discrete_lyapunov(F, Q)
    Gz0 = 0.5 * (Gz0 + Gz0.T)            # simetriza por estabilidad numérica
    acov = []
    Gzh = Gz0.copy()
    for h in range(hmax + 1):
        # Γ_z(h) = Fʰ Γ_z(0); E[y_t y_{t-h}ᵀ] = bloque (0,0) de Γ_z(h)
        acov.append(Gzh[:k, :k].copy())
        Gzh = F @ Gzh                    # avanza a Γ_z(h+1) = F Γ_z(h)
    return acov, stable, eig


def theoretical_ccf(acov, kmax=KMAX):
    """CCF teórica corr(profits[t], investment[t+k]) a partir de Γ_y(h).
    Γ_y(h)=E[y_t y_{t-h}ᵀ]; columnas [profits=0, investment=1].
      k≥0 (profits lidera, investment en t+k): E[profits_{t+k} investment_t]... cuidado
      con la convención. Usamos la definición de P7 directamente:
        γ(k) = corr(profits[t], investment[t+k]).
      Para un proceso estacionario:
        cov(profits[t], investment[t+k]) = E[profits_t investment_{t+k}]
        = (Γ_y(k))[investment, profits]  si k≥0   (porque Γ_y(k)=E[y_t y_{t-k}ᵀ]
          => E[y_{t+k} y_tᵀ] => E[investment_{t+k} profits_t] = Γ_y(k)[inv, prof])
        = (Γ_y(-k))[profits, investment] si k<0.
    """
    s_p = np.sqrt(acov[0][0, 0])
    s_i = np.sqrt(acov[0][1, 1])
    ks = np.arange(-kmax, kmax + 1)
    c = np.empty_like(ks, float)
    for m, k in enumerate(ks):
        if k >= 0:
            cov = acov[k][1, 0]          # E[investment_{t+k} profits_t]
        else:
            cov = acov[-k][0, 1]         # E[profits_{t-k} investment_t] = E[profits_t inv_{t+k}]... k<0
        c[m] = cov / (s_p * s_i)
    return ks, c


# --------------------------------------------------------------------------- joroba
def hump_metrics(ks, c_theo, c_emp):
    """Métricas de la joroba negativa en lag -4/-5 (investment lidera)."""
    def g(c, k):
        return float(c[np.where(ks == k)[0][0]])
    out = {}
    for tag, c in [("theo", c_theo), ("emp", c_emp)]:
        out[tag] = dict(lag0=g(c, 0), lag_4=g(c, -4), lag_5=g(c, -5),
                        lag_3=g(c, -3), lag_6=g(c, -6))
        # ubicación del mínimo entre lags negativos (donde vive la joroba)
        neg_ks = ks[ks < 0]
        neg_c = np.array([g(c, k) for k in neg_ks])
        out[tag]["argmin_neglag"] = int(neg_ks[np.argmin(neg_c)])
        out[tag]["min_neglag"] = float(neg_c.min())
    return out


def reproduces_hump(m):
    """Criterio: la CCF teórica reproduce la joroba si (i) tiene mínimo en lag -3..-6,
    (ii) ese mínimo es negativo y de magnitud comparable (>=50% del empírico)."""
    th = m["theo"]; em = m["emp"]
    loc_ok = -6 <= th["argmin_neglag"] <= -3
    sign_ok = th["min_neglag"] < 0
    mag_ok = th["min_neglag"] <= 0.5 * em["min_neglag"]   # ambos negativos -> <= 50% del empírico
    return loc_ok and sign_ok and mag_ok, dict(loc_ok=loc_ok, sign_ok=sign_ok, mag_ok=mag_ok)


# --------------------------------------------------------------------------- por p
def run_one_p(q, p, kmax=KMAX, verbose=True):
    """Ajusta VAR(p), deriva CCF teórica por Lyapunov, compara con empírica."""
    model = VAR(q[["profits", "investment"]])
    res = model.fit(p)
    F, Q, k, pp = companion_matrices(res)
    acov, stable, eig = theoretical_acov(F, Q, k, hmax=kmax)
    ks, c_theo = theoretical_ccf(acov, kmax)
    _, c_emp = empirical_ccf(q, kmax)
    m = hump_metrics(ks, c_theo, c_emp)
    repro, parts = reproduces_hump(m)
    if verbose:
        print(f"\n  --- VAR({p}) ---  (n={res.nobs}, AIC={res.aic:.3f}, BIC={res.bic:.3f}, "
              f"|λ|max={eig:.3f}, {'ESTABLE' if stable else 'NO ESTABLE'})")
        print(f"    {'lag k':>6} | {'CCF teórica':>11} | {'CCF empírica':>12}")
        for kk in range(-6, 7):
            idx = np.where(ks == kk)[0][0]
            mark = "  <- joroba" if kk in (-4, -5) else ("  <- contemp" if kk == 0 else "")
            print(f"    {kk:>+6} | {c_theo[idx]:>+11.3f} | {c_emp[idx]:>+12.3f}{mark}")
        th = m["theo"]; em = m["emp"]
        print(f"    mínimo (lags neg): teórica {th['min_neglag']:+.3f} @ lag{th['argmin_neglag']:+d}"
              f"  |  empírica {em['min_neglag']:+.3f} @ lag{em['argmin_neglag']:+d}")
        print(f"    ¿reproduce la joroba? {'SÍ' if repro else 'NO'}  "
              f"(ubicación={parts['loc_ok']}, signo={parts['sign_ok']}, magnitud≥50%={parts['mag_ok']})")
    return dict(p=p, res=res, ks=ks, c_theo=c_theo, c_emp=c_emp, metrics=m,
                repro=repro, parts=parts, stable=stable, eig=eig,
                aic=res.aic, bic=res.bic, nobs=res.nobs)


# --------------------------------------------------------------------------- plot / csv
def plot_ccfs(results, ks_emp, c_emp):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="k", lw=0.8)
    ax.axvspan(-5.5, -3.5, color="tab:red", alpha=0.08)
    ax.annotate("joroba\nempírica\nlag -4/-5", (-4.5, -0.28), fontsize=8,
                color="tab:red", ha="center")
    ax.plot(ks_emp, c_emp, "k-o", lw=2.0, ms=5, label="CCF empírica (QoQ)", zorder=5)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(results)))
    for d, col in zip(results, cmap):
        ax.plot(d["ks"], d["c_theo"], "--", color=col, lw=1.4,
                label=f"VAR({d['p']}) teórica (Lyapunov)")
    ax.set_xlabel("lag k (trim)   ← investment lidera   |   profits lidera →")
    ax.set_ylabel("corr(profits[t], investment[t+k])")
    ax.set_title("CCF teórica del VAR(p) (Lyapunov) vs CCF empírica\n"
                 "¿el VAR(p) lineal-estocástico ya genera la joroba negativa de lag 4-5?",
                 fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = C.OUTDIR / "p12_var_lyapunov_ccf.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def dump_csv(results):
    rows = []
    for d in results:
        for idx, kk in enumerate(d["ks"]):
            rows.append(dict(p=d["p"], lag=int(kk), ccf_theo=float(d["c_theo"][idx]),
                             ccf_emp=float(d["c_emp"][idx]), stable=d["stable"],
                             eig_max=d["eig"], aic=d["aic"], bic=d["bic"], reproduces=d["repro"]))
    path = C.OUTDIR / "p12_var_lyapunov_ccf.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- conclusión
def conclude(results):
    print("\n" + "=" * 78)
    print("  CONCLUSIÓN — equivalencia observacional kernel-de-delay vs lineal-memoria-larga")
    print("=" * 78)
    repro_ps = [d["p"] for d in results if d["repro"]]
    print(f"  La CCF teórica reproduce la joroba negativa de lag 4-5 para p = "
          f"{repro_ps if repro_ps else '(ninguno)'}.")
    for d in results:
        th = d["metrics"]["theo"]
        print(f"    VAR({d['p']}): joroba teórica mín {th['min_neglag']:+.3f} @ lag"
              f"{th['argmin_neglag']:+d}  (reproduce={d['repro']})  "
              f"[BIC={d['bic']:.2f}, |λ|max={d['eig']:.2f}]")
    if repro_ps:
        print(f"\n  => EQUIVALENCIA OBSERVACIONAL (a nivel CCF de 2º orden). Un VAR(p≥{min(repro_ps)})")
        print("     lineal y estacionario YA genera la joroba negativa de lag 4-5 a partir de")
        print("     sus coeficientes (Lyapunov), SIN postular un kernel de delay explícito.")
        print("     La joroba NO es evidencia, por sí sola, de un mecanismo de retardo de")
        print("     maduración POR ENCIMA de un sistema lineal estocástico de memoria larga:")
        print("     ambas descripciones son indistinguibles en la autocovarianza de 2º orden.")
        print("     El 'kernel de delay' es una RE-PARAMETRIZACIÓN interpretable del mismo")
        print("     contenido lineal, no un mecanismo adicional identificable con estos datos.")
    else:
        print("\n  => La joroba NO emerge de la CCF teórica del VAR(p) lineal: el patrón")
        print("     requeriría estructura más allá de la autocovarianza lineal de 2º orden.")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help=f"smoke test: solo VAR({SMOKE_P})")
    args = ap.parse_args()

    q = load_qoq()
    ks_emp, c_emp = empirical_ccf(q, KMAX)
    print("=" * 78)
    print("  Pieza 12 — CCF teórica del VAR(p) vía Lyapunov (profit-investment, QoQ)")
    print("=" * 78)
    print(f"  n obs QoQ = {len(q)}")
    print(f"  CCF empírica de referencia: lag0={c_emp[np.where(ks_emp==0)[0][0]]:+.3f}  "
          f"lag-4={c_emp[np.where(ks_emp==-4)[0][0]]:+.3f}  "
          f"lag-5={c_emp[np.where(ks_emp==-5)[0][0]]:+.3f}")

    if args.smoke:
        print(f"\n  [SMOKE TEST] un solo p = {SMOKE_P}")
        d = run_one_p(q, SMOKE_P)
        print("\n  (smoke OK — no se generan figuras/CSV; corré sin --smoke para p=2..6)")
        return

    results = [run_one_p(q, p) for p in range(PMIN, PMAX + 1)]
    fig = plot_ccfs(results, ks_emp, c_emp)
    csv = dump_csv(results)
    conclude(results)
    print(f"\n  figura -> {fig}")
    print(f"  csv    -> {csv}")


if __name__ == "__main__":
    main()
