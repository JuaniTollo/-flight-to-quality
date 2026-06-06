"""Pieza 4 — Evaluación dinámica (herramientas de la tradición testing-Goodwin).

Evalúa el oscilador como sistema dinámico, NO por pronóstico. Usa GRADIENT MATCHING
(Ramsay & Hooker): ajusta los parámetros para que el campo vectorial f(u,p) reproduzca
la velocidad observada du/dt — evita el single-shooting (que degeneraba) y es el método
con que esta literatura testea modelos tipo Goodwin/Lotka-Volterra.

Diagnósticos:
  1. Período implícito (autovalores en el punto fijo) vs período observado (FFT)  [Harvie]
  2. Tipo de ciclo por la parte real de los autovalores: amortiguado / neutral / límite
  3. R² del campo de fase: ¿los datos siguen el campo del oscilador?
  4. Robustez de los parámetros entre ciclos NBER

Run:  uv run python -m src.experiment.p4_dynamical_eval
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, fsolve

from src.experiment import common as C


def derivatives(t, Z):
    """du/dt por diferencias finitas, con un suavizado leve (3 puntos) anti-ruido."""
    Zs = pd.DataFrame(Z).rolling(3, center=True, min_periods=1).mean().to_numpy()
    return np.gradient(Zs, t, axis=0)


def grad_match(model, U, dU, p0=None):
    """Ajusta p minimizando ||du/dt - f(u,p)|| sobre todos los puntos (campo vectorial)."""
    p0 = model.p0 if p0 is None else p0

    def resid(p):
        if not model.guard(p):
            return np.full(U.size, 1e3)
        F = np.array([model.rhs(u, p) for u in U])
        return (F - dU).ravel()

    res = least_squares(resid, p0, max_nfev=20000)
    return res.x


def field_r2(model, U, dU, p):
    F = np.array([model.rhs(u, p) for u in U])
    ss_res = np.sum((dU - F) ** 2, axis=0)
    ss_tot = np.sum((dU - dU.mean(0)) ** 2, axis=0)
    return 1 - ss_res / ss_tot                      # por ecuación [profits, investment]


def jacobian(model, p, u, eps=1e-5):
    J = np.zeros((2, 2))
    for j in range(2):
        d = np.zeros(2); d[j] = eps
        J[:, j] = (model.rhs(u + d, p) - model.rhs(u - d, p)) / (2 * eps)
    return J


def classify(eigs):
    re, im = eigs.real, np.abs(eigs.imag)
    period = 2 * np.pi / im.max() if im.max() > 1e-6 else np.inf   # años
    re_dom = re[np.argmax(im)] if im.max() > 1e-6 else re.max()
    if abs(re_dom) < 0.02:
        kind = "neutral (Goodwin/LV)"
    elif re_dom < 0:
        kind = "amortiguado (espirala al equilibrio)"
    else:
        kind = "autosostenido (ciclo límite)"
    return period, re_dom, kind


def observed_period(df):
    """Período dominante por FFT, por serie (sobre la serie z-scoreada, detrendeada)."""
    out = {}
    for c in C.COLS:
        v = df[c].to_numpy(); v = v - v.mean()
        n = len(v); freq = np.fft.rfftfreq(n, d=0.25)        # 0.25 años/trimestre
        P = np.abs(np.fft.rfft(v)) ** 2
        k = 1 + np.argmax(P[1:])
        out[c] = 1 / freq[k]
    return out


def phase_lag(eigvals, eigvecs, period):
    """Rezago de fase profits↔investment del modo oscilatorio (trimestres). >0 ⇒ profits lidera."""
    i = int(np.argmax(np.abs(eigvals.imag)))           # modo complejo dominante
    if abs(eigvals[i].imag) < 1e-6:
        return np.nan
    v = eigvecs[:, i]
    dphi = np.angle(v[0]) - np.angle(v[1])             # fase profits − investment (rad)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi        # a (−π, π]
    return dphi / (2 * np.pi) * period * 4             # rad → fracción de período → trimestres


def eval_model(model, df):
    to_z, to_raw, off, mu, sd = C.zscorer(df, model.needs_offset)
    U = to_z(df[C.COLS].to_numpy())
    t = (df.index - df.index[0]).days.to_numpy() / 365.25
    dU = derivatives(t, U)
    p = grad_match(model, U, dU)
    r2 = field_r2(model, U, dU, p)
    fp = fsolve(lambda u: model.rhs(u, p), U.mean(0), full_output=False)
    eigvals, eigvecs = np.linalg.eig(jacobian(model, p, fp))
    period, re_dom, kind = classify(eigvals)
    lag = phase_lag(eigvals, eigvecs, period)
    return dict(p=p, r2=r2, fp=fp, eigs=eigvals, period=period, re_dom=re_dom, kind=kind,
                lag=lag, U=U, dU=dU, to_raw=to_raw)


def per_cycle_params(model, df):
    """Gradient matching por ciclo NBER → dispersión de parámetros (robustez)."""
    rows = []
    for k in range(1, len(C.TROUGHS)):
        lo, hi = C.TROUGHS[k - 1], C.TROUGHS[k]
        sub = df.loc[(df.index > lo) & (df.index <= hi)]
        if len(sub) < 8:
            continue
        to_z, *_ = C.zscorer(sub, model.needs_offset)
        U = to_z(sub[C.COLS].to_numpy())
        t = (sub.index - sub.index[0]).days.to_numpy() / 365.25
        rows.append(grad_match(model, U, derivatives(t, U)))
    return np.array(rows)


def main():
    df = C.load()
    obs = observed_period(df)
    pnames = {"FN": ["a", "b", "logc", "I"], "LV": ["alpha", "beta", "delta", "gamma"],
              "LIN": ["a11", "a12", "a21", "a22", "c1", "c2"]}

    print("=== Pieza 4 — evaluación dinámica (gradient matching) ===")
    print(f"Período OBSERVADO (FFT): profits={obs['PROFITS_YOY']:.1f} años  "
          f"investment={obs['INVEST_YOY']:.1f} años\n")

    results = {}
    for m in (C.FN, C.LV, C.LIN):
        r = eval_model(m, df)
        results[m.name] = r
        eg = ", ".join(f"{e.real:+.2f}{e.imag:+.2f}i" for e in r["eigs"])
        print(f"--- {m.name} ---")
        print(f"  R² del campo de fase: profits={r['r2'][0]:.2f}  investment={r['r2'][1]:.2f}")
        print(f"  autovalores en el punto fijo: [{eg}]")
        print(f"  período IMPLÍCITO = {r['period']:.1f} años   (vs observado ~{obs['PROFITS_YOY']:.1f}–{obs['INVEST_YOY']:.1f})")
        print(f"  amortiguamiento Re(λ) = {r['re_dom']:+.3f}  →  {r['kind']}")
        if np.isfinite(r["lag"]):
            quien = "profits lidera investment" if r["lag"] > 0 else "investment lidera profits"
            print(f"  rezago de fase = {abs(r['lag']):.1f} trimestres  ({quien})")
        P = per_cycle_params(m, df)
        cv = np.std(P, axis=0) / (np.abs(np.mean(P, axis=0)) + 1e-9)
        print(f"  robustez entre {len(P)} ciclos (CV por parámetro): "
              + ", ".join(f"{k}={v:.2f}" for k, v in zip(pnames[m.name], cv)))
        print()

    print("=== Comparación R² del campo de fase (¿qué oscilador describe el movimiento?) ===")
    for s, j in [("profits", 0), ("investment", 1)]:
        print(f"  {s:11s}: " + "  ".join(f"{n}={results[n]['r2'][j]:+.2f}" for n in ("FN", "LV", "LIN")))
    print()

    # ---- figura: campo vectorial (quiver) + datos ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, name in zip(axes, ("FN", "LV")):
        r = results[name]
        U = r["U"]
        x = np.linspace(U[:, 0].min(), U[:, 0].max(), 18)
        y = np.linspace(U[:, 1].min(), U[:, 1].max(), 18)
        X, Y = np.meshgrid(x, y)
        m = C.FN if name == "FN" else C.LV
        UU = np.zeros_like(X); VV = np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                d = m.rhs(np.array([X[i, j], Y[i, j]]), r["p"])
                n = np.hypot(*d) + 1e-9
                UU[i, j], VV[i, j] = d / n
        ax.quiver(X, Y, UU, VV, color="tab:gray", alpha=0.55, width=0.003)
        ax.plot(U[:, 0], U[:, 1], color="black", lw=0.8, alpha=0.7, label="datos (órbita)")
        ax.plot(*r["fp"], "r*", ms=14, label="punto fijo")
        ax.set_title(f"Campo de fase ajustado — {name}\n{r['kind']}, T≈{r['period']:.1f}a", fontsize=10)
        ax.set_xlabel("Profits (z)"); ax.set_ylabel("Investment (z)")
        ax.legend(fontsize=8)
    fig.suptitle("Evaluación dinámica: campo vectorial del oscilador vs trayectoria observada", fontsize=12)
    fig.tight_layout(); fig.savefig(C.OUTDIR / "p4_vector_field.png", dpi=130); plt.close(fig)
    print(f"✓ figura: {C.OUTDIR}/p4_vector_field.png")


if __name__ == "__main__":
    main()
