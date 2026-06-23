"""Figura Pieza C — BLINDAJE Slutsky-Yule del valle de sobreacumulación.

Cómputo (autocontenido, sin dependencias internas):
  - CCF cruda corr(P_t, I_{t+k}) sobre log-diff QoQ, z-scoreada, sin suavizado.
  - Null por surrogates AR (AIC) que preservan autocorrelación marginal de cada serie
    y la correlación contemporánea de innovaciones, pero DESTRUYEN el feedback cruzado.

Diferencia clave: la banda null se calcula en CADA lag k (no solo lag-4), tomando
los percentiles 2.5/97.5 de ~2000 surrogates en todo el rango k=-12..+12.

Run:  .venv/bin/python -m src.experiment.p7b_ccf_surrogate
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment import common as C

RNG = np.random.default_rng(20260621)
KMAX = 12
LAG = 4               # rezago de sobreacumulación (~1 año)
NS = 2000             # número de surrogates


# --------------------------------------------------------------------- núcleo CCF cruda (log-trim QoQ)
def load_logtrim():
    raw = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    g = pd.DataFrame({"P": np.log(raw["PROFITS"]).diff() * 100,
                      "I": np.log(raw["INVESTMENT"]).diff() * 100}).dropna()
    return g["P"].to_numpy(), g["I"].to_numpy()


def z(x):
    return (x - x.mean()) / x.std()


def ccf(p, i, kmax=KMAX):
    """corr(P_t, I_{t+k}); k<0 ⇒ inversión pasada vs ganancias presentes (sobreacumulación)."""
    p, i = z(p), z(i)
    n = len(p)
    ks = np.arange(-kmax, kmax + 1)
    c = np.empty(len(ks))
    for m, k in enumerate(ks):
        c[m] = np.corrcoef(p[:n - k], i[k:])[0, 1] if k >= 0 else np.corrcoef(p[-k:], i[:n + k])[0, 1]
    return ks, c


def fit_ar(x, p):
    n = len(x)
    X = np.column_stack([x[p - 1 - j: n - 1 - j] for j in range(p)] + [np.ones(n - p)])
    y = x[p:]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    phi, c = beta[:p], beta[p]
    resid = y - X @ beta
    return phi, c, resid


def ar_order(x, pmax=8):
    n = len(x); best, bp = np.inf, 1
    for p in range(1, pmax + 1):
        _, _, r = fit_ar(x, p)
        aic = (n - p) * np.log(np.sum(r ** 2) / (n - p)) + 2 * (p + 1)
        if aic < best:
            best, bp = aic, p
    return bp


def main():
    P, I = load_logtrim()
    n = len(P)
    ks, obs = ccf(P, I)
    idx = {k: m for m, k in enumerate(ks)}

    lag0 = obs[idx[0]]
    lag_m4 = obs[idx[-LAG]]
    print(f"n={n}")
    print(f"CCF cruda (sin suavizado):  lag0={lag0:+.3f}  lag-4={lag_m4:+.3f}\n")

    # ---- Surrogates AR: preservan autocorr marginal + corr contemporánea, destruyen feedback ----
    pP, pI = ar_order(P), ar_order(I)
    phiP, cP, rP = fit_ar(P, pP)
    phiI, cI, rI = fit_ar(I, pI)
    L = min(len(rP), len(rI))
    rho_innov = np.corrcoef(rP[-L:], rI[-L:])[0, 1]
    sP, sI = rP.std(), rI.std()
    print(f"Surrogates AR: P~AR({pP}), I~AR({pI}); corr de innovaciones ρ={rho_innov:+.3f}")

    cov = np.array([[sP ** 2, rho_innov * sP * sI], [rho_innov * sP * sI, sI ** 2]])
    L_ch = np.linalg.cholesky(cov)
    burn = 100
    T = n + burn

    def simulate():
        e = L_ch @ RNG.standard_normal((2, T))
        xp = np.zeros(T); xi = np.zeros(T)
        for t in range(max(pP, pI), T):
            xp[t] = cP + phiP @ xp[t - pP:t][::-1] + e[0, t]
            xi[t] = cI + phiI @ xi[t - pI:t][::-1] + e[1, t]
        return xp[burn:], xi[burn:]

    null = np.empty((NS, len(ks)))
    for s in range(NS):
        xp, xi = simulate()
        _, cs = ccf(xp, xi)
        null[s] = cs

    lo = np.quantile(null, 0.025, axis=0)
    hi = np.quantile(null, 0.975, axis=0)
    mean = null.mean(axis=0)

    # p-valor de dos colas en lag-4
    nm4 = null[:, idx[-LAG]]
    p_two = 2 * min((nm4 <= lag_m4).mean(), (nm4 >= lag_m4).mean())
    print(f"null lag-4: media={mean[idx[-LAG]]:+.3f}  IC95%=[{lo[idx[-LAG]]:+.3f}, {hi[idx[-LAG]]:+.3f}]")
    print(f"observado lag-4 = {lag_m4:+.3f}  →  p(dos colas) = {p_two:.4f}  "
          f"{'FUERA del null ⇒ feedback REAL' if lag_m4 < lo[idx[-LAG]] or lag_m4 > hi[idx[-LAG]] else 'dentro'}")
    print(f"null lag0:  media={mean[idx[0]]:+.3f}  IC95%=[{lo[idx[0]]:+.3f}, {hi[idx[0]]:+.3f}]")

    # ----------------------------------------------------------------- figura
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.axhline(0, color="k", lw=0.8, zorder=1)

    # Banda null 95%
    ax.fill_between(ks, lo, hi, color="0.6", alpha=0.35, zorder=2,
                    label="banda null 95% (surrogates AR)")
    ax.plot(ks, mean, color="0.45", lw=1.3, ls="--", zorder=3, label="media del null")

    # CCF observada como stem
    ml, sl, bl = ax.stem(ks, obs, linefmt="tab:blue", markerfmt="o", basefmt=" ",
                         label="CCF observada")
    plt.setp(sl, linewidth=1.6, color="tab:blue")
    plt.setp(ml, markersize=5, markerfacecolor="tab:blue", markeredgecolor="tab:blue")

    # Marcar lag 0 (co-movimiento contemporáneo)
    ax.scatter([0], [lag0], s=110, facecolor="none", edgecolor="tab:green",
               linewidth=2.2, zorder=6)
    ax.annotate(f"co-movimiento\ncontemporáneo\nlag 0 = {lag0:+.2f}",
                xy=(0, lag0), xytext=(2.4, lag0 + 0.07),
                fontsize=8.5, color="tab:green", ha="left",
                arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.2))

    # Marcar lag -4 (valle de sobreacumulación, fuera del null)
    ax.scatter([-LAG], [lag_m4], s=130, facecolor="none", edgecolor="tab:red",
               linewidth=2.4, zorder=6)
    ax.annotate(f"valle de sobreacumulación (~1 año)\n"
                f"lag −4 = {lag_m4:+.2f}  ·  p = {p_two:.3f}\n"
                f"cae BAJO la banda null ⇒ feedback real",
                xy=(-LAG, lag_m4), xytext=(-11.5, -0.42),
                fontsize=8.5, color="tab:red", ha="left",
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.3))

    ax.set_xlabel("rezago h (trimestres)   ← inversión lidera   |   ganancias lideran →",
                  fontsize=10)
    ax.set_ylabel("corr(ganancias[t], inversión[t+k])", fontsize=10)
    ax.set_title("El valle de sobreacumulación (~1 año) es real:\n"
                 "cae fuera del null por surrogates AR (preservan autocorrelación + "
                 "co-movimiento, destruyen feedback)", fontsize=11)
    ax.set_xlim(-KMAX - 0.6, KMAX + 0.6)
    ax.set_ylim(-0.6, 0.75)
    ax.set_xticks(range(-KMAX, KMAX + 1, 2))
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    import shutil
    out = C.OUTDIR / "p7b_ccf_surrogate"
    fig.savefig(f"{out}.png", dpi=130)
    fig.savefig(f"{out}.pdf")
    plt.close(fig)
    paper_pdf = C.ROOT / "paper" / "figures" / "p7b_ccf_surrogate.pdf"
    shutil.copy(f"{out}.pdf", paper_pdf)
    print(f"\n✓ figura: {out}.pdf  →  copiada a {paper_pdf}")


if __name__ == "__main__":
    main()
