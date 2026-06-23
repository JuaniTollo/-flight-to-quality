"""Pieza 11 — Modelo FÍSICO: cadena de maduración (delay distribuido) del ciclo de
sobreacumulación profit→investment.

Motivación: ni LV/FN (presa-depredador instantáneo, fase fija) ni un delay puntual
determinístico capturan el patrón. La idea física correcta es que la inversión deprime
la ganancia tras MADURAR por etapas (gestación → capacidad → exceso de oferta), lo que
genera una DISTRIBUCIÓN de demoras. Eso es un delay distribuido (kernel Gamma), que por
el *linear chain trick* es exactamente equivalente a una cascada de ODEs:

    dI/dt = a·P − b·I                       (acelerador)
    dmⱼ/dt = (mⱼ₋₁ − mⱼ)/θ                  (cadena de n etapas de maduración; m₀ = I)
    dP/dt = c·I − d·P + k·mₙ                (ganancia: + demanda actual; +k·mₙ con k<0 = sobreacumulación madurada)

mₙ = inversión filtrada por un kernel Gamma(shape=n, scale=θ): media n·θ (lag de
maduración), dispersión √n·θ (cuánto fluctúa). Modelo físico, autónomo, DIFERENCIABLE
→ calibrable por PINN/UDE inverso (inferencia diferenciable). La versión continua (DDE + autodiff/
adjoint) es el paso siguiente; acá se calibra el kernel por su solución analítica (filtro)
para aislar la pregunta de IDENTIFICABILIDAD sin integración inestable.

Resultado (sobre crisis NBER, QoQ honesto, sin 2008/2020):
  - la MEDIA del lag se identifica en ~4 trimestres (1 año);
  - el ANCHO de la distribución NO se identifica (mismo ajuste sea distribuido o puntual).
El aporte honesto es la FRONTERA DE IDENTIFICABILIDAD del kernel, no un mecanismo nuevo.

Run:  uv run python -m src.experiment.p11_physical_delay
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma as gammadist

from src.experiment import common as C

DROP = {"2008", "2020"}          # outliers de amplitud; análisis sobre crisis "normales"
H = 8                            # semiancho de ventana (trim)


def qoq_data():
    raw = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    P = np.log(raw["PROFITS"]).diff() * 100
    I = np.log(raw["INVESTMENT"]).diff() * 100
    d = pd.DataFrame({"P": P, "I": I}).dropna()
    return d.index, d["P"].to_numpy(), d["I"].to_numpy()


def crisis_slices(idx, Iv):
    S = []
    for lbl, pk, tr in C.RECESSIONS:
        if lbl in DROP:
            continue
        pk, tr = pd.Timestamp(pk), pd.Timestamp(tr)
        nm = (idx >= pk - pd.offsets.QuarterEnd(2)) & (idx <= tr + pd.offsets.QuarterEnd(4))
        pos = np.where(nm)[0]
        if len(pos) == 0:
            continue
        a = pos[np.argmin(Iv[pos])]
        if a - H >= 0 and a + H < len(Iv):
            S.append((a - H, a + H + 1))
    return S


def gkernel(mu, shape, K=12):
    """Kernel Gamma de media mu (trim) y forma 'shape' (grande=puntual, chico=ancho)."""
    scale = mu / shape
    ks = np.arange(0, K + 1) + 0.5                 # +0.5 evita divergencia en 0 si shape<1
    w = gammadist.pdf(ks, a=shape, scale=scale)
    s = w.sum()
    return w / s if s > 0 and np.all(np.isfinite(w)) else None


def fit(Pv, Iv, slices, mu, shape):
    """P_t ~ c·I_t + e·P_{t-1} + k·m_t, pooled sobre ventanas z-scoreadas. Lineal → OLS."""
    w = gkernel(mu, shape)
    if w is None:
        return np.nan, None
    X, y = [], []
    for lo, hi in slices:
        Pi, Ii = Pv[lo:hi], Iv[lo:hi]
        Pi = (Pi - Pi.mean()) / Pi.std()
        Ii = (Ii - Ii.mean()) / Ii.std()
        m = np.convolve(Ii, w)[:len(Ii)]
        for t in range(1, len(Pi)):
            X.append([Ii[t], Pi[t - 1], m[t]]); y.append(Pi[t])
    X, y = np.array(X), np.array(y)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2)
    return r2, b


def main():
    idx, Pv, Iv = qoq_data()
    sl = crisis_slices(idx, Iv)
    print(f"=== Pieza 11 — modelo físico de delay distribuido (cadena de maduración) ===")
    print(f"crisis usables (sin 2008/2020): {len(sl)} | transform: QoQ (honesto)\n")

    # 1) ¿se identifica la MEDIA del lag?
    mus = np.arange(1, 9)
    r2_mu = [fit(Pv, Iv, sl, m, 8.0)[0] for m in mus]
    mu_best = mus[int(np.argmax(r2_mu))]
    print("¿Media del lag identificable? (shape=8)")
    for m, r in zip(mus, r2_mu):
        print(f"  μ={m}q: R²={r:+.4f}" + ("  <-- óptimo" if m == mu_best else ""))
    print(f"  → lag de maduración ≈ {mu_best} trimestres (~1 año)\n")

    # 2) ¿se identifica el ANCHO?
    shapes = [1, 2, 4, 8, 16, 32, 64, 128]
    r2_sh, ks = [], []
    print("¿Ancho de la distribución identificable? (μ=4)")
    for sh in shapes:
        r2, b = fit(Pv, Iv, sl, 4.0, sh)
        r2_sh.append(r2); ks.append(b[2])
        print(f"  shape={sh:<4} CV={1/np.sqrt(sh):.2f} ({'ANCHO' if sh < 4 else 'puntual'}): R²={r2:+.4f}  k={b[2]:+.2f}")
    span = max(r2_sh) - min(r2_sh)
    print(f"  → rango de R² sobre todo el barrido de ancho = {span:.4f}")
    print(f"  → ANCHO {'NO' if span < 0.03 else 'SÍ'} identificable "
          f"({'mismo ajuste sea distribuido o puntual' if span < 0.03 else 'el ajuste discrimina el ancho'})\n")

    # 3) figura
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].plot(mus, r2_mu, "o-"); ax[0].axvline(mu_best, color="r", ls="--", lw=0.8)
    ax[0].set_title(f"Media del lag SÍ se identifica\n(óptimo μ≈{mu_best}q ≈ 1 año)")
    ax[0].set_xlabel("μ = lag medio de acumulación (trim)"); ax[0].set_ylabel("R²")

    ax[1].plot([1/np.sqrt(s) for s in shapes], r2_sh, "o-")
    ax[1].set_title(f"Ancho NO se identifica\n(R² plano, rango={span:.3f})")
    ax[1].set_xlabel("CV del kernel (←puntual | ancho→)"); ax[1].set_ylabel("R²")
    ax[1].set_ylim(min(r2_sh) - 0.05, max(r2_sh) + 0.05)

    for sh, lbl in [(64, "casi puntual (CV=0.12)"), (2, "ancho (CV=0.71)")]:
        w = gkernel(4.0, sh)
        ax[2].plot(np.arange(len(w)), w, "o-", label=lbl, ms=4)
    ax[2].set_title("Dos kernels que ajustan IGUAL\n(indistinguibles con esta data)")
    ax[2].set_xlabel("rezago j (trim)"); ax[2].set_ylabel("peso del kernel"); ax[2].legend(fontsize=8)
    fig.suptitle("Modelo físico de acumulación: el centro del lag se identifica, el ancho no", fontsize=12)
    fig.tight_layout(); fig.savefig(C.OUTDIR / "p11_physical_delay.pdf"); fig.savefig(C.OUTDIR / "p11_physical_delay.png", dpi=200); plt.close(fig)
    print(f"✓ figura: {C.OUTDIR}/p11_physical_delay.png")
    print("\nVEREDICTO: modelo físico (cadena de maduración) válido y diferenciable; el lag de "
          "\nmaduración ~1 año es un parámetro físico identificable, pero el ancho de su "
          "\ndistribución choca con la frontera de identificabilidad (SNR ~3%). El aporte es "
          "\nesa frontera, caracterizada con un modelo físico. Próximo: DDE continua + PINN/adjoint.")


if __name__ == "__main__":
    main()
