"""Pieza 5 — Oscilador NO-ESTACIONARIO: ¿deriva la periodicidad del ciclo?

La periodicidad de las crisis es irregular y parece alargarse (Gran Moderación). Un
oscilador autónomo de frecuencia fija no puede capturarla. Acá se caracteriza la
no-estacionariedad por dos vías:
  (1) STFT (tiempo-frecuencia, model-free): ridge del período dominante en el tiempo.
  (2) VAR rodante (= oscilador lineal de parámetros variables, TVP): por ventana, el
      autovalor complejo dominante del VAR → período y persistencia (1=no amortiguado).

Hipótesis: período se alarga + persistencia baja (ciclos más largos y suaves) = Gran
Moderación. Caveat: ventana 12a + dato trimestral ⇒ resolución gruesa, track ruidoso.

Run:  uv run python -m src.experiment.p5_nonstationary
"""
from __future__ import annotations

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import stft
from statsmodels.tsa.api import VAR

from src.experiment import common as C

warnings.filterwarnings("ignore")
FS = 4.0          # trimestres/año
WIN = 48          # ventana 12 años


def stft_ridge(v):
    f, t, Z = stft(v, fs=FS, nperseg=WIN, noverlap=WIN - 4)
    P = np.abs(Z); band = (f >= 1 / 15) & (f <= 1 / 2)
    ridge = np.array([1 / f[band][np.argmax(P[band, j])] for j in range(P.shape[1])])
    return t, ridge


def rolling_var_resonance(df):
    """VAR rodante: período y persistencia del autovalor complejo dominante por ventana."""
    raw = df[C.COLS].to_numpy(); idx = df.index
    centers, periods, persist = [], [], []
    for s in range(0, len(raw) - WIN, 2):
        w = raw[s:s + WIN]
        try:
            res = VAR(w).fit(maxlags=3, ic="bic")
            ev = np.linalg.eigvals(_companion(res.coefs))
        except Exception:
            continue
        cpx = ev[np.abs(ev.imag) > 1e-6]
        if len(cpx) == 0:
            continue
        z = cpx[np.argmax(np.abs(cpx))]                       # autovalor dominante
        ang = abs(np.angle(z))
        if ang < 1e-6:
            continue
        centers.append(idx[s + WIN // 2].year + (idx[s + WIN // 2].dayofyear - 1) / 365.25)
        periods.append(2 * np.pi / ang / FS)                  # período en años
        persist.append(abs(z))                                 # 1 = no amortiguado
    return np.array(centers), np.array(periods), np.array(persist)


def _companion(coefs):
    """Matriz compañía de un VAR(p) con coefs (p, k, k)."""
    p, k, _ = coefs.shape
    top = np.hstack([coefs[i] for i in range(p)])
    if p == 1:
        return top
    bottom = np.hstack([np.eye(k * (p - 1)), np.zeros((k * (p - 1), k))])
    return np.vstack([top, bottom])


def main():
    df = C.load(); yr0 = df.index[0].year
    crises = [pd.Timestamp(t).year for _, _, t in C.RECESSIONS]

    # (1) STFT spectrogram + ridge
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, (col, lbl) in zip(axes, [("PROFITS_YOY", "Profits"), ("INVEST_YOY", "Investment")]):
        v = df[col].to_numpy(); v = (v - v.mean()) / v.std()
        f, t, Z = stft(v, fs=FS, nperseg=WIN, noverlap=WIN - 4)
        P = np.abs(Z); band = (f >= 1 / 15) & (f <= 1 / 2)
        ax.pcolormesh(yr0 + t, 1 / f[band], P[band], shading="auto", cmap="viridis")
        tr, ridge = stft_ridge(v)
        ax.plot(yr0 + tr, ridge, "w.-", lw=1.4, ms=4, label="período dominante")
        for c in crises:
            ax.axvline(c, color="red", lw=0.7, alpha=0.5)
        ax.set_ylabel(f"{lbl}\nperíodo (años)"); ax.set_ylim(2, 15)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title("Tiempo-frecuencia (STFT, ventana 12a): deriva de la periodicidad\n"
                      "(líneas rojas = crisis NBER)", fontsize=11)
    axes[1].set_xlabel("Año")
    fig.tight_layout(); fig.savefig(C.OUTDIR / "p5_timefreq.png", dpi=130); plt.close(fig)

    # (2) VAR rodante: período + persistencia
    cen, per, per_z = rolling_var_resonance(df)
    sm = lambda a: pd.Series(a).rolling(5, center=True, min_periods=1).mean().to_numpy()
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax[0].plot(cen, per, ".", color="gray", alpha=0.4); ax[0].plot(cen, sm(per), "b-", lw=2, label="suavizado")
    ax[0].set_ylabel("período (años)"); ax[0].set_ylim(0, 20); ax[0].legend(fontsize=8)
    ax[0].set_title("Oscilador lineal de parámetros variables (VAR rodante 12a)", fontsize=11)
    ax[1].plot(cen, per_z, ".", color="gray", alpha=0.4); ax[1].plot(cen, sm(per_z), "g-", lw=2, label="suavizado")
    ax[1].set_ylabel("persistencia |λ|\n(1 = no amortiguado)"); ax[1].set_xlabel("Año"); ax[1].legend(fontsize=8)
    for a in ax:
        for c in crises:
            a.axvline(c, color="red", lw=0.6, alpha=0.4)
    fig.tight_layout(); fig.savefig(C.OUTDIR / "p5_tvp_resonance.png", dpi=130); plt.close(fig)

    def trend(x, y):
        sl = np.polyfit(x, y, 1)[0]; r = np.corrcoef(x, y)[0, 1]
        return sl * 10, r
    sp, rp = trend(cen, per); sz, rz = trend(cen, per_z)
    print("=== Pieza 5 — oscilador no-estacionario (VAR rodante) ===")
    print(f"  n ventanas: {len(cen)}  ({cen[0]:.0f}–{cen[-1]:.0f})")
    print(f"  período:      media={np.nanmean(per):.1f}a  tendencia={sp:+.2f} a/década  (r={rp:+.2f})")
    print(f"  persistencia: media={np.nanmean(per_z):.2f}   tendencia={sz:+.3f} /década  (r={rz:+.2f})")
    print(f"  → período {'se alarga' if sp>0 else 'se acorta'}; amortiguamiento "
          f"{'aumenta (más suave)' if sz<0 else 'baja'}")
    print(f"✓ figuras: {C.OUTDIR}/p5_timefreq.png , p5_tvp_resonance.png")


if __name__ == "__main__":
    main()
