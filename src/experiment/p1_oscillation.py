"""Pieza 1 — La oscilación característica (grafica; el ajuste se hace en Julia).

Lee output/experiment/julia_oscillation.csv (FN y LV ajustados por solver sobre una
ventana representativa, producido por `oscillation.jl`) y dibuja:
  (1) la trayectoria del oscilador sobre los datos, y
  (2) el retrato de fase (investment vs profits).

Observación esperada y on-message: FN (ciclo límite) ajusta una oscilación genuina; LV
(neutral, tipo Goodwin) DEGENERA bajo el solver single-shooting (mal-condicionado) — la
misma ill-posedness que la PINN inversa regulariza (Pieza 3).

Pre-requisito:  julia --project=.../08_LV_inverse_PINN src/experiment/oscillation.jl
Run:            uv run python -m src.experiment.p1_oscillation
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.experiment import common as C

OSC = C.OUTDIR / "julia_oscillation.csv"
COLOR = {"FN": "tab:blue", "LV": "tab:green"}


def main():
    if not OSC.exists():
        raise SystemExit(f"Falta {OSC}. Corré: julia --project=... src/experiment/oscillation.jl")
    df = pd.read_csv(OSC)
    data = df[df.kind == "data"].sort_values("decyear")

    # ---- Figura 1: trayectorias ----
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for j, (col, lbl) in enumerate([("profits", "Profits"), ("investment", "Investment")]):
        ax[j].plot(data.decyear, data[col], "o-", color="black", ms=3, lw=1.2, label="Real", zorder=3)
        for name in ("FN", "LV"):
            s = df[df.kind == name].sort_values("decyear")
            ax[j].plot(s.decyear, s[col], color=COLOR[name], lw=1.8, label=f"{name} (solver)")
        ax[j].set_ylabel(f"{lbl} YoY %"); ax[j].axhline(0, color="gray", lw=0.6, ls=":")
        ax[j].legend(loc="upper left", fontsize=8, ncol=3)
    ax[0].set_title("Oscilación característica (ventana representativa 1991–2002)\n"
                    "FN ajusta una oscilación; LV degenera bajo el solver (mal-condicionado)",
                    fontsize=11)
    ax[1].set_xlabel("Año")
    fig.tight_layout(); fig.savefig(C.OUTDIR / "p1_trajectory.png", dpi=130); plt.close(fig)

    # ---- Figura 2: retrato de fase ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for axp, name in zip(axes, ("FN", "LV")):
        axp.plot(data.profits, data.investment, "o-", color="black", ms=4, lw=0.8, alpha=0.6, label="Real")
        s = df[df.kind == name].sort_values("decyear")
        axp.plot(s.profits, s.investment, color=COLOR[name], lw=1.8, label=f"{name} (solver)")
        axp.set_xlabel("Profits YoY %"); axp.set_ylabel("Investment YoY %")
        axp.set_title(f"Espacio de fase — {name}")
        axp.axhline(0, color="gray", lw=0.5, ls=":"); axp.axvline(0, color="gray", lw=0.5, ls=":")
        axp.legend(loc="upper left", fontsize=8)
    fig.suptitle("Retrato de fase (1991–2002): FN traza un ciclo; LV colapsa", fontsize=12)
    fig.tight_layout(); fig.savefig(C.OUTDIR / "p1_phase.png", dpi=130); plt.close(fig)

    phase_data()
    print(f"✓ figuras: {C.OUTDIR}/p1_phase_data.png , p1_trajectory.png , p1_phase.png")


def phase_data():
    """Retrato de fase MODEL-FREE: el ciclo existe como rotación, sin ajustar nada.
    Suaviza 4 trimestres para revelar los lazos; colorea por tiempo para ver la dirección."""
    df = C.load()
    sm = df.rolling(4, center=True, min_periods=1).mean()
    yr = df.index.year + (df.index.dayofyear - 1) / 365.25
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(sm["PROFITS_YOY"], sm["INVEST_YOY"], color="gray", lw=0.6, alpha=0.5, zorder=1)
    sc = ax.scatter(sm["PROFITS_YOY"], sm["INVEST_YOY"], c=yr, cmap="viridis", s=14, zorder=2)
    ax.axhline(0, color="k", lw=0.5, ls=":"); ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Profits YoY %"); ax.set_ylabel("Investment YoY %")
    ax.set_title("El ciclo existe como rotación en el plano de fase (model-free)\n"
                 "profits→investment; suavizado 4T, color = año", fontsize=11)
    fig.colorbar(sc, ax=ax, label="año")
    fig.tight_layout(); fig.savefig(C.OUTDIR / "p1_phase_data.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
