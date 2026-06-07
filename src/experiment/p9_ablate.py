"""Pieza 9 — Ablations del claim "el oscilador es un modelo de RÉGIMEN DE CRISIS".

Barre la grilla de variantes para ver si el resultado "R² del campo de fase mejor en
crisis que en expansión" es ROBUSTO a las decisiones de diseño, o un artefacto de una
elección particular. Reusa el gradient matching de p4 (Ramsay-Hooker), que es el método
correcto para ventanas cortas (el single-shooting degenera).

Knobs ablados:
  - model:     FN, LV, LIN
  - anchor:    invmin, profmin, peak, trough   (dónde se centra la ventana de crisis)
  - halfwidth: 4, 5, 6, 8 trimestres
  - exclude:   none | big2 (saca 2008 y 2020)
  - smooth:    1 (sin), 3, 5  (puntos del suavizado de du/dt)

Métrica por celda: R² de campo medio en ventanas de crisis vs en ventanas de expansión
de control (mismo largo, mitad de cada expansión), y el contraste (crisis − expansión).

Run:  uv run python -m src.experiment.p9_ablate
"""
from __future__ import annotations

import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.experiment import common as C

MODELS = {"FN": C.FN, "LV": C.LV, "LIN": C.LIN}


# --------------------------------------------------------------------------- núcleo (de p4)
def derivatives(t, Z, win):
    Zs = pd.DataFrame(Z).rolling(win, center=True, min_periods=1).mean().to_numpy() if win > 1 else Z
    return np.gradient(Zs, t, axis=0)


def grad_match(model, U, dU):
    def resid(p):
        if not model.guard(p):
            return np.full(U.size, 1e3)
        F = np.array([model.rhs(u, p) for u in U])
        return (F - dU).ravel()
    return least_squares(resid, model.p0, max_nfev=8000).x


def field_r2(model, U, dU, p):
    F = np.array([model.rhs(u, p) for u in U])
    ss_res = np.sum((dU - F) ** 2, axis=0)
    ss_tot = np.sum((dU - dU.mean(0)) ** 2, axis=0) + 1e-12
    return float(np.mean(1 - ss_res / ss_tot))          # escalar: promedio de las 2 ecuaciones


def window_r2(model, sub, smooth):
    if len(sub) < 6:
        return np.nan
    to_z, *_ = C.zscorer(sub, model.needs_offset)
    U = to_z(sub[C.COLS].to_numpy())
    t = (sub.index - sub.index[0]).days.to_numpy() / 365.25
    dU = derivatives(t, U, smooth)
    return field_r2(model, U, dU, grad_match(model, U, dU))


# --------------------------------------------------------------------------- ventanas
def crisis_windows(df, p, i, anchor, H, exclude):
    idx = df.index
    drop = {"2008", "2020"} if exclude == "big2" else set()
    wins = []
    for lbl, pk, tr in C.RECESSIONS:
        if lbl in drop:
            continue
        pk, tr = pd.Timestamp(pk), pd.Timestamp(tr)
        near = (idx >= pk - pd.offsets.QuarterEnd(2)) & (idx <= tr + pd.offsets.QuarterEnd(4))
        pos = np.where(near)[0]
        if len(pos) == 0:
            continue
        if anchor == "invmin":
            a = pos[np.argmin(i[pos])]
        elif anchor == "profmin":
            a = pos[np.argmin(p[pos])]
        elif anchor == "peak":
            a = pos[0] + 2
        else:                                            # trough NBER
            a = pos[-1] - 4 if len(pos) >= 5 else pos[-1]
        if a - H >= 0 and a + H < len(p):
            wins.append(df.iloc[a - H:a + H + 1])
    return wins


def expansion_windows(df, H):
    """Ventanas de control: mitad de cada período inter-crisis (lejos de toda crisis)."""
    idx = df.index
    wins = []
    for k in range(1, len(C.TROUGHS)):
        lo, hi = C.TROUGHS[k - 1], C.TROUGHS[k]
        mid = lo + (hi - lo) / 2
        a = int(np.argmin(np.abs(idx - mid)))
        if a - H >= 0 and a + H < len(df):
            wins.append(df.iloc[a - H:a + H + 1])
    return wins


# --------------------------------------------------------------------------- barrido
def main():
    df = C.load()
    p, i = df["PROFITS_YOY"].to_numpy(), df["INVEST_YOY"].to_numpy()
    anchors = ["invmin", "profmin", "peak", "trough"]
    Hs = [4, 5, 6, 8]
    excludes = ["none", "big2"]
    smooths = [1, 3, 5]

    rows = []
    print("=== Pieza 9 — ABLATIONS: R² de campo crisis vs expansión ===")
    print(f"{'model':4s} {'anchor':8s} {'H':>2s} {'excl':5s} {'sm':>2s} | "
          f"{'R²_crisis':>9s} {'R²_exp':>7s} {'Δ(c−e)':>7s} {'n_c':>3s}")
    for name, anchor, H, excl, sm in itertools.product(MODELS, anchors, Hs, excludes, smooths):
        m = MODELS[name]
        cw = crisis_windows(df, p, i, anchor, H, excl)
        ew = expansion_windows(df, H)
        rc = np.nanmean([window_r2(m, w, sm) for w in cw]) if cw else np.nan
        re = np.nanmean([window_r2(m, w, sm) for w in ew]) if ew else np.nan
        delta = rc - re
        rows.append(dict(model=name, anchor=anchor, H=H, excl=excl, sm=sm,
                         r2_crisis=rc, r2_exp=re, delta=delta, n=len(cw)))
        print(f"{name:4s} {anchor:8s} {H:2d} {excl:5s} {sm:2d} | "
              f"{rc:+9.3f} {re:+7.3f} {delta:+7.3f} {len(cw):3d}")

    R = pd.DataFrame(rows)
    print("\n=== RESUMEN ===")
    print(f"celdas con Δ>0 (crisis mejor): {(R.delta > 0).sum()}/{len(R)} "
          f"({100*(R.delta > 0).mean():.0f}%)")
    print(f"Δ medio por modelo:")
    for nm in MODELS:
        sub = R[R.model == nm]
        print(f"   {nm}: Δ medio={sub.delta.mean():+.3f}  "
              f"(crisis>exp en {100*(sub.delta > 0).mean():.0f}% de celdas)")
    print(f"Δ medio por ancla:")
    for a in anchors:
        sub = R[R.anchor == a]
        print(f"   {a:8s}: Δ medio={sub.delta.mean():+.3f}")
    # veredicto por modelo NO LINEAL (LIN es el control: esperamos que NO muestre ventaja)
    nl = R[R.model.isin(["FN", "LV"])]
    frac = (nl.delta > 0).mean()
    print(f"\nrobustez (solo no lineales FN/LV, LIN es el control): "
          f"{'SE SOSTIENE' if frac > 0.75 else 'FRÁGIL'} — crisis>exp en {100*frac:.0f}% de celdas")
    print(f"  (LIN, control: crisis>exp en solo {100*(R[R.model=='LIN'].delta>0).mean():.0f}% — "
          f"como se espera, el oscilador lineal no distingue régimen)")

    R.to_csv(C.OUTDIR / "p9_ablations.csv", index=False)

    # heatmap: Δ por (anchor × H), promediando sobre modelo/excl/smooth
    piv = R.pivot_table(index="anchor", columns="H", values="delta", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-abs(piv.values).max(), vmax=abs(piv.values).max())
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("half-width H (trim)"); ax.set_ylabel("ancla")
    for r in range(piv.shape[0]):
        for c in range(piv.shape[1]):
            ax.text(c, r, f"{piv.values[r, c]:+.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Ablation: Δ R² (crisis − expansión)\n>0 = el oscilador describe mejor las crisis")
    fig.colorbar(im, ax=ax, label="Δ R²")
    fig.tight_layout(); fig.savefig(C.OUTDIR / "p9_ablation_heatmap.png", dpi=130); plt.close(fig)
    print(f"\n✓ {C.OUTDIR}/p9_ablations.csv , p9_ablation_heatmap.png")


if __name__ == "__main__":
    main()
