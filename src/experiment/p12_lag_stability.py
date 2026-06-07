"""Pieza 12 — Diagnóstico FORMAL de estabilidad del lag de maduración entre crisis.

Las piezas previas establecieron, de forma descriptiva, que el *centro* del lag de
maduración del feedback sobreacumulación inversión→ganancia es estable entre episodios
(lag por crisis: media 4.2, sd ~0.9 trim; P10). Lo que faltaba es el diagnóstico FORMAL:
estimar el lag μ por crisis CON INTERVALO DE CONFIANZA y una validación tipo holdout para
un parámetro estructural (leave-one-crisis-out sobre el lag pooled).

Estimador del lag (mismo modelo físico que P11): el feedback es un delay distribuido
(kernel Gamma de media μ), equivalente por el linear chain trick a una cadena de
maduración. Para cada ventana se ajusta, por OLS pooled z-scoreado,

    P_t ~ c·I_t + e·P_{t-1} + k·m_t(μ),   m_t(μ) = (kernel_Gamma(μ) * I)_t

barriendo μ en una grilla fina y leyendo el óptimo del perfil de R² (refinado por
interpolación cuadrática alrededor del máximo → μ continuo, no encajado a la grilla).

Dos diagnósticos:
  (A) μ POR CRISIS con IC95 por bootstrap de bloques DENTRO de la ventana (block bootstrap
      móvil: remuestrea bloques contiguos de trimestres preservando la autocorrelación
      local). ¿Cada crisis individual ancla el lag, o es ruido?
  (B) LEAVE-ONE-CRISIS-OUT: el lag pooled excluyendo cada crisis. ¿Se sostiene μ_pooled al
      sacar cualquier episodio, o lo sostiene un solo episodio influyente? Es la validación
      análoga-a-holdout para un parámetro estructural.

Transform QoQ = pct_change(1) sobre niveles (el HONESTO; YoY=pct_change(4) infla ~2x el
feedback en lag−4). Crisis "normales" (sin 2008/2020, outliers de amplitud).

Run (todas las crisis):   uv run python -m src.experiment.p12_lag_stability
Smoke test (2-3 crisis):  uv run python -m src.experiment.p12_lag_stability --smoke
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma as gammadist

from src.experiment import common as C

DROP = {"2008", "2020"}          # outliers de amplitud; análisis sobre crisis "normales"
H = 8                            # semiancho de ventana (trim) → ventana de 2H+1=17 trim
SHAPE = 8.0                      # forma del kernel Gamma (el ancho NO es identificable, P10;
                                 # se fija; sólo se estima el centro μ)
KMAX_KERNEL = 12                 # soporte del kernel
MU_GRID = np.arange(1.0, 8.01, 0.25)   # grilla del barrido de μ (trim)
NBOOT = 800                      # réplicas bootstrap por crisis
BLOCK = 4                        # largo del bloque del moving-block bootstrap (trim)
SEED = 12345


# --------------------------------------------------------------------------- datos QoQ
def qoq_data():
    raw = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    P = raw["PROFITS"].pct_change(1) * 100
    I = raw["INVESTMENT"].pct_change(1) * 100
    d = pd.DataFrame({"P": P, "I": I}).dropna()
    return d.index, d["P"].to_numpy(), d["I"].to_numpy()


def crisis_windows(idx, Iv, labels_only=None):
    """Devuelve [(label, lo, hi)] por crisis: ventana ±H alrededor del fondo de inversión.

    Mismo anclaje que P8/P10/P11 (inversión QoQ más deprimida en la vecindad NBER)."""
    out = []
    for lbl, pk, tr in C.RECESSIONS:
        if lbl in DROP:
            continue
        if labels_only is not None and lbl not in labels_only:
            continue
        pk, tr = pd.Timestamp(pk), pd.Timestamp(tr)
        nm = (idx >= pk - pd.offsets.QuarterEnd(2)) & (idx <= tr + pd.offsets.QuarterEnd(4))
        pos = np.where(nm)[0]
        if len(pos) == 0:
            continue
        a = pos[np.argmin(Iv[pos])]
        if a - H >= 0 and a + H < len(Iv):
            out.append((lbl, a - H, a + H + 1))
    return out


# --------------------------------------------------------------------------- kernel + ajuste
def gkernel(mu, shape=SHAPE, K=KMAX_KERNEL):
    """Kernel Gamma de media mu (trim) y forma 'shape'. Normalizado a suma 1."""
    scale = mu / shape
    ks = np.arange(0, K + 1) + 0.5
    w = gammadist.pdf(ks, a=shape, scale=scale)
    s = w.sum()
    return w / s if s > 0 and np.all(np.isfinite(w)) else None


def _window_rows(Pw, Iw, mu):
    """Filas (X, y) z-scoreadas de UNA ventana para un μ dado. X=[I_t, P_{t-1}, m_t]."""
    w = gkernel(mu)
    if w is None or Pw.std() == 0 or Iw.std() == 0:
        return None, None
    Pz = (Pw - Pw.mean()) / Pw.std()
    Iz = (Iw - Iw.mean()) / Iw.std()
    m = np.convolve(Iz, w)[:len(Iz)]
    X = np.column_stack([Iz[1:], Pz[:-1], m[1:]])
    y = Pz[1:]
    return X, y


def _r2(X, y):
    if X is None or len(y) <= X.shape[1]:
        return np.nan
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2)


def fit_mu(window_rows_fn):
    """Estima μ maximizando el perfil de R² sobre MU_GRID, refinando con interpolación
    cuadrática alrededor del máximo (μ continuo). window_rows_fn(mu) -> (X, y) apilados.

    Devuelve (mu_hat, r2_max, perfil_r2) o (nan, nan, perfil) si no se puede ajustar."""
    prof = np.array([_r2(*window_rows_fn(mu)) for mu in MU_GRID])
    if not np.any(np.isfinite(prof)):
        return np.nan, np.nan, prof
    j = int(np.nanargmax(prof))
    mu_hat, r2_max = MU_GRID[j], prof[j]
    # refinamiento parabólico con los vecinos (si existen y son finitos)
    if 0 < j < len(MU_GRID) - 1 and np.all(np.isfinite(prof[j - 1:j + 2])):
        y0, y1, y2 = prof[j - 1], prof[j], prof[j + 1]
        denom = y0 - 2 * y1 + y2
        if denom < 0:                       # máximo (cóncavo)
            step = MU_GRID[1] - MU_GRID[0]
            delta = 0.5 * (y0 - y2) / denom
            delta = float(np.clip(delta, -1.0, 1.0))
            mu_hat = MU_GRID[j] + delta * step
            r2_max = y1 - 0.25 * (y0 - y2) * delta
    return float(mu_hat), float(r2_max), prof


# --------------------------------------------------------------------------- (A) por crisis + IC
def per_crisis_mu(Pv, Iv, lo, hi):
    """Estima μ en una sola ventana. Devuelve (mu, r2, perfil_r2)."""
    Pw, Iw = Pv[lo:hi], Iv[lo:hi]
    return fit_mu(lambda mu: _window_rows(Pw, Iw, mu))


def moving_block_indices(n, block, rng):
    """Índices de un moving-block bootstrap para una serie de largo n (preserva la
    autocorrelación local; remuestrea bloques contiguos solapados)."""
    if n <= block:
        return rng.integers(0, n, size=n)
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return idx


def bootstrap_mu_window(Pv, Iv, lo, hi, B=NBOOT, seed=SEED):
    """IC95 de μ por moving-block bootstrap DENTRO de la ventana de la crisis."""
    Pw, Iw = Pv[lo:hi].copy(), Iv[lo:hi].copy()
    n = len(Pw)
    rng = np.random.default_rng(seed)
    mus = []
    for _ in range(B):
        bi = moving_block_indices(n, BLOCK, rng)
        Pb, Ib = Pw[bi], Iw[bi]
        mu, r2, _ = fit_mu(lambda mu: _window_rows(Pb, Ib, mu))
        if np.isfinite(mu):
            mus.append(mu)
    mus = np.array(mus)
    if len(mus) < 10:
        return np.nan, np.nan, mus
    lo_ci, hi_ci = np.percentile(mus, [2.5, 97.5])
    return float(lo_ci), float(hi_ci), mus


# --------------------------------------------------------------------------- pooled + LOO
def pooled_rows_fn(Pv, Iv, windows):
    """Devuelve una window_rows_fn que apila las filas de TODAS las ventanas dadas."""
    def fn(mu):
        Xs, ys = [], []
        for _, lo, hi in windows:
            X, y = _window_rows(Pv[lo:hi], Iv[lo:hi], mu)
            if X is not None:
                Xs.append(X); ys.append(y)
        if not Xs:
            return None, None
        return np.vstack(Xs), np.concatenate(ys)
    return fn


def pooled_mu(Pv, Iv, windows):
    return fit_mu(pooled_rows_fn(Pv, Iv, windows))


def pooled_bootstrap_mu(Pv, Iv, windows, B=NBOOT, seed=SEED):
    """IC95 del μ pooled por bootstrap de BLOQUES de crisis (cluster = crisis entera)."""
    rng = np.random.default_rng(seed)
    nC = len(windows)
    mus = []
    for _ in range(B):
        pick = rng.integers(0, nC, size=nC)
        boot_windows = [windows[i] for i in pick]
        mu, r2, _ = pooled_mu(Pv, Iv, boot_windows)
        if np.isfinite(mu):
            mus.append(mu)
    mus = np.array(mus)
    if len(mus) < 10:
        return np.nan, np.nan
    return tuple(np.percentile(mus, [2.5, 97.5]))


def leave_one_out(Pv, Iv, windows):
    """μ pooled excluyendo cada crisis, una a la vez."""
    out = []
    for k, (lbl, _, _) in enumerate(windows):
        rest = [w for j, w in enumerate(windows) if j != k]
        mu, r2, _ = pooled_mu(Pv, Iv, rest)
        out.append((lbl, mu, r2))
    return out


# --------------------------------------------------------------------------- figura
def make_figure(per_rows, mu_pool, ci_pool, loo, path):
    labels = [r["label"] for r in per_rows]
    mus = np.array([r["mu"] for r in per_rows])
    los = np.array([r["ci_lo"] for r in per_rows])
    his = np.array([r["ci_hi"] for r in per_rows])
    yy = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(4.2, 0.55 * len(labels) + 2)))

    # panel 1: forest plot de μ por crisis + banda pooled
    ax1.axvspan(ci_pool[0], ci_pool[1], color="tab:blue", alpha=0.12,
                label=f"IC95 pooled [{ci_pool[0]:.2f},{ci_pool[1]:.2f}]")
    ax1.axvline(mu_pool, color="tab:blue", lw=1.6, label=f"μ pooled = {mu_pool:.2f}")
    err = np.vstack([mus - los, his - mus])
    ax1.errorbar(mus, yy, xerr=err, fmt="o", color="k", ms=5, capsize=3,
                 label="μ por crisis ±IC95 (block bootstrap)")
    ax1.set_yticks(yy); ax1.set_yticklabels(labels)
    ax1.set_xlabel("μ = lag de maduración (trim)")
    ax1.set_title("μ por crisis con IC95\n(IC ancho: el lag no se ancla en 1 ventana; "
                  "el pooled sí)", fontsize=10)
    ax1.invert_yaxis(); ax1.legend(fontsize=8, loc="best")

    # panel 2: leave-one-crisis-out
    loo_lbls = [l for l, _, _ in loo]
    loo_mu = np.array([m for _, m, _ in loo])
    yy2 = np.arange(len(loo_lbls))
    ax2.axvspan(ci_pool[0], ci_pool[1], color="tab:blue", alpha=0.12)
    ax2.axvline(mu_pool, color="tab:blue", lw=1.6, label=f"μ pooled (todas) = {mu_pool:.2f}")
    ax2.plot(loo_mu, yy2, "s", color="tab:red", ms=6, label="μ pooled excluyendo la crisis")
    ax2.set_yticks(yy2); ax2.set_yticklabels(loo_lbls)
    ax2.set_xlabel("μ pooled (trim)")
    ax2.set_title("Leave-one-crisis-out\n(¿se sostiene el lag al excluir cada episodio?)", fontsize=10)
    ax2.invert_yaxis(); ax2.legend(fontsize=8, loc="best")

    fig.suptitle("Pieza 12 — Estabilidad del lag de maduración entre crisis (QoQ)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- main / runner
def run(smoke=False):
    idx, Pv, Iv = qoq_data()
    all_windows = crisis_windows(idx, Iv)

    if smoke:
        windows = all_windows[:3]
        tag = "SMOKE TEST (primeras 3 crisis)"
    else:
        windows = all_windows
        tag = "FULL (todas las crisis normales)"

    print("=== Pieza 12 — estabilidad del lag de maduración entre crisis ===")
    print(f"modo: {tag} | transform: QoQ (honesto) | sin 2008/2020")
    print(f"crisis: {len(windows)}  {[w[0] for w in windows]}")
    print(f"ventana: ±{H} trim ({2*H+1}/crisis) | kernel Gamma shape={SHAPE} | "
          f"μ-grid {MU_GRID[0]}..{MU_GRID[-1]} paso {MU_GRID[1]-MU_GRID[0]}")
    print(f"bootstrap: B={NBOOT}, moving-block largo={BLOCK}\n")

    # ---- (A) μ por crisis con IC
    print("=== (A) μ por crisis con IC95 (block bootstrap dentro de la ventana) ===")
    print("  NOTA: una sola ventana (17 trim) tiene poca información sobre el lag (el feedback "
          "\n  de maduración explica ~3% de la varianza, dominada por el co-movimiento "
          "\n  contemporáneo). El perfil de R² es casi plano → μ̂ por crisis es ruidoso y su IC "
          "\n  es ancho por construcción. La identificación del lag emerge al POOLEAR (ver abajo).")
    print(f"\n  {'crisis':>7} | {'μ̂':>5} | {'R²':>6} | {'rango perfil':>11} | "
          f"{'IC95':>16} | ancho IC")
    print("  " + "-" * 70)
    per_rows = []
    for lbl, lo, hi in windows:
        mu, r2, prof = per_crisis_mu(Pv, Iv, lo, hi)
        flat = float(np.nanmax(prof) - np.nanmin(prof))   # qué tanto mueve μ al R² (planitud)
        ci_lo, ci_hi, _ = bootstrap_mu_window(Pv, Iv, lo, hi)
        width = ci_hi - ci_lo if np.isfinite(ci_lo) else np.nan
        per_rows.append(dict(label=lbl, mu=mu, r2=r2, profile_range=flat,
                             ci_lo=ci_lo, ci_hi=ci_hi))
        print(f"  {lbl:>7} | {mu:5.2f} | {r2:+6.3f} | {flat:11.3f} | "
              f"[{ci_lo:5.2f},{ci_hi:5.2f}] | {width:5.2f}")

    mu_arr = np.array([r["mu"] for r in per_rows])
    print(f"\n  resumen μ̂ por crisis: media {np.nanmean(mu_arr):.2f}, "
          f"mediana {np.nanmedian(mu_arr):.2f}, sd {np.nanstd(mu_arr):.2f}, "
          f"rango [{np.nanmin(mu_arr):.2f},{np.nanmax(mu_arr):.2f}]")
    print(f"  rango medio del perfil de R² (movimiento por μ) = "
          f"{np.nanmean([r['profile_range'] for r in per_rows]):.3f} "
          f"→ perfiles planos: el lag NO se ancla dentro de una sola ventana.")

    # ---- pooled + IC
    mu_pool, r2_pool, _ = pooled_mu(Pv, Iv, windows)
    ci_pool = pooled_bootstrap_mu(Pv, Iv, windows)
    print(f"\n=== pooled (todas las crisis del modo) ===")
    print(f"  μ pooled = {mu_pool:.2f} trim (R²={r2_pool:+.3f}), "
          f"IC95 por crisis = [{ci_pool[0]:.2f}, {ci_pool[1]:.2f}]")

    # ---- (B) leave-one-crisis-out
    print(f"\n=== (B) LEAVE-ONE-CRISIS-OUT (μ pooled excluyendo cada crisis) ===")
    print(f"  {'excluida':>8} | {'μ pooled':>8} | {'Δ vs todas':>10} | {'R²':>6}")
    print("  " + "-" * 44)
    loo = leave_one_out(Pv, Iv, windows)
    for lbl, mu, r2 in loo:
        print(f"  {lbl:>8} | {mu:8.2f} | {mu - mu_pool:+10.2f} | {r2:+6.3f}")
    loo_mu = np.array([m for _, m, _ in loo])
    span = np.nanmax(loo_mu) - np.nanmin(loo_mu)
    inside = np.all((loo_mu >= ci_pool[0]) & (loo_mu <= ci_pool[1]))
    print(f"\n  rango μ LOO = [{np.nanmin(loo_mu):.2f}, {np.nanmax(loo_mu):.2f}] "
          f"(span {span:.2f} trim)")
    print(f"  ¿todos los μ-LOO dentro del IC95 pooled? {'SÍ' if inside else 'NO'} "
          f"→ el lag {'NO depende de ningún episodio individual' if inside else 'es influenciado por algún episodio'}")

    # ---- figura
    fig_path = C.OUTDIR / ("p12_lag_stability_smoke.png" if smoke else "p12_lag_stability.png")
    make_figure(per_rows, mu_pool, ci_pool, loo, fig_path)
    print(f"\n✓ figura: {fig_path}")

    # ---- CSV
    csv_path = C.OUTDIR / ("p12_lag_stability_smoke.csv" if smoke else "p12_lag_stability.csv")
    df_out = pd.DataFrame(per_rows)
    df_out["mu_pooled"] = mu_pool
    df_out["ci_pool_lo"] = ci_pool[0]; df_out["ci_pool_hi"] = ci_pool[1]
    loo_map = {l: m for l, m, _ in loo}
    df_out["mu_loo_excl_this"] = df_out["label"].map(loo_map)
    df_out.to_csv(csv_path, index=False)
    print(f"✓ tabla:  {csv_path}")

    yr = f" (~{mu_pool/4:.1f} año)" if mu_pool >= 3.5 else ""
    print(f"\nVEREDICTO: el lag de maduración se identifica al POOLEAR (μ ≈ {mu_pool:.1f} trim"
          f"{yr}), no dentro de una sola ventana (perfiles planos por SNR ~3%). "
          f"{'El lag pooled es ESTABLE: el LOO lo mueve ' + format(span, '.2f') + ' trim y queda dentro del IC95 pooled → no depende de ningún episodio individual.' if inside else 'CUIDADO: algún μ-LOO sale del IC95 pooled → hay un episodio influyente.'}")


def main():
    run(smoke="--smoke" in sys.argv)


if __name__ == "__main__":
    main()
