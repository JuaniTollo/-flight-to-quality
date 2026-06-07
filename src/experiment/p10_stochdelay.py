"""Pieza 10 — DELAY ESTOCÁSTICO / KERNEL DE LAG DISTRIBUIDO (hipótesis central).

Hipótesis: el feedback de sobreacumulación (inversión pasada
deprime la ganancia) NO es un retardo determinístico de τ fijo ni un oscilador de fase
fija. Es un KERNEL DE LAG DISTRIBUIDO: la demora τ a la que la acumulación golpea la
ganancia es una VARIABLE ALEATORIA con distribución propia (media μ ~4, dispersión s),
que además fluctúa entre crisis.

Modelo (todo en QoQ = pct_change(1) sobre NIVELES, la transform honesta; YoY infla ~2x):

    profits_t = a + b·invest_t  −  k · Σ_{τ=1}^{L} w(τ; μ, s) · invest_{t-τ}  + ε

  - b·invest_t : co-movimiento contemporáneo (el lag0 = +0.62 que DOMINA el patrón).
  - k > 0      : fuerza total del feedback de sobreacumulación (signo negativo => deprime).
  - w(τ; μ, s) : kernel-distribución DISCRETA, normalizado Σ w = 1, con media μ y desvío s.
                 Implementado como Gamma discretizada (pmf ∝ Gamma_pdf en τ=1..L).
                 μ = lag medio (~4 esperado) ; s = dispersión del lag (¿es punto o distribución?).

Test central:
  (1) Estimar (μ, s) POOLED sobre las crisis (QoQ) y POR crisis. ¿s > 0?  ¿μ ≈ 4?
  (2) KERNEL-DISTRIBUCIÓN (s libre) vs DELAY PUNTUAL (s → 0, todo el peso en un solo τ).
      ¿Mejora el ajuste el kernel distribuido? (ΔSSE, R2, AICc por nº de parámetros).
  (3) IDENTIFICABILIDAD honesta de μ vs s con ~13 puntos por crisis: superficie de perfil
      de SSE(μ, s), y CI por bootstrap de bloques sobre el pool. Si s no se identifica,
      decirlo claro.

Reglas: QoQ principal; n=9 crisis usables (sin 2008/2020 por el protocolo establecido);
ser crítico/honesto sobre identificabilidad (1 crisis ~ 1 oscilación ~ 13 pts ruidosos).

Run:  uv run python -m src.experiment.p10_stochdelay
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from src.experiment import common as C

HW = 6                     # media-ventana: ±6 trim del fondo (mismo ancla que p8/p9)
WIN = 2 * HW + 1           # 13 puntos por ventana
LMAX = 8                   # soporte del kernel: τ = 1..8 trimestres
DROP = ("2008", "2020")    # protocolo: muestra "endógena" sin las dos crisis financieras
TAU = np.arange(1, LMAX + 1)


# --------------------------------------------------------------------------- kernel
def gamma_kernel(mu, s):
    """Kernel-distribución DISCRETA sobre τ=1..LMAX con media≈μ y desvío≈s.

    Gamma(shape=k, scale=θ) tiene media kθ y var kθ². Fijamos k=(μ/s)², θ=s²/μ y
    evaluamos la pdf en los enteros τ, luego normalizamos a Σ=1 (pmf discreta).
    s→0  => toda la masa colapsa al τ entero más cercano a μ (DELAY PUNTUAL).
    Devuelve w (LMAX,) con Σ w = 1, o None si los parámetros son inválidos.
    """
    if not (np.isfinite(mu) and np.isfinite(s)) or mu <= 0:
        return None
    if s <= 1e-3:                                   # límite puntual: masa en el τ ~ μ
        w = np.zeros(LMAX)
        j = int(np.clip(round(mu), 1, LMAX)) - 1
        w[j] = 1.0
        return w
    k = (mu / s) ** 2
    theta = s ** 2 / mu
    # log pdf Gamma en TAU (evita overflow): (k-1)ln τ - τ/θ - k ln θ - lnΓ(k)
    logpdf = (k - 1) * np.log(TAU) - TAU / theta - k * np.log(theta) - gammaln(k)
    w = np.exp(logpdf - logpdf.max())
    ssum = w.sum()
    if not np.isfinite(ssum) or ssum <= 0:
        return None
    return w / ssum


def kernel_moments(w):
    """media y desvío EFECTIVOS del kernel discretizado/normalizado (lo que de verdad pesa)."""
    m = float(np.sum(TAU * w))
    v = float(np.sum((TAU - m) ** 2 * w))
    return m, np.sqrt(max(v, 0.0))


# --------------------------------------------------------------------------- diseño
def lag_matrix(inv):
    """Precomputa, para una ventana, la matriz de rezagos Lmat (WIN × LMAX) con
    Lmat[j, τ-1] = inv_{t-τ}, donde t recorre los WIN puntos de la ventana. `inv` es el
    vector EXTENDIDO (cola de LMAX + ventana). Con esto c = Lmat @ w (mat-vec, rápido)."""
    n = len(inv)
    base = n - WIN
    Lmat = np.zeros((WIN, LMAX))
    for j in range(WIN):
        t = base + j
        for li, tau in enumerate(TAU):
            if t - tau >= 0:
                Lmat[j, li] = inv[t - tau]
    return Lmat


def build_design(Lmat, mu, s):
    """Columna de feedback distribuido c = Σ_τ w(τ) inv_{t-τ} = Lmat @ w. Devuelve c o None."""
    w = gamma_kernel(mu, s)
    if w is None:
        return None
    return Lmat @ w


# --------------------------------------------------------------------------- datos
def crisis_windows_qoq(df):
    """Ventanas ±HW del fondo de inversión, QoQ desde NIVELES. Devuelve, por crisis usable:
    (label, prof_win(WIN), inv_ext(LMAX+WIN)) donde inv_ext incluye la cola de rezagos
    REAL (no z-scoreada, para que el kernel opere sobre la serie cruda QoQ de la crisis).
    Cada serie se z-scorea POR VENTANA al final para comparabilidad de escala entre crisis.
    """
    idx = df.index
    prof_lvl = df["PROFITS"].to_numpy()
    inv_lvl = df["INVESTMENT"].to_numpy()
    # QoQ = pct_change(1) sobre niveles (transform honesta)
    prof = np.concatenate([[np.nan], np.diff(prof_lvl) / prof_lvl[:-1]]) * 100
    inv = np.concatenate([[np.nan], np.diff(inv_lvl) / inv_lvl[:-1]]) * 100

    out = []
    for lbl, pk, tr in C.RECESSIONS:
        if lbl in DROP:
            continue
        pk, tr = pd.Timestamp(pk), pd.Timestamp(tr)
        win = (idx >= pk - pd.offsets.QuarterEnd(2)) & (idx <= tr + pd.offsets.QuarterEnd(4))
        pos = np.where(win)[0]
        if len(pos) == 0:
            continue
        a = pos[np.argmin(inv[pos])] if not np.isnan(inv[pos]).all() else pos[len(pos) // 2]
        # necesitamos LMAX rezagos antes del inicio de ventana, y QoQ válido (idx>=1)
        if a - HW - LMAX < 1 or a + HW >= len(prof):
            continue
        win_sl = slice(a - HW, a + HW + 1)
        ext_sl = slice(a - HW - LMAX, a + HW + 1)
        P = prof[win_sl].copy()
        Iext = inv[ext_sl].copy()
        if np.isnan(P).any() or np.isnan(Iext).any():
            continue
        # z-score por ventana (la regresión lineal a,b absorbe media; z solo iguala escala)
        Pz = (P - P.mean()) / P.std()
        inv_now = Iext[-WIN:].copy()
        Lmat = lag_matrix(Iext)
        out.append((lbl, Pz, inv_now, Lmat, P.std(), P.mean()))
    return out


# --------------------------------------------------------------------------- ajuste lineal
def fit_linear_given_kernel(windows, mu, s, pooled=True):
    """Dado (μ, s), el modelo es LINEAL en (a, b, k) por ventana o pooled:
        Pz_t = a + b·inv_t − k·c_t(μ,s)
    Pooled: a_c por crisis (intercepto libre), b y k COMPARTIDOS entre crisis.
    Indep : a, b, k por crisis. Devuelve SSE total y dict por crisis.
    Resuelto por mínimos cuadrados cerrado (sin optimizador) para velocidad y estabilidad.
    """
    designs = []
    for lbl, Pz, inv_now, Lmat, _, _ in windows:
        c = build_design(Lmat, mu, s)
        if c is None:
            return np.inf, None, None
        designs.append((lbl, Pz, inv_now, c))

    if pooled:
        # variables compartidas: b (inv_now), k (−c). interceptos a_c por crisis (dummies).
        nC = len(designs)
        rows_y, rows_X = [], []
        for ci, (lbl, Pz, inv_now, c) in enumerate(designs):
            dummies = np.zeros((WIN, nC)); dummies[:, ci] = 1.0
            X = np.column_stack([dummies, inv_now, -c])   # [a_1..a_nC, b, k]
            rows_X.append(X); rows_y.append(Pz)
        X = np.vstack(rows_X); y = np.concatenate(rows_y)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        b, k = coef[-2], coef[-1]
        resid = y - X @ coef
        sse = float(resid @ resid)
        per = {}
        off = 0
        for ci, (lbl, Pz, inv_now, c) in enumerate(designs):
            a_c = coef[ci]
            pred = a_c + b * inv_now - k * c
            r = Pz - pred
            per[lbl] = dict(sse=float(r @ r), a=a_c, b=b, k=k, pred=pred, obs=Pz, c=c)
            off += WIN
        return sse, per, dict(b=b, k=k)
    else:
        sse = 0.0; per = {}
        for lbl, Pz, inv_now, c in designs:
            X = np.column_stack([np.ones(WIN), inv_now, -c])
            coef, *_ = np.linalg.lstsq(X, Pz, rcond=None)
            pred = X @ coef
            r = Pz - pred
            sse += float(r @ r)
            per[lbl] = dict(sse=float(r @ r), a=coef[0], b=coef[1], k=coef[2],
                            pred=pred, obs=Pz, c=c)
        return sse, per, None


# --------------------------------------------------------------------------- estimación μ,s
def estimate_mu_s(windows, pooled=True, s_floor=1e-3, s_cap=4.0, n_mu=71, n_s=39):
    """Estima (μ, s) minimizando SSE (perfilando a,b,k por mínimos cuadrados).
    Búsqueda en grilla (robusta, evita óptimos locales del kernel).
    Devuelve (mu*, s*, sse*, per, shared, surf). n_mu/n_s controlan la resolución."""
    mus = np.linspace(1.0, float(LMAX), n_mu)
    ss = np.concatenate([[s_floor], np.linspace(0.2, s_cap, n_s)])
    best = (None, None, np.inf, None, None)
    surf = np.full((len(mus), len(ss)), np.nan)
    for im, mu in enumerate(mus):
        for js, s in enumerate(ss):
            sse, per, shared = fit_linear_given_kernel(windows, mu, s, pooled=pooled)
            surf[im, js] = sse
            if sse < best[2]:
                best = (mu, s, sse, per, shared)
    return best[0], best[1], best[2], best[3], best[4], (mus, ss, surf)


def best_point_delay(windows, pooled=True):
    """DELAY PUNTUAL: kernel con toda la masa en un solo τ entero (s→0). Barre τ=1..LMAX
    y elige el mejor. Equivale a μ=τ, s→0. Devuelve (tau*, sse*, per, shared)."""
    best = (None, np.inf, None, None)
    for tau in TAU:
        sse, per, shared = fit_linear_given_kernel(windows, float(tau), 1e-3, pooled=pooled)
        if sse < best[1]:
            best = (tau, sse, per, shared)
    return best


# --------------------------------------------------------------------------- métricas
def aicc(sse, n, p):
    """AICc gaussiano. p = nº de parámetros (incluye σ). n = nº de observaciones."""
    if sse <= 0 or n - p - 1 <= 0:
        return np.nan
    ll = -0.5 * n * (np.log(2 * np.pi * sse / n) + 1)
    return 2 * p - 2 * ll + (2 * p * (p + 1)) / (n - p - 1)


# --------------------------------------------------------------------------- bootstrap CI
def bootstrap_mu_s(windows, B=200, seed=20260606):
    """CI de (μ, s) por bootstrap de CRISIS (re-muestreo de las 9 ventanas con reemplazo).
    Honesto sobre identificabilidad: si s* pega en el piso o el CI de s incluye ~0, el lag
    NO se distingue de un punto. Determinista (LCG sembrado)."""
    nC = len(windows)
    s = seed
    mus_b, ss_b = [], []
    for _ in range(B):
        rows = []
        for _ in range(nC):
            s = (1103515245 * s + 12345) % (2 ** 31)
            rows.append(s % nC)
        boot = [windows[r] for r in rows]
        mu_b, s_b, *_ = estimate_mu_s(boot, pooled=True, n_mu=29, n_s=19)
        mus_b.append(mu_b); ss_b.append(s_b)
    return np.array(mus_b), np.array(ss_b)


# --------------------------------------------------------------------------- figuras
def plot_kernel_and_surface(mu, s, surf_pack, mus_b, ss_b, per_pooled):
    mus, ss, surf = surf_pack
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (1) kernel estimado vs delay puntual
    ax = axes[0]
    w = gamma_kernel(mu, s)
    ax.bar(TAU - 0.15, w, width=0.3, color="tab:red", alpha=0.8, label=f"kernel-dist (μ={mu:.1f}, s={s:.2f})")
    wp = gamma_kernel(round(mu), 1e-3)
    ax.bar(TAU + 0.15, wp, width=0.3, color="grey", alpha=0.6, label=f"delay puntual (τ={int(round(mu))})")
    ax.set_xlabel("τ (trimestres de rezago)"); ax.set_ylabel("peso w(τ)")
    ax.set_title("Kernel de lag distribuido estimado (pooled, QoQ)", fontsize=10)
    ax.legend(fontsize=8)

    # (2) superficie de perfil SSE(μ, s) -> identificabilidad
    ax = axes[1]
    Z = surf.T
    cs = ax.contourf(mus, ss, Z, levels=25, cmap="viridis")
    ax.plot(mu, s, "r*", ms=14)
    ax.set_xlabel("μ (lag medio)"); ax.set_ylabel("s (dispersión del lag)")
    ax.set_title("Perfil SSE(μ, s)  ·  estrella = óptimo\nvalle plano en s => s no identificable", fontsize=10)
    fig.colorbar(cs, ax=ax, shrink=0.85, label="SSE pooled")

    # (3) bootstrap de (μ, s)
    ax = axes[2]
    ax.scatter(mus_b, ss_b, s=12, alpha=0.4, color="tab:blue")
    ax.plot(mu, s, "r*", ms=14)
    ax.axhline(0.2, color="grey", ls=":", lw=1)
    ax.set_xlabel("μ bootstrap"); ax.set_ylabel("s bootstrap")
    ax.set_title(f"Bootstrap de crisis (B={len(mus_b)})\nμ CI95=[{np.percentile(mus_b,2.5):.1f},"
                 f"{np.percentile(mus_b,97.5):.1f}]  s CI95=[{np.percentile(ss_b,2.5):.2f},"
                 f"{np.percentile(ss_b,97.5):.2f}]", fontsize=9)

    fig.tight_layout()
    path = C.OUTDIR / "p10_stochdelay.png"
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


# --------------------------------------------------------------------------- main
def main():
    df = C.load_raw() if hasattr(C, "load_raw") else pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    windows = crisis_windows_qoq(df)
    labels = [w[0] for w in windows]
    nObs = len(windows) * WIN
    print("=" * 74)
    print("  Pieza 10 — DELAY ESTOCÁSTICO / KERNEL DE LAG DISTRIBUIDO (QoQ, hipótesis central)")
    print("=" * 74)
    print(f"  Crisis usables (sin {DROP}): {len(windows)} -> {labels}")
    print(f"  {WIN} pts/ventana ; soporte kernel τ=1..{LMAX} ; nObs total = {nObs}")

    # ---- POOLED: kernel distribuido (s libre) vs delay puntual (s->0)
    mu_p, s_p, sse_kd, per_kd, shared_kd, surf_pack = estimate_mu_s(windows, pooled=True)
    w_p = gamma_kernel(mu_p, s_p)
    m_eff, s_eff = kernel_moments(w_p)
    tau_pt, sse_pt, per_pt, shared_pt = best_point_delay(windows, pooled=True)

    # nº de parámetros pooled: interceptos a_c (nC) + b + k + (μ,s o τ) + σ
    nC = len(windows)
    p_kd = nC + 2 + 2 + 1          # a_c, b, k, μ, s, σ
    p_pt = nC + 2 + 1 + 1          # a_c, b, k, τ(entero), σ
    aic_kd = aicc(sse_kd, nObs, p_kd)
    aic_pt = aicc(sse_pt, nObs, p_pt)
    tss = sum(float((w[1] - w[1].mean()) @ (w[1] - w[1].mean())) for w in windows)
    r2_kd = 1 - sse_kd / tss
    r2_pt = 1 - sse_pt / tss

    print("\n" + "-" * 74)
    print("  (1) POOLED — KERNEL DISTRIBUIDO (s libre)  vs  DELAY PUNTUAL (s→0)")
    print("-" * 74)
    print(f"  KERNEL-DIST:  μ*={mu_p:.2f}  s*={s_p:.2f}  (efectivo: media={m_eff:.2f}, sd={s_eff:.2f})")
    print(f"                b(co-mov contemp)={shared_kd['b']:+.3f}  k(feedback)={shared_kd['k']:+.3f}")
    print(f"                SSE={sse_kd:.2f}  R2={r2_kd:+.3f}  AICc={aic_kd:.1f}")
    print(f"  DELAY-PUNTUAL: τ*={tau_pt}        SSE={sse_pt:.2f}  R2={r2_pt:+.3f}  AICc={aic_pt:.1f}")
    print(f"  ΔSSE (punt−dist)={sse_pt - sse_kd:+.2f}  "
          f"({100*(sse_pt - sse_kd)/sse_pt:+.1f}% del punt)  ΔAICc(punt−dist)={aic_pt - aic_kd:+.1f}")
    print(f"  kernel pooled w(τ): " + "  ".join(f"τ{t}={wv:.2f}" for t, wv in zip(TAU, w_p)))

    # ---- POR CRISIS
    print("\n" + "-" * 74)
    print("  (2) POR CRISIS — (μ, s) idiosincrásicos (¿el lag fluctúa entre crisis?)")
    print("-" * 74)
    print(f"  {'crisis':>7} | {'μ':>5} | {'s':>5} | {'μ_eff':>5} | {'s_eff':>5} | {'k':>6} | {'SSE':>6}")
    print("  " + "-" * 56)
    mu_each, s_each = [], []
    for w in windows:
        lbl = w[0]
        mu_c, s_c, sse_c, per_c, _, _ = estimate_mu_s([w], pooled=True)
        wc = gamma_kernel(mu_c, s_c)
        me, sd = kernel_moments(wc)
        kk = per_c[lbl]["k"]
        mu_each.append(mu_c); s_each.append(s_c)
        print(f"  {lbl:>7} | {mu_c:5.2f} | {s_c:5.2f} | {me:5.2f} | {sd:5.2f} | {kk:+6.2f} | {sse_c:6.2f}")
    mu_each, s_each = np.array(mu_each), np.array(s_each)
    print(f"\n  μ por crisis: media={mu_each.mean():.2f}  sd={mu_each.std():.2f}  "
          f"rango=[{mu_each.min():.1f},{mu_each.max():.1f}]")
    print(f"  s por crisis: media={s_each.mean():.2f}  sd={s_each.std():.2f}  "
          f"(¿cuántas pegan el piso s<0.25? {int((s_each < 0.25).sum())}/{len(s_each)})")

    # ---- IDENTIFICABILIDAD: bootstrap + perfil de s a μ fijo
    print("\n" + "-" * 74)
    print("  (3) IDENTIFICABILIDAD de μ vs s  (n=9 crisis, 13 pts c/u)")
    print("-" * 74)
    mus_b, ss_b = bootstrap_mu_s(windows, B=300)
    mu_lo, mu_hi = np.percentile(mus_b, [2.5, 97.5])
    s_lo, s_hi = np.percentile(ss_b, [2.5, 97.5])
    print(f"  bootstrap μ: media={mus_b.mean():.2f}  CI95=[{mu_lo:.2f},{mu_hi:.2f}]")
    print(f"  bootstrap s: media={ss_b.mean():.2f}  CI95=[{s_lo:.2f},{s_hi:.2f}]  "
          f"(piso s≤0.05 en {int((ss_b<=0.05).sum())}/{len(ss_b)} reps)")

    # perfil de SSE en s, con μ fijo en el óptimo -> ¿hay valle plano? (no identificable)
    s_grid = np.concatenate([[1e-3], np.linspace(0.2, 4.0, 20)])
    prof = []
    for s in s_grid:
        sse, _, _ = fit_linear_given_kernel(windows, mu_p, s, pooled=True)
        prof.append(sse)
    prof = np.array(prof)
    flat = (prof.max() - prof.min()) / prof.min()
    print(f"  perfil SSE(s | μ={mu_p:.1f}): min={prof.min():.2f} (s={s_grid[np.argmin(prof)]:.2f})  "
          f"max={prof.max():.2f}  variación relativa={100*flat:.1f}%")
    print(f"    -> {'VALLE PLANO: s mal identificada' if flat < 0.05 else 'curvatura en s: s tiene información'}")

    path = plot_kernel_and_surface(mu_p, s_p, surf_pack, mus_b, ss_b, per_kd)

    # ---- VEREDICTO
    print("\n" + "=" * 74)
    print("  VEREDICTO (honesto)")
    print("=" * 74)
    s_sig = s_lo > 0.1 and (s_each < 0.25).sum() < len(s_each) / 2
    improves = (aic_pt - aic_kd) > 2
    print(f"  ¿s>0 significativo (CI95 lejos de 0)?  {'SÍ' if s_sig else 'NO/dudoso'}  "
          f"(s CI95=[{s_lo:.2f},{s_hi:.2f}])")
    print(f"  ¿kernel-distribución mejora vs delay puntual (ΔAICc>2)?  "
          f"{'SÍ' if improves else 'NO'}  (ΔAICc={aic_pt - aic_kd:+.1f}, ΔSSE={sse_pt - sse_kd:+.2f})")
    print(f"  μ pooled ≈ {mu_p:.1f} (esperado ~4 por el trough empírico); μ_eff={m_eff:.1f}")
    print(f"  figura -> {path}")

    return dict(mu=mu_p, s=s_p, sse_kd=sse_kd, sse_pt=sse_pt, tau_pt=tau_pt,
                aic_kd=aic_kd, aic_pt=aic_pt, mu_ci=(mu_lo, mu_hi), s_ci=(s_lo, s_hi),
                r2_kd=r2_kd, mu_each=mu_each, s_each=s_each, shared=shared_kd, flat=flat)


if __name__ == "__main__":
    main()
