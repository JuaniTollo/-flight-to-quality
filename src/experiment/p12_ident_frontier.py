"""Pieza 12 — FRONTERA DE IDENTIFICABILIDAD del lag de maduración (sintético).

Resultado central a hoy (P10/P11): con ~9 episodios y un feedback que explica ~3% de la
varianza (SNR≈0.03), el CENTRO del lag de maduración (μ≈4 trim ≈ 1 año) se identifica de
forma robusta, pero el ANCHO de la distribución del delay NO. Esta pieza convierte ese
muro en un resultado POSITIVO: mapea *a partir de cuántos episodios y a qué SNR* el centro
y el ancho del lag se vuelven recuperables.

Diseño (todo determinista, RNG sembrado):

  1. Generación. Para cada celda (n_ep, L, SNR) se simula un ENSEMBLE de n_ep episodios
     desde un kernel de lag CONOCIDO Gamma(μ*, ancho* via shape*):
         I_t  = driver AR(1) (proxy del ciclo de inversión, escala unitaria),
         m_t  = Σ_τ w(τ; μ*, shape*) · I_{t-τ}        (convolución = cadena de maduración),
         P_t  = b·I_t  −  k·m_t  +  ε_t,
     donde k se CALIBRA por episodio para que la fracción de varianza explicada por el
     término de feedback (−k·m) sobre la varianza total de P sea exactamente SNR. ε es
     ruido gaussiano. Esto reproduce la estructura de p10/p11: co-movimiento contemporáneo
     dominante (b) + feedback de maduración débil (k·m) ahogado en ruido.

  2. Estimación. Se reusa EXACTAMENTE el estimador de p10_stochdelay (filtro Gamma /
     distributed-lag): se barre (μ, s) en grilla, perfilando (a, b, k) por mínimos
     cuadrados, y se elige el (μ̂, ŝ) de mínima SSE pooled. El ancho efectivo del kernel
     estimado es el sd del kernel discretizado.

  3. Métricas por celda (sobre n_rep réplicas con semillas distintas):
       - error/RMSE del CENTRO  : μ̂_eff − μ*  (efectivo del kernel discretizado).
       - error/RMSE del ANCHO   : sd(ŵ) − sd(w*).
       - COBERTURA del IC del centro y del ancho: bootstrap de episodios por réplica;
         fracción de réplicas cuyo IC95 contiene el valor verdadero. Recuperable ≈
         cobertura ≈ 0.95 con RMSE chico; no recuperable ≈ cobertura colapsa o RMSE grande.

Barrido: n_episodios={9,20,50,100} × longitud={13,26,52} × SNR={0.03,0.1,0.3}.

Salida: tabla CSV (results/p12_ident_frontier.csv) + figura de mapas de calor
(output/experiment/p12_ident_frontier.png) mostrando la frontera.

SMOKE TEST:  uv run python -m src.experiment.p12_ident_frontier --smoke
FULL RUN:    uv run python -m src.experiment.p12_ident_frontier
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammaln

from src.experiment import common as C

# --------------------------------------------------------------------------- config
LMAX = 8                      # soporte del kernel: τ = 1..8 trimestres (igual que p10)
TAU = np.arange(1, LMAX + 1)

# verdad sintética calibrada a los hallazgos reales:
MU_TRUE = 4.0                 # centro del lag ~ 1 año (trough empírico convergente)
SHAPE_TRUE = 4.0              # ancho moderado: sd ≈ 2 trim (entre puntual y muy ancho)
B_TRUE = 0.6                  # co-movimiento contemporáneo dominante (lag0 ≈ +0.6 real)
AR_PHI = 0.6                  # persistencia del driver de inversión (ciclo)

RESULTS = C.ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- kernel
def gamma_kernel(mu: float, s: float):
    """Kernel-distribución DISCRETA sobre τ=1..LMAX, media≈μ y desvío≈s (idéntico a p10).
    s→0 colapsa al delay puntual. Devuelve w (LMAX,) con Σ w = 1, o None si inválido."""
    if not (np.isfinite(mu) and np.isfinite(s)) or mu <= 0:
        return None
    if s <= 1e-3:
        w = np.zeros(LMAX)
        j = int(np.clip(round(mu), 1, LMAX)) - 1
        w[j] = 1.0
        return w
    k = (mu / s) ** 2
    theta = s ** 2 / mu
    logpdf = (k - 1) * np.log(TAU) - TAU / theta - k * np.log(theta) - gammaln(k)
    w = np.exp(logpdf - logpdf.max())
    ssum = w.sum()
    if not np.isfinite(ssum) or ssum <= 0:
        return None
    return w / ssum


def kernel_from_shape(mu: float, shape: float):
    """Kernel verdadero parametrizado por (media μ, shape de la Gamma). shape grande =>
    casi puntual; shape chico => ancho. Devuelve w normalizado sobre τ=1..LMAX."""
    s = mu / np.sqrt(shape)            # sd de Gamma(shape, scale=μ/shape) = μ/√shape
    return gamma_kernel(mu, s)


def kernel_moments(w):
    """media y sd EFECTIVOS del kernel discretizado/normalizado."""
    m = float(np.sum(TAU * w))
    v = float(np.sum((TAU - m) ** 2 * w))
    return m, np.sqrt(max(v, 0.0))


# --------------------------------------------------------------------------- generación
def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def gen_episode(rng: np.random.Generator, L: int, snr: float, w_true):
    """Genera UN episodio (P, I) de longitud L con feedback de kernel w_true y SNR dado.

    SNR := Var(−k·m) / Var(P). Se calibra k para alcanzarlo exactamente: con
    P = b·I − k·m + ε, fijamos la escala del término de feedback respecto al total.
    Para que SNR sea estable, fijamos Var(P_señal)=Var(b·I − k·m) y elegimos
    Var(ε) = Var(P_señal)·(1/(1−r)−1)... — más simple y robusto: construimos las
    componentes y escalamos. Devuelve (P (L,), I (L+LMAX,)) con cola de rezagos.
    """
    n_ext = L + LMAX
    # driver AR(1) (proxy del ciclo de inversión), media 0, varianza ~1
    eps_i = rng.standard_normal(n_ext)
    I = np.zeros(n_ext)
    for t in range(1, n_ext):
        I[t] = AR_PHI * I[t - 1] + eps_i[t]
    I = (I - I.mean()) / (I.std() + 1e-12)

    # feedback de maduración: m = convolución del kernel con I (causal)
    m_full = np.convolve(I, w_true)[:n_ext]

    I_win = I[LMAX:]                       # ventana observada (L pts)
    m_win = m_full[LMAX:]
    comov = B_TRUE * I_win                 # término contemporáneo
    fb = m_win - m_win.mean()              # forma del feedback (signo lo pone −k)

    # calibrar k para que Var(k·fb) = snr · Var(P). Resolvemos con P = comov − k·fb + ε,
    # imponiendo Var(−k·fb)/Var(P) = snr y Var(ε) eligida para fijar la varianza total.
    # Fijamos Var(P)=1 (escala arbitraria): Var(−k·fb)=snr -> k = sqrt(snr/Var(fb)).
    # El resto de la varianza la reparten comov y ε; ε absorbe el residuo no determinista.
    vfb = float(np.var(fb)) + 1e-12
    k = np.sqrt(snr / vfb)
    fb_term = -k * fb
    # varianza objetivo total = 1; comov aporta lo suyo; ε completa hasta 1.
    var_signal = float(np.var(comov + fb_term))
    var_noise = max(1.0 - var_signal, 0.05)    # piso para no degenerar el ruido
    noise = rng.standard_normal(L) * np.sqrt(var_noise)
    P = comov + fb_term + noise

    # devolver I extendido (cola de LMAX + ventana) y P de la ventana, como en p10
    I_ext = I.copy()                       # ya incluye LMAX de cola + L de ventana
    return P, I_ext


# --------------------------------------------------------------------------- diseño/ajuste
def lag_matrix(inv_ext, L):
    """Matriz de rezagos (L × LMAX): Lmat[j, τ-1] = inv_{t-τ}, t recorre la ventana."""
    n = len(inv_ext)
    base = n - L
    Lmat = np.zeros((L, LMAX))
    for j in range(L):
        t = base + j
        for li, tau in enumerate(TAU):
            if t - tau >= 0:
                Lmat[j, li] = inv_ext[t - tau]
    return Lmat


def build_windows(episodes, L):
    """Convierte episodios crudos (P, I_ext) en (Pz, inv_now, Lmat) z-scoreados por ventana,
    en el mismo formato que consume el estimador pooled de p10."""
    out = []
    for ei, (P, I_ext) in enumerate(episodes):
        Pz = (P - P.mean()) / (P.std() + 1e-12)
        inv_now = I_ext[-L:].copy()
        Lmat = lag_matrix(I_ext, L)
        out.append((f"ep{ei}", Pz, inv_now, Lmat))
    return out


def fit_given_kernel(windows, mu, s, L):
    """Modelo lineal pooled Pz = a_c + b·inv_now − k·c, c = Lmat@w(μ,s). SSE cerrado."""
    w = gamma_kernel(mu, s)
    if w is None:
        return np.inf
    nC = len(windows)
    rows_X, rows_y = [], []
    for ci, (_, Pz, inv_now, Lmat) in enumerate(windows):
        c = Lmat @ w
        dummies = np.zeros((L, nC)); dummies[:, ci] = 1.0
        X = np.column_stack([dummies, inv_now, -c])
        rows_X.append(X); rows_y.append(Pz)
    X = np.vstack(rows_X); y = np.concatenate(rows_y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return float(resid @ resid)


def estimate_mu_s(windows, L, n_mu=71, n_s=33, s_cap=4.0):
    """Estima (μ, s) por grilla (mínima SSE). Devuelve (μ̂, ŝ, μ_eff, sd_eff).

    La resolución de la grilla acota el sesgo de discretización del estimador: con n_mu=71
    el paso en μ es ~0.1 trim, muy por debajo de las diferencias que mide la frontera, de
    modo que la cobertura del IC no se contamina con sesgo de grilla (importa cuando la
    varianza muestral se achica, p.ej. n_ep=100). El bootstrap usa la MISMA grilla para que
    el punto-estimado caiga dentro de su propia distribución bootstrap."""
    mus = np.linspace(1.0, float(LMAX), n_mu)
    ss = np.concatenate([[1e-3], np.linspace(0.2, s_cap, n_s)])
    best_mu, best_s, best_sse = mus[0], ss[0], np.inf
    for mu in mus:
        for s in ss:
            sse = fit_given_kernel(windows, mu, s, L)
            if sse < best_sse:
                best_sse, best_mu, best_s = sse, mu, s
    w = gamma_kernel(best_mu, best_s)
    me, sd = kernel_moments(w)
    return best_mu, best_s, me, sd


# --------------------------------------------------------------------------- bootstrap CI
def bootstrap_ci(windows, L, rng: np.random.Generator, B=120, n_mu=71, n_s=33):
    """IC95 de (μ_eff, sd_eff) por bootstrap de episodios (resample con reemplazo).
    Usa la MISMA grilla que el punto-estimado para no inyectar sesgo de discretización en
    la cobertura. Devuelve (mu_lo,mu_hi, sd_lo,sd_hi)."""
    nC = len(windows)
    mus_b, sds_b = [], []
    for _ in range(B):
        idx = rng.integers(0, nC, size=nC)
        boot = [windows[i] for i in idx]
        _, _, me, sd = estimate_mu_s(boot, L, n_mu=n_mu, n_s=n_s)
        mus_b.append(me); sds_b.append(sd)
    mu_lo, mu_hi = np.percentile(mus_b, [2.5, 97.5])
    sd_lo, sd_hi = np.percentile(sds_b, [2.5, 97.5])
    return mu_lo, mu_hi, sd_lo, sd_hi


# --------------------------------------------------------------------------- una celda
def run_cell(n_ep, L, snr, n_rep, base_seed, B=120, grid=(71, 33), boot_grid=(71, 33),
             verbose=False):
    """Una celda del barrido: n_rep réplicas. Mide error y cobertura de centro y ancho.
    Determinista: la semilla de cada réplica deriva de (base_seed, n_ep, L, snr, rep)."""
    w_true = kernel_from_shape(MU_TRUE, SHAPE_TRUE)
    mu_star, sd_star = kernel_moments(w_true)

    mu_err, sd_err = [], []
    mu_cov, sd_cov = 0, 0
    n_mu, n_s = grid
    bn_mu, bn_s = boot_grid
    for rep in range(n_rep):
        seed = abs(hash((base_seed, n_ep, L, round(snr, 4), rep))) % (2 ** 32)
        rng = make_rng(seed)
        episodes = [gen_episode(rng, L, snr, w_true) for _ in range(n_ep)]
        windows = build_windows(episodes, L)
        _, _, me, sd = estimate_mu_s(windows, L, n_mu=n_mu, n_s=n_s)
        mu_err.append(me - mu_star)
        sd_err.append(sd - sd_star)
        mu_lo, mu_hi, sd_lo, sd_hi = bootstrap_ci(windows, L, rng, B=B,
                                                  n_mu=bn_mu, n_s=bn_s)
        mu_cov += int(mu_lo <= mu_star <= mu_hi)
        sd_cov += int(sd_lo <= sd_star <= sd_hi)
        if verbose:
            print(f"    rep{rep}: μ̂_eff={me:.2f} (err {me-mu_star:+.2f}, IC[{mu_lo:.2f},{mu_hi:.2f}]) "
                  f"sd={sd:.2f} (err {sd-sd_star:+.2f}, IC[{sd_lo:.2f},{sd_hi:.2f}])")

    mu_err = np.array(mu_err); sd_err = np.array(sd_err)
    return dict(
        n_ep=n_ep, L=L, snr=snr, n_rep=n_rep,
        mu_true=mu_star, sd_true=sd_star,
        mu_bias=float(mu_err.mean()), mu_rmse=float(np.sqrt(np.mean(mu_err ** 2))),
        sd_bias=float(sd_err.mean()), sd_rmse=float(np.sqrt(np.mean(sd_err ** 2))),
        mu_cov=mu_cov / n_rep, sd_cov=sd_cov / n_rep,
    )


# --------------------------------------------------------------------------- figura
def plot_frontier(df: pd.DataFrame, path: Path):
    """Mapas de calor: para cada longitud L, RMSE y cobertura del centro y del ancho
    como función de (n_ep, SNR). Muestra la frontera de recuperabilidad."""
    Ls = sorted(df["L"].unique())
    metrics = [("mu_rmse", "RMSE centro (μ̂_eff−μ*)", "viridis_r"),
               ("mu_cov", "cobertura IC centro", "RdYlGn"),
               ("sd_rmse", "RMSE ancho (sd−sd*)", "viridis_r"),
               ("sd_cov", "cobertura IC ancho", "RdYlGn")]
    nrow, ncol = len(metrics), len(Ls)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False)
    n_eps = sorted(df["n_ep"].unique())
    snrs = sorted(df["snr"].unique())
    for r, (col, title, cmap) in enumerate(metrics):
        for c, L in enumerate(Ls):
            ax = axes[r][c]
            sub = df[df["L"] == L]
            M = np.full((len(snrs), len(n_eps)), np.nan)
            for _, row in sub.iterrows():
                i = snrs.index(row["snr"]); j = n_eps.index(row["n_ep"])
                M[i, j] = row[col]
            vmin, vmax = (0, 1) if col.endswith("cov") else (None, None)
            im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(n_eps))); ax.set_xticklabels(n_eps)
            ax.set_yticks(range(len(snrs))); ax.set_yticklabels(snrs)
            for i in range(len(snrs)):
                for j in range(len(n_eps)):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                                fontsize=7, color="black")
            if r == 0:
                ax.set_title(f"L={L} trim", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{title}\nSNR", fontsize=8)
            ax.set_xlabel("n_episodios", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Pieza 12 — Frontera de identificabilidad del lag de maduración (sintético)\n"
                 "verde = recuperable (cobertura→0.95, RMSE→0)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="celda chica (1 punto, pocas réplicas/bootstrap) para validar el pipeline")
    args = ap.parse_args()

    w_true = kernel_from_shape(MU_TRUE, SHAPE_TRUE)
    mu_star, sd_star = kernel_moments(w_true)
    print("=" * 74)
    print("  Pieza 12 — FRONTERA DE IDENTIFICABILIDAD del lag de maduración (sintético)")
    print("=" * 74)
    print(f"  kernel verdadero: μ*={MU_TRUE} shape*={SHAPE_TRUE}  ->  μ_eff*={mu_star:.2f}  "
          f"sd_eff*={sd_star:.2f}  (b={B_TRUE}, AR φ={AR_PHI})")

    if args.smoke:
        print("\n  [SMOKE] una celda chica: n_ep=9, L=13, SNR=0.03, 2 réplicas, B=20 bootstrap")
        t0 = time.time()
        res = run_cell(n_ep=9, L=13, snr=0.03, n_rep=2, base_seed=20260607,
                       B=20, grid=(21, 13), boot_grid=(13, 9), verbose=True)
        dt = time.time() - t0
        print(f"\n  resultado celda: μ_bias={res['mu_bias']:+.2f} μ_rmse={res['mu_rmse']:.2f} "
              f"cov_centro={res['mu_cov']:.2f} | sd_bias={res['sd_bias']:+.2f} "
              f"sd_rmse={res['sd_rmse']:.2f} cov_ancho={res['sd_cov']:.2f}")
        print(f"  tiempo celda smoke: {dt:.1f}s")
        # estimación de costo del full (el coste crece con L y n_ep; esto es orden de magnitud)
        cells = 4 * 3 * 3
        fits_smoke = 2 * (1 + 20)          # n_rep × (1 estimate + B bootstrap), grilla chica
        fits_full = 40 * (1 + 120)         # n_rep × (1 + B) del full, grilla 71×33
        scale_grid = (71 * 33) / (21 * 13)  # grilla full vs smoke
        scale_L = 6                         # celdas grandes (L=52, n_ep=100) dominan
        per_cell_full = dt / fits_smoke * fits_full * scale_grid
        print(f"  estimación full (≈{cells} celdas × {40} réplicas × B=120, grilla 71×33): "
              f"~{cells * per_cell_full * scale_L / 2 / 60:.0f}–{cells * per_cell_full * scale_L / 60:.0f} min "
              f"(las celdas L=52/n_ep=100 dominan)")
        return

    # ----------------------------------------------------------------- BARRIDO COMPLETO
    n_eps = [9, 20, 50, 100]
    Ls = [13, 26, 52]
    snrs = [0.03, 0.1, 0.3]
    n_rep = 12               # réplicas por celda (reducido de 40 para viabilidad; frontera visible)
    B = 60                   # réplicas bootstrap por IC (reducido de 120; IC95 algo más ruidoso)
    base_seed = 20260607

    rows = []
    t0 = time.time()
    total = len(n_eps) * len(Ls) * len(snrs)
    done = 0
    for L in Ls:
        for snr in snrs:
            for n_ep in n_eps:
                tc = time.time()
                res = run_cell(n_ep, L, snr, n_rep=n_rep, base_seed=base_seed, B=B)
                rows.append(res)
                done += 1
                print(f"  [{done:>2}/{total}] n_ep={n_ep:>3} L={L:>2} SNR={snr:<4} | "
                      f"μ: bias={res['mu_bias']:+.2f} rmse={res['mu_rmse']:.2f} "
                      f"cov={res['mu_cov']:.2f} | sd: bias={res['sd_bias']:+.2f} "
                      f"rmse={res['sd_rmse']:.2f} cov={res['sd_cov']:.2f} "
                      f"({time.time()-tc:.0f}s)")

    df = pd.DataFrame(rows)
    out_csv = RESULTS / "p12_ident_frontier.csv"
    df.to_csv(out_csv, index=False)
    out_png = C.OUTDIR / "p12_ident_frontier.png"
    plot_frontier(df, out_png)

    print("\n" + "=" * 74)
    print("  RESUMEN — ¿dónde se vuelve recuperable el lag?")
    print("=" * 74)
    rec_c = df[(df["mu_cov"] >= 0.9) & (df["mu_rmse"] <= 0.5)]
    rec_w = df[(df["sd_cov"] >= 0.9) & (df["sd_rmse"] <= 0.5)]
    print(f"  CENTRO recuperable (cov≥0.90 y rmse≤0.5) en {len(rec_c)}/{len(df)} celdas")
    if len(rec_c):
        print(f"    mínimo n_ep que lo logra: {rec_c['n_ep'].min()} "
              f"(a SNR={rec_c.loc[rec_c['n_ep'].idxmin(),'snr']})")
    print(f"  ANCHO recuperable  (cov≥0.90 y rmse≤0.5) en {len(rec_w)}/{len(df)} celdas")
    if len(rec_w):
        print(f"    mínimo n_ep que lo logra: {rec_w['n_ep'].min()} "
              f"(a SNR={rec_w.loc[rec_w['n_ep'].idxmin(),'snr']})")
    else:
        print("    -> el ANCHO no se recupera en ninguna celda del barrido (consistente con P10/P11)")
    print(f"\n  tabla -> {out_csv}")
    print(f"  figura -> {out_png}")
    print(f"  tiempo total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
