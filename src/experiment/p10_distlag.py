"""Pieza 10 — KERNEL DE LAG DISTRIBUIDO explícito (no oscilador, no τ fijo).

Hipótesis: el feedback de sobreacumulación (inversión pasada
-> deprime ganancia) NO es un retardo determinístico de τ fijo NI un oscilador de fase fija,
sino un KERNEL DE LAG DISTRIBUIDO estocástico: la demora a la que la acumulación golpea la
ganancia es una distribución, y esa distribución FLUCTÚA entre crisis.

Modelo (explícito, lineal, finito):

    profits_growth_t = a·investment_t + Σ_{k=1..8} β_k · investment_{t-k} + ε_t

donde profits_growth y investment son QoQ = pct_change(1) sobre los NIVELES (transform
HONESTO; el YoY=pct_change(4) infla ~2x y mete estructura espuria de orden 4).

El vector (a=β_0, β_1, ..., β_8) ES el kernel. Lo leemos como una "distribución" de lag:
  - centro: lag medio ponderado por |β_k| (o por la parte negativa).
  - dispersión: sd del lag ponderado.
  - ¿la joroba negativa está en k=4-5? -> masa negativa concentrada ahí.

Tres estimaciones:
  (1) POOLED libre — OLS de β_k sobre el ensemble de crisis (pares dentro de ventana).
  (2) POOLED Almon — β_k = polinomio de grado d en k (suaviza, reduce varianza/colinealidad).
  (3) PER-CRISIS — un kernel por crisis -> mide la FLUCTUACIÓN del kernel entre crisis.

CRÍTICO (lo reportamos explícito, sin inflar):
  - colinealidad entre rezagos: número de condición y VIF de la matriz de diseño.
  - n chico: 9 crisis × ~13 trim por ventana; per-crisis el kernel de 9 coef es casi singular.
  - intervalos de confianza de β_k: analíticos (HAC/Newey-West) Y bootstrap por bloques de crisis.

Run:  uv run python -m src.experiment.p10_distlag
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment import common as C

KMAX = 8                    # rezagos 1..8 (más a=β0 contemporáneo)  -> kernel de 9 coef
HW = 6                      # media-ventana ±6 trim del fondo (mismo ancla que p8/p9)
SEED = 12345
NBOOT = 4000


# --------------------------------------------------------------------------- datos QoQ
def load_qoq():
    """QoQ = pct_change(1) sobre NIVELES PROFITS, INVESTMENT (transform honesto)."""
    df = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    out = pd.DataFrame(index=df.index)
    out["PROF"] = np.log(df["PROFITS"]).diff() * 100.0
    out["INV"] = np.log(df["INVESTMENT"]).diff() * 100.0
    return out.dropna()


def crisis_anchors(df):
    """Ancla por recesión = inversión QoQ más deprimida en su vecindad NBER (igual que p8)."""
    idx = df.index
    inv = df["INV"].to_numpy()
    out = []
    for lbl, pk, tr in C.RECESSIONS:
        pk, tr = pd.Timestamp(pk), pd.Timestamp(tr)
        win = (idx >= pk - pd.offsets.QuarterEnd(2)) & (idx <= tr + pd.offsets.QuarterEnd(4))
        pos = np.where(win)[0]
        if len(pos) == 0:
            continue
        a = pos[np.argmin(inv[pos])]
        out.append((lbl, a))
    return out


def crisis_window_slices(df, anchors, drop_labels=()):
    """Ventana de cada crisis para el regresor distribuido.

    Para regresar profits_t sobre investment_{t-k}, k=0..KMAX, cada fila t necesita KMAX
    rezagos hacia atrás. La ventana 'útil' del análisis es ±HW del fondo; extendemos el
    borde IZQUIERDO en KMAX para que las filas centradas en [-HW, +HW] tengan sus rezagos
    dentro de la MISMA ventana de crisis (sin cruzar a otra crisis ni a expansión lejana).
    Devuelve lista de (label, slice_objetivo, slice_extendido_para_rezagos)."""
    n = len(df)
    out = []
    for lbl, a in anchors:
        if lbl in drop_labels:
            continue
        lo_t = a - HW            # primer t objetivo
        hi_t = a + HW            # último t objetivo (inclusive)
        lo_ext = lo_t - KMAX     # necesitamos KMAX rezagos antes del primer objetivo
        if lo_ext < 0 or hi_t >= n:
            continue
        out.append((lbl, lo_t, hi_t, lo_ext))
    return out


# --------------------------------------------------------------------------- diseño
def build_design(df, win_slices, standardize=True):
    """Apila las filas (y_t, [inv_t, inv_{t-1}, ..., inv_{t-KMAX}]) de TODAS las crisis.

    Cada fila usa SOLO rezagos dentro de su propia ventana de crisis (no cruza bordes).
    standardize: z-score por ventana (cada crisis comparable en escala antes de apilar).
    Devuelve X (m × (KMAX+1)), y (m,), y un vector de etiquetas de crisis por fila."""
    prof = df["PROF"].to_numpy()
    inv = df["INV"].to_numpy()
    X_all, y_all, lab_all = [], [], []
    per = {}
    for lbl, lo_t, hi_t, lo_ext in win_slices:
        # serie local de la ventana extendida (para z-score por crisis)
        sl_ext = slice(lo_ext, hi_t + 1)
        pe = prof[sl_ext]; ie = inv[sl_ext]
        if standardize:
            pm, ps = pe.mean(), pe.std()
            im, is_ = ie.mean(), ie.std()
            if ps == 0 or is_ == 0:
                continue
        else:
            pm, ps, im, is_ = 0.0, 1.0, 0.0, 1.0
        Xc, yc = [], []
        for t in range(lo_t, hi_t + 1):
            y = (prof[t] - pm) / ps
            row = [(inv[t - k] - im) / is_ for k in range(0, KMAX + 1)]
            yc.append(y); Xc.append(row)
        Xc = np.array(Xc); yc = np.array(yc)
        X_all.append(Xc); y_all.append(yc); lab_all += [lbl] * len(yc)
        per[lbl] = (Xc, yc)
    X = np.vstack(X_all); y = np.concatenate(y_all)
    return X, y, np.array(lab_all), per


# --------------------------------------------------------------------------- OLS libre
def ols(X, y, add_const=True):
    """OLS con intercepto opcional. Devuelve beta (incluye const al final si add_const),
    residuos, matriz (X'X)^-1, y la matriz de diseño usada."""
    Xd = np.column_stack([X, np.ones(len(X))]) if add_const else X
    XtX = Xd.T @ Xd
    XtXi = np.linalg.pinv(XtX)
    beta = XtXi @ Xd.T @ y
    resid = y - Xd @ beta
    return beta, resid, XtXi, Xd


def newey_west_se(Xd, resid, XtXi, L=4):
    """SE HAC (Newey-West) para β: robustas a autocorrelación (filas QoQ correlacionadas)."""
    n, k = Xd.shape
    u = resid.reshape(-1, 1)
    Xu = Xd * u                       # n × k
    S = Xu.T @ Xu                     # lag 0
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    cov = XtXi @ S @ XtXi
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return se, cov


# --------------------------------------------------------------------------- Almon
def almon_basis(degree):
    """Base polinomial de Almon: cada β_k = Σ_{j=0..degree} γ_j k^j.
    Devuelve matriz Z ((KMAX+1) × (degree+1)) tal que beta_kernel = Z @ gamma."""
    ks = np.arange(0, KMAX + 1)
    Z = np.vander(ks, degree + 1, increasing=True)   # columnas 1, k, k^2, ...
    return Z, ks


def fit_almon(X, y, degree, add_const=True):
    """Regresión restringida: y = (X Z) gamma + const. Recupera el kernel beta = Z gamma.
    SE del kernel por delta-method: cov(beta) = Z cov(gamma) Z'."""
    Z, ks = almon_basis(degree)
    XZ = X @ Z                                       # m × (degree+1)
    g, resid, XtXi, Xd = ols(XZ, y, add_const=add_const)
    se_g, cov_g = newey_west_se(Xd, resid, XtXi, L=4)
    ng = degree + 1
    gamma = g[:ng]
    cov_gamma = cov_g[:ng, :ng]
    beta = Z @ gamma                                 # kernel suavizado
    cov_beta = Z @ cov_gamma @ Z.T
    se_beta = np.sqrt(np.clip(np.diag(cov_beta), 0, None))
    # R2
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, se_beta, r2, resid


# --------------------------------------------------------------------------- diagnostico kernel
def kernel_stats(beta):
    """Lee el kernel como 'distribución' de lag.
    Reporta: centro (lag medio pond. por |beta|), dispersión, y el centro/masa de la parte
    NEGATIVA (la joroba de sobreacumulación)."""
    ks = np.arange(0, KMAX + 1)
    w = np.abs(beta)
    center_abs = float(np.sum(ks * w) / np.sum(w)) if np.sum(w) > 0 else np.nan
    var_abs = float(np.sum(((ks - center_abs) ** 2) * w) / np.sum(w)) if np.sum(w) > 0 else np.nan
    sd_abs = np.sqrt(var_abs) if np.isfinite(var_abs) else np.nan
    # parte negativa (joroba de sobreacumulación)
    negmask = beta < 0
    wn = -beta[negmask]
    kn = ks[negmask]
    if wn.sum() > 0:
        center_neg = float(np.sum(kn * wn) / wn.sum())
        var_neg = float(np.sum(((kn - center_neg) ** 2) * wn) / wn.sum())
        sd_neg = np.sqrt(var_neg)
        mass_neg = float(wn.sum())
        # cuanta masa negativa cae en {4,5}
        m45 = float(np.sum(-beta[(ks >= 4) & (ks <= 5) & negmask]))
        frac45 = m45 / mass_neg if mass_neg > 0 else np.nan
    else:
        center_neg = sd_neg = mass_neg = frac45 = np.nan
    # lag del minimo (trough del kernel) ignorando k=0 (que es el co-movimiento +)
    k_trough = int(ks[1:][np.argmin(beta[1:])])
    return dict(center_abs=center_abs, sd_abs=sd_abs, center_neg=center_neg,
                sd_neg=sd_neg, mass_neg=mass_neg, frac45=frac45, k_trough=k_trough)


def collinearity_report(X):
    """Colinealidad entre rezagos: número de condición y VIF por rezago."""
    # correlacion entre columnas (rezagos)
    Xc = X - X.mean(0)
    # numero de condicion de X (escala razonable: columnas ya z-scoreadas por ventana)
    s = np.linalg.svd(Xc, compute_uv=False)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    # VIF: 1/(1-R2_j) regresando cada rezago contra los otros
    k = X.shape[1]
    vifs = []
    for j in range(k):
        others = np.delete(X, j, axis=1)
        Xd = np.column_stack([others, np.ones(len(X))])
        beta = np.linalg.pinv(Xd.T @ Xd) @ Xd.T @ X[:, j]
        resid = X[:, j] - Xd @ beta
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((X[:, j] - X[:, j].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs.append(1.0 / (1.0 - r2) if r2 < 1 else np.inf)
    return cond, np.array(vifs)


# --------------------------------------------------------------------------- bootstrap por crisis
def block_bootstrap_kernel(per, labels_unique, estimator, B=NBOOT, seed=SEED):
    """Bootstrap por BLOQUES de crisis: remuestrea crisis enteras (cluster = crisis), reestima
    el kernel. Captura la incertidumbre que viene del n chico de crisis (lo que de verdad limita).
    estimator(X,y) -> beta (KMAX+1,). Devuelve array B × (KMAX+1)."""
    rng = np.random.default_rng(seed)
    nC = len(labels_unique)
    out = []
    for _ in range(B):
        pick = rng.integers(0, nC, size=nC)
        Xs, ys = [], []
        for idx in pick:
            Xc, yc = per[labels_unique[idx]]
            Xs.append(Xc); ys.append(yc)
        X = np.vstack(Xs); y = np.concatenate(ys)
        try:
            beta = estimator(X, y)
        except Exception:
            continue
        if beta is not None and np.all(np.isfinite(beta)):
            out.append(beta)
    return np.array(out)


# --------------------------------------------------------------------------- main
def print_kernel(name, beta, se=None, ci=None):
    print(f"\n  --- kernel {name} ---")
    print("    k  |  beta    " + ("|  SE     " if se is not None else "")
          + ("|  IC95 (bootstrap por crisis)   " if ci is not None else ""))
    for k in range(KMAX + 1):
        line = f"    {k:>2} | {beta[k]:+7.3f} "
        if se is not None:
            line += f"| {se[k]:6.3f} "
        if ci is not None:
            lo, hi = ci[0][k], ci[1][k]
            sig = "  *" if (lo > 0 or hi < 0) else "   "
            line += f"| [{lo:+6.3f}, {hi:+6.3f}]{sig}"
        print(line)


def main():
    np.set_printoptions(precision=3, suppress=True)
    df = load_qoq()
    anchors = crisis_anchors(df)
    print(f"Datos QoQ (pct_change(1) de NIVELES): {len(df)} trim, {df.index[0].date()}..{df.index[-1].date()}")
    print(f"Rezagos modelados: a=β0 (contemporáneo) + β1..β{KMAX}  -> kernel de {KMAX+1} coef")

    # ====================================================================== POOLED (sin 2008/2020)
    win = crisis_window_slices(df, anchors, drop_labels=("2008", "2020"))
    labels = [w[0] for w in win]
    print(f"\nCrisis utilizables (±{HW} trim, +{KMAX} de cola para rezagos, sin 2008/2020): "
          f"{len(win)} -> {labels}")
    X, y, labrow, per = build_design(df, win, standardize=True)
    print(f"Matriz de diseño apilada: {X.shape[0]} filas × {X.shape[1]} rezagos "
          f"({len(labels)} crisis × {2*HW+1} filas obj.)")

    # ---- colinealidad
    cond, vifs = collinearity_report(X)
    print(f"\n=== COLINEALIDAD ENTRE REZAGOS (crítico) ===")
    print(f"  número de condición de X = {cond:.1f}  "
          f"({'ALTA colinealidad (>30)' if cond > 30 else 'moderada' if cond > 10 else 'baja'})")
    print(f"  VIF por rezago k=0..{KMAX}: " + " ".join(f"{v:.1f}" for v in vifs)
          + f"   (max {vifs.max():.1f}; VIF>5 = inflación seria de varianza)")

    # ---- (1) POOLED libre OLS + HAC SE
    beta_f, resid_f, XtXi_f, Xd_f = ols(X, y, add_const=True)
    se_f, _ = newey_west_se(Xd_f, resid_f, XtXi_f, L=4)
    beta_free = beta_f[:KMAX + 1]; se_free = se_f[:KMAX + 1]
    ss_res = float(np.sum(resid_f ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_free = 1 - ss_res / ss_tot

    # bootstrap por crisis del kernel libre
    est_free = lambda Xb, yb: ols(Xb, yb, add_const=True)[0][:KMAX + 1]
    boot_free = block_bootstrap_kernel(per, labels, est_free)
    ci_free = np.percentile(boot_free, [2.5, 97.5], axis=0) if len(boot_free) else None

    print(f"\n=== (1) POOLED LIBRE — OLS, kernel de {KMAX+1} coef (R²={r2_free:.3f}) ===")
    print_kernel("POOLED libre", beta_free, se=se_free, ci=ci_free)
    ks_free = kernel_stats(beta_free)
    print(f"\n  trough del kernel (min β para k≥1): k={ks_free['k_trough']}")
    print(f"  centro |β|-pond. = {ks_free['center_abs']:.2f} lag, sd = {ks_free['sd_abs']:.2f}")
    print(f"  joroba NEGATIVA: centro={ks_free['center_neg']:.2f} lag, sd={ks_free['sd_neg']:.2f}, "
          f"masa={ks_free['mass_neg']:.2f}, fracción en k∈{{4,5}}={ks_free['frac45']:.0%}")

    # ---- (2) POOLED Almon (grado 2 y 3)
    print(f"\n=== (2) POOLED ALMON — kernel = polinomio en k (suaviza, baja varianza) ===")
    almon_betas = {}
    for d in (2, 3):
        beta_a, se_a, r2_a, _ = fit_almon(X, y, degree=d, add_const=True)
        almon_betas[d] = beta_a
        est_a = (lambda dd: (lambda Xb, yb: fit_almon(Xb, yb, degree=dd, add_const=True)[0]))(d)
        boot_a = block_bootstrap_kernel(per, labels, est_a)
        ci_a = np.percentile(boot_a, [2.5, 97.5], axis=0) if len(boot_a) else None
        ks_a = kernel_stats(beta_a)
        print(f"\n  Almon grado {d} (R²={r2_a:.3f}):")
        print_kernel(f"Almon d={d}", beta_a, se=se_a, ci=ci_a)
        print(f"    trough k={ks_a['k_trough']}, joroba neg centro={ks_a['center_neg']:.2f} "
              f"sd={ks_a['sd_neg']:.2f}, fracción en {{4,5}}={ks_a['frac45']:.0%}")

    # ====================================================================== PER-CRISIS (fluctuación)
    print(f"\n{'='*70}\n  (3) PER-CRISIS — un kernel por crisis -> FLUCTUACIÓN del kernel\n{'='*70}")
    print(f"  ADVERTENCIA: por crisis hay {2*HW+1}={2*HW+1} filas y {KMAX+1} coef -> casi singular.")
    print(f"  El kernel libre per-crisis es inestable; usamos ALMON grado 2 (3 coef) por crisis.")
    per_betas = {}
    print(f"\n  {'crisis':>7} | " + " ".join(f"β{k}".rjust(6) for k in range(KMAX + 1)) + " | k_trough")
    print("  " + "-" * 78)
    for lbl in labels:
        Xc, yc = per[lbl]
        try:
            beta_c, _, _, _ = fit_almon(Xc, yc, degree=2, add_const=True)
        except Exception:
            continue
        per_betas[lbl] = beta_c
        kt = kernel_stats(beta_c)["k_trough"]
        print(f"  {lbl:>7} | " + " ".join(f"{b:+6.2f}" for b in beta_c) + f" | {kt:>3}")

    PB = np.array([per_betas[l] for l in labels if l in per_betas])
    mean_k = PB.mean(0); sd_k = PB.std(0)
    print(f"\n  --- distribución del kernel ENTRE crisis (Almon d=2 por crisis) ---")
    print("    k  |  media β |  sd β   |  cv(|β|)")
    for k in range(KMAX + 1):
        cv = sd_k[k] / abs(mean_k[k]) if abs(mean_k[k]) > 1e-6 else np.inf
        print(f"    {k:>2} | {mean_k[k]:+7.3f} | {sd_k[k]:6.3f} | {cv:6.2f}")

    # fluctuación del trough entre crisis
    troughs = [kernel_stats(per_betas[l])["k_trough"] for l in labels if l in per_betas]
    print(f"\n  lag del trough del kernel por crisis: {dict(zip([l for l in labels if l in per_betas], troughs))}")
    print(f"  -> media {np.mean(troughs):.2f}, mediana {np.median(troughs):.0f}, "
          f"sd {np.std(troughs):.2f}, rango [{min(troughs)},{max(troughs)}]")
    # cuanto varia el kernel: distancia media al kernel pooled
    pooled_a2 = almon_betas[2]
    dists = [np.linalg.norm(per_betas[l] - pooled_a2) for l in labels if l in per_betas]
    print(f"  ‖kernel_crisis − kernel_pooled‖ (L2): media {np.mean(dists):.2f}, "
          f"rango [{min(dists):.2f},{max(dists):.2f}]")

    # ====================================================================== contraste YoY (solo referencia)
    print(f"\n{'='*70}\n  CONTRASTE YoY (referencia — infla ~2x, NO es el principal)\n{'='*70}")
    dyoy = C.load()  # PROFITS_YOY, INVEST_YOY ya están
    dfy = pd.DataFrame({"PROF": dyoy["PROFITS_YOY"], "INV": dyoy["INVEST_YOY"]})
    any_y = crisis_anchors(dfy)
    winy = crisis_window_slices(dfy, any_y, drop_labels=("2008", "2020"))
    Xy, yy, _, _ = build_design(dfy, winy, standardize=True)
    by, ry, _, _ = ols(Xy, yy, add_const=True)
    by = by[:KMAX + 1]
    print("    k  | β(QoQ)  | β(YoY)   (YoY infla y mete orden-4 espurio)")
    for k in range(KMAX + 1):
        print(f"    {k:>2} | {beta_free[k]:+6.3f} | {by[k]:+6.3f}")

    # ====================================================================== figura
    make_figure(beta_free, se_free, ci_free, almon_betas, per_betas, labels, pooled_a2)

    # ====================================================================== CSV
    save_csv(beta_free, se_free, ci_free, almon_betas, per_betas, labels, vifs, cond)

    # ====================================================================== veredicto
    print(f"\n{'='*70}\n  VEREDICTO\n{'='*70}")
    lag0 = beta_free[0]
    neg45 = beta_free[4] < 0 and beta_free[5] < 0
    print(f"  - lag0 (co-movimiento contemporáneo) = {lag0:+.3f} "
          f"{'(DOMINA, positivo)' if lag0 > 0.3 else ''}")
    print(f"  - joroba negativa en k∈{{4,5}}: β4={beta_free[4]:+.3f}, β5={beta_free[5]:+.3f} "
          f"-> {'PRESENTE' if neg45 else 'AUSENTE/parcial'}")
    sig_any = ci_free is not None and any((ci_free[0][k] > 0 or ci_free[1][k] < 0) for k in range(1, KMAX + 1))
    print(f"  - ¿algún β_k (k≥1) significativo (IC95 bootstrap excluye 0)? "
          f"{'SÍ' if sig_any else 'NO — los rezagos NO son individualmente distinguibles de 0'}")
    print(f"  - colinealidad (cond={cond:.0f}, VIFmax={vifs.max():.1f}) + n chico = β_k mal "
          f"identificados individualmente; lo robusto es la FORMA agregada del kernel.")


def make_figure(beta_free, se_free, ci_free, almon_betas, per_betas, labels, pooled_a2):
    ks = np.arange(0, KMAX + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # panel 1: pooled libre + Almon, con IC bootstrap
    ax1.axhline(0, color="grey", lw=0.6)
    if ci_free is not None:
        ax1.fill_between(ks, ci_free[0], ci_free[1], color="tab:blue", alpha=0.15,
                         label="IC95 bootstrap por crisis")
    ax1.errorbar(ks, beta_free, yerr=1.96 * se_free, fmt="-o", color="tab:blue",
                 ms=5, capsize=3, label="POOLED libre ±1.96·SE(HAC)")
    ax1.plot(ks, almon_betas[2], "-s", color="tab:red", ms=4, label="Almon grado 2")
    ax1.plot(ks, almon_betas[3], "--^", color="tab:orange", ms=4, label="Almon grado 3")
    ax1.axvspan(3.5, 5.5, color="tab:green", alpha=0.08)
    ax1.annotate("joroba de\nsobreacumulación\n(k=4-5)", (4.5, ax1.get_ylim()[0]),
                 fontsize=8, color="tab:green", ha="center")
    ax1.set_title("Kernel de lag distribuido POOLED (QoQ, sin 2008/2020)\n"
                  "β_k de profits_t ~ Σ investment_{t-k}", fontsize=10)
    ax1.set_xlabel("rezago k (trimestres)"); ax1.set_ylabel("β_k")
    ax1.set_xticks(ks); ax1.legend(fontsize=8)

    # panel 2: per-crisis -> fluctuacion del kernel
    ax2.axhline(0, color="grey", lw=0.6)
    for lbl in labels:
        if lbl in per_betas:
            ax2.plot(ks, per_betas[lbl], "-", lw=0.9, alpha=0.5, label=lbl)
    ax2.plot(ks, pooled_a2, "-o", color="k", lw=2.2, ms=5, label="POOLED (Almon d=2)")
    ax2.axvspan(3.5, 5.5, color="tab:green", alpha=0.08)
    ax2.set_title("FLUCTUACIÓN del kernel entre crisis (Almon d=2 por crisis)\n"
                  "cada línea = una crisis; negro = pooled", fontsize=10)
    ax2.set_xlabel("rezago k (trimestres)"); ax2.set_ylabel("β_k")
    ax2.set_xticks(ks); ax2.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    path = C.OUTDIR / "p10_distlag.png"
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"\n  figura -> {path}")


def save_csv(beta_free, se_free, ci_free, almon_betas, per_betas, labels, vifs, cond):
    ks = np.arange(0, KMAX + 1)
    d = {"k": ks, "beta_pooled_free": beta_free, "se_HAC": se_free,
         "vif": vifs, "beta_almon_d2": almon_betas[2], "beta_almon_d3": almon_betas[3]}
    if ci_free is not None:
        d["ci_lo"] = ci_free[0]; d["ci_hi"] = ci_free[1]
    for lbl in labels:
        if lbl in per_betas:
            d[f"beta_{lbl}"] = per_betas[lbl]
    out = pd.DataFrame(d)
    path = C.OUTDIR / "p10_distlag.csv"
    out.to_csv(path, index=False)
    print(f"  CSV -> {path}  (cond. número global = {cond:.1f})")


if __name__ == "__main__":
    main()
