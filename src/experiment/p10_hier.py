"""Pieza 10 — Kernel de lag distribuido JERÁRQUICO (partial pooling entre crisis).

Hipótesis a evaluar: el feedback de sobreacumulación (inversión pasada -> deprime la
ganancia) NO es un retardo τ fijo ni un oscilador de fase fija, sino un KERNEL DE LAG
DISTRIBUIDO cuya forma FLUCTÚA entre crisis. El objeto metodológicamente correcto no es
"el lag" sino la DISTRIBUCIÓN del kernel sobre el ensemble de episodios: una media
poblacional μ_k y una dispersión entre-crisis τ_k por cada lag k.

Modelo (regresión de lag distribuido, una por crisis, con pooling parcial):

    profits_t^(c)  =  Σ_{k=0}^{K} β_k^(c) · investment_{t-k}^(c)  +  ε_t^(c)
    β_k^(c)  ~  N(μ_k , τ_k²)            (cada crisis es un DRAW de la poblacional)
    ε_t^(c)  ~  N(0 , σ²)

Todo en QoQ = pct_change(1) (el transform HONESTO; el YoY infla ~2× y mete estructura
de orden 4 espuria). Ventana de cada crisis = ±6 trim del fondo de inversión (mismo
ancla que p8/p9). Para no inflar la dimensión con 9×(K+1) parámetros sobre ~13 puntos,
K=5 (lags 0..5: cubre el lag0=+ co-movimiento y la joroba negativa en 4–5).

Sin pymc/numpyro instalado -> partial pooling FRECUENTISTA por EM / empirical Bayes
gaussiano jerárquico (cerrado dado (σ², τ²); EM sobre los hiperparámetros). Esto es
exactamente el MAP/posterior-mean del modelo bayesiano gaussiano de arriba con priors
planos sobre μ. Se reporta:
  - kernel poblacional μ_k (media) + su error estándar,
  - dispersión entre-crisis τ_k por lag (cuánto fluctúa el feedback),
  - shrinkage por crisis (cuánto se tira cada β_k^(c) hacia μ_k): factor 0..1,
  - INCERTIDUMBRE de τ_k vía leave-one-crisis-out + bootstrap de crisis (n=9 -> τ
    está MAL estimado; se reporta el rango honesto),
  - ¿1980/2020 son colas? (distancia de Mahalanobis de su kernel a la poblacional).

Run:  uv run python -m src.experiment.p10_hier
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment import common as C

KMAX = 5                     # lags 0..5 (co-movimiento en 0 + joroba negativa en 4-5)
HW = 6                       # media-ventana ±6 trim del fondo (igual que p9)
RNG = np.random.default_rng(20260606)


# --------------------------------------------------------------------------- datos QoQ
def qoq_levels():
    """Construye QoQ = pct_change(1) (en %) desde los NIVELES PROFITS, INVESTMENT."""
    df = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    out = pd.DataFrame(index=df.index)
    out["P"] = np.log(df["PROFITS"]).diff() * 100.0
    out["I"] = np.log(df["INVESTMENT"]).diff() * 100.0
    return out.dropna()


def crisis_design(df, kmax=KMAX, hw=HW):
    """Para cada recesión: ancla = inversión más deprimida en su vecindad; ventana ±hw.
    Devuelve lista de (label, X (n×(K+1)), y (n,)) con X_tk = investment_{t-k} y y=profits_t,
    estandarizando X e y POR VENTANA (z-score) para que el feedback se compare en forma,
    no en escala. Los lags se toman del histórico completo (no se pierden por el borde)."""
    idx = df.index
    p, i = df["P"].to_numpy(), df["I"].to_numpy()
    out = []
    for lbl, pk, tr in C.RECESSIONS:
        pk, tr = pd.Timestamp(pk), pd.Timestamp(tr)
        win = (idx >= pk - pd.offsets.QuarterEnd(2)) & (idx <= tr + pd.offsets.QuarterEnd(4))
        pos = np.where(win)[0]
        if len(pos) == 0:
            continue
        a = pos[np.argmin(i[pos])]
        lo, hi = a - hw, a + hw + 1
        if lo - kmax < 0 or hi > len(p):       # necesito i_{t-K} para el primer t
            continue
        ts = np.arange(lo, hi)
        y = p[ts]
        X = np.column_stack([i[ts - k] for k in range(kmax + 1)])
        # z-score por ventana (cada crisis comparable en escala -> kernel = forma)
        y = (y - y.mean()) / y.std()
        Xm, Xs = X.mean(0), X.std(0)
        Xs[Xs == 0] = 1.0
        X = (X - Xm) / Xs
        out.append((lbl, X, y))
    return out


# --------------------------------------------------------------------------- EM jerárquico
def fit_hierarchical(design, kmax=KMAX, n_em=300, tol=1e-8):
    """Empirical-Bayes gaussiano jerárquico vía EM (partial pooling).

    Modelo: y^(c) = X^(c) β^(c) + ε,  ε~N(0,σ²I),  β^(c)~N(μ, T),  T=diag(τ²).
    E-step (cerrado, gaussiano): posterior de β^(c) | datos, μ, σ², T es gaussiano
        Σ_c = (X'X/σ² + T^{-1})^{-1},   m_c = Σ_c (X'y/σ² + T^{-1} μ).
    M-step:
        μ      = mean_c m_c
        τ_k²   = mean_c [ (m_ck - μ_k)² + Σ_c[k,k] ]            (varianza entre-crisis + incertidumbre)
        σ²     = (Σ_c [ ||y-Xm_c||² + tr(X Σ_c X') ]) / (Σ_c n_c)

    Devuelve μ, τ² (entre-crisis), σ², los m_c (kernels por crisis encogidos), Σ_c,
    y el factor de shrinkage por crisis/lag.
    """
    p1 = kmax + 1
    Xs = [d[1] for d in design]
    ys = [d[2] for d in design]
    labels = [d[0] for d in design]
    nC = len(design)
    XtX = [X.T @ X for X in Xs]
    Xty = [X.T @ y for X, y in zip(Xs, ys)]
    ntot = sum(len(y) for y in ys)

    # init: OLS por crisis (ridge ligero para estabilidad), μ=media, τ²=var entre, σ² del residuo
    ols = []
    for X, y in zip(Xs, ys):
        b = np.linalg.solve(X.T @ X + 1e-3 * np.eye(p1), X.T @ y)
        ols.append(b)
    ols = np.array(ols)
    mu = ols.mean(0)
    tau2 = np.maximum(ols.var(0), 1e-3)
    sig2 = float(np.mean([np.mean((y - X @ b) ** 2) for X, y, b in zip(Xs, ys, ols)]))
    sig2 = max(sig2, 1e-3)

    prev = np.inf
    for _ in range(n_em):
        Tinv = np.diag(1.0 / tau2)
        m_list, S_list = [], []
        for c in range(nC):
            S = np.linalg.inv(XtX[c] / sig2 + Tinv)
            m = S @ (Xty[c] / sig2 + Tinv @ mu)
            m_list.append(m); S_list.append(S)
        M = np.array(m_list)
        mu_new = M.mean(0)
        tau2_new = np.array([
            np.mean([(M[c, k] - mu_new[k]) ** 2 + S_list[c][k, k] for c in range(nC)])
            for k in range(p1)
        ])
        tau2_new = np.maximum(tau2_new, 1e-6)
        num = 0.0
        for c in range(nC):
            r = ys[c] - Xs[c] @ M[c]
            num += float(r @ r) + float(np.trace(Xs[c] @ S_list[c] @ Xs[c].T))
        sig2_new = max(num / ntot, 1e-6)

        delta = (np.abs(mu_new - mu).sum() + np.abs(tau2_new - tau2).sum()
                 + abs(sig2_new - sig2))
        mu, tau2, sig2 = mu_new, tau2_new, sig2_new
        if abs(prev - delta) < tol:
            break
        prev = delta

    # shrinkage por crisis/lag: cuánto pesa el pooling. Para cada crisis comparamos el
    # ancho posterior con el del prior poblacional: s = 1 - diag(S_c)/τ²  (≈ peso de la
    # verosimilitud propia; s->0 = todo pooling, s->1 = crisis informa sola).
    Tinv = np.diag(1.0 / tau2)
    shrink = np.zeros((nC, p1))   # cuánto SE ENCOGE hacia μ (1=todo a la media, 0=OLS propio)
    M = np.zeros((nC, p1)); S_diag = np.zeros((nC, p1))
    for c in range(nC):
        S = np.linalg.inv(XtX[c] / sig2 + Tinv)
        m = S @ (Xty[c] / sig2 + Tinv @ mu)
        M[c] = m; S_diag[c] = np.diag(S)
        # encogimiento = 1 - Var_post/Var_prior (cuánto cerró la incertidumbre = cuánto aprendió,
        # complemento del peso del prior). Reportamos el factor de pull-to-mean por lag.
        shrink[c] = np.diag(S) / tau2   # fracción de la varianza poblacional que QUEDA = peso del prior
    return dict(mu=mu, tau2=tau2, sig2=sig2, M=M, S_diag=S_diag, labels=labels,
                shrink=shrink, ols=ols, Xs=Xs, ys=ys)


# --------------------------------------------------------------------------- incertidumbre
def se_population_mean(res):
    """Error estándar de μ_k. Con jerárquico gaussiano y crisis ~iid de la poblacional,
    SE(μ_k) ≈ sqrt( (τ_k² + mean_c S_c[k,k]) / nC ). Domina τ_k²/nC con nC=9."""
    nC = len(res["labels"])
    extra = res["S_diag"].mean(0)
    return np.sqrt((res["tau2"] + extra) / nC)


def tau_uncertainty(design, kmax=KMAX, B=400):
    """INCERTIDUMBRE de τ_k (dispersión entre-crisis). Con nC=9 está MAL estimada.
    Dos vías honestas:
      (a) leave-one-crisis-out: re-estima τ_k dejando una crisis afuera -> rango LOCO.
      (b) bootstrap de crisis (remuestreo con reemplazo del ensemble) -> IC percentil.
    Devuelve τ̂, rango LOCO, IC95 bootstrap. (sqrt de τ² para reportar en unidades de β.)"""
    nC = len(design)
    base = fit_hierarchical(design, kmax)
    tau_hat = np.sqrt(base["tau2"])

    loco = []
    for c in range(nC):
        sub = [design[j] for j in range(nC) if j != c]
        r = fit_hierarchical(sub, kmax)
        loco.append(np.sqrt(r["tau2"]))
    loco = np.array(loco)                       # (nC, p1)

    boot = []
    idxs = np.arange(nC)
    for _ in range(B):
        pick = RNG.choice(idxs, size=nC, replace=True)
        sub = [design[j] for j in pick]
        try:
            r = fit_hierarchical(sub, kmax, n_em=150)
            boot.append(np.sqrt(r["tau2"]))
        except np.linalg.LinAlgError:
            continue
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return dict(tau_hat=tau_hat, loco=loco, boot_lo=lo, boot_hi=hi,
                loco_lo=loco.min(0), loco_hi=loco.max(0))


def tail_crises(res):
    """¿Qué crisis son COLAS del kernel poblacional? Distancia de Mahalanobis del kernel
    encogido m_c a μ usando la covarianza poblacional diag(τ²). Reporta z por lag también."""
    mu, tau2, M, labels = res["mu"], res["tau2"], res["M"], res["labels"]
    maha = np.sqrt(((M - mu) ** 2 / tau2).sum(1))
    z = (M - mu) / np.sqrt(tau2)
    return labels, maha, z


# --------------------------------------------------------------------------- figura
def plot(res, unc, path):
    mu, tau = res["mu"], np.sqrt(res["tau2"])
    se = se_population_mean(res)
    ks = np.arange(KMAX + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # panel 1: kernel poblacional + draws por crisis + banda de dispersión entre-crisis
    for c, lbl in enumerate(res["labels"]):
        ax1.plot(ks, res["M"][c], "-", color="grey", lw=0.7, alpha=0.5)
    ax1.fill_between(ks, mu - tau, mu + tau, color="tab:orange", alpha=0.18,
                     label="±τ (dispersión ENTRE crisis)")
    ax1.errorbar(ks, mu, yerr=se, fmt="-o", color="tab:red", lw=2, ms=5, capsize=3,
                 label="μ poblacional ± SE(μ)")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_title("Kernel de lag distribuido JERÁRQUICO (QoQ)\n"
                  "líneas grises = β^(c) por crisis (encogido) · rojo = media poblacional",
                  fontsize=10)
    ax1.set_xlabel("lag k (trimestres: inversión_{t-k} -> profits_t)")
    ax1.set_ylabel("β_k (estandarizado)"); ax1.legend(fontsize=8)

    # panel 2: τ_k con su incertidumbre (LOCO + bootstrap)
    ax2.bar(ks, unc["tau_hat"], color="tab:orange", alpha=0.5, label="τ_k estimado")
    # el bootstrap de una varianza está sesgado: el IC puede no contener al estimador puntual;
    # dibujamos la banda IC95 cruda con barras (no como yerr alrededor de τ̂, que daría neg.)
    lo_err = np.clip(unc["tau_hat"] - unc["boot_lo"], 0, None)
    hi_err = np.clip(unc["boot_hi"] - unc["tau_hat"], 0, None)
    ax2.errorbar(ks, unc["tau_hat"], yerr=[lo_err, hi_err],
                 fmt="none", ecolor="k", capsize=4, label="IC95 bootstrap de crisis")
    ax2.plot(ks, unc["boot_lo"], "_", color="dimgrey", ms=10)
    ax2.plot(ks, unc["boot_hi"], "_", color="dimgrey", ms=10)
    ax2.plot(ks, unc["loco_lo"], "v", color="tab:blue", ms=6, label="rango leave-one-crisis-out")
    ax2.plot(ks, unc["loco_hi"], "^", color="tab:blue", ms=6)
    ax2.set_title("Dispersión entre crisis τ_k y su INCERTIDUMBRE (n=9 -> mal estimada)",
                  fontsize=10)
    ax2.set_xlabel("lag k"); ax2.set_ylabel("τ_k (sd entre crisis)"); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    df = qoq_levels()
    design = crisis_design(df)
    labels = [d[0] for d in design]
    print(f"=== KERNEL DE LAG DISTRIBUIDO JERÁRQUICO (QoQ, K={KMAX}, ±{HW} trim) ===")
    print(f"Crisis utilizables (n={len(design)}): {labels}")
    print("Inferencia: empirical-Bayes gaussiano jerárquico por EM (partial pooling).")
    print("  profits_t = Σ_k β_k^(c) · invest_{t-k} + ε ;  β_k^(c) ~ N(μ_k, τ_k²)\n")

    res = fit_hierarchical(design)
    mu, tau, sig2 = res["mu"], np.sqrt(res["tau2"]), res["sig2"]
    se = se_population_mean(res)
    unc = tau_uncertainty(design)

    print("--- KERNEL POBLACIONAL (media μ_k) + dispersión entre crisis τ_k ---")
    print(f"  {'lag':>3} | {'μ_k':>7} | {'SE(μ)':>6} | {'τ_k':>6} | "
          f"{'τ IC95 boot':>16} | {'τ rango LOCO':>16}")
    print("  " + "-" * 70)
    for k in range(KMAX + 1):
        print(f"  {k:>3} | {mu[k]:+7.3f} | {se[k]:6.3f} | {tau[k]:6.3f} | "
              f"[{unc['boot_lo'][k]:5.2f},{unc['boot_hi'][k]:5.2f}]   | "
              f"[{unc['loco_lo'][k]:5.2f},{unc['loco_hi'][k]:5.2f}]")
    print(f"\n  σ (ruido intra-ventana) = {np.sqrt(sig2):.3f}")

    print("\n--- FORMA: ¿pulso + en 0 y joroba negativa en 4-5? ---")
    sig_pos = [k for k in range(KMAX + 1) if mu[k] - 1.96 * se[k] > 0]
    sig_neg = [k for k in range(KMAX + 1) if mu[k] + 1.96 * se[k] < 0]
    print(f"  lags con μ_k > 0 (IC95): {sig_pos}")
    print(f"  lags con μ_k < 0 (IC95): {sig_neg}")
    print(f"  μ = {np.array2string(mu, precision=2, suppress_small=True)}")

    print("\n--- SHRINKAGE por crisis (peso del PRIOR poblacional = Var_post/τ²; "
          "1=encoge todo a μ, 0=crisis informa sola) ---")
    print(f"  {'crisis':>7} | " + " ".join(f"k{k:>4}" for k in range(KMAX + 1)) + " | media")
    print("  " + "-" * 60)
    for c, lbl in enumerate(labels):
        s = res["shrink"][c]
        print(f"  {lbl:>7} | " + " ".join(f"{v:5.2f}" for v in s) + f" | {s.mean():.2f}")
    print(f"  shrinkage medio global = {res['shrink'].mean():.2f}  "
          f"(alto = data débil por crisis -> el pooling domina)")

    print("\n--- ¿1980/2020 son COLAS? (Mahalanobis del kernel a la poblacional) ---")
    labs, maha, z = tail_crises(res)
    order = np.argsort(-maha)
    for j in order:
        flag = "  <-- cola" if maha[j] > np.median(maha) + maha.std() else ""
        print(f"  {labs[j]:>7}: D_Maha={maha[j]:.2f}{flag}")
    # 2020 cae fuera de la ventana (a+6 fuera de muestra? chequear); reportar presencia
    for tag in ("1980", "2020", "2008"):
        if tag in labs:
            j = labs.index(tag)
            print(f"    {tag}: z por lag = {np.array2string(z[j], precision=1)}")
        else:
            print(f"    {tag}: NO entró (ventana fuera de muestra o lags insuficientes)")

    path = C.OUTDIR / "p10_hier_kernel.png"
    plot(res, unc, path)

    # CSV con todo
    rows = []
    for k in range(KMAX + 1):
        rows.append(dict(lag=k, mu=mu[k], se_mu=se[k], tau=tau[k],
                         tau_boot_lo=unc["boot_lo"][k], tau_boot_hi=unc["boot_hi"][k],
                         tau_loco_lo=unc["loco_lo"][k], tau_loco_hi=unc["loco_hi"][k]))
    pd.DataFrame(rows).to_csv(C.OUTDIR / "p10_hier_kernel.csv", index=False)
    perc = pd.DataFrame(res["M"], index=labels,
                        columns=[f"beta_lag{k}" for k in range(KMAX + 1)])
    perc.to_csv(C.OUTDIR / "p10_hier_per_crisis.csv")
    print(f"\n✓ figura -> {path}")
    print(f"✓ CSV -> p10_hier_kernel.csv , p10_hier_per_crisis.csv")

    # --------- veredicto honesto sobre identificabilidad
    print("\n" + "=" * 70)
    print("VEREDICTO (crítico):")
    print(f"  - n={len(design)} crisis × {2*HW+1} pts; K+1={KMAX+1} coef por crisis -> "
          f"τ_k apoyado en {len(design)} 'observaciones' de β.")
    print(f"  - El IC bootstrap de τ es ANCHO (ver arriba): la DISPERSIÓN entre crisis es")
    print(f"    el objeto peor identificado. La MEDIA μ_k está mucho mejor.")
    return res, unc


if __name__ == "__main__":
    main()
