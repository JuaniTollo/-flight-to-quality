"""Pieza 12 — Modelo BAYESIANO JERÁRQUICO del lag central, con UQ propia.

Objeto: el *lag central* del feedback de sobreacumulación (inversión pasada deprime la
ganancia) estimado POR crisis, y su variación ENTRE crisis. A diferencia de p10_hier
(que hace partial pooling de TODO el vector β_k por lag k), acá el objeto jerárquico es
un ESCALAR por episodio —el lag central μ_c— y la pregunta es si ese lag fluctúa de
crisis a crisis o es estructuralmente constante.

    μ_c            = lag central de la crisis c            (estimado del distributed-lag)
    μ_c | μ_pop,τ  ~  Normal(μ_pop, τ²)                    (partial pooling entre crisis)
    μ̂_c | μ_c     ~  Normal(μ_c, σ_c²)                    (μ̂_c = estimación con su SE)

Es el modelo gaussiano jerárquico / random-effects (estilo meta-análisis): cada crisis
aporta un lag central μ̂_c con su error de muestreo σ_c, y se infiere la media poblacional
μ_pop y la dispersión ENTRE crisis τ (la "fluctuación del lag entre crisis").

Tres salidas:
  (1) μ_pop con IC: el lag central poblacional (esperado ~4 trim = 1 año).
  (2) τ con IC: la variación entre crisis. ¿τ distinguible de 0?  -> ¿el lag fluctúa?
  (3) COMPARACIÓN PUNTO-vs-DISTRIBUCIÓN para el ancho τ:
        - H0 (PUNTO):        τ = 0   (un único lag común a todas las crisis)
        - H1 (DISTRIBUCIÓN): τ > 0   (cada crisis con su propio lag)
      vía (a) cociente de verosimilitudes REML / Bayes factor aproximado (BIC), y
      (b) si hay MCMC, la masa posterior de τ cerca de 0.

Inferencia:
  - Si pymc o numpyro están instalados -> NUTS (semipooling completo, posterior de μ_pop y τ).
  - Si no -> empirical Bayes / REML cerrado para el modelo gaussiano random-effects
    (DerSimonian-Laird + perfil de verosimilitud REML para el IC de τ). La versión NUTS
    es el paso siguiente (el modelo gaussiano lo hace casi idéntico; NUTS solo agrega la
    incertidumbre de los σ_c y un prior débil en τ).

Todo en QoQ = pct_change(1) sobre NIVELES (la transform insesgada; YoY infla ~2x el
feedback en lag-4). n = 9 crisis usables (sin 2008/2020). Ventana = ±6 trim del fondo.

Run (completo):  uv run python -m src.experiment.p12_hier_bayes
Run (smoke):     uv run python -m src.experiment.p12_hier_bayes --smoke
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from src.experiment import common as C

KMAX = 8                    # lags 0..8 (cubre el co-movimiento en 0 y la joroba en 4-5)
LAG_LO, LAG_HI = 1, 8       # rango de lags donde se busca el lag central del FEEDBACK (>0)
HW = 6                      # media-ventana ±6 trim del fondo (igual que p8/p9/p10)
DROP = ("2008", "2020")     # protocolo: muestra sin las dos crisis financieras atípicas
RNG = np.random.default_rng(20260607)


# --------------------------------------------------------------------------- datos QoQ
def qoq_levels():
    """QoQ = pct_change(1) en % desde los NIVELES PROFITS, INVESTMENT."""
    df = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    out = pd.DataFrame(index=df.index)
    out["P"] = df["PROFITS"].pct_change(1) * 100.0
    out["I"] = df["INVESTMENT"].pct_change(1) * 100.0
    return out.dropna()


def crisis_windows(df, kmax=KMAX, hw=HW, drop=DROP):
    """Para cada recesión: ancla = inversión más deprimida en su vecindad; ventana ±hw.
    Devuelve lista de (label, X (n×(K+1)), y (n,)) con X_tk = invest_{t-k}, y = profits_t,
    z-score por ventana (cada crisis comparable en FORMA, no en escala)."""
    idx = df.index
    p, i = df["P"].to_numpy(), df["I"].to_numpy()
    out = []
    for lbl, pk, tr in C.RECESSIONS:
        if lbl in drop:
            continue
        pk, tr = pd.Timestamp(pk), pd.Timestamp(tr)
        win = (idx >= pk - pd.offsets.QuarterEnd(2)) & (idx <= tr + pd.offsets.QuarterEnd(4))
        pos = np.where(win)[0]
        if len(pos) == 0:
            continue
        a = pos[np.argmin(i[pos])]
        lo, hi = a - hw, a + hw + 1
        if lo - kmax < 0 or hi > len(p):
            continue
        ts = np.arange(lo, hi)
        y = p[ts]
        X = np.column_stack([i[ts - k] for k in range(kmax + 1)])
        y = (y - y.mean()) / y.std()
        Xm, Xs = X.mean(0), X.std(0)
        Xs[Xs == 0] = 1.0
        X = (X - Xm) / Xs
        out.append((lbl, X, y))
    return out


# --------------------------------------------------------------------- lag central por crisis
def central_lag_per_crisis(design, n_boot=400):
    """Estima, por crisis, el LAG CENTRAL del feedback negativo + su error de muestreo σ_c.

    Para cada crisis se ajusta el distributed-lag (OLS con ridge ligero) y se define el
    lag central como el CENTROIDE de la masa negativa del kernel sobre los lags >0:
        μ_c = Σ_{k in [LAG_LO,LAG_HI]} k · w_k  /  Σ w_k ,   w_k = max(-β_k, 0)
    (el feedback de sobreacumulación es negativo; tomamos dónde se concentra esa masa).
    El error σ_c se obtiene por bootstrap de bloques residual dentro de la ventana
    (remuestreo de residuos OLS) -> SD del μ_c re-estimado. σ_c es el ruido de MEDICIÓN
    del lag de cada crisis, que el jerárquico necesita para separar señal de ruido.

    Devuelve labels, mu_hat (nC,), sigma (nC,)  y los kernels β por crisis (para la figura).
    """
    labels, mu_hat, sigma, betas = [], [], [], []
    ks = np.arange(LAG_LO, LAG_HI + 1)
    p1 = KMAX + 1
    ridge = 1e-2 * np.eye(p1)

    def centroid(beta):
        w = np.clip(-beta[LAG_LO:LAG_HI + 1], 0, None)
        if w.sum() <= 1e-9:
            return np.nan
        return float(np.sum(ks * w) / w.sum())

    for lbl, X, y in design:
        XtX = X.T @ X + ridge
        beta = np.linalg.solve(XtX, X.T @ y)
        m = centroid(beta)
        if not np.isfinite(m):
            continue
        # bootstrap residual de bloques (preserva autocorrelación intra-ventana)
        resid = y - X @ beta
        fitted = X @ beta
        n = len(y)
        bl = 4                                       # bloque ~1 año
        boots = []
        for _ in range(n_boot):
            # remuestreo de bloques de residuos
            rb = []
            while len(rb) < n:
                s0 = RNG.integers(0, n)
                rb.extend(resid[s0:s0 + bl])
            rb = np.array(rb[:n])
            yb = fitted + rb
            bb = np.linalg.solve(XtX, X.T @ yb)
            mb = centroid(bb)
            if np.isfinite(mb):
                boots.append(mb)
        sc = float(np.std(boots)) if len(boots) > 3 else 1.0
        sc = max(sc, 0.2)                             # piso: nunca SE cero (mal puesto)
        labels.append(lbl)
        mu_hat.append(m)
        sigma.append(sc)
        betas.append(beta)
    return labels, np.array(mu_hat), np.array(sigma), np.array(betas)


# ----------------------------------------------------- empirical-Bayes / REML (random effects)
def reml_loglik(tau2, y, s2):
    """Log-verosimilitud REML del modelo random-effects gaussiano (perfilada en μ_pop).
        y_c ~ N(μ_pop, s2_c + tau2).  REML integra μ_pop con prior plano.
    """
    v = s2 + tau2
    w = 1.0 / v
    mu = np.sum(w * y) / np.sum(w)
    ll = -0.5 * np.sum(np.log(v)) - 0.5 * np.sum(w * (y - mu) ** 2) - 0.5 * np.log(np.sum(w))
    return ll, mu


def fit_random_effects(y, s2):
    """Empirical Bayes del modelo gaussiano random-effects (estilo meta-análisis).

    Estima τ² maximizando la verosimilitud REML; μ_pop = media ponderada por precisión.
    Devuelve μ_pop, SE(μ_pop), τ, y todo lo necesario para el IC por perfil REML.
    """
    # maximiza REML en log(tau2) sobre un rango amplio (incluye tau2->0 = punto)
    grid = np.concatenate([[0.0], np.exp(np.linspace(np.log(1e-4), np.log(50.0), 600))])
    lls = np.array([reml_loglik(t, y, s2)[0] for t in grid])
    j = int(np.argmax(lls))
    tau2_hat = grid[j]
    ll_hat, mu_hat = reml_loglik(tau2_hat, y, s2)
    v = s2 + tau2_hat
    se_mu = np.sqrt(1.0 / np.sum(1.0 / v))
    return dict(tau2=tau2_hat, tau=np.sqrt(tau2_hat), mu_pop=mu_hat, se_mu=se_mu,
                grid=grid, ll=lls, ll_hat=ll_hat)


def tau_profile_ci(y, s2, fit, level=0.95):
    """IC de τ por perfil de verosimilitud REML (cutoff χ²_1). Devuelve [lo, hi] en τ."""
    cut = stats.chi2.ppf(level, 1) / 2.0
    target = fit["ll_hat"] - cut
    grid, lls = fit["grid"], fit["ll"]
    above = lls >= target
    tau_grid = np.sqrt(grid)
    if not above.any():
        return 0.0, 0.0
    lo = tau_grid[above].min()
    hi = tau_grid[above].max()
    return float(lo), float(hi)


def point_vs_distribution(y, s2, fit):
    """COMPARACIÓN punto (τ=0) vs distribución (τ>0) del lag entre crisis.

    (a) Cociente de verosimilitudes: 2·(ll(τ̂) - ll(τ=0)). Bajo H0 (τ=0, en la frontera)
        el estadístico ~ ½χ²_0 + ½χ²_1 (mezcla, Self & Liang 1987): p = ½·P(χ²_1 > LR).
    (b) Bayes factor aproximado por BIC: BF_10 ≈ exp(ΔBIC/2), ΔBIC = BIC_0 - BIC_1,
        con BIC = -2·ll + k·ln(nC).  H1 tiene un parámetro más (τ).
        BF_10 > 1 favorece distribución; < 1 favorece punto.
    """
    nC = len(y)
    ll1 = fit["ll_hat"]
    ll0 = reml_loglik(0.0, y, s2)[0]
    lr = 2.0 * (ll1 - ll0)
    lr = max(lr, 0.0)
    p_mix = 0.5 * stats.chi2.sf(lr, 1)               # test en la frontera
    bic1 = -2 * ll1 + 1 * np.log(nC)                 # +1 param: τ
    bic0 = -2 * ll0 + 0 * np.log(nC)
    dbic = bic0 - bic1
    bf10 = float(np.exp(dbic / 2.0))
    return dict(lr=lr, p=float(p_mix), bf10=bf10, dbic=float(dbic), ll0=ll0, ll1=ll1)


# ----------------------------------------------------------------------------- NUTS (si hay)
def has_mcmc():
    for mod in ("pymc", "numpyro"):
        try:
            __import__(mod)
            return mod
        except ImportError:
            continue
    return None


def fit_nuts(y, s2, draws=2000, tune=2000, chains=4, seed=20260607):
    """Modelo gaussiano jerárquico por NUTS (si pymc está instalado). Non-centered.
        μ_pop ~ Normal(4, 5);  τ ~ HalfNormal(2);  μ_c = μ_pop + τ·z_c, z_c ~ N(0,1)
        y_c ~ Normal(μ_c, σ_c).
    Devuelve dict con resúmenes posteriores (mean/IC) de μ_pop y τ, y P(τ < 0.5)."""
    import pymc as pm
    sigma = np.sqrt(s2)
    with pm.Model():
        mu_pop = pm.Normal("mu_pop", mu=4.0, sigma=5.0)
        tau = pm.HalfNormal("tau", sigma=2.0)
        z = pm.Normal("z", 0.0, 1.0, shape=len(y))
        mu_c = pm.Deterministic("mu_c", mu_pop + tau * z)
        pm.Normal("obs", mu=mu_c, sigma=sigma, observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                          target_accept=0.95, progressbar=False)
    post = idata.posterior
    mp = post["mu_pop"].values.ravel()
    tt = post["tau"].values.ravel()
    return dict(
        mu_pop=float(mp.mean()), mu_ci=(float(np.percentile(mp, 2.5)), float(np.percentile(mp, 97.5))),
        tau=float(tt.mean()), tau_ci=(float(np.percentile(tt, 2.5)), float(np.percentile(tt, 97.5))),
        p_tau_small=float((tt < 0.5).mean()), idata=idata)


# ----------------------------------------------------------------------------- figura
def plot(labels, mu_hat, sigma, fit, ci_tau, path):
    nC = len(labels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [2, 1]})

    order = np.argsort(mu_hat)
    yy = np.arange(nC)
    ax1.errorbar(mu_hat[order], yy, xerr=1.96 * sigma[order], fmt="o", color="tab:blue",
                 capsize=3, label="lag central por crisis  μ̂_c ± 1.96 σ_c")
    ax1.axvline(fit["mu_pop"], color="tab:red", lw=2, label=f"μ_pop = {fit['mu_pop']:.2f}")
    ax1.axvspan(fit["mu_pop"] - 1.96 * fit["se_mu"], fit["mu_pop"] + 1.96 * fit["se_mu"],
                color="tab:red", alpha=0.15, label="IC95 μ_pop")
    ax1.fill_betweenx([-1, nC], fit["mu_pop"] - fit["tau"], fit["mu_pop"] + fit["tau"],
                      color="tab:orange", alpha=0.12, label=f"±τ entre crisis (τ={fit['tau']:.2f})")
    ax1.set_yticks(yy); ax1.set_yticklabels([labels[i] for i in order])
    ax1.set_xlabel("lag central (trimestres)")
    ax1.set_title("Lag central por crisis + media poblacional (jerárquico, QoQ)", fontsize=10)
    ax1.set_ylim(-0.8, nC - 0.2); ax1.legend(fontsize=8, loc="lower right")

    # perfil REML de τ con cutoff e IC
    tau_grid = np.sqrt(fit["grid"])
    ax2.plot(tau_grid, fit["ll"], color="tab:orange")
    ax2.axhline(fit["ll_hat"] - stats.chi2.ppf(0.95, 1) / 2, color="grey", ls="--",
                label="cutoff IC95")
    ax2.axvline(fit["tau"], color="tab:red", label=f"τ̂={fit['tau']:.2f}")
    ax2.axvspan(ci_tau[0], ci_tau[1], color="tab:red", alpha=0.12,
                label=f"IC95 τ=[{ci_tau[0]:.2f},{ci_tau[1]:.2f}]")
    ax2.set_xlim(0, max(tau_grid[fit['ll'] >= fit['ll_hat'] - 4].max() * 1.1, 1.0))
    ax2.set_xlabel("τ (dispersión ENTRE crisis)"); ax2.set_ylabel("log-verosimilitud REML")
    ax2.set_title("Perfil de τ: ¿punto (0) o distribución (>0)?", fontsize=10)
    ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------- run core
def run(smoke=False):
    df = qoq_levels()
    design = crisis_windows(df)
    labels0 = [d[0] for d in design]
    n_boot = 60 if smoke else 400

    print(f"=== P12 — JERÁRQUICO BAYESIANO DEL LAG CENTRAL (QoQ, ±{HW} trim) ===")
    print(f"Crisis (sin {DROP}): {labels0}  (n={len(design)})")
    mode = "SMOKE TEST (pocas muestras/iteraciones)" if smoke else "RUN COMPLETO"
    print(f"Modo: {mode}\n")

    labels, mu_hat, sigma, betas = central_lag_per_crisis(design, n_boot=n_boot)
    s2 = sigma ** 2
    print("--- LAG CENTRAL POR CRISIS (μ̂_c) y su error de muestreo (σ_c) ---")
    for lbl, m, s in zip(labels, mu_hat, sigma):
        print(f"  {lbl:>6}:  μ̂_c = {m:5.2f} trim   σ_c = {s:4.2f}")
    print(f"  media simple = {mu_hat.mean():.2f}   sd simple = {mu_hat.std(ddof=1):.2f}  "
          f"(sd cruda mezcla señal + ruido de medición)")

    # empirical Bayes / REML
    fit = fit_random_effects(mu_hat, s2)
    ci_tau = tau_profile_ci(mu_hat, s2, fit)
    pvd = point_vs_distribution(mu_hat, s2, fit)

    mcmc = has_mcmc()
    used = "EB/REML (frecuentista; partial pooling cerrado)"
    nuts_res = None
    if mcmc == "pymc":
        try:
            draws = 150 if smoke else 2000
            tune = 150 if smoke else 2000
            chains = 2 if smoke else 4
            print(f"\n[MCMC] pymc detectado -> NUTS ({chains} cadenas, "
                  f"{tune} tune + {draws} draws)...")
            nuts_res = fit_nuts(mu_hat, s2, draws=draws, tune=tune, chains=chains)
            used = f"NUTS (pymc) + EB/REML de contraste"
        except Exception as e:
            print(f"[MCMC] pymc falló ({e}); se usa EB/REML.")
            mcmc = None
    elif mcmc == "numpyro":
        print("\n[MCMC] numpyro detectado pero el wrapper NUTS implementado es para pymc; "
              "se usa EB/REML. (NUTS-numpyro = paso siguiente.)")
        mcmc = None
    else:
        print("\n[MCMC] ni pymc ni numpyro instalados -> EB/REML (partial pooling cerrado).")
        print("       La versión NUTS es el paso siguiente: el modelo gaussiano la vuelve")
        print("       casi idéntica (NUTS agrega prior débil en τ e incertidumbre de σ_c).")

    print("\n" + "=" * 64)
    print("MEDIA POBLACIONAL DEL LAG  μ_pop  y  DISPERSIÓN ENTRE CRISIS  τ")
    print("=" * 64)
    print(f"  [EB/REML]  μ_pop = {fit['mu_pop']:.2f}  "
          f"IC95 [{fit['mu_pop']-1.96*fit['se_mu']:.2f}, {fit['mu_pop']+1.96*fit['se_mu']:.2f}]  "
          f"trimestres")
    print(f"  [EB/REML]  τ     = {fit['tau']:.2f}  IC95 [{ci_tau[0]:.2f}, {ci_tau[1]:.2f}]  "
          f"(sd ENTRE crisis del lag)")
    if nuts_res is not None:
        print(f"  [NUTS  ]  μ_pop = {nuts_res['mu_pop']:.2f}  "
              f"IC95 [{nuts_res['mu_ci'][0]:.2f}, {nuts_res['mu_ci'][1]:.2f}]")
        print(f"  [NUTS  ]  τ     = {nuts_res['tau']:.2f}  "
              f"IC95 [{nuts_res['tau_ci'][0]:.2f}, {nuts_res['tau_ci'][1]:.2f}]   "
              f"P(τ<0.5) = {nuts_res['p_tau_small']:.2f}")

    print("\n--- ¿τ DISTINGUIBLE DE 0?  (¿el lag fluctúa entre crisis?) ---")
    print(f"  punto-vs-distribución:  LR = {pvd['lr']:.2f}  "
          f"p (frontera, mezcla χ²) = {pvd['p']:.3f}")
    print(f"  Bayes factor (BIC)  BF_10 = {pvd['bf10']:.2f}   ΔBIC = {pvd['dbic']:+.2f}  "
          f"({'favorece DISTRIBUCIÓN (τ>0)' if pvd['bf10']>1 else 'favorece PUNTO (τ=0)'})")
    tau_zero = ci_tau[0] <= 1e-6
    verdict = ("τ NO distinguible de 0: el lag central es estadísticamente CONSTANTE entre "
               "crisis (un único lag común; la 'fluctuación' es ruido de identificación)."
               if (tau_zero or pvd['bf10'] < 1)
               else "τ > 0 con soporte: el lag central FLUCTÚA entre crisis.")
    print(f"\n  VEREDICTO: {verdict}")
    print(f"  Inferencia usada: {used}")

    # salidas
    path = C.OUTDIR / "p12_hier_bayes.png"
    plot(labels, mu_hat, sigma, fit, ci_tau, path)
    rows = [dict(crisis=l, mu_hat=m, sigma=s) for l, m, s in zip(labels, mu_hat, sigma)]
    pd.DataFrame(rows).to_csv(C.OUTDIR / "p12_per_crisis_lag.csv", index=False)
    summ = dict(mu_pop=fit["mu_pop"], se_mu=fit["se_mu"], tau=fit["tau"],
                tau_ci_lo=ci_tau[0], tau_ci_hi=ci_tau[1], lr=pvd["lr"], p=pvd["p"],
                bf10=pvd["bf10"], dbic=pvd["dbic"], inference=used, n_crisis=len(labels))
    if nuts_res is not None:
        summ.update(nuts_mu_pop=nuts_res["mu_pop"], nuts_tau=nuts_res["tau"],
                    nuts_p_tau_small=nuts_res["p_tau_small"])
    pd.DataFrame([summ]).to_csv(C.OUTDIR / "p12_summary.csv", index=False)
    print(f"\n  figura -> {path}")
    print(f"  CSV    -> p12_per_crisis_lag.csv , p12_summary.csv")
    return fit, pvd, nuts_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test: pocas muestras/iteraciones (no es el run completo)")
    args = ap.parse_args()
    run(smoke=args.smoke)


if __name__ == "__main__":
    main()
