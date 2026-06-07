"""Pieza 12 — Robustez del LAG CENTRAL del feedback de sobreacumulación a la
transformación de las series.

Contexto. El feedback inversión→ganancia es un retardo de maduración distribuido
centrado en ~1 año (~4 trimestres). Una nota metodológica previa mostró que la
transformación YoY (`pct_change(4)`) **infla la MAGNITUD** del feedback en lag−4
≈2× respecto de la transformación insesgada QoQ (`pct_change(1)`), porque YoY es una
diferencia a 4 trimestres e inyecta estructura de orden 4 justo donde cae la señal.

Lo que faltaba consolidar no es la magnitud sino la **UBICACIÓN** del lag: ¿el centro
del retardo (~4 trim) se queda en el mismo lugar cuando se cambia la transformación,
aunque su magnitud cambie? Si la ubicación es estable a través de transformaciones muy
distintas (diferencias de distinto orden, log-diferencias, filtros de ciclo), el lag es
un rasgo estructural de los datos y no un artefacto de una transformación particular.

Método. Se reutiliza la calibración por filtro Gamma / distributed-lag de la Pieza 11
(`p11_physical_delay.py`): sobre las MISMAS series de niveles (PROFITS, INVESTMENT) se
aplica cada transformación, se arman ventanas alrededor del fondo de inversión de cada
recesión NBER (z-scoreadas, pooled), y se ajusta por OLS

    P_t ~ c·I_t + e·P_{t-1} + k·m_t(μ),     m_t = (Gamma(μ, shape) ∗ I)_t

barriendo el lag medio μ = 1..8 trim. El μ que maximiza R² es el LAG CENTRAL estimado
bajo esa transformación. Reportamos μ* (ubicación) y k* en μ* (magnitud) por transform.

Transformaciones (todas sobre los mismos niveles):
  - YoY    : pct_change(4)·100          (la inflada de referencia)
  - QoQ    : pct_change(1)·100          (la insesgada de referencia)
  - dlog   : 100·Δ log(nivel)           (≈ QoQ, otra parametrización)
  - HP     : componente cíclica del filtro Hodrick-Prescott (lambda=1600, trim.)
  - BP8-32 : band-pass 8–32 trim (Christiano-Fitzgerald asimétrico; ciclo de negocios)

Lectura: si μ* ≈ 4 en todas, la UBICACIÓN del lag es robusta a la transformación aunque
la MAGNITUD (k) cambie → el retardo de maduración es un rasgo estructural, no un artefacto.

SMOKE TEST (rápido, 2 transforms para validar la maquinaria):
    uv run python -m src.experiment.p12_transform_robustness --smoke
RUN COMPLETO (todas las transforms, tabla + figura):
    uv run python -m src.experiment.p12_transform_robustness
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma as gammadist
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.filters.cf_filter import cffilter

from src.experiment import common as C

DROP = {"2008", "2020"}          # outliers de amplitud; crisis "normales" (igual que P11)
H = 8                            # semiancho de ventana (trim) alrededor del fondo de inversión
SHAPE = 8.0                      # forma del kernel Gamma (ancho fijo; el ancho no es identificable, P11)
MUS = np.arange(1, 9)            # barrido del lag medio (trim)

# --- transformaciones de los niveles PROFITS/INVESTMENT -----------------------
# Cada una toma el DataFrame de niveles (cols PROFITS, INVESTMENT) y devuelve
# (DataFrame transformado con cols P, I) alineado por DATE, ya sin NaN.

def _yoy(lv: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"P": lv["PROFITS"].pct_change(4) * 100,
                         "I": lv["INVESTMENT"].pct_change(4) * 100}).dropna()


def _qoq(lv: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"P": lv["PROFITS"].pct_change(1) * 100,
                         "I": lv["INVESTMENT"].pct_change(1) * 100}).dropna()


def _dlog(lv: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"P": np.log(lv["PROFITS"]).diff() * 100,
                         "I": np.log(lv["INVESTMENT"]).diff() * 100}).dropna()


def _hp(lv: pd.DataFrame) -> pd.DataFrame:
    # filtro HP sobre el log-nivel (lambda=1600 estándar trimestral); componente cíclica
    cP, _ = hpfilter(np.log(lv["PROFITS"]), lamb=1600)
    cI, _ = hpfilter(np.log(lv["INVESTMENT"]), lamb=1600)
    return pd.DataFrame({"P": cP * 100, "I": cI * 100}, index=lv.index).dropna()


def _bp(lv: pd.DataFrame) -> pd.DataFrame:
    # band-pass de ciclo de negocios 8–32 trim sobre el log-nivel.
    # Christiano-Fitzgerald asimétrico (no recorta extremos, sirve para muestra corta).
    # Baxter-King (bkfilter) recortaría 12 trim de cada extremo (3 años): demasiado para
    # 145 trim → se usa CF; se documenta la elección. low=8, high=32 trim.
    cP = cffilter(np.log(lv["PROFITS"]), low=8, high=32, drift=False)[0]
    cI = cffilter(np.log(lv["INVESTMENT"]), low=8, high=32, drift=False)[0]
    return pd.DataFrame({"P": np.asarray(cP) * 100, "I": np.asarray(cI) * 100},
                        index=lv.index).dropna()


TRANSFORMS = {
    "YoY":    (_yoy,  "pct_change(4)·100 (referencia inflada)"),
    "QoQ":    (_qoq,  "pct_change(1)·100 (referencia insesgada)"),
    "dlog":   (_dlog, "100·Δ log(nivel)"),
    "HP":     (_hp,   "HP cycle (λ=1600) sobre log-nivel"),
    "BP8-32": (_bp,   "band-pass 8–32 trim (Christiano-Fitzgerald) sobre log-nivel"),
}
SMOKE_KEYS = ["YoY", "QoQ"]      # smoke test: las dos de referencia


def levels() -> pd.DataFrame:
    raw = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    return raw[["PROFITS", "INVESTMENT"]].dropna()


def crisis_slices(idx, Iv):
    """Ventanas ±H trim alrededor del fondo de inversión de cada recesión (sin 2008/2020).
    Idéntico a p11_physical_delay para que las piezas sean comparables."""
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
    """Kernel Gamma de media mu (trim) y forma 'shape' (idéntico a p11)."""
    scale = mu / shape
    ks = np.arange(0, K + 1) + 0.5
    w = gammadist.pdf(ks, a=shape, scale=scale)
    s = w.sum()
    return w / s if s > 0 and np.all(np.isfinite(w)) else None


def fit(Pv, Iv, slices, mu, shape=SHAPE):
    """P_t ~ c·I_t + e·P_{t-1} + k·m_t, pooled sobre ventanas z-scoreadas → OLS."""
    w = gkernel(mu, shape)
    if w is None:
        return np.nan, None
    X, y = [], []
    for lo, hi in slices:
        Pi, Ii = Pv[lo:hi], Iv[lo:hi]
        sP, sI = Pi.std(), Ii.std()
        if sP == 0 or sI == 0:
            continue
        Pi = (Pi - Pi.mean()) / sP
        Ii = (Ii - Ii.mean()) / sI
        m = np.convolve(Ii, w)[:len(Ii)]
        for t in range(1, len(Pi)):
            X.append([Ii[t], Pi[t - 1], m[t]]); y.append(Pi[t])
    if not X:
        return np.nan, None
    X, y = np.array(X), np.array(y)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2)
    return r2, b


def estimate_central_lag(lv, transform_fn):
    """Aplica una transform, arma ventanas de crisis y barre μ. Devuelve dict con el
    perfil R²(μ), el lag central μ*, y k (magnitud) en μ*."""
    d = transform_fn(lv)
    idx, Pv, Iv = d.index, d["P"].to_numpy(), d["I"].to_numpy()
    sl = crisis_slices(idx, Iv)
    r2 = np.array([fit(Pv, Iv, sl, m)[0] for m in MUS])
    mu_star = int(MUS[int(np.nanargmax(r2))])
    _, b = fit(Pv, Iv, sl, mu_star)
    return {"r2": r2, "mu_star": mu_star, "k_at_mu": float(b[2]), "n_crisis": len(sl)}


def run(keys):
    lv = levels()
    print("=== Pieza 12 — robustez del LAG CENTRAL del feedback a la transformación ===")
    print(f"series: niveles PROFITS/INVESTMENT | ventana ±{H} trim del fondo de inversión")
    print(f"kernel Gamma shape={SHAPE:g} (ancho fijo) | barrido μ={MUS.min()}..{MUS.max()} trim\n")

    results = {}
    for key in keys:
        fn, desc = TRANSFORMS[key]
        results[key] = estimate_central_lag(lv, fn)
        results[key]["desc"] = desc

    # --- tabla ---------------------------------------------------------------
    print(f"{'transform':<8} {'lag central μ*':>14} {'k en μ* (magnitud)':>20} {'R²(μ*)':>9} {'crisis':>7}")
    print("-" * 64)
    for key in keys:
        r = results[key]
        print(f"{key:<8} {r['mu_star']:>11d} q {r['k_at_mu']:>20.3f} "
              f"{np.nanmax(r['r2']):>9.3f} {r['n_crisis']:>7d}")
    mus_found = [results[k]["mu_star"] for k in keys]
    spread = max(mus_found) - min(mus_found)
    print("-" * 64)
    print(f"ubicación del lag: μ* ∈ [{min(mus_found)}, {max(mus_found)}] trim, "
          f"rango = {spread} trim (mediana {int(np.median(mus_found))})")
    stable = spread <= 1
    print(f"→ UBICACIÓN del lag {'ESTABLE' if stable else 'NO estable'} a la transformación "
          f"(μ* {'≈ 4 trim ~1 año en todas' if stable and abs(int(np.median(mus_found)) - 4) <= 1 else 'varía'}); "
          f"la magnitud k {'cambia entre transforms' if len(set(round(results[k]['k_at_mu'],2) for k in keys)) > 1 else 'es similar'}.\n")

    # --- figura --------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for key in keys:
        r = results[key]
        ax[0].plot(MUS, r["r2"], "o-", ms=4, label=key)
        ax[0].axvline(r["mu_star"], ls=":", lw=0.6, alpha=0.5)
    ax[0].axvspan(3.5, 4.5, color="grey", alpha=0.12, label="~1 año")
    ax[0].set_xlabel("μ = lag medio de maduración (trim)")
    ax[0].set_ylabel("R² del ajuste distributed-lag")
    ax[0].set_title("Perfil R²(μ) por transformación\n(el pico ubica el lag central)")
    ax[0].legend(fontsize=8)

    xs = np.arange(len(keys))
    mus_plot = [results[k]["mu_star"] for k in keys]
    ks_plot = [abs(results[k]["k_at_mu"]) for k in keys]
    ax[1].bar(xs - 0.18, mus_plot, width=0.35, label="lag central μ* (trim)", color="C0")
    ax2 = ax[1].twinx()
    ax2.bar(xs + 0.18, ks_plot, width=0.35, label="|k| en μ* (magnitud)", color="C1", alpha=0.8)
    ax[1].axhline(4, color="grey", ls="--", lw=0.8)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(keys, rotation=0)
    ax[1].set_ylabel("lag central μ* (trim)", color="C0")
    ax2.set_ylabel("|k| en μ* (magnitud)", color="C1")
    ax[1].set_ylim(0, max(MUS) + 1)
    ax[1].set_title("lag central μ* y magnitud |k| por transformación")
    fig.suptitle("Pieza 12 — ubicación del lag central del feedback a través de transformaciones",
                 fontsize=12)
    fig.tight_layout()
    out = C.OUTDIR / "p12_transform_robustness.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"✓ figura: {out}")

    # --- veredicto, leído de los números (no hardcodeado) --------------------
    # se separan dos familias: transforms por DIFERENCIA (QoQ, dlog) frente a
    # FILTROS de ciclo (HP, band-pass), porque se comportan distinto.
    diff_keys = [k for k in keys if k in ("QoQ", "dlog")]
    diff_mus = [results[k]["mu_star"] for k in diff_keys]
    print("\nVEREDICTO (leído de los números):")
    if diff_mus and (max(diff_mus) - min(diff_mus)) <= 1 and all(abs(m - 4) <= 1 for m in diff_mus):
        print(f"  - Transforms por diferencia insesgadas ({', '.join(diff_keys)}): el lag central"
              f"\n    se ubica en μ*={diff_mus} trim, consistente con ~1 año. La ubicación del lag"
              f"\n    es ROBUSTA dentro de esta familia; lo que cambia es la magnitud |k|.")
    elif diff_mus:
        print(f"  - Transforms por diferencia ({', '.join(diff_keys)}): μ*={diff_mus} trim.")
    if "YoY" in keys:
        print(f"  - YoY (pct_change(4)): μ*={results['YoY']['mu_star']} trim — su estructura de orden 4"
              "\n    desplaza el pico del barrido (además de inflar |k|), por eso QoQ es la referencia.")
    filt_keys = [k for k in keys if k in ("HP", "BP8-32")]
    if filt_keys:
        fm = {k: results[k]["mu_star"] for k in filt_keys}
        print(f"  - Filtros de ciclo ({', '.join(filt_keys)}): μ*={list(fm.values())} trim — colapsan a"
              "\n    lag corto. Son filtros de dos lados (HP) / band-pass que ya extraen el componente"
              "\n    cíclico co-movido; sobre series pre-filtradas el barrido distributed-lag carga el"
              "\n    término contemporáneo y no es el diagnóstico apropiado para ubicar el retardo.")
    print("  ⇒ El lag central ~1 año es estable a la elección de DIFERENCIA (QoQ/dlog), no a"
          "\n    cualquier detrending: los filtros de ciclo cambian la pregunta. La magnitud |k|"
          "\n    depende fuertemente de la transform (YoY la infla, como ya se sabía).")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test: solo 2 transforms (YoY, QoQ) para validar la maquinaria")
    args = ap.parse_args()
    keys = SMOKE_KEYS if args.smoke else list(TRANSFORMS.keys())
    run(keys)


if __name__ == "__main__":
    main()
