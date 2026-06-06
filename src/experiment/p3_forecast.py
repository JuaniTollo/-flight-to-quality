"""Pieza 3 — Pronóstico alineado (evaluación; los modelos se ajustan en Julia).

Lee output/experiment/julia_forecasts.csv (FN/LV × solver/PINN inversa, producidos por
`models.jl`) y agrega los baselines lineales —RW, media, VAR— calculados sobre EXACTAMENTE
los mismos (crisis, h, target). Reporta Theil's U = RMSE/RMSE_RW y el test Diebold-Mariano
(con corrección Harvey-Leybourne-Newbold) vs RW.

Pre-requisito:  julia --project=.../08_LV_inverse_PINN src/experiment/models.jl
Run:            uv run python -m src.experiment.p3_forecast
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR

from src.experiment import common as C

warnings.filterwarnings("ignore")

JF = C.OUTDIR / "julia_forecasts.csv"
TARGETS = ["2001", "2008", "2020"]
HORIZONS = [1, 2, 4]
LOOKBACK = 12
SMAP = {"profits": "PROFITS_YOY", "invest": "INVEST_YOY"}


def dm_hln(e_model, e_rw, h):
    """Diebold-Mariano (pérdida cuadrática) con corrección HLN. <0 ⇒ modelo mejor que RW."""
    d = e_model ** 2 - e_rw ** 2
    n = len(d)
    if n < 4:
        return np.nan, np.nan
    dbar = d.mean()
    g = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (g[0] + 2 * sum(g[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    hln = dm * np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-9))
    p = 2 * (1 - stats.t.cdf(abs(hln), df=n - 1))
    return hln, p


def baselines(df):
    """Predicciones de RW, media y VAR sobre el episodio de cada crisis (mismos targets)."""
    raw = df[C.COLS].to_numpy()
    rmap = {l: (pd.Timestamp(p), pd.Timestamp(t)) for l, p, t in C.RECESSIONS}
    recs = []
    for cr in TARGETS:
        peak, trough = rmap[cr]
        w_lo, w_hi = peak - pd.DateOffset(months=6), trough + pd.DateOffset(months=12)
        tr = np.asarray((df.index >= w_lo - pd.DateOffset(years=LOOKBACK)) & (df.index < w_lo))
        tmean = raw[tr].mean(0)
        vr = VAR(raw[tr]); p = max(1, int(getattr(vr.select_order(maxlags=4), "bic", 1)))
        vres = vr.fit(p)
        epi = np.where((df.index >= w_lo) & (df.index <= w_hi))[0]
        for ti in epi:
            for h in HORIZONS:
                oi = ti - h
                if oi < p:
                    continue
                td = df.index[ti]
                vf = vres.forecast(raw[oi - p + 1:oi + 1], steps=h)[h - 1]
                for j, s in enumerate(["profits", "invest"]):
                    yt = raw[ti, j]
                    recs += [
                        dict(crisis=cr, model="rw", h=h, target=td, series=s, y_true=yt, y_pred=raw[oi, j]),
                        dict(crisis=cr, model="mean", h=h, target=td, series=s, y_true=yt, y_pred=tmean[j]),
                        dict(crisis=cr, model="var", h=h, target=td, series=s, y_true=yt, y_pred=vf[j]),
                    ]
    return pd.DataFrame(recs)


def main():
    if not JF.exists():
        raise SystemExit(f"Falta {JF}. Corré primero: julia --project=... src/experiment/models.jl")
    df = C.load()
    jf = pd.read_csv(JF, parse_dates=["target_date"]).rename(columns={"target_date": "target"})
    jf["model"] = jf["model"] + "_" + jf["method"]               # FN_solver, FN_pinn, LV_solver, LV_pinn
    jf = jf[["crisis", "model", "h", "target", "series", "y_true", "y_pred"]]
    jf["crisis"] = jf["crisis"].astype(str)
    allf = pd.concat([baselines(df), jf], ignore_index=True)
    allf["err"] = allf["y_pred"] - allf["y_true"]
    allf.to_csv(C.OUTDIR / "p3_forecast_raw.csv", index=False)

    models = ["rw", "mean", "var", "FN_solver", "FN_pinn", "LV_solver", "LV_pinn"]
    rows = []
    for cr in TARGETS:
        for h in HORIZONS:
            for s in ["profits", "invest"]:
                sub = allf[(allf.crisis == cr) & (allf.h == h) & (allf.series == s)]
                rw = sub[sub.model == "rw"].set_index("target")["err"]
                for m in models:
                    em = sub[sub.model == m].set_index("target")["err"]
                    common = em.index.intersection(rw.index)               # alineado
                    if len(common) < 2:
                        continue
                    e_m, e_rw = em.loc[common].to_numpy(), rw.loc[common].to_numpy()
                    rmse = np.sqrt(np.mean(e_m ** 2)); rmse_rw = np.sqrt(np.mean(e_rw ** 2))
                    _, pval = dm_hln(e_m, e_rw, h)
                    rows.append(dict(crisis=cr, model=m, h=h, series=s, n=len(common),
                                     rmse=rmse, theil_u=rmse / rmse_rw if rmse_rw else np.nan,
                                     dm_p_vs_rw=pval))
    res = pd.DataFrame(rows)
    res.to_csv(C.OUTDIR / "p3_forecast_summary.csv", index=False)

    print("=== Pieza 3 — Theil's U (=RMSE/RMSE_RW; <1 ⇒ le gana a RW; * = DM p<0.05) ===")
    for s in ["profits", "invest"]:
        print(f"\n  {s.upper()}")
        print("  " + " " * 11 + " | " + " | ".join(f"{cr} h{h}" for cr in TARGETS for h in HORIZONS))
        for m in models:
            cells = []
            for cr in TARGETS:
                for h in HORIZONS:
                    r = res[(res.crisis == cr) & (res.model == m) & (res.h == h) & (res.series == s)]
                    if not len(r):
                        cells.append("  -  "); continue
                    u = r.iloc[0]["theil_u"]; sig = "*" if r.iloc[0]["dm_p_vs_rw"] < 0.05 else " "
                    cells.append(f"{u:4.2f}{sig}")
            print(f"  {m:11s} | " + " | ".join(cells))
    print(f"\n✓ {C.OUTDIR}/p3_forecast_summary.csv")


if __name__ == "__main__":
    main()
