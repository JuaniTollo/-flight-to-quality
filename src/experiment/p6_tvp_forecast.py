"""Pieza 6 — Métrica: ¿modelar la no-estacionariedad mejora el pronóstico?

Compara, sobre los MISMOS objetivos (backtest origen-móvil, h={1,2,4}):
  - RW                : persistencia (baseline)
  - VAR expanding     : estacionario, re-estimado con TODA la historia ≤ origen
  - VAR rodante 12a   : NO-estacionario (TVP) — solo los últimos 48 trimestres → se adapta
                        al período/dinámica local cambiante.

Si VAR_rodante < VAR_expanding (RMSE), entonces incorporar la no-estacionariedad ANDA
MEJOR — y el DM dice si es significativo. Se reporta full-sample, por época, y en crisis.

Run:  uv run python -m src.experiment.p6_tvp_forecast
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR

from src.experiment import common as C

warnings.filterwarnings("ignore")

HORIZONS = [1, 2, 4]
ROLL = 48                       # ventana rodante = 12 años
MINTRAIN = 40                   # mínimo para expanding
OOS_START = "1962-01-01"
CRISIS = [("2008", "2007-01-01", "2010-12-31"), ("2020", "2019-01-01", "2021-12-31"),
          ("2001", "2000-06-01", "2002-12-31")]


def var_forecast(block, h):
    res = VAR(block).fit(maxlags=4, ic="bic")
    p = max(1, res.k_ar)
    return res.forecast(block[-p:], steps=h)[h - 1]


def dm(e1, e2, h):
    d = e1 ** 2 - e2 ** 2; n = len(d)
    if n < 5:
        return np.nan
    db = d.mean(); g = [np.mean((d[k:] - db) * (d[:n - k] - db)) for k in range(h)]
    var = (g[0] + 2 * sum(g[1:])) / n
    if var <= 0:
        return np.nan
    s = db / np.sqrt(var) * np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-9))
    return 2 * (1 - stats.t.cdf(abs(s), df=n - 1))


def main():
    df = C.load(); raw = df[C.COLS].to_numpy(); dates = df.index
    o0 = dates.get_indexer([pd.Timestamp(OOS_START)], method="bfill")[0]
    recs = []
    for oi in range(o0, len(raw) - max(HORIZONS)):
        if oi < MINTRAIN:
            continue
        exp_block, roll_block = raw[:oi + 1], raw[max(0, oi - ROLL + 1):oi + 1]
        try:
            fe = {h: var_forecast(exp_block, h) for h in HORIZONS}
            fr = {h: var_forecast(roll_block, h) for h in HORIZONS}
        except Exception:
            continue
        for h in HORIZONS:
            ti = oi + h; td = dates[ti]; yt = raw[ti]
            for j, s in enumerate(["profits", "invest"]):
                recs.append(dict(target=td, h=h, series=s, y_true=yt[j],
                                 rw=raw[oi, j], var_exp=fe[h][j], var_roll=fr[h][j]))
    R = pd.DataFrame(recs)

    def block(sub, tag):
        out = []
        for h in HORIZONS:
            for s in ["profits", "invest"]:
                d = sub[(sub.h == h) & (sub.series == s)]
                if len(d) < 5:
                    continue
                e_rw = (d.rw - d.y_true).to_numpy()
                e_ex = (d.var_exp - d.y_true).to_numpy()
                e_ro = (d.var_roll - d.y_true).to_numpy()
                rr = np.sqrt(np.mean(e_rw ** 2))
                out.append(dict(scope=tag, h=h, series=s, n=len(d),
                                U_exp=np.sqrt(np.mean(e_ex ** 2)) / rr,
                                U_roll=np.sqrt(np.mean(e_ro ** 2)) / rr,
                                dm_roll_vs_exp=dm(e_ro, e_ex, h)))
        return out

    R["yr"] = R["target"].dt.year
    rows = block(R, "full")
    rows += block(R[R.yr < 1985], "pre1985")
    rows += block(R[R.yr >= 1985], "post1985")     # era de período cambiante (Gran Moderación)
    cris = pd.concat([R[(R.target >= a) & (R.target <= b)] for _, a, b in CRISIS])
    rows += block(cris, "crisis")
    out = pd.DataFrame(rows)
    out.to_csv(C.OUTDIR / "p6_tvp_forecast.csv", index=False)

    print("=== Pieza 6 — ¿el VAR rodante (no-estacionario) anda mejor? Theil's U vs RW ===")
    print("   (U<1 ⇒ le gana a RW; * = DM rodante-vs-expanding p<0.05)\n")
    for scope in ["full", "pre1985", "post1985", "crisis"]:
        print(f"  [{scope}]")
        for s in ["profits", "invest"]:
            cells = []
            for h in HORIZONS:
                r = out[(out.scope == scope) & (out.series == s) & (out.h == h)]
                if not len(r):
                    continue
                r = r.iloc[0]; sig = "*" if (r.dm_roll_vs_exp or 1) < 0.05 else " "
                better = "↓" if r.U_roll < r.U_exp else "↑"
                cells.append(f"h{h}: exp={r.U_exp:.2f} roll={r.U_roll:.2f}{better}{sig}")
            print(f"    {s:8s} | " + "  ".join(cells))
        print()
    print(f"✓ {C.OUTDIR}/p6_tvp_forecast.csv   (↓ = rodante mejor que expanding)")


if __name__ == "__main__":
    main()
