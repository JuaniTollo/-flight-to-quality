"""EDA 02 — Cross-correlation diagnostics for the Tapia causal structure.

We expect:
  - Corr(P_t, I_{t-k}) > 0 at k<0  →  current profits predict future investment
  - Corr(I_t, P_{t-k}) < 0 at k>0  →  past investment depresses current profits

Run:  uv run python -m src.eda.eda_02_rezagos_causalidad
"""
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda._style import DATA_PROCESSED, PLOTS_DIR, setup

MAX_LAG = 12


def cross_corr(x: pd.Series, y: pd.Series, max_lag: int) -> Tuple[List[int], List[float]]:
    lags = list(range(-max_lag, max_lag + 1))
    corrs = [x.corr(y.shift(k)) for k in lags]
    return lags, corrs


def main() -> None:
    setup()
    df = pd.read_csv(DATA_PROCESSED / "lotka_volterra.csv", index_col=0, parse_dates=True)
    print(f"[EDA 02] {df.index[0].date()} → {df.index[-1].date()} | N={len(df)}")

    lags_pi, corrs_pi = cross_corr(df["PROFITS_YOY"], df["INVEST_YOY"], MAX_LAG)
    lags_ip, corrs_ip = cross_corr(df["INVEST_YOY"], df["PROFITS_YOY"], MAX_LAG)

    ci = 1.96 / np.sqrt(len(df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(lags_pi, corrs_pi, color=["#c0392b" if c > 0 else "#2980b9" for c in corrs_pi],
            alpha=0.8, width=0.7)
    ax1.axhline(ci, color="black", ls="--", lw=1, label=f"95% CI ±{ci:.2f}")
    ax1.axhline(-ci, color="black", ls="--", lw=1)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.axvline(0, color="gray", lw=0.8, ls=":")
    ax1.set_title("Corr(P_t, I_{t−k}) — profits as leading indicator", loc="left")
    ax1.set_xlabel("Lag k (quarters)  [k<0 = P predicts future I]")
    ax1.set_ylabel("Pearson correlation")
    ax1.legend(fontsize=9)

    ax2.bar(lags_ip, corrs_ip, color=["#c0392b" if c > 0 else "#2980b9" for c in corrs_ip],
            alpha=0.8, width=0.7)
    ax2.axhline(ci, color="black", ls="--", lw=1, label=f"95% CI ±{ci:.2f}")
    ax2.axhline(-ci, color="black", ls="--", lw=1)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.axvline(0, color="gray", lw=0.8, ls=":")
    ax2.set_title("Corr(I_t, P_{t−k}) — over-accumulation depresses profits", loc="left")
    ax2.set_xlabel("Lag k (quarters)  [k>0 = past I depresses current P]")
    ax2.legend(fontsize=9)

    fig.suptitle("Cross-correlation — empirical validation of Tapia's causal claim", y=1.02)
    fig.text(0.08, -0.04,
             "Source: FRED A053RC1Q027SBEA, GPDI. YoY % change.",
             fontsize=8, color="gray", style="italic")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_ccf_global.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # === Phase-segmented CCF ===
    LAGS_SEG = list(range(0, MAX_LAG + 1))
    df_exp = df[df["RECESSION"] < 0.5]
    df_rec = df[df["RECESSION"] > 0.5]
    print(f"  expansion: N={len(df_exp)}  |  recession: N={len(df_rec)}")
    corr_exp = [df_exp["PROFITS_YOY"].corr(df_exp["INVEST_YOY"].shift(k)) for k in LAGS_SEG]
    corr_rec = [df_rec["PROFITS_YOY"].corr(df_rec["INVEST_YOY"].shift(k)) for k in LAGS_SEG]

    x = np.arange(len(LAGS_SEG))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width / 2, corr_exp, width, label="Expansion", color="#27ae60", alpha=0.8)
    ax.bar(x + width / 2, corr_rec, width, label="Recession", color="#c0392b", alpha=0.8)
    ci_exp = 1.96 / np.sqrt(len(df_exp))
    ax.axhline(ci_exp, color="#27ae60", ls=":", lw=1)
    ax.axhline(-ci_exp, color="#27ae60", ls=":", lw=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in LAGS_SEG])
    ax.set_xlabel("Lag k (quarters)")
    ax.set_ylabel("Corr(P_t, I_{t−k})")
    ax.set_title("Phase-segmented lags — does the causal regime shift?", loc="left")
    ax.legend()
    fig.text(0.08, -0.04,
             "Source: FRED A053RC1Q027SBEA, GPDI, USREC. Phases by NBER indicator.",
             fontsize=8, color="gray", style="italic")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_ccf_segmentado.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # === Summary ===
    ccf_df = pd.DataFrame({"lag": lags_pi, "corr_P_I": corrs_pi, "corr_I_P": corrs_ip})
    neg = ccf_df[ccf_df["lag"] < 0]
    pos = ccf_df[ccf_df["lag"] > 0]
    best_lead = neg.loc[neg["corr_P_I"].idxmax(), "lag"]
    best_suppress = pos.loc[pos["corr_I_P"].idxmin(), "lag"]
    print(f"  profits lead investment: best lag = {abs(int(best_lead))} quarters")
    print(f"  past investment suppresses profits: best lag = {int(best_suppress)} quarters")
    print(f"[EDA 02] wrote 2 figures to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
