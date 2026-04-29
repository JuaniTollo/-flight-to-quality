"""Replicates Fig 2.12 (Corporate Profits), Fig 2.15 (World GDP Growth) and the
capital/inventory cycle plot from Tapia (2023, *Six Crises of the World Economy*).

Run:  uv run python -m src.eda.eda_00_tapia_figures
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from src.eda._style import DATA_PROCESSED, PLOTS_DIR, setup


def fig_corporate_profits() -> None:
    df = pd.read_csv(DATA_PROCESSED / "corporate_profits.csv", index_col=0, parse_dates=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df.index, df["PROFITS_BEFORE_TAX"], color="black", lw=1.5, label="Profits Before Tax")
    ax.plot(df.index, df["PROFITS_AFTER_TAX"], color="black", lw=1.5, ls="--", label="Profits After Tax")
    if "RECESSION" in df.columns:
        ax.fill_between(df.index, 0, 1, where=df["RECESSION"] > 0.5,
                        transform=ax.get_xaxis_transform(),
                        color="#d3d3d3", alpha=0.5, lw=0, label="Recession")
    ax.set_yscale("log")
    ax.set_ylabel("Billions of Dollars (Log Scale)")
    ax.set_title("Fig 2.12: Corporate Profits Before & After Tax", loc="left")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax.legend(loc="upper left")
    fig.text(0.13, 0.02,
             "Source: US BEA via FRED (A446RC1Q027SBEA, A448RC1Q027SBEA)",
             fontsize=8, color="gray", style="italic")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(PLOTS_DIR / "00_corporate_profits.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_global_growth() -> None:
    df = pd.read_csv(DATA_PROCESSED / "global_growth.csv", index_col=0, parse_dates=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df.index, df["GROWTH"], color="#34495e", alpha=0.6, width=250, label="Annual Growth (%)")
    ax.plot(df.index, df["TREND_10Y"], color="#c0392b", lw=2, label="10-year Structural Trend")
    ax.axhline(0, color="black", linewidth=1)

    contractions = df[df["GROWTH"] < 0]
    if not contractions.empty:
        ax.scatter(contractions.index, contractions["GROWTH"], color="red", s=20, zorder=5)
        for row in contractions.itertuples():
            ax.annotate(f"{row.Index.year}", xy=(row.Index, row.GROWTH),
                        xytext=(row.Index, row.GROWTH - 0.5), ha="center", fontsize=8,
                        arrowprops=dict(arrowstyle="-", color="gray"))

    ax.set_title("Fig 2.15: World GDP per Capita Growth", loc="left")
    ax.set_ylabel("Annual Growth (%)")
    ax.legend(loc="upper right", frameon=True)
    fig.text(0.13, 0.02,
             "Source: FRED NYGDPPCAPKDWLD (constant 2015 US$)",
             fontsize=8, color="gray", style="italic")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(PLOTS_DIR / "00_global_growth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_capital_cycle() -> None:
    df = pd.read_csv(DATA_PROCESSED / "capital_cycle.csv", index_col=0, parse_dates=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if "INV_RATIO" in df.columns:
        ax1.plot(df.index, df["INV_RATIO"], color="#8e44ad", lw=2, label="Investment / GDP (%)")
        ax1.axhline(df["INV_RATIO"].mean(), color="black", ls=":", label="Long-term Mean")
        ax1.set_title("A. Capital Over-Accumulation (Investment / GDP)", loc="left")
        ax1.set_ylabel("Share of GDP (%)")
    if "RECESSION" in df.columns:
        ax1.fill_between(df.index, 0, 1, where=df["RECESSION"] > 0.5,
                         transform=ax1.get_xaxis_transform(),
                         color="#7f8c8d", alpha=0.2, lw=0, label="NBER Recession")
    ax1.legend(loc="lower right")

    if "REAL_INVENTORY_CHANGE" in df.columns:
        colors = np.where(df["REAL_INVENTORY_CHANGE"] >= 0, "#2980b9", "#c0392b")
        ax2.bar(df.index, df["REAL_INVENTORY_CHANGE"], color=colors, width=100, alpha=0.7)
        ax2.axhline(0, color="black", lw=1)
        ax2.set_title("B. Inventory Cycle (Real Change in Private Inventories)", loc="left")
        ax2.set_ylabel("Billions of 2017 Dollars")
    if "RECESSION" in df.columns:
        ax2.fill_between(df.index, 0, 1, where=df["RECESSION"] > 0.5,
                         transform=ax2.get_xaxis_transform(),
                         color="#7f8c8d", alpha=0.2, lw=0)

    fig.text(0.13, 0.02,
             "Source: FRED GPDIC1, GDPC1, CBIC1",
             fontsize=8, color="gray", style="italic")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(PLOTS_DIR / "00_capital_inventory_cycle.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup()
    fig_corporate_profits()
    fig_global_growth()
    fig_capital_cycle()
    print(f"[EDA 00] wrote 3 figures to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
