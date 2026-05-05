"""EDA 03 — Phase-space topology: profit vs investment z-scored trajectory.

Lotka-Volterra dynamics produce anticlockwise orbits. We compare raw YoY vs
z-score, plot the full orbit, and a per-cycle small-multiples panel.

Run:  uv run python -m src.eda.eda_03_espacio_fases
"""
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

from src.eda._style import DATA_PROCESSED, PLOTS_DIR, setup


def main() -> None:
    setup()
    df = pd.read_csv(DATA_PROCESSED / "lotka_volterra.csv", index_col=0, parse_dates=True)
    print(f"[EDA 03] {df.index[0].date()} → {df.index[-1].date()} | N={len(df)}")
    print(df[["P_z", "I_z"]].describe().round(3))

    norm = mcolors.Normalize(vmin=df.index.year.min(), vmax=df.index.year.max())
    cmap = cm.get_cmap("RdYlBu_r")
    rec_mask = df["RECESSION"] > 0.5 if "RECESSION" in df.columns else pd.Series(False, index=df.index)

    def draw(ax, x_col, y_col, xlabel, ylabel, title):
        for i in range(len(df) - 1):
            color = cmap(norm(df.index.year[i]))
            ax.plot([df[x_col].iloc[i], df[x_col].iloc[i + 1]],
                    [df[y_col].iloc[i], df[y_col].iloc[i + 1]],
                    color=color, lw=0.8, alpha=0.7)
        ax.scatter(df.loc[~rec_mask, x_col], df.loc[~rec_mask, y_col],
                   c=df.index[~rec_mask].year, cmap="RdYlBu_r",
                   vmin=norm.vmin, vmax=norm.vmax, s=12, alpha=0.6, zorder=3)
        ax.scatter(df.loc[rec_mask, x_col], df.loc[rec_mask, y_col],
                   color="#c0392b", s=30, marker="x", alpha=0.9, zorder=4, label="NBER Recession")
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.legend(fontsize=9)

    # === A vs B: raw YoY vs z-score ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    draw(ax1, "PROFITS_YOY", "INVEST_YOY",
         "Profits YoY % (prey)", "Investment YoY % (predator)",
         "A. Raw YoY — flattened ellipse")
    draw(ax2, "P_z", "I_z",
         "Profits z-score (prey)", "Investment z-score (predator)",
         "B. Z-score — symmetric cycle")
    sm = cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax1, ax2], label="Year", shrink=0.8)
    fig.suptitle("Phase space: impact of z-score normalization (Tapia — Lotka-Volterra)", y=1.01)
    fig.text(0.08, -0.02,
             "Source: FRED A053RC1Q027SBEA, GPDI. Quarterly YoY % change + z-score.",
             fontsize=8, color="gray", style="italic")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_comparacion_yoy_vs_zscore.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # === Full z-score orbit with quadrant labels ===
    fig, ax = plt.subplots(figsize=(9, 7))
    for i in range(len(df) - 1):
        color = cmap(norm(df.index.year[i]))
        ax.plot([df["P_z"].iloc[i], df["P_z"].iloc[i + 1]],
                [df["I_z"].iloc[i], df["I_z"].iloc[i + 1]],
                color=color, lw=0.9, alpha=0.75)
    ax.scatter(df.loc[~rec_mask, "P_z"], df.loc[~rec_mask, "I_z"],
               c=df.index[~rec_mask].year, cmap="RdYlBu_r",
               vmin=norm.vmin, vmax=norm.vmax, s=14, alpha=0.65, zorder=3)
    ax.scatter(df.loc[rec_mask, "P_z"], df.loc[rec_mask, "I_z"],
               color="#c0392b", s=35, marker="x", alpha=0.9, zorder=4, label="NBER Recession")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)

    for year in [1958, 1970, 1980, 1990, 2001, 2009, 2020]:
        yr_data = df[df.index.year == year]
        if not yr_data.empty:
            ax.annotate(str(year), xy=(yr_data["P_z"].iloc[0], yr_data["I_z"].iloc[0]),
                        fontsize=8, color="#2c3e50", xytext=(6, 4), textcoords="offset points")

    lim = 2.8
    ax.text(lim, lim * 0.85, "I\nExpansion", ha="center", fontsize=9, color="#27ae60", alpha=0.7)
    ax.text(-lim, lim * 0.85, "II\nOver-accumulation", ha="center", fontsize=9, color="#e67e22", alpha=0.7)
    ax.text(-lim, -lim * 0.85, "III\nContraction", ha="center", fontsize=9, color="#c0392b", alpha=0.7)
    ax.text(lim, -lim * 0.85, "IV\nRecovery", ha="center", fontsize=9, color="#2980b9", alpha=0.7)

    sm = cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Year")
    ax.set_xlabel("Corporate profits — z-score (prey P)")
    ax.set_ylabel("Private investment — z-score (predator I)")
    ax.set_title("Phase space (z-score): economic cycle", loc="left")
    ax.legend(loc="upper left")
    fig.text(0.08, -0.02,
             "Source: FRED A053RC1Q027SBEA, GPDI. Z-score on quarterly YoY % change.",
             fontsize=8, color="gray", style="italic")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_espacio_fases_zscore.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # === Per-cycle small multiples ===
    cycles = {
        "1970 recession": ("1967", "1973"),
        "1980–82 crisis": ("1978", "1984"),
        "Dot-com 2001": ("1997", "2004"),
        "GFC 2008–09": ("2004", "2012"),
        "COVID 2020": ("2017", "2022"),
    }
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    cycle_cmaps = ["Purples", "Reds", "Blues", "Greens", "Oranges"]
    cycle_dark = ["#6c3483", "#922b21", "#1f618d", "#1e8449", "#b9770e"]
    arrow_every = 6  # quarters between arrowheads (~1.5 years)

    for ax, (label, (start, end)), cmap_name, dark in zip(
        axes, cycles.items(), cycle_cmaps, cycle_dark
    ):
        sub = df.loc[start:end]
        if sub.empty or len(sub) < 3:
            ax.set_visible(False)
            continue

        x = sub["P_z"].to_numpy()
        y = sub["I_z"].to_numpy()
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_norm = mcolors.Normalize(vmin=0, vmax=len(segments) - 1)
        seg_cmap = plt.get_cmap(cmap_name)
        # restrict colormap range so the lightest segments are still readable
        seg_colors = seg_cmap(0.35 + 0.6 * seg_norm(np.arange(len(segments))))
        lc = LineCollection(segments, colors=seg_colors, linewidths=1.6, alpha=0.9)
        ax.add_collection(lc)

        for i in range(arrow_every, len(x) - 1, arrow_every):
            arrow = FancyArrowPatch(
                (x[i - 1], y[i - 1]), (x[i], y[i]),
                arrowstyle="-|>,head_length=5,head_width=3.5",
                mutation_scale=1.6,
                color=dark, lw=0, alpha=0.85, zorder=4,
            )
            ax.add_patch(arrow)

        ax.scatter(x[0], y[0], color="#27ae60", s=70, zorder=5,
                   edgecolor="white", linewidth=1.2, label="start")
        ax.scatter(x[-1], y[-1], color="#c0392b", s=70, marker="s", zorder=5,
                   edgecolor="white", linewidth=1.2, label="end")
        ax.axhline(0, color="black", lw=0.6)
        ax.axvline(0, color="black", lw=0.6)
        pad = 0.15
        ax.set_xlim(x.min() - pad * (x.max() - x.min()),
                    x.max() + pad * (x.max() - x.min()))
        ax.set_ylim(y.min() - pad * (y.max() - y.min()),
                    y.max() + pad * (y.max() - y.min()))
        ax.set_title(f"{label}  ({sub.index[0].year}–{sub.index[-1].year})", loc="left")
        ax.set_xlabel("Profits z-score")
        ax.set_ylabel("Investment z-score")
        ax.legend(fontsize=8, loc="best")
    axes[-1].set_visible(False)
    fig.suptitle(
        "Per-cycle z-score orbits — arrows show direction; "
        "anticlockwise rotation = Lotka-Volterra topology",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_orbitas_zscore_por_ciclo.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA 03] wrote 3 figures to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
