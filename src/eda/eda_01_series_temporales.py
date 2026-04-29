"""EDA 01 — Time-series of YoY growth in profits and investment, with NBER bands.

Tapia's hypothesis: profitability is a leading indicator — it peaks and falls
*before* investment.

We render three views, because a 1-quarter lead is invisible on an axis with
300+ quarters:

  - 01_series_temporales.png            : full history, single row (overview)
  - 01_series_temporales_split.png      : same data split into 3 rows
  - 01_series_temporales.html           : interactive Plotly version (zoom/scroll)
  - 01_zoom_ciclos.png                  : 3 hand-picked cycles, ~10 years each

Run:  uv run python -m src.eda.eda_01_series_temporales
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import plotly.graph_objects as go

from src.eda._style import DATA_PROCESSED, PLOTS_DIR, setup


def _draw_panel(ax, sub: pd.DataFrame, title: str | None = None) -> None:
    ax.plot(sub.index, sub["PROFITS_YOY"], color="#c0392b", lw=1.8, label="Profits (prey)")
    ax.plot(sub.index, sub["INVEST_YOY"], color="#2980b9", lw=1.8, ls="--", label="Investment (predator)")
    ax.axhline(0, color="black", lw=0.8)
    if "RECESSION" in sub.columns:
        ax.fill_between(sub.index, 0, 1, where=sub["RECESSION"] > 0.5,
                        transform=ax.get_xaxis_transform(),
                        color="#bdc3c7", alpha=0.4, lw=0, label="NBER Recession")
    if title:
        ax.set_title(title, loc="left")
    ax.set_ylabel("YoY %")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0f}%"))


def main() -> None:
    setup()
    df = pd.read_csv(DATA_PROCESSED / "lotka_volterra.csv", index_col=0, parse_dates=True)
    print(f"[EDA 01] {df.index[0].date()} → {df.index[-1].date()} | N={len(df)}")

    # === A) Full history (overview) ===
    fig, ax = plt.subplots(figsize=(14, 5))
    _draw_panel(ax, df, title="Economic cycle: profits vs investment (Tapia — Lotka-Volterra)")
    ax.legend(loc="upper right", frameon=True)
    fig.text(0.08, -0.02,
             "Source: FRED A053RC1Q027SBEA, GPDI, USREC.  At 313 quarters wide a 1Q lag is invisible — see split / interactive views below.",
             fontsize=8, color="gray", style="italic")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_series_temporales.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # === B) Split view: 3 rows, ~26 years each, so 1Q gaps are readable ===
    years = df.index.year
    edges = [years.min(), 1974, 2000, years.max() + 1]
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    for ax, lo, hi in zip(axes, edges[:-1], edges[1:]):
        sub = df[(years >= lo) & (years < hi)]
        _draw_panel(ax, sub, title=f"{lo}–{hi - 1}")
    axes[0].legend(loc="upper right", frameon=True, fontsize=9)
    fig.suptitle("Profits vs investment — split into 26-year panels (1Q lead-lag now legible)", y=1.0)
    fig.text(0.08, -0.01,
             "Source: FRED A053RC1Q027SBEA, GPDI, USREC.",
             fontsize=8, color="gray", style="italic")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_series_temporales_split.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # === C) Plotly interactive HTML (zoom + range slider) ===
    rec_periods = []
    if "RECESSION" in df.columns:
        rec = df["RECESSION"] > 0.5
        in_rec = False
        start = None
        for ts, flag in rec.items():
            if flag and not in_rec:
                start, in_rec = ts, True
            elif not flag and in_rec:
                rec_periods.append((start, ts))
                in_rec = False
        if in_rec:
            rec_periods.append((start, df.index[-1]))

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=df.index, y=df["PROFITS_YOY"],
        name="Profits (prey)", line=dict(color="#c0392b", width=2),
        hovertemplate="%{x|%Y-Q%q}<br>Profits YoY: %{y:.1f}%<extra></extra>",
    ))
    fig_p.add_trace(go.Scatter(
        x=df.index, y=df["INVEST_YOY"],
        name="Investment (predator)", line=dict(color="#2980b9", width=2, dash="dash"),
        hovertemplate="%{x|%Y-Q%q}<br>Investment YoY: %{y:.1f}%<extra></extra>",
    ))
    shapes = [
        dict(type="rect", xref="x", yref="paper", x0=lo, x1=hi, y0=0, y1=1,
             fillcolor="#bdc3c7", opacity=0.35, line_width=0, layer="below")
        for lo, hi in rec_periods
    ]
    shapes.append(dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=0,
                       line=dict(color="black", width=1)))
    fig_p.update_layout(
        title="Profits vs Investment — interactive (drag to zoom, scroll the slider)",
        xaxis=dict(title="Quarter", rangeslider=dict(visible=True), type="date"),
        yaxis=dict(title="YoY change (%)", ticksuffix="%"),
        hovermode="x unified",
        shapes=shapes,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
        height=560,
    )
    html_path = PLOTS_DIR / "01_series_temporales.html"
    fig_p.write_html(str(html_path), include_plotlyjs="cdn")

    # === Per-cycle zoom ===
    cycles = [
        ("1965", "1975", "1970s crisis"),
        ("1995", "2005", "Dot-com & 2001"),
        ("2003", "2012", "GFC 2008"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
    for ax, (start, end, title) in zip(axes, cycles):
        mask = df.loc[start:end]
        ax.plot(mask.index, mask["PROFITS_YOY"], color="#c0392b", lw=2, label="Profits")
        ax.plot(mask.index, mask["INVEST_YOY"], color="#2980b9", lw=2, ls="--", label="Investment")
        ax.axhline(0, color="black", lw=0.8)
        if "RECESSION" in mask.columns:
            ax.fill_between(mask.index, 0, 1, where=mask["RECESSION"] > 0.5,
                            transform=ax.get_xaxis_transform(),
                            color="#bdc3c7", alpha=0.5, lw=0)
        ax.set_title(title, loc="left")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    axes[0].legend(fontsize=9)
    fig.suptitle("Per-cycle zoom: profits lead investment", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_zoom_ciclos.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA 01] wrote 3 PNGs + 1 HTML to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
