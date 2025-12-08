import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import numpy as np
import argparse
from pathlib import Path
import os

# --- NEURIPS STYLE ---
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

def plot_corporate_profits(df, output_dir):
    """Genera la Fig 2.12: Beneficios Antes vs Después de Impuestos"""
    fig, ax = plt.subplots()
    
    if 'PROFITS_BEFORE_TAX' not in df.columns:
        print("Skipping Profits plot: Columns missing.")
        return

    # 1. Plot Lines
    ax.plot(df.index, df['PROFITS_BEFORE_TAX'], color='black', lw=1.5, label='Profits Before Tax')
    ax.plot(df.index, df['PROFITS_AFTER_TAX'], color='black', lw=1.5, ls='--', label='Profits After Tax')
    
    # 2. Recessions
    if 'RECESSION' in df.columns:
        ax.fill_between(df.index, 0, 1, where=df['RECESSION']>0.5, transform=ax.get_xaxis_transform(),
                        color='#d3d3d3', alpha=0.5, lw=0, label='Recession')

    # 3. Log Scale & Formatting
    ax.set_yscale('log')
    ax.set_ylabel('Billions of Dollars (Log Scale)')
    ax.set_title('Fig 2.12: Corporate Profits Before & After Tax', loc='left')
    
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: '{:,.0f}'.format(y)))
    
    fig.text(0.13, 0.02, "Source: US Bureau of Economic Analysis (FRED Series A446RC1/A448RC1)", 
             fontsize=8, color='gray', style='italic')

    plt.legend(loc='upper left')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_dir / "fig2_12_corporate_profits.png", dpi=300)
    print(f"✅ Saved: {output_dir}/fig2_12_corporate_profits.png")

def plot_global_growth(df, output_dir):
    """Replicates Fig 2.15: Annual growth of WGDP per capita."""
    fig, ax = plt.subplots()
    
    # Calculate Growth if needed
    col_plot = None
    if 'WGDP_PC_LEVEL' in df.columns:
        df['Growth'] = df['WGDP_PC_LEVEL'].pct_change() * 100
        col_plot = 'Growth'
    
    if not col_plot:
        print("❌ Missing WGDP column.")
        return

    df = df.dropna()

    # Bar Chart
    ax.bar(df.index, df[col_plot], color='#34495e', alpha=0.6, width=250, label='Annual Growth (%)')
    
    # Trend
    df['Trend'] = df[col_plot].rolling(window=10, center=True).mean()
    ax.plot(df.index, df['Trend'], color='#c0392b', lw=2, label='10y Structural Trend')

    ax.axhline(0, color='black', linewidth=1)
    
    # Contractions
    contractions = df[df[col_plot] < 0]
    if not contractions.empty:
        ax.scatter(contractions.index, contractions[col_plot], color='red', s=20, zorder=5)
        for row in contractions.itertuples():
            val = float(getattr(row, col_plot))
            ax.annotate(f"{row.Index.year}", xy=(row.Index, val), xytext=(row.Index, val - 0.5),
                        ha='center', fontsize=8, arrowprops=dict(arrowstyle='-', color='gray'))

    ax.set_title('Fig 2.15: World GDP per Capita Growth (Calculated)', loc='left')
    ax.set_ylabel('Annual Growth (%)')
    
    fig.text(0.13, 0.02, "Source: Derived from World GDP per Capita Constant 2015 US$ (FRED: NYGDPPCAPKDWLD).", 
             fontsize=8, color='gray', style='italic')

    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_dir / "fig2_15_global_growth.png", dpi=300)
    print(f"✅ Saved: {output_dir}/fig2_15_global_growth.png")

def plot_capital_cycle(df, output_dir):
    """Plots Investment/GDP and Inventory Cycles"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- PANEL 1: Investment/GDP ---
    if 'REAL_INVESTMENT' in df.columns and 'REAL_GDP' in df.columns:
        df['Inv_Ratio'] = (df['REAL_INVESTMENT'] / df['REAL_GDP']) * 100
        ax1.plot(df.index, df['Inv_Ratio'], color='#8e44ad', lw=2, label='Investment as % of GDP')
        ax1.set_title(r'\textbf{A. Capital Over-Accumulation (Investment/GDP Ratio)}', loc='left')
        ax1.set_ylabel('Share of GDP (%)')
        ax1.axhline(df['Inv_Ratio'].mean(), color='black', ls=':', label='Long-term Average')
    
    if 'RECESSION' in df.columns:
        ax1.fill_between(df.index, 0, 1, where=df['RECESSION']>0.5, transform=ax1.get_xaxis_transform(),
                        color='#7f8c8d', alpha=0.2, lw=0, label='NBER Recession')
    ax1.legend(loc='lower right')

    # --- PANEL 2: Inventory Cycle ---
    col_inv = 'REAL_INVENTORY_CHANGE'
    if col_inv in df.columns:
        colors = np.where(df[col_inv] >= 0, '#2980b9', '#c0392b')
        ax2.bar(df.index, df[col_inv], color=colors, width=100, alpha=0.7, label='Inventory Change')
        ax2.set_title(r'\textbf{B. Inventory Cycle (Real Change in Private Inventories)}', loc='left')
        ax2.set_ylabel('Billions of 2017 Dollars')
        ax2.axhline(0, color='black', lw=1)

    if 'RECESSION' in df.columns:
        ax2.fill_between(df.index, 0, 1, where=df['RECESSION']>0.5, transform=ax2.get_xaxis_transform(),
                        color='#7f8c8d', alpha=0.2, lw=0)

    fig.text(0.13, 0.02, "Source: FRED Series GPDIC1 (Investment), GDPC1 (GDP), CBIC1 (Inventories).", 
             fontsize=8, color='gray', style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_dir / "fig_capital_inventory_cycle.png", dpi=300)
    print(f"✅ Saved: {output_dir}/fig_capital_inventory_cycle.png")

def main(dataset_name):
    processed_path = Path(f"data/processed/{dataset_name}.csv")
    output_dir = Path(f"out/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not processed_path.exists():
        print(f"❌ Data file not found: {processed_path}. Run processor first.")
        return

    print(f"📊 Visualizing: {dataset_name}...")
    df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
    
    # DISPATCHER LIMPIO
    if dataset_name == 'corporate_profits':
        plot_corporate_profits(df, output_dir)
    elif dataset_name == 'global_growth':
        plot_global_growth(df, output_dir)
    elif dataset_name == 'capital_cycle':
        plot_capital_cycle(df, output_dir)
    else:
        print(f"No specific plot defined for {dataset_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name from config")
    args = parser.parse_args()
    main(args.dataset)