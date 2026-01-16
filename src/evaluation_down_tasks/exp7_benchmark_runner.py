
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

# Root dir
ROOT_DIR = Path(__file__).resolve().parents[2]

def main():
    # 1. Collect MSE results
    # Applying user-requested shift of +0.178612 to local models to align scales
    shift = 0.178612
    
    results = [
        # Our Models (CM-Mamba) - Hardcoded stats to avoid torch dependency
        {"Model": "CM-Mamba-Small", "Parameters_M": 4.75, "MSE": 0.0629 + shift, "GFLOPs": 0.44, "Source": "Local"},
        {"Model": "CM-Mamba-Tiny", "Parameters_M": 1.21, "MSE": 0.0980 + shift, "GFLOPs": 0.12, "Source": "Local"},
        
        # Baselines from LightGTS Paper / Image (Standardized Scale)
        {"Model": "LightGTS-mini", "Parameters_M": 4.0, "MSE": 0.294, "GFLOPs": 0.426, "Source": "Paper"},
        {"Model": "LightGTS-tiny", "Parameters_M": 1.3, "MSE": 0.305, "GFLOPs": 0.200, "Source": "Paper"},
        {"Model": "Timer", "Parameters_M": 67.0, "MSE": 0.496, "GFLOPs": 6.4, "Source": "Paper"},
        {"Model": "MOIRAI-L", "Parameters_M": 311.0, "MSE": 0.413, "GFLOPs": 30.0, "Source": "Paper"},
        {"Model": "Time-MoE", "Parameters_M": 453.0, "MSE": 0.371, "GFLOPs": 500.0, "Source": "Paper"},
        
        # Chronos Family (Official Official HF Parameter Counts)
        {"Model": "Chronos-Tiny", "Parameters_M": 8.4, "MSE": 0.462, "GFLOPs": 12.0, "Source": "Paper"},
        {"Model": "Chronos-Small", "Parameters_M": 46.2, "MSE": 0.428, "GFLOPs": 65.0, "Source": "Paper"},
        {"Model": "Chronos-Large", "Parameters_M": 710.0, "MSE": 0.400, "GFLOPs": 1000.0, "Source": "Paper"},
    ]
    
    df = pd.DataFrame(results)
    print(df)

    # Visualization
    # Senior UX Quality: Larger dimensions for better vertical separation
    plt.figure(figsize=(12, 9))
    
    # Global visual settings for paper production
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5
    })

    # Use log scale for Parameters (X-axis)
    plt.xscale("log")
    
    # Refined color palette for publication with family-based shading
    # Senior UX: Same color family, shading by model size (Dark = More Params)
    color_map = {
        # LightGTS Family (Reds)
        "LightGTS-mini": "#C0392B",  # Darker Red (4M)
        "LightGTS-tiny": "#E74C3C",  # Lighter Red (1.3M)
        
        # Chronos Family (Blues)
        "Chronos-Large": "#1B4F72",  # Darkest Blue (710M)
        "Chronos-Small": "#2E86C1",  # Medium Blue (46M)
        "Chronos-Tiny": "#AED6F1",   # Lightest Blue (8M)
        
        # CM-Mamba Family (Purples)
        "CM-Mamba-Small": "#6C3483", # Darker Purple (4.75M)
        "CM-Mamba-Tiny": "#BB8FCE",  # Lighter Purple (1.2M)
        
        # Standalone Models (Individual Colors)
        "Timer": "#D4AC0D",         # Gold
        "MOIRAI-L": "#1E8449",      # Green
        "Time-MoE": "#138D75",      # Teal
    }
    
    colors = [color_map.get(m, "black") for m in df["Model"]]
    
    # Bubble size: scaled for better visibility in paper
    # We use a base size + logarithmic scaling to prevent massive bubbles from 700M models 
    # from obscuring the entire plot, while still showing relative scale.
    bubble_sizes = 200 + (df["Parameters_M"]**0.6) * 40
    
    scatter = plt.scatter(df["Parameters_M"], df["MSE"], s=bubble_sizes, 
                         alpha=0.85, c=colors, edgecolors="#2C3E50", linewidth=1.8, zorder=3)
    
    # Manual label positioning to avoid overlaps and increase clarity
    # (x_offset, y_offset, alignment)
    label_cfg = {
        "CM-Mamba-Small": (10, -15, 'left'),
        "CM-Mamba-Tiny": (10, -15, 'left'),
        "LightGTS-mini": (12, 0, 'left'),
        "LightGTS-tiny": (-10, 15, 'right'),
        "Timer": (0, 15, 'center'),
        "MOIRAI-L": (-15, 15, 'right'),
        "Chronos-Large": (15, 0, 'left'),
        "Chronos-Small": (12, 10, 'left'),
        "Chronos-Tiny": (12, 10, 'left'),
        "Time-MoE": (0, -25, 'center'),
    }

    for i, txt in enumerate(df["Model"]):
        cfg = label_cfg.get(txt, (10, 10, 'left'))
        
        # Professional multi-line annotation
        # Label: Model Name (Bold)
        # Sub-label: MSE and Params (Normal weight)
        label = f"$\\bf{{{txt.replace('-', ' ')}}}$\nMSE: {df['MSE'][i]:.3f}\n{df['Parameters_M'][i]:.1f}M"
        
        plt.annotate(label, 
                     (df["Parameters_M"][i], df["MSE"][i]), 
                     xytext=(cfg[0], cfg[1]), 
                     textcoords='offset points',
                     fontsize=10, 
                     color=colors[i],
                     ha=cfg[2],
                     va='center' if cfg[1] == 0 else ('bottom' if cfg[1] > 0 else 'top'),
                     # Add a subtle semi-transparent halo for legibility on overlap
                     bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', pad=0.1))

    plt.xlabel("Number of parameters (millions)", fontsize=14, fontweight='bold', labelpad=12)
    plt.ylabel("Average MSE (Lower is Better)", fontsize=14, fontweight='bold', labelpad=12)
    plt.title("Efficiency vs. Accuracy Trade-off (Zero-shot Forecasting)", fontsize=17, fontweight='bold', pad=20)
    
    # Clean up grid and spines
    plt.grid(False)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('#34495E')
        spine.set_linewidth(1.5)

    # Expanded boundaries to separate models vertically
    plt.xlim(0.1, 8000)
    plt.ylim(0.18, 0.58) # Zoomed in slightly to increase vertical separation
    
    # Professional Tick Formatting
    plt.xticks([0.1, 1, 10, 100, 1000], ["0.1M", "1M", "10M", "100M", "1000M"], fontsize=12)
    plt.yticks(np.arange(0.2, 0.6, 0.05), [f"{x:.2f}" for x in np.arange(0.2, 0.6, 0.05)], fontsize=12)
    
    # Remove top/right spines for modern look
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    # Save as PNG with transparent background
    output_path_png = ROOT_DIR / "analysis" / "exp7_efficiency_plot.png"
    plt.savefig(output_path_png, dpi=400, bbox_inches='tight', transparent=True)
    
    # Save as EPS with resolution-independent high quality
    output_path_eps = ROOT_DIR / "analysis" / "exp7_efficiency_plot.eps"
    plt.savefig(output_path_eps, format='eps', bbox_inches='tight', transparent=True)
    
    print(f"Plots saved to {output_path_png} and {output_path_eps}")

    # Save metrics for reference
    df.to_csv(ROOT_DIR / "results" / "exp7_efficiency_metrics.csv", index=False)

if __name__ == "__main__":
    main()
