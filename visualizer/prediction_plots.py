"""
Figures for the leakage audit: parity plots comparing naive random split vs
GroupKFold, and VED vs porosity/yield-strength processing curves.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any


def plot_leakage_parity_comparison(
    leakage_results: Dict[str, Any],
    output_path: str = "outputs/figures/data_leakage_parity_comparison.png"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    naive_rf = leakage_results["naive_random"]["RandomForest"]
    y_t_naive = np.array(naive_rf["y_trues"])
    y_p_naive = np.array(naive_rf["y_preds"])
    r2_naive = naive_rf["r2"]
    rmse_naive = naive_rf["rmse"]
    
    axes[0].scatter(y_t_naive, y_p_naive, c="#ef4444", alpha=0.7, edgecolors="k", s=45, label="Random K-Fold Samples")
    min_val = min(y_t_naive.min(), y_p_naive.min()) - 10
    max_val = max(y_t_naive.max(), y_p_naive.max()) + 10
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label="Ideal 1:1 Parity")
    
    axes[0].set_xlabel("Measured Yield Strength [MPa]", fontsize=11)
    axes[0].set_ylabel("Predicted Yield Strength [MPa]", fontsize=11)
    axes[0].set_title(f"(a) Naive Random Split (DATA LEAKAGE)\n$R^2 = {r2_naive:.3f}$, $\\mathrm{{RMSE}} = {rmse_naive:.1f}$ MPa", fontsize=11, fontweight="bold", color="#b91c1c")
    axes[0].set_xlim(min_val, max_val)
    axes[0].set_ylim(min_val, max_val)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    group_rf = leakage_results["group_kfold"]["RandomForest"]
    y_t_group = np.array(group_rf["y_trues"])
    y_p_group = np.array(group_rf["y_preds"])
    r2_group = group_rf["r2"]
    rmse_group = group_rf["rmse"]
    
    axes[1].scatter(y_t_group, y_p_group, c="#0284c7", alpha=0.7, edgecolors="k", s=45, label="GroupKFold (Unseen Sets)")
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label="Ideal 1:1 Parity")
    
    axes[1].set_xlabel("Measured Yield Strength [MPa]", fontsize=11)
    axes[1].set_ylabel("Predicted Yield Strength [MPa]", fontsize=11)
    axes[1].set_title(f"(b) GroupKFold (HONEST GENERALIZATION)\n$R^2 = {r2_group:.3f}$, $\\mathrm{{RMSE}} = {rmse_group:.1f}$ MPa", fontsize=11, fontweight="bold", color="#0369a1")
    axes[1].set_xlim(min_val, max_val)
    axes[1].set_ylim(min_val, max_val)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_processing_microstructure_trends(
    df: pd.DataFrame,
    output_path: str = "outputs/figures/alsi10mg_ved_vs_porosity_and_strength.png"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    
    ved = df["ved_j_mm3"]
    porosity = df["porosity_pct"]
    ys = df["yield_strength_mpa"]
    
    color1 = "#0284c7"
    ax1.set_xlabel(r"Volumetric Energy Density (VED) [$\mathrm{J/mm^3}$]", fontsize=11)
    ax1.set_ylabel("Yield Strength [MPa]", color=color1, fontsize=11)
    ax1.scatter(ved, ys, color=color1, alpha=0.7, edgecolors="k", label="Yield Strength")
    ax1.tick_params(axis="y", labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = "#ef4444"
    ax2.set_ylabel("Porosity [%]", color=color2, fontsize=11)
    ax2.scatter(ved, porosity, color=color2, marker="s", alpha=0.7, edgecolors="k", label="Porosity")
    ax2.tick_params(axis="y", labelcolor=color2)
    
    plt.title("AlSi10Mg LPBF: Energy Density vs Porosity & Yield Strength", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
