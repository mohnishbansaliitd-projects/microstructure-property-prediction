"""
Runs the full Project 4 pipeline: loads the AlSi10Mg LPBF dataset, benchmarks
naive random split vs GroupKFold to quantify leakage, fits the physics baseline,
and generates the comparison figures.
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from data.alsi10mg_loader import load_alsi10mg_psp_dataset
from core.physics_models import PhysicsInformedStrengthModel
from core.leakage_audit import run_leakage_audit_experiment, compute_r2_bootstrap_ci
from visualizer.prediction_plots import plot_leakage_parity_comparison, plot_processing_microstructure_trends


def run_full_pipeline():
    print("Project 4: microstructure-to-property prediction, leakage audit")

    df = load_alsi10mg_psp_dataset()
    n_sets = df["parameter_set"].nunique()
    print(f"\nLoaded AlSi10Mg dataset: {len(df)} specimens across {n_sets} parameter sets")
    print(f"  mean yield strength: {df['yield_strength_mpa'].mean():.1f} +/- {df['yield_strength_mpa'].std():.1f} MPa")
    print(f"  mean cell size: {df['cell_size_um'].mean():.2f} um | mean porosity: {df['porosity_pct'].mean():.2f}%")

    results = run_leakage_audit_experiment(df, n_splits=5)

    print("\nModel comparison (naive random split vs GroupKFold):")
    print(f"{'Model':<20} | {'Naive R2 (leakage)':<20} | {'GroupKFold R2 (honest)':<22} | {'Physics R2':<12}")
    print("-" * 80)

    for m in ["RandomForest", "GradientBoosting", "RidgeRegression"]:
        r2_naive = results["naive_random"][m]["r2"]
        r2_group = results["group_kfold"][m]["r2"]
        print(f"{m:<20} | {r2_naive:<20.3f} | {r2_group:<22.3f} | {results['physics_baseline']['r2']:<12.3f}")

    naive_rf = results["naive_random"]["RandomForest"]
    group_rf = results["group_kfold"]["RandomForest"]

    mean_naive, ci_low_naive, ci_up_naive = compute_r2_bootstrap_ci(naive_rf["y_trues"], naive_rf["y_preds"], n_bootstraps=1000)
    mean_group, ci_low_group, ci_up_group = compute_r2_bootstrap_ci(group_rf["y_trues"], group_rf["y_preds"], n_bootstraps=1000)

    print("\nBootstrap 95% CI for Random Forest R2:")
    print(f"  naive random split (with leakage): {mean_naive:.3f}  [95% CI: {ci_low_naive:.3f}, {ci_up_naive:.3f}]")
    print(f"  GroupKFold split (leakage-free):    {mean_group:.3f}  [95% CI: {ci_low_group:.3f}, {ci_up_group:.3f}]")
    print(f"  inflation: +{(mean_naive - mean_group):.3f} R2 points ({(mean_naive/mean_group - 1)*100:.1f}% artificial boost)")

    figures_dir = os.path.join(PROJECT_DIR, "outputs", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_leakage_parity_comparison(
        results,
        output_path=os.path.join(figures_dir, "data_leakage_parity_comparison.png")
    )

    plot_processing_microstructure_trends(
        df,
        output_path=os.path.join(figures_dir, "alsi10mg_ved_vs_porosity_and_strength.png")
    )
    print(f"\nFigures saved to {figures_dir}")


if __name__ == "__main__":
    run_full_pipeline()
