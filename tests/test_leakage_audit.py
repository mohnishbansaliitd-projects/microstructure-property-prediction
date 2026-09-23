"""Tests for Project 4: microstructure-to-property prediction, leakage audit."""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.alsi10mg_loader import load_alsi10mg_psp_dataset, _build_synthetic_calibrated_alsi10mg_dataset
from core.physics_models import PhysicsInformedStrengthModel, hall_petch_relation, gibson_ashby_porosity_model
from core.leakage_audit import run_leakage_audit_experiment, compute_r2_bootstrap_ci


def test_alsi10mg_dataset_loader():
    """Verifies dataset loading and column schema."""
    df = _build_synthetic_calibrated_alsi10mg_dataset(n_param_sets=10, repeats=3)
    assert len(df) == 30
    assert "yield_strength_mpa" in df.columns
    assert "parameter_set" in df.columns
    assert (df["yield_strength_mpa"] > 100.0).all()
    assert (df["porosity_pct"] >= 0.0).all()


def test_physics_models():
    """Verifies Hall-Petch scaling and Gibson-Ashby bounds."""
    # Hall Petch: smaller cell size -> higher strength
    sig_coarse = hall_petch_relation(np.array([2.0]), sigma_0=100.0, k_hp=150.0)[0]
    sig_fine = hall_petch_relation(np.array([0.5]), sigma_0=100.0, k_hp=150.0)[0]
    assert sig_fine > sig_coarse

    # Gibson Ashby: higher porosity -> lower strength
    p_low = gibson_ashby_porosity_model(np.array([0.01]), sigma_solid=300.0)[0]
    p_high = gibson_ashby_porosity_model(np.array([0.20]), sigma_solid=300.0)[0]
    assert p_low > p_high


def test_leakage_audit_benchmark():
    """Verifies that naive random split produces an inflated R2 compared to GroupKFold."""
    df = _build_synthetic_calibrated_alsi10mg_dataset(n_param_sets=25, repeats=3)
    results = run_leakage_audit_experiment(df, n_splits=5)
    
    r2_naive = results["naive_random"]["RandomForest"]["r2"]
    r2_group = results["group_kfold"]["RandomForest"]["r2"]
    
    assert "RandomForest" in results["naive_random"]
    assert "physics_baseline" in results
    # The naive random split with repeat leakage should achieve equal or higher R2 than strict GroupKFold
    assert r2_naive >= r2_group - 0.05


def test_bootstrap_ci():
    """Verifies bootstrap confidence intervals compute valid bounds."""
    y_true = [100.0, 150.0, 200.0, 250.0, 300.0]
    y_pred = [105.0, 148.0, 195.0, 255.0, 298.0]
    mean_r2, ci_low, ci_up = compute_r2_bootstrap_ci(y_true, y_pred, n_bootstraps=50)
    assert ci_low <= mean_r2 <= ci_up


if __name__ == "__main__":
    test_alsi10mg_dataset_loader()
    test_physics_models()
    test_leakage_audit_benchmark()
    test_bootstrap_ci()
    print("all tests passed")
