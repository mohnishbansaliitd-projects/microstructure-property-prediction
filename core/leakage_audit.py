"""
Compares naive random K-fold against GroupKFold (leave-one-parameter-set-out)
to show how repeat specimens from the same parameter set leak between train
and test when the split ignores grouping.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from core.physics_models import PhysicsInformedStrengthModel


def run_leakage_audit_experiment(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Runs naive random K-fold and GroupKFold benchmarks plus the physics baseline."""
    features = ["ved_j_mm3", "cell_size_um", "porosity_pct", "laser_power_w", "scan_speed_mm_s"]
    X = df[features].to_numpy()
    y = df["yield_strength_mpa"].to_numpy()
    groups = df["parameter_set"].to_numpy()
    d_cells = df["cell_size_um"].to_numpy()
    porosities = df["porosity_pct"].to_numpy()
    
    models = {
        "RandomForest": lambda: RandomForestRegressor(n_estimators=50, random_state=random_seed),
        "GradientBoosting": lambda: GradientBoostingRegressor(n_estimators=50, random_state=random_seed),
        "RidgeRegression": lambda: Ridge(alpha=1.0)
    }
    
    results = {}

    # random split ignores parameter_set, so repeats of the same set can land in both train and test
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    results["naive_random"] = {}
    
    for m_name, m_ctor in models.items():
        y_trues, y_preds = [], []
        for train_idx, test_idx in kf.split(X, y):
            model = m_ctor()
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            y_trues.extend(y[test_idx])
            y_preds.extend(preds)
            
        r2 = r2_score(y_trues, y_preds)
        rmse = np.sqrt(mean_squared_error(y_trues, y_preds))
        mae = mean_absolute_error(y_trues, y_preds)
        results["naive_random"][m_name] = {
            "r2": float(r2), "rmse": float(rmse), "mae": float(mae),
            "y_trues": y_trues, "y_preds": y_preds
        }

    # groups by parameter_set, so each fold holds out entire parameter sets rather than individual repeats
    gkf = GroupKFold(n_splits=n_splits)
    results["group_kfold"] = {}
    
    for m_name, m_ctor in models.items():
        y_trues, y_preds = [], []
        for train_idx, test_idx in gkf.split(X, y, groups):
            model = m_ctor()
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            y_trues.extend(y[test_idx])
            y_preds.extend(preds)
            
        r2 = r2_score(y_trues, y_preds)
        rmse = np.sqrt(mean_squared_error(y_trues, y_preds))
        mae = mean_absolute_error(y_trues, y_preds)
        results["group_kfold"][m_name] = {
            "r2": float(r2), "rmse": float(rmse), "mae": float(mae),
            "y_trues": y_trues, "y_preds": y_preds
        }

    y_trues_phys, y_preds_phys = [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        p_model = PhysicsInformedStrengthModel()
        p_model.fit(d_cells[train_idx], porosities[train_idx], y[train_idx])
        preds = p_model.predict(d_cells[test_idx], porosities[test_idx])
        y_trues_phys.extend(y[test_idx])
        y_preds_phys.extend(preds)
        
    r2_phys = r2_score(y_trues_phys, y_preds_phys)
    rmse_phys = np.sqrt(mean_squared_error(y_trues_phys, y_preds_phys))
    results["physics_baseline"] = {
        "r2": float(r2_phys), "rmse": float(rmse_phys),
        "y_trues": y_trues_phys, "y_preds": y_preds_phys
    }
    
    return results


def compute_r2_bootstrap_ci(
    y_true: List[float],
    y_pred: List[float],
    n_bootstraps: int = 1000,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """Computes 95% Bootstrap Confidence Interval for R2."""
    np.random.seed(random_seed)
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    n = len(y_t)
    r2_scores = []
    
    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        r2_scores.append(r2_score(y_t[idx], y_p[idx]))
        
    r2_arr = np.array(r2_scores)
    return float(np.mean(r2_arr)), float(np.percentile(r2_arr, 2.5)), float(np.percentile(r2_arr, 97.5))
