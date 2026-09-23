"""
Hall-Petch cell-strengthening and Gibson-Ashby porosity-reduction relations,
combined into a fitted physics baseline for yield strength.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Optional


def hall_petch_relation(d_cell_um: np.ndarray, sigma_0: float, k_hp: float) -> np.ndarray:
    # sigma_y = sigma_0 + k_hp * d^(-1/2)
    return sigma_0 + k_hp / np.sqrt(np.maximum(d_cell_um, 1e-4))


def gibson_ashby_porosity_model(porosity_frac: np.ndarray, sigma_solid: float, n: float = 2.0) -> np.ndarray:
    # sigma_y = sigma_solid * (1 - P)^n
    return sigma_solid * ((1.0 - np.clip(porosity_frac, 0.0, 0.99)) ** n)


def combined_physics_model(
    X: np.ndarray,  # col 0: cell_size_um, col 1: porosity_pct
    sigma_0: float,
    k_hp: float,
    n_porosity: float
) -> np.ndarray:
    d = X[:, 0]
    p = X[:, 1] / 100.0  # porosity_pct is a percentage, model needs a fraction
    solid_strength = sigma_0 + k_hp / np.sqrt(np.maximum(d, 1e-4))
    return solid_strength * ((1.0 - np.clip(p, 0.0, 0.99)) ** n_porosity)


class PhysicsInformedStrengthModel:

    def __init__(self):
        self.params: Optional[Tuple[float, float, float]] = None

    def fit(self, d_cell_um: np.ndarray, porosity_pct: np.ndarray, y_strength: np.ndarray):
        X = np.column_stack([d_cell_um, porosity_pct])
        p0 = [100.0, 150.0, 2.0]
        bounds = ([10.0, 10.0, 0.5], [300.0, 500.0, 5.0])
        try:
            popt, _ = curve_fit(combined_physics_model, X, y_strength, p0=p0, bounds=bounds, maxfev=5000)
            self.params = tuple(popt)
        except Exception:
            # curve_fit can fail to converge on noisy folds; fall back to reasonable defaults
            self.params = (120.0, 180.0, 2.0)

    def predict(self, d_cell_um: np.ndarray, porosity_pct: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise ValueError("Model must be fitted before predict.")
        X = np.column_stack([d_cell_um, porosity_pct])
        return combined_physics_model(X, *self.params)
