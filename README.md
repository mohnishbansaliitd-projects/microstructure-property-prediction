# Microstructure-Property Prediction with a Leakage Audit

A study of yield-strength prediction for AlSi10Mg parts made by laser powder-bed fusion (LPBF). It compares a naive random cross-validation split with a grouped split (GroupKFold by parameter set) for three regressors, and adds a physics baseline (Hall-Petch cell strengthening combined with a Gibson-Ashby porosity term).

## Motivation

The dataset contains up to three tested specimens per parameter set, and these repeats share processing conditions, cell size and porosity. A random split can place repeats of the same set in both training and test folds. This project reports scores under both splitting schemes so that the effect of the split on the reported R2 is visible.

## Method

- Load the processing, microstructure and porosity features and the per-specimen tensile results, and merge them (`data/alsi10mg_loader.py`).
- Features: volumetric energy density, cell size, porosity, laser power, scan speed. Target: yield strength.
- Models: Random Forest, Gradient Boosting, Ridge (`core/leakage_audit.py`), each with 5-fold random `KFold` and 5-fold `GroupKFold` grouped by `parameter_set`.
- Physics baseline: `sigma_y = (sigma_0 + k_hp / sqrt(d)) * (1 - P)^n`, fitted by curve fit inside each GroupKFold fold (`core/physics_models.py`).
- Bootstrap 95% confidence interval on R2 for the Random Forest predictions.

## Dataset

AlSi10Mg LPBF processing-structure-property data from Zenodo: https://zenodo.org/records/10008435. Per `data/alsi10mg_loader.py`, 60 parameter sets joined with per-specimen mechanical tests give 177 tested specimens. The files are expected in `../data_check/alsi10mg/` and are not included here. Cell size is missing for some parameter sets and is filled with the median of the measured ones.

## Results

Source: the saved figure `outputs/figures/data_leakage_parity_comparison.png` (titles of the two panels, Random Forest). The repository has no saved log or table, so no other numbers are quoted.

| Split | R2 | RMSE (MPa) |
|---|---|---|
| Naive random split | 0.324 | 22.3 |
| GroupKFold by parameter set | 0.415 | 20.8 |

In this saved run the grouped split gives a higher R2 than the random split. The figure shows only Random Forest; Gradient Boosting, Ridge, the physics baseline and the bootstrap intervals are printed by `main.py` but not saved. `outputs/figures/alsi10mg_ved_vs_porosity_and_strength.png` plots yield strength and porosity against volumetric energy density.

## Repository structure

```
main.py                 full pipeline
data/alsi10mg_loader.py data loading and merging
core/                   leakage audit, physics models
visualizer/             figures
tests/                  unit tests
outputs/figures/        saved figures
```

## How to run

No requirements file is included. From the imports: numpy, pandas, scipy, scikit-learn, matplotlib, openpyxl (for `read_excel`) and pytest.

```
pip install numpy pandas scipy scikit-learn matplotlib openpyxl pytest
python -m pytest tests
python main.py
```

Download the Zenodo files `AlSi10Mg PSP feature table.xlsx` and `[FR] Mechanical property table.xlsx` into `../data_check/alsi10mg/` first. If the files are missing or fail to parse, the loader silently falls back to a synthetic dataset generated in the code, and `main.py` does not print which source was used. The saved figures show 177 points, which matches the real data.

## Limitations and next steps

- Only 60 parameter sets; per-set scores are sensitive to which sets land in each fold.
- A single random seed and one grouped split; no repeated cross-validation.
- The tests use the synthetic generator, not the real data, and assert only that the random-split R2 is not much lower than the grouped R2.
- Next steps: save all model scores to a file, print the data source in use, add repeated grouped CV, and add a requirements file.

Mohnish Bansal, B.Tech Materials Engineering, IIT Delhi
