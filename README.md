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

Source: `outputs/run_log.txt` (output of `python main.py` on the Zenodo dataset: 177 specimens, 60 parameter sets, 5 folds). Values are R2 / RMSE in MPa for yield strength.

| Model | Random 5-fold split | GroupKFold by parameter set |
|---|---|---|
| Random Forest | 0.324 / 22.4 | 0.414 / 20.8 |
| Gradient Boosting | 0.356 / 21.8 | 0.485 / 19.5 |
| Ridge regression | 0.373 / 21.5 | 0.317 / 22.5 |
| Physics baseline | 0.391 / 21.2 | |

Bootstrap 95% CI for Random Forest R2: random split 0.318 [0.121, 0.487], GroupKFold 0.409 [0.242, 0.542].

On this dataset the grouped split does not give lower scores than the random split for the tree models; the confidence intervals overlap, so the difference between split strategies is not significant at this sample size. The physics baseline is competitive with the ML models. `outputs/figures/data_leakage_parity_comparison.png` shows the Random Forest parity plots for both splits, and `outputs/figures/alsi10mg_ved_vs_porosity_and_strength.png` plots yield strength and porosity against volumetric energy density.

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

Dependencies are listed in `requirements.txt`.

```
pip install -r requirements.txt
python -m pytest tests
python main.py
```

Download the Zenodo files `AlSi10Mg PSP feature table.xlsx` and `[FR] Mechanical property table.xlsx` into `../data_check/alsi10mg/` first. If the files are missing or fail to parse, the loader silently falls back to a synthetic dataset generated in the code, and `main.py` does not print which source was used. The saved figures show 177 points, which matches the real data.

## Limitations and next steps

- Only 60 parameter sets; per-set scores are sensitive to which sets land in each fold.
- A single random seed and one grouped split; no repeated cross-validation.
- The tests use the synthetic generator, not the real data, and assert only that the random-split R2 is not much lower than the grouped R2.
- Next steps: save all model scores to a file, print the data source in use, add repeated grouped CV.

Mohnish Bansal, B.Tech Materials Engineering, IIT Delhi
