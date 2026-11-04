# fraud-anomaly-augmentation

Code, run-level results and figures for the article

> C. Aslanyürek and M. Aslanyürek, *Reassessing Unsupervised Anomaly Augmentation for Fraud Detection: The Roles of Distribution Shift, Learner Instability, and Run-to-Run Variability*, Applied Sciences (submitted).

The repository contains everything needed to inspect the reported numbers, redraw the figures and rerun the experiments. The datasets are public but not redistributed here; see `data/README.md`.

## Contents

```
data/                 place the datasets here (not included)
experiments/          experiments 16-22 and the shared module common.py
experiments/legacy/   earlier scripts whose results are reported in the article
results/              run-level results of every experiment (CSV and settings.json)
figures/              Figures 1-9 (PNG, 300 dpi, and editable SVG)
scripts/              figure scripts
run_all.py            runs experiments 16-22 in order
```

## Experiments and run order

The experiments build on each other and are numbered in the order in which they should be run. Experiment 22 reads the output of 18-21, and the figure scripts read the output of all of them.

| Step | Script | Dataset | What it does | Output folder |
|---|---|---|---|---|
| 1 | `experiment_16_variance_controls.py` | CreditCard | anomaly features and controls with LightGBM and XGBoost over 20 replicates; separate runs for the regularized regime and the leaf-value mechanism | `experiment_16`, `experiment_16_reg`, `experiment_16_mech` |
| 2 | `experiment_17_crossfit_psi.py` | CreditCard | in-sample and cross-fitted PSI of the anomaly features | `experiment_17` |
| 3 | `experiment_18_baf_variance_controls.py` | BAF | multi-seed replication with controls, Base and Variants I-V | `experiment_18` |
| 4 | `experiment_19_controls_detectors.py` | both | six further controls, four further detectors (LOF, Extended IF, One-Class SVM, autoencoder); saves test predictions | `experiment_19_cc`, `experiment_19_baf` |
| 5 | `experiment_20_mechanism.py` | CreditCard | leaf values and leaf Hessian sums, reverse experiment with XGBoost, sensitivity analysis, histogram-strategy test | `experiment_20` |
| 6 | `experiment_21_rolling_origin.py` | both | rolling-origin evaluation (3 CreditCard and 4 BAF windows) | `experiment_21_cc`, `experiment_21_baf` |
| 7 | `experiment_22_analysis.py` | — | paired bootstrap intervals, equivalence tests, Holm correction, summary tables | `experiment_22` |
| 8 | `scripts/make_figures.py`, `scripts/make_revision_figures.py` | — | Figures 1-6 and 7-9 | `figures/` |

## Running

```bash
pip install -r requirements.txt
# put the datasets into data/, then from the repository root:
python run_all.py --quick       # short check, two replicates (about 30-60 minutes)
python run_all.py               # full run into results_repro/
python scripts/make_figures.py
python scripts/make_revision_figures.py --results results
```

`run_all.py` writes to `results_repro/` so that the published results are left untouched; `--out results` writes next to them instead, and `--only 19 20` runs selected experiments. Each experiment can also be started on its own, for example

```bash
python experiments/experiment_20_mechanism.py --data data/creditcard.csv --out results_repro/experiment_20
```

All experiments append one row per fitted model to `runs.csv` and skip rows that already exist, so an interrupted run continues where it stopped. With 24 CPU cores a full run took about 10 hours; experiment 19 on BAF is the longest step. Experiments 19 and 21 keep an encoded copy of the BAF files in `data/cache/`, and experiment 19 stores test predictions (about 3 GB in total); these files are not part of the repository.

## Notes on reproducibility

- Results were produced with Python 3.14, LightGBM 4.7.0, XGBoost 3.2.0, scikit-learn 1.8.0, NumPy 2.4.3 and pandas 3.0.1 on Windows 11.
- Experiments 19-22 train LightGBM with `deterministic=True` and a fixed column-wise histogram strategy, so their results repeat exactly on the same software versions.
- Experiments 16-18 used LightGBM's automatic choice of histogram strategy. For the unstable LightGBM configurations on CreditCard, this choice alone can change the result of a run (experiment 20, part D). Stable configurations are not affected.
- The 286 LightGBM runs with `Regime == "reg"` in `results/experiment_16/runs.csv` used a misconfigured early-stopping rule and are not used in the article; the corrected runs are in `results/experiment_16_reg`.

## Earlier scripts

`experiments/legacy/` keeps the four earlier scripts whose results are reported in the article: the distribution-shift diagnostics of the BAF variants (06), the earlier multi-seed analysis (12), the SHAP analysis (14) and the single-run CreditCard comparison (15). They read the data files from the working directory:

```bash
cd data
python ../experiments/legacy/experiment_06_distribution_shift.py
```

The complete set of exploratory scripts is archived with version 1.0.0 on Zenodo.

## Result files

Each `runs.csv` has one row per fitted model. The main columns are the experimental cell (`Learner`, `Pipeline` or `Regime`, `Config`, `Replicate`), test and validation AUPRC, ROC-AUC, recall at fixed false positive rates, the logarithm of the 5% FPR threshold (`Log10_Thr5` or `Log10_ValNeg_q95`) and, where computed, the SHAP rank of the added feature. `settings.json` records the arguments and package versions of each run. The tables in `results/experiment_22` are the ones reported in the article and its Supplementary Materials.

## Citation and license

Please cite the article (see `CITATION.cff`). Code, results and figures are released under the MIT license. The datasets remain under the licenses of their providers.
