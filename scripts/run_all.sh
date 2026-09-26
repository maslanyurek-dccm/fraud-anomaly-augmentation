#!/usr/bin/env bash
# Reproduce experiments 16-18 from the repository root:  bash scripts/run_all.sh
# Outputs go to results_repro/ so that the published results/ are not touched.
set -e
python experiments/experiment_16_variance_controls.py --data data/creditcard.csv --out results_repro/experiment_16
python experiments/experiment_16_variance_controls.py --data data/creditcard.csv --learners lgbm --regimes reg --out results_repro/experiment_16_reg
python experiments/experiment_16_variance_controls.py --data data/creditcard.csv --learners lgbm --pipelines raw --regimes bagged bag_hess bag_mds bag_l2 --configs M0 M0_noise M2_pct --out results_repro/experiment_16_mech
python experiments/experiment_17_crossfit_psi.py --data data/creditcard.csv --out results_repro/experiment_17
python experiments/experiment_18_baf_variance_controls.py --data_dir data --out results_repro/experiment_18
echo "Done. Compare results_repro/ with results/."
