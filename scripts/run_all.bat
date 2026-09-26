@echo off
REM Reproduce experiments 16-18 on Windows. Run from the repository root:  scripts\run_all.bat
REM Outputs go to results_repro\ so that the published results\ are not touched.
python experiments\experiment_16_variance_controls.py --data data\creditcard.csv --out results_repro\experiment_16 || goto :error
python experiments\experiment_16_variance_controls.py --data data\creditcard.csv --learners lgbm --regimes reg --out results_repro\experiment_16_reg || goto :error
python experiments\experiment_16_variance_controls.py --data data\creditcard.csv --learners lgbm --pipelines raw --regimes bagged bag_hess bag_mds bag_l2 --configs M0 M0_noise M2_pct --out results_repro\experiment_16_mech || goto :error
python experiments\experiment_17_crossfit_psi.py --data data\creditcard.csv --out results_repro\experiment_17 || goto :error
python experiments\experiment_18_baf_variance_controls.py --data_dir data --out results_repro\experiment_18 || goto :error
echo Done. Compare results_repro\ with results\.
exit /b 0
:error
echo A step failed. Re-running the script resumes from where it stopped.
exit /b 1
