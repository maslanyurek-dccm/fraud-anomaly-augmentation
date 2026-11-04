"""Run experiments 16-22 in the order used for the article.

Usage, from the repository root with the datasets in data/:

    python run_all.py                  # all experiments, results in results_repro/
    python run_all.py --quick          # short check with two replicates
    python run_all.py --only 19 20     # selected experiments only
    python run_all.py --out results    # write next to the published results

Every experiment skips runs that are already in its runs.csv, so an
interrupted run can be resumed with the same command.
"""
import argparse
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='results_repro')
ap.add_argument('--data', default='data')
ap.add_argument('--quick', action='store_true')
ap.add_argument('--only', nargs='+', default=None)
args = ap.parse_args()

cc = os.path.join(args.data, 'creditcard.csv')
o = lambda name: os.path.join(args.out, name)
Q = ['--quick'] if args.quick else []

STEPS = [
    ('16', 'experiment_16_variance_controls.py', ['--data', cc, '--regimes', 'orig', 'bagged', '--out', o('experiment_16')] + Q),
    ('16', 'experiment_16_variance_controls.py', ['--data', cc, '--learners', 'lgbm', '--regimes', 'reg', '--out', o('experiment_16_reg')] + Q),
    ('16', 'experiment_16_variance_controls.py', ['--data', cc, '--learners', 'lgbm', '--pipelines', 'raw',
                                                  '--regimes', 'bagged', 'bag_hess', 'bag_mds', 'bag_l2',
                                                  '--configs', 'M0', 'M0_noise', 'M2_pct', '--out', o('experiment_16_mech')] + Q),
    ('17', 'experiment_17_crossfit_psi.py', ['--data', cc, '--out', o('experiment_17')] + (['--replicates', '2'] if args.quick else [])),
    ('18', 'experiment_18_baf_variance_controls.py', ['--data_dir', args.data, '--out', o('experiment_18')] + Q),
    ('19', 'experiment_19_controls_detectors.py', ['--dataset', 'cc', '--data', cc, '--out', o('experiment_19_cc')] + Q),
    ('19', 'experiment_19_controls_detectors.py', ['--dataset', 'baf', '--data_dir', args.data, '--out', o('experiment_19_baf')] + Q),
    ('20', 'experiment_20_mechanism.py', ['--data', cc, '--out', o('experiment_20')] + Q),
    ('21', 'experiment_21_rolling_origin.py', ['--dataset', 'cc', '--data', cc, '--out', o('experiment_21_cc')] + Q),
    ('21', 'experiment_21_rolling_origin.py', ['--dataset', 'baf', '--data_dir', args.data, '--out', o('experiment_21_baf')] + Q),
    ('22', 'experiment_22_analysis.py', ['--results', args.out, '--exp18', os.path.join(args.out, 'experiment_18', 'runs.csv')]
            + (['--boot_cc', '100', '--boot_baf', '50'] if args.quick else [])),
]

here = os.path.dirname(os.path.abspath(__file__))
for num, script, extra in STEPS:
    if args.only and num not in args.only:
        continue
    cmd = [sys.executable, os.path.join(here, 'experiments', script)] + extra
    print('\n>>', ' '.join(cmd), flush=True)
    if subprocess.call(cmd) != 0:
        sys.exit(f'experiment {num} stopped with an error; run the same command again to resume')
print('\nAll steps finished. Figures: python scripts/make_revision_figures.py --results', args.out)
