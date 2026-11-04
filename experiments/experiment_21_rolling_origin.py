"""Experiment 21: rolling-origin evaluation.

CreditCard, fractions of the time-ordered data (train / validation / test):
  W1  0-45% / 45-55% / 55-70%
  W2  0-55% / 55-65% / 65-80%
  W3  0-60% / 60-80% / 80-100%   (main split of the article)
BAF, months (train / validation / test, scored on Base and Variants I-V):
  W1  0-2 / 3 / 4,  W2  0-3 / 4 / 5,  W3  0-4 / 5 / 6,  W4  0-5 / 6 / 7
Configurations M0, noise and consensus; 20 (CreditCard) or 5 (BAF) replicates.
"""
import os
import argparse
import numpy as np
import common as rc

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', choices=['cc', 'baf'], required=True)
ap.add_argument('--data', default=os.path.join('data', 'creditcard.csv'))
ap.add_argument('--data_dir', default='data')
ap.add_argument('--out', default=None)
ap.add_argument('--replicates', type=int, default=None)
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
R = args.replicates or (20 if args.dataset == 'cc' else 5)
if args.quick:
    R = 2
out = args.out or os.path.join('results', f'experiment_21_{args.dataset}')
os.makedirs(os.path.join(out, 'preds'), exist_ok=True)
RUNS = os.path.join(out, 'runs.csv')
KEYCOLS = ['Origin', 'Learner', 'Regime', 'Config', 'Replicate']
CONFIGS = ['M0', 'noise_normal', 'M2_pct']
if args.dataset == 'cc':
    ORIGINS = {'O1': (0.45, 0.55, 0.70), 'O2': (0.55, 0.65, 0.80), 'O3': (0.60, 0.80, 1.00)}
    SETTINGS = [('lgbm', 'orig'), ('lgbm', 'bagged'), ('lgbm', 'bag_hess'), ('xgb', 'orig'), ('xgb', 'bagged')]
else:
    ORIGINS = {'O1': ([0, 1, 2], 3, 4), 'O2': ([0, 1, 2, 3], 4, 5), 'O3': ([0, 1, 2, 3, 4], 5, 6), 'O4': ([0, 1, 2, 3, 4, 5], 6, 7)}
    SETTINGS = [('lgbm', 'orig'), ('lgbm', 'bagged'), ('xgb', 'orig'), ('xgb', 'bagged')]
    store, cols = rc.baf_store(args.data_dir, os.path.join(args.data_dir, 'cache'), fit_months=(0, 1, 2))

rc.save_settings(out, dict(args=vars(args), origins=ORIGINS, settings=SETTINGS, configs=CONFIGS, replicates=R))
done = rc.done_keys(RUNS, KEYCOLS)
for o, spec in ORIGINS.items():
    if args.dataset == 'cc':
        XR, XS, Y, EVAL = rc.load_creditcard(args.data, spec)
    else:
        XR, XS, Y, EVAL = rc.baf_split(store, *spec)
    rc.log(f'origin {o}: ' + ', '.join(f'{k} {len(Y[k]):,}/{int(Y[k].sum())}' for k in Y))
    if args.dataset == 'cc':
        np.save(os.path.join(out, f'labels_{o}.npy'), Y['test'])
    for r in range(1, R + 1):
        todo = [(L, G, c) for (L, G) in SETTINGS for c in CONFIGS
                if not (c == 'M0' and G == 'orig' and r > 1) and (o, L, G, c, str(r)) not in done]
        if not todo:
            continue
        F = rc.replicate_features(XS, Y, r, set(f for (_, _, c) in todo for f in rc.CONFIGS[c]))
        for (L, G, c) in todo:
            X = rc.build_X(XR, F, c)
            model, preds, fit_sec = rc.fit_predict(L, rc.learner_params(L, G, r), X, Y, EVAL)
            row = dict(Origin=o, Learner=L, Regime=G, Config=c, Replicate=r, Fit_sec=fit_sec,
                       N_test_pos=int(Y[EVAL[0]].sum()), N_val_pos=int(Y['val'].sum()), N_train_pos=int(Y['train'].sum()))
            row.update(rc.run_metrics(Y, preds, EVAL))
            rc.append_row(row, RUNS)
            if args.dataset == 'cc':
                np.save(os.path.join(out, 'preds', f'{o}__{L}__{G}__{c}__r{r}.npy'), preds['test'])
            rc.log(f'  {o} r={r:2d} {L:4s} {G:8s} {c:12s} AUPRC={row["AUPRC_" + EVAL[0].replace(" ", "_")]:.4f}')
        del F
rc.log(f'done -> {RUNS}')
