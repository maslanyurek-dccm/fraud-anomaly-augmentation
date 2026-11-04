"""Experiment 19: additional negative controls and anomaly detectors.

  --dataset cc   CreditCard, unscaled pipeline, 20 replicates
  --dataset baf  BAF, ordinal-encoded pipeline, months 0-5 / 6 / 7, 10 replicates

Learner settings: LightGBM original, LightGBM with a minimum Hessian sum of 1
(CreditCard) or with bagging (BAF), XGBoost original and XGBoost with bagging.
Configurations: M0, M1, M2, M3, Mahalanobis; six noise controls, a permuted
and a distribution-matched consensus; LOF, Extended IF, One-Class SVM and an
autoencoder. Test predictions are saved for the bootstrap of experiment 22.
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
ap.add_argument('--configs', nargs='+', default=None)
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()

ALL_CONFIGS = ['M0', 'M1_pct', 'M2_pct', 'M3', 'maha_pct', 'noise_normal', 'noise_uniform', 'noise_lognormal', 'noise_t3',
               'noise_corr', 'noise5', 'cons_perm', 'cons_distmatch', 'lof_pct', 'eif_pct', 'ocsvm_pct', 'ae_pct']
configs = args.configs or ALL_CONFIGS
R = args.replicates or (20 if args.dataset == 'cc' else 10)
if args.quick:
    R = 2
SETTINGS = ([('lgbm', 'orig'), ('lgbm', 'bag_hess'), ('xgb', 'orig'), ('xgb', 'bagged')] if args.dataset == 'cc'
            else [('lgbm', 'orig'), ('lgbm', 'bagged'), ('xgb', 'orig'), ('xgb', 'bagged')])
out = args.out or os.path.join('results', f'experiment_19_{args.dataset}')
os.makedirs(os.path.join(out, 'preds'), exist_ok=True)
RUNS = os.path.join(out, 'runs.csv')
KEYCOLS = ['Learner', 'Regime', 'Config', 'Replicate']

rc.log(f'EXPERIMENT 19 - dataset={args.dataset} replicates={R} configs={len(configs)}')
if args.dataset == 'cc':
    XR, XS, Y, EVAL = rc.load_creditcard(args.data)
else:
    store, cols = rc.baf_store(args.data_dir, os.path.join(args.data_dir, 'cache'))
    XR, XS, Y, EVAL = rc.baf_split(store, [0, 1, 2, 3, 4, 5], 6, 7)
    del store
for k in Y:
    rc.log(f'  {k:12s} n={len(Y[k]):>9,} fraud={int(Y[k].sum()):>6,}')
for k in EVAL:
    np.save(os.path.join(out, f'labels_{k.replace(" ", "_")}.npy'), Y[k])
rc.save_settings(out, dict(args=vars(args), settings=SETTINGS, configs=configs, replicates=R,
                           lgbm_base=rc.LGBM_BASE, xgb_base=rc.XGB_BASE))

# SHAP sample on the first evaluation set: all frauds + 5,000 normals
rng = np.random.default_rng(42)
e0 = EVAL[0]
pos, neg = np.nonzero(Y[e0] == 1)[0], np.nonzero(Y[e0] == 0)[0]
SHAP_IDX = np.concatenate([pos, rng.choice(neg, min(5000, len(neg)), replace=False)])

done = rc.done_keys(RUNS, KEYCOLS)
need_all = set(f for c in configs for f in rc.CONFIGS[c])
cache = {}
for r in range(1, R + 1):
    todo = [(L, G, c) for (L, G) in SETTINGS for c in configs
            if not (c in rc.DETERMINISTIC_CONFIGS and G == 'orig' and r > 1)
            and (L, G, c, str(r)) not in done]
    if not todo:
        continue
    need = set(f for (_, _, c) in todo for f in rc.CONFIGS[c])
    rc.log(f'replicate {r}: computing features {sorted(need)}')
    F = rc.replicate_features(XS, Y, r, need, cache)
    for k in [x for x in ('maha_pct', 'lof_pct', 'if_pct', 'cons_pct', 'eif_pct', 'ocsvm_pct', 'ae_pct') if x in F]:
        for e in EVAL:   # standalone quality of each anomaly feature (once per replicate)
            rc.append_row(dict(Replicate=r, Feature=k, EvalSet=e,
                               Standalone_AUPRC=rc.average_precision_score(Y[e], F[k][e]),
                               Standalone_ROC=rc.roc_auc_score(Y[e], F[k][e])), os.path.join(out, 'standalone.csv'))
    for (L, G, c) in todo:
        X = rc.build_X(XR, F, c)
        p = rc.learner_params(L, G, r)
        model, preds, fit_sec = rc.fit_predict(L, p, X, Y, EVAL)
        row = dict(Learner=L, Regime=G, Config=c, Replicate=r, Fit_sec=fit_sec)
        row.update(rc.run_metrics(Y, preds, EVAL))
        row['Added_SHAP_Rank'] = rc.shap_rank_added(L, model, X[e0][SHAP_IDX], XR[e0].shape[1])
        row['Noise_corr_feature'] = F['noise_corr_feature'] if c == 'noise_corr' else np.nan   # fixed CSV schema
        for e in EVAL:
            np.save(os.path.join(out, 'preds', f'{L}__{G}__{c}__r{r}__{e.replace(" ", "_")}.npy'), preds[e])
        rc.append_row(row, RUNS)
        rc.log(f'  r={r:2d} {L:4s} {G:8s} {c:15s} AUPRC[{e0}]={row["AUPRC_" + e0.replace(" ", "_")]:.4f}')
    del F
rc.log(f'done -> {RUNS}')
