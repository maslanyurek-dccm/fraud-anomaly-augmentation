"""Experiment 20: leaf-value diagnostics, reverse experiment and sensitivity analysis.

CreditCard, unscaled pipeline, configurations M0 and noise.
  Part A  LightGBM with bagging, alone and with one leaf-value safeguard
          (minimum Hessian sum 1, output cap 1, L2 = 1); all leaf values and
          leaf Hessian sums are stored. 20 seeds.
  Part B  XGBoost moved towards the LightGBM defaults (minimum child weight
          1e-3, lambda 0, both, both with leaf-wise growth). 20 seeds.
  Part C  One factor at a time: number of leaves or depth, learning rate,
          number of trees. 10 seeds.
  Part D  Deterministic LightGBM baseline under the automatic, column-wise and
          row-wise histogram strategies, with all threads and with one thread.
"""
import os
import argparse
import numpy as np
import pandas as pd
import common as rc

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=os.path.join('data', 'creditcard.csv'))
ap.add_argument('--out', default=os.path.join('results', 'experiment_20'))
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
RUNS = os.path.join(args.out, 'runs.csv')
KEYCOLS = ['Part', 'Learner', 'Regime', 'Variant', 'Config', 'Replicate']
R_AB, R_C = (2, 2) if args.quick else (20, 10)

plan = []
for r in range(1, R_AB + 1):
    for G in ['bagged', 'bag_hess', 'bag_mds', 'bag_l2']:
        for c in ['M0', 'noise_normal']:
            plan.append(('A', 'lgbm', G, 'default', c, r, None))
    for G in ['bagged', 'xrev_mcw', 'xrev_l2', 'xrev_both', 'xrev_leafwise']:
        for c in ['M0', 'noise_normal']:
            plan.append(('B', 'xgb', G, 'default', c, r, None))
SENS_L = {'leaves15': {'num_leaves': 15}, 'leaves63': {'num_leaves': 63}, 'lr0.02': {'learning_rate': 0.02},
          'lr0.1': {'learning_rate': 0.1}, 'trees100': {'n_estimators': 100}, 'trees600': {'n_estimators': 600}}
SENS_X = {'depth4': {'max_depth': 4}, 'depth8': {'max_depth': 8}, 'lr0.02': {'learning_rate': 0.02},
          'lr0.1': {'learning_rate': 0.1}, 'trees100': {'n_estimators': 100}, 'trees600': {'n_estimators': 600}}
for r in range(1, R_C + 1):
    for v, o in SENS_L.items():
        for G in ['bagged', 'bag_hess']:
            for c in ['M0', 'noise_normal']:
                plan.append(('C', 'lgbm', G, v, c, r, o))
    for v, o in SENS_X.items():
        for c in ['M0', 'noise_normal']:
            plan.append(('C', 'xgb', 'bagged', v, c, r, o))

# Part D: does the histogram strategy / thread count alone change the deterministic baseline?
#         LightGBM, original (no bagging) regime, M0 and noise_normal, each setting fitted twice.
MODES = {'auto_tall': {'deterministic': False, 'force_col_wise': False},
         'col_tall': {}, 'row_tall': {'force_col_wise': False, 'force_row_wise': True},
         'auto_t1': {'deterministic': False, 'force_col_wise': False, 'n_jobs': 1},
         'col_t1': {'n_jobs': 1}, 'row_t1': {'force_col_wise': False, 'force_row_wise': True, 'n_jobs': 1}}
for rep in (1, 2):
    for v, o in MODES.items():
        for c in ['M0', 'noise_normal']:
            plan.append(('D', 'lgbm', 'orig', v, c, rep, o))
rc.log(f'EXPERIMENT 20 - {len(plan)} fits')
XR, XS, Y, EVAL = rc.load_creditcard(args.data)
rc.save_settings(args.out, dict(args=vars(args), sens_lgbm=SENS_L, sens_xgb=SENS_X, lgbm_base=rc.LGBM_BASE, xgb_base=rc.XGB_BASE))
done = rc.done_keys(RUNS, KEYCOLS)


def leaves_of(L, model):
    """DataFrame tree, leaf_value, leaf_hessian (sum of Hessians), leaf_count."""
    if L == 'lgbm':
        rows = []
        for t, ti in enumerate(model.booster_.dump_model()['tree_info']):
            stack = [ti['tree_structure']]
            while stack:
                nd = stack.pop()
                if 'leaf_value' in nd:
                    rows.append((t, nd['leaf_value'], nd.get('leaf_weight', np.nan), nd.get('leaf_count', np.nan)))
                else:
                    stack += [nd['left_child'], nd['right_child']]
        return pd.DataFrame(rows, columns=['tree', 'leaf_value', 'leaf_hessian', 'leaf_count'])
    df = model.get_booster().trees_to_dataframe()
    df = df[df['Feature'] == 'Leaf']
    return pd.DataFrame({'tree': df['Tree'].values, 'leaf_value': df['Gain'].values,
                         'leaf_hessian': df['Cover'].values, 'leaf_count': np.nan})


cur_r, F = None, None
for (part, L, G, v, c, r, o) in plan:
    if (part, L, G, v, c, str(r)) in done:
        continue
    rf = 1 if part == 'D' else r          # Part D: same noise column in every fit
    if rf != cur_r:
        F = rc.replicate_features(XS, Y, rf, {'noise_normal'})
        cur_r = rf
    X = rc.build_X(XR, F, c)
    p = rc.learner_params(L, G, r, o)
    model, preds, fit_sec = rc.fit_predict(L, p, X, Y, EVAL)
    row = dict(Part=part, Learner=L, Regime=G, Variant=v, Config=c, Replicate=r, Fit_sec=fit_sec)
    row.update(rc.run_metrics(Y, preds, EVAL))
    lv = leaves_of(L, model)
    # LightGBM stores the initial log-odds (boost_from_average) inside the leaves of tree 0,
    # so leaf statistics are computed on trees >= 1 for both learners (XGBoost keeps it in base_score).
    st = lv[lv['tree'] >= 1]
    a = np.abs(st['leaf_value'].values)
    h = st['leaf_hessian'].values
    raw_test = np.log(np.clip(preds['test'], 1e-300, 1)) - np.log(np.clip(1 - preds['test'], 1e-300, 1))
    row.update(N_leaves=len(lv), Max_abs_leaf=float(a.max()), P99_abs_leaf=float(np.quantile(a, 0.99)),
               Median_leaf_hess=float(np.nanmedian(h)), Min_leaf_hess=float(np.nanmin(h)),
               Frac_leaf_hess_lt_1e2=float(np.nanmean(h < 1e-2)), Frac_abs_leaf_gt_5=float(np.mean(a > 5)),
               Min_test_logit=float(raw_test.min()), Median_test_logit=float(np.median(raw_test)))
    rc.append_row(row, RUNS)
    if part in ('A', 'B') and c == 'M0':   # full leaf table and per-tree trace for the M0 models
        lv.insert(0, 'Replicate', r); lv.insert(0, 'Regime', G); lv.insert(0, 'Learner', L)
        lv.to_csv(os.path.join(args.out, 'leaves.csv.gz'), mode='a', index=False,
                  header=not os.path.exists(os.path.join(args.out, 'leaves.csv.gz')), compression='gzip')
        tr = lv[lv['tree'] >= 1].groupby(['Learner', 'Regime', 'Replicate', 'tree']).agg(
            max_abs_leaf=('leaf_value', lambda x: float(np.abs(x).max())),
            min_leaf_hess=('leaf_hessian', 'min')).reset_index()
        tr.to_csv(os.path.join(args.out, 'trace.csv.gz'), mode='a', index=False,
                  header=not os.path.exists(os.path.join(args.out, 'trace.csv.gz')), compression='gzip')
    rc.log(f'{part} r={r:2d} {L:4s} {G:13s} {v:9s} {c:12s} AUPRC={row["AUPRC_test"]:.4f} max|leaf|={row["Max_abs_leaf"]:.2f}')
rc.log(f'done -> {RUNS}')
