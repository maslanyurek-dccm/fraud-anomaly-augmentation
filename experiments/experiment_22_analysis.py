"""Experiment 22: analysis of experiments 19-21 and Holm correction of experiment 18.

  - paired two-level bootstrap of AUPRC differences (replicates and test rows
    resampled, rows stratified by class), 95% intervals, Holm-adjusted p-values
    within each family, and equivalence tests (90% interval inside the margin)
  - summary tables for experiments 19, 20 and 21
  - Holm-adjusted Wilcoxon tests of experiment 18, with and without Variant IV

Reads results/experiment_19_*, _20, _21_* and _18; writes results/experiment_22.
"""
import os
import warnings
warnings.filterwarnings("ignore")
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import common as rc

ap = argparse.ArgumentParser()
ap.add_argument('--results', default='results')
ap.add_argument('--exp18', default=os.path.join('results', 'experiment_18', 'runs.csv'))
ap.add_argument('--boot_cc', type=int, default=2000)
ap.add_argument('--boot_baf', type=int, default=1000)
ap.add_argument('--delta_cc', type=float, default=0.01)
ap.add_argument('--delta_baf', type=float, default=0.005)
ap.add_argument('--skip_boot', action='store_true')
args = ap.parse_args()
OUT = os.path.join(args.results, 'experiment_22')
os.makedirs(OUT, exist_ok=True)
ANOMALY = ['M1_pct', 'M2_pct', 'M3', 'maha_pct', 'lof_pct', 'eif_pct', 'ocsvm_pct', 'ae_pct']
CONTROLS = ['noise_normal', 'noise_uniform', 'noise_lognormal', 'noise_t3', 'noise_corr', 'noise5', 'cons_perm', 'cons_distmatch']


def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m); tmp[idx] = adj
    out[ok] = tmp
    return out


def q(x, a):
    return float(np.quantile(x, a))


# 1. bootstrap
def bootstrap_dataset(ds, B, delta):
    d = os.path.join(args.results, f'experiment_19_{ds}')
    runs_path = os.path.join(d, 'runs.csv')
    if not os.path.exists(runs_path):
        rc.log(f'skip bootstrap {ds}: {runs_path} not found')
        return
    if not os.path.isdir(os.path.join(d, 'preds')):
        rc.log(f'skip bootstrap {ds}: no saved predictions in {d}')
        return
    runs = pd.read_csv(runs_path)
    evals = ['test'] if ds == 'cc' else list(rc.BAF_FILES)
    labels = {e: np.load(os.path.join(d, f'labels_{e.replace(" ", "_")}.npy')) for e in evals}
    metrics = {'test': ['test']} if ds == 'cc' else {
        'Base': ['Base'], 'MeanVariant_I_V': rc.VARIANTS,
        'MeanVariant_I_II_III_V': ['Variant I', 'Variant II', 'Variant III', 'Variant V']}
    rows = []
    rng = np.random.default_rng(2026)
    for (L, G), g in runs.groupby(['Learner', 'Regime']):
        cfgs = sorted(g.Config.unique())
        reps = sorted(g.Replicate.unique())
        runs_of = {c: sorted(g[g.Config == c].Replicate.unique()) for c in cfgs}
        rc.log(f'bootstrap {ds} {L}/{G}: {len(cfgs)} configs, {len(reps)} replicates, B={B}')
        idx = {}
        for c in cfgs:
            for r in runs_of[c]:
                for e in evals:
                    f = os.path.join(d, 'preds', f'{L}__{G}__{c}__r{r}__{e.replace(" ", "_")}.npy')
                    idx[(c, r, e)] = rc.APIndex(np.load(f), labels[e])

        def rep_of(c, r):   # deterministic configs have a single run (replicate 1)
            return r if r in runs_of[c] else runs_of[c][0]

        def metric_values(w_by_eval, rep_list):
            apc = {}   # each (config, run, eval set) AP computed once per bootstrap draw
            for c in cfgs:
                for rr in set(rep_of(c, r) for r in rep_list):
                    for e in evals:
                        apc[(c, rr, e)] = idx[(c, rr, e)].ap(w_by_eval[e])
            out = {}
            for c in cfgs:
                for mname, sets in metrics.items():
                    out[(c, mname)] = np.array([np.mean([apc[(c, rep_of(c, r), e)] for e in sets]) for r in rep_list])
            return out
        ones = {e: np.ones(len(labels[e])) for e in evals}
        point = metric_values(ones, reps)
        comps = [(a, 'M0', 'anomaly_vs_M0') for a in ANOMALY] + [(a, 'noise_normal', 'anomaly_vs_noise') for a in ANOMALY] + \
                [(k, 'M0', 'control_vs_M0') for k in CONTROLS]
        comps = [x for x in comps if x[0] in cfgs and x[1] in cfgs]
        boot = {(a, b, m): np.empty(B) for (a, b, _) in comps for m in metrics}
        for bi in range(B):
            w = {e: rc.stratified_weights(labels[e], rng) for e in evals}
            rep_s = list(rng.choice(reps, len(reps), replace=True))
            vals = metric_values(w, rep_s)
            for (a, b, _) in comps:
                for m in metrics:
                    boot[(a, b, m)][bi] = np.mean(vals[(a, m)] - vals[(b, m)])
            if (bi + 1) % 200 == 0:
                rc.log(f'   {bi + 1}/{B}')
        for (a, b, fam) in comps:
            for m in metrics:
                x = boot[(a, b, m)]
                est = float(np.mean(point[(a, m)] - point[(b, m)]))
                base = float(np.mean(point[(b, m)]))
                p = max(2 * min(np.mean(x <= 0), np.mean(x >= 0)), 1.0 / B)
                lo90, hi90 = q(x, 0.05), q(x, 0.95)
                rows.append(dict(Dataset=ds, Learner=L, Regime=G, Metric=m, Family=fam, Config=a, Reference=b,
                                 Delta=est, Delta_rel_pct=100 * est / base if base else np.nan,
                                 CI95_lo=q(x, 0.025), CI95_hi=q(x, 0.975), CI90_lo=lo90, CI90_hi=hi90,
                                 p_boot=p, Equivalence_delta=delta, Equivalent=bool(-delta < lo90 and hi90 < delta),
                                 Reference_AUPRC=base))
        del idx
    res = pd.DataFrame(rows)
    res['p_holm'] = np.nan
    for _, gi in res.groupby(['Dataset', 'Learner', 'Regime', 'Metric', 'Family']).groups.items():
        res.loc[gi, 'p_holm'] = holm(res.loc[gi, 'p_boot'].values)
    res.to_csv(os.path.join(OUT, f'bootstrap_{ds}.csv'), index=False)
    rc.log(f'-> bootstrap_{ds}.csv ({len(res)} rows)')


if not args.skip_boot:
    bootstrap_dataset('cc', args.boot_cc, args.delta_cc)
    bootstrap_dataset('baf', args.boot_baf, args.delta_baf)

# 2. summaries
def summarise(df, by, col):
    return df.groupby(by)[col].agg(N='size', Median='median', Q1=lambda x: q(x, .25), Q3=lambda x: q(x, .75),
                                    Min='min', Max='max', Collapses=lambda x: int((x < 0.5).sum())).reset_index()


for ds in ('cc', 'baf'):
    p = os.path.join(args.results, f'experiment_19_{ds}', 'runs.csv')
    if os.path.exists(p):
        r = pd.read_csv(p)
        if ds == 'baf':
            r['AUPRC_MeanVariant'] = r[[f'AUPRC_{v.replace(" ", "_")}' for v in rc.VARIANTS]].mean(1)
            r['AUPRC_MeanVariant_noIV'] = r[[f'AUPRC_{v.replace(" ", "_")}' for v in ['Variant I', 'Variant II', 'Variant III', 'Variant V']]].mean(1)
            cols = ['AUPRC_Base', 'AUPRC_MeanVariant', 'AUPRC_MeanVariant_noIV']
        else:
            cols = ['AUPRC_test']
        s = pd.concat([summarise(r, ['Learner', 'Regime', 'Config'], c).assign(Metric=c) for c in cols])
        s = s.merge(r.groupby(['Learner', 'Regime', 'Config']).Added_SHAP_Rank.median().rename('Median_SHAP_rank').reset_index(),
                    on=['Learner', 'Regime', 'Config'], how='left')
        s.to_csv(os.path.join(OUT, f'summary_exp19_{ds}.csv'), index=False)
        sp = os.path.join(args.results, f'experiment_19_{ds}', 'standalone.csv')
        if os.path.exists(sp):
            pd.read_csv(sp).groupby(['Feature', 'EvalSet'])[['Standalone_AUPRC', 'Standalone_ROC']].agg(['median', 'min', 'max']) \
                .to_csv(os.path.join(OUT, f'standalone_exp19_{ds}.csv'))

p = os.path.join(args.results, 'experiment_20', 'runs.csv')
if os.path.exists(p):
    r = pd.read_csv(p)
    s = r.groupby(['Part', 'Learner', 'Regime', 'Variant', 'Config']).agg(
        N=('AUPRC_test', 'size'), Median_AUPRC=('AUPRC_test', 'median'), Q1=('AUPRC_test', lambda x: q(x, .25)),
        Q3=('AUPRC_test', lambda x: q(x, .75)), Min_AUPRC=('AUPRC_test', 'min'), Collapses=('AUPRC_test', lambda x: int((x < .5).sum())),
        Median_max_abs_leaf=('Max_abs_leaf', 'median'), Median_P99_abs_leaf=('P99_abs_leaf', 'median'),
        Median_min_leaf_hess=('Min_leaf_hess', 'median'), Median_frac_hess_lt_1e2=('Frac_leaf_hess_lt_1e2', 'median'),
        Median_Log10_Thr5=('Log10_Thr5', 'median'), Median_min_test_logit=('Min_test_logit', 'median')).reset_index()
    s.to_csv(os.path.join(OUT, 'summary_exp20.csv'), index=False)
    a = r[(r.Part == 'A') & (r.Regime == 'bagged')]
    if len(a) > 3:
        rho, pv = stats.spearmanr(np.log10(a.Max_abs_leaf), a.AUPRC_test)
        pd.DataFrame([dict(Subset='lgbm bagged (Part A)', Spearman_log_max_leaf_vs_AUPRC=rho, p=pv, N=len(a))]) \
            .to_csv(os.path.join(OUT, 'exp20_leaf_vs_auprc.csv'), index=False)
    for f in ('leaves.csv.gz', 'trace.csv.gz'):   # compact versions for plotting
        fp = os.path.join(args.results, 'experiment_20', f)
        if os.path.exists(fp):
            d = pd.read_csv(fp)
            if f == 'leaves.csv.gz':
                d = d[d.tree >= 1]   # tree 0 of LightGBM carries the initial log-odds
                d['log10_abs_leaf'] = np.log10(np.abs(d.leaf_value) + 1e-12)
                d['log10_leaf_hess'] = np.log10(d.leaf_hessian.clip(lower=1e-12))
                h = d.groupby(['Learner', 'Regime', pd.cut(d.log10_leaf_hess, np.arange(-12, 6.5, 0.5)),
                               pd.cut(d.log10_abs_leaf, np.arange(-12, 4.5, 0.5))], observed=True).size().rename('count').reset_index()
                h.astype(str).to_csv(os.path.join(OUT, 'exp20_leaf_hist2d.csv'), index=False)
            else:
                d.groupby(['Learner', 'Regime', 'tree']).agg(median_max_abs_leaf=('max_abs_leaf', 'median'),
                                                             q90_max_abs_leaf=('max_abs_leaf', lambda x: q(x, .9))) \
                    .reset_index().to_csv(os.path.join(OUT, 'exp20_trace.csv'), index=False)

for ds in ('cc', 'baf'):
    p = os.path.join(args.results, f'experiment_21_{ds}', 'runs.csv')
    if os.path.exists(p):
        r = pd.read_csv(p)
        if ds == 'baf':
            r['AUPRC_MeanVariant'] = r[[f'AUPRC_{v.replace(" ", "_")}' for v in rc.VARIANTS]].mean(1)
            col = ['AUPRC_Base', 'AUPRC_MeanVariant']
        else:
            col = ['AUPRC_test']
        s = pd.concat([summarise(r, ['Origin', 'Learner', 'Regime', 'Config'], c).assign(Metric=c) for c in col])
        pos = r.groupby('Origin')[['N_train_pos', 'N_val_pos', 'N_test_pos']].first().reset_index()
        s.merge(pos, on='Origin').to_csv(os.path.join(OUT, f'summary_exp21_{ds}.csv'), index=False)
        # paired differences (same replicate) M2 - M0 and M2 - noise per origin
        rows = []
        for (o, L, G), g in r.groupby(['Origin', 'Learner', 'Regime']):
            for c_ in col:
                w = g.pivot_table(index='Replicate', columns='Config', values=c_)
                for a, b in (('M2_pct', 'M0'), ('M2_pct', 'noise_normal'), ('noise_normal', 'M0')):
                    if a in w and b in w:
                        ref = w[b] if w[b].notna().sum() > 1 else w[b].dropna().iloc[0]
                        dd = (w[a] - ref).dropna()
                        pw = stats.wilcoxon(dd)[1] if len(dd) >= 5 and np.any(dd != 0) else np.nan
                        rows.append(dict(Origin=o, Learner=L, Regime=G, Metric=c_, Config=a, Reference=b, N=len(dd),
                                         Median_diff=dd.median(), Mean_diff=dd.mean(), Wilcoxon_p=pw))
        pr = pd.DataFrame(rows)
        if len(pr):
            pr['p_holm'] = np.nan
            for _, gi in pr.groupby(['Learner', 'Regime', 'Metric']).groups.items():
                pr.loc[gi, 'p_holm'] = holm(pr.loc[gi, 'Wilcoxon_p'].values)
            pr.to_csv(os.path.join(OUT, f'paired_exp21_{ds}.csv'), index=False)

# 3. experiment 18: Holm correction and Variant iii/iv merged sensitivity
if os.path.exists(args.exp18):
    r = pd.read_csv(args.exp18)
    keys = ['Learner', 'Pipeline', 'Regime', 'Config', 'Replicate']
    w = r.pivot_table(index=keys, columns='EvalSet', values='AUPRC').reset_index()
    w['MV_I_V'] = w[rc.VARIANTS].mean(1)
    w['MV_noIV'] = w[['Variant I', 'Variant II', 'Variant III', 'Variant V']].mean(1)
    rows = []
    for (L, P, G), g in w.groupby(['Learner', 'Pipeline', 'Regime']):
        m0 = g[g.Config == 'M0'].set_index('Replicate')
        for c in g.Config.unique():
            if c == 'M0':
                continue
            gc_ = g[g.Config == c].set_index('Replicate')
            for met in ('Base', 'MV_I_V', 'MV_noIV'):
                if len(m0) == 1:
                    dd = gc_[met] - float(m0[met].iloc[0])
                else:
                    ii = gc_.index.intersection(m0.index)
                    dd = gc_.loc[ii, met] - m0.loc[ii, met]
                pw = stats.wilcoxon(dd)[1] if len(dd) >= 5 and np.any(dd != 0) else np.nan
                rows.append(dict(Learner=L, Pipeline=P, Regime=G, Config=c, Metric=met, N=len(dd), Median_diff=dd.median(),
                                 Mean_diff=dd.mean(), CI95_lo_t=dd.mean() - stats.t.ppf(.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else np.nan,
                                 CI95_hi_t=dd.mean() + stats.t.ppf(.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else np.nan,
                                 Frac_positive=(dd > 0).mean(), Wilcoxon_p=pw))
    pr = pd.DataFrame(rows)
    pr['p_holm'] = np.nan
    for _, gi in pr.groupby(['Learner', 'Pipeline', 'Regime', 'Metric']).groups.items():
        pr.loc[gi, 'p_holm'] = holm(pr.loc[gi, 'Wilcoxon_p'].values)
    pr.to_csv(os.path.join(OUT, 'exp18_paired_holm_merged.csv'), index=False)
    rc.log('-> exp18_paired_holm_merged.csv')
rc.log(f'analysis finished -> {OUT}')
