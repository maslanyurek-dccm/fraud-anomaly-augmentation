"""Figures 7-9 of the article.

Reads results/experiment_20, _21_cc, _21_baf and _22 and writes
figures/Figure_7 to Figure_9 (PNG and SVG).
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--results', default='results')
ap.add_argument('--out', default='figures')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
A = os.path.join(args.results, 'experiment_22')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
                     'legend.fontsize': 7.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
LABEL = {'M1_pct': 'Single IF', 'M2_pct': 'IF consensus', 'M3': 'Consensus + disagr.', 'maha_pct': 'Mahalanobis',
         'lof_pct': 'LOF', 'eif_pct': 'Extended IF', 'ocsvm_pct': 'One-Class SVM', 'ae_pct': 'Autoencoder'}
ORDER = list(LABEL)
SET_LABEL = {('lgbm', 'bag_hess'): 'LightGBM, Hessian-constrained', ('xgb', 'bagged'): 'XGBoost, bagging',
             ('xgb', 'orig'): 'XGBoost, original', ('lgbm', 'orig'): 'LightGBM, original', ('lgbm', 'bagged'): 'LightGBM, bagging'}
COL = {('lgbm', 'bag_hess'): '#2E6DB4', ('xgb', 'bagged'): '#D9822B', ('xgb', 'orig'): '#7F7F7F',
       ('lgbm', 'orig'): '#3A9D5D', ('lgbm', 'bagged'): '#9B59B6'}


def save(fig, name):
    fig.savefig(os.path.join(args.out, name + '.png'), bbox_inches='tight')
    fig.savefig(os.path.join(args.out, name + '.svg'), bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------ Figure 7
def forest(ax, df, settings, margin, title):
    n = len(settings)
    offs = np.linspace(-0.25, 0.25, n) if n > 1 else [0]
    ax.axvspan(-margin * 100, margin * 100, color='#E8F1E4', zorder=0)
    ax.axvline(0, color='black', lw=0.7, zorder=1)
    for k, st in enumerate(settings):
        g = df[(df.Learner == st[0]) & (df.Regime == st[1])].set_index('Config')
        for i, c in enumerate(ORDER):
            if c not in g.index:
                continue
            r = g.loc[c]
            y = i + offs[k]
            ax.plot([r.CI90_lo * 100, r.CI90_hi * 100], [y, y], color=COL[st], lw=1.6, zorder=2)
            ax.plot(r.Delta * 100, y, 'o', color=COL[st], ms=3.6, zorder=3, label=SET_LABEL[st] if i == 0 else None)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([LABEL[c] for c in ORDER])
    ax.invert_yaxis()
    ax.set_xlabel('Difference vs. noise (AUPRC, pp)')
    sel = df[df[['Learner', 'Regime']].apply(tuple, axis=1).isin(settings)]
    lim = max(margin * 100 * 1.45, np.nanmax(np.abs(sel[['CI90_lo', 'CI90_hi']].values)) * 100 * 1.1)
    ax.set_xlim(-lim, lim)
    ax.set_title(title, loc='left')
    ax.text(margin * 100, len(ORDER) - 0.35, f'±{margin * 100:g} pp\nequivalence margin', ha='right', va='bottom',
            fontsize=6.5, color='#4D7A3A')


bc = pd.read_csv(os.path.join(A, 'bootstrap_cc.csv'))
bb = pd.read_csv(os.path.join(A, 'bootstrap_baf.csv'))
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.5))
forest(axs[0], bc[bc.Family == 'anomaly_vs_noise'], [('lgbm', 'bag_hess'), ('xgb', 'bagged'), ('xgb', 'orig')], 0.01,
       '(a) CreditCard, test AUPRC')
forest(axs[1], bb[(bb.Family == 'anomaly_vs_noise') & (bb.Metric == 'MeanVariant_I_V')],
       [('lgbm', 'orig'), ('lgbm', 'bagged'), ('xgb', 'orig'), ('xgb', 'bagged')], 0.005, '(b) BAF, mean-variant AUPRC')
axs[1].set_yticklabels([])
h, l = [], []
for ax in axs:
    hh, ll = ax.get_legend_handles_labels()
    for a, b in zip(hh, ll):
        if b not in l:
            h.append(a); l.append(b)
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout()
save(fig, 'Figure_7')

# ------------------------------------------------------------ Figure 8
r = pd.read_csv(os.path.join(args.results, 'experiment_20', 'runs.csv'))
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={'width_ratios': [1.15, 1]})
cond = [('A', 'lgbm', 'bagged', 'LightGBM default', '#C0392B', 'o'),
        ('A', 'lgbm', 'bag_hess', 'LightGBM, min. Hessian = 1', '#2E6DB4', 'o'),
        ('A', 'lgbm', 'bag_l2', 'LightGBM, L2 = 1', '#16A085', 'o'),
        ('A', 'lgbm', 'bag_mds', 'LightGBM, output cap = 1', '#8E44AD', 'o'),
        ('B', 'xgb', 'bagged', 'XGBoost default', '#D9822B', 's'),
        ('B', 'xgb', 'xrev_both', 'XGBoost, LightGBM defaults', '#7B241C', 's')]
ax = axs[0]
for part, L, G, lab, col, mk in cond:
    g = r[(r.Part == part) & (r.Learner == L) & (r.Regime == G)]
    ax.scatter(g.Max_abs_leaf, g.AUPRC_test, s=12, color=col, marker=mk, alpha=0.75, label=lab, edgecolor='none')
ax.set_xscale('log')
ax.axhline(0.5, color='#999999', lw=0.7, ls=':')
ax.set_xlabel('Largest absolute leaf value per model (log scale)')
ax.set_ylabel('Test AUPRC')
ax.set_title('(a) Leaf values and test AUPRC', loc='left')
ax.legend(frameon=False, loc='lower left', fontsize=6.5)
ax = axs[1]
xc = [('bagged', 'Default'), ('xrev_mcw', 'Min. child\nweight 10⁻³'), ('xrev_l2', 'λ = 0'),
      ('xrev_both', 'Both'), ('xrev_leafwise', 'Both +\nleaf-wise')]
rng = np.random.default_rng(1)
for i, (G, lab) in enumerate(xc):
    for j, (c, col) in enumerate((('M0', '#444444'), ('noise_normal', '#AAAAAA'))):
        g = r[(r.Part == 'B') & (r.Regime == G) & (r.Config == c)].AUPRC_test.values
        x = i + (j - 0.5) * 0.28 + rng.uniform(-0.07, 0.07, len(g))
        ax.scatter(x, g, s=9, color=col, alpha=0.8, edgecolor='none', label=('M0' if c == 'M0' else 'Noise control') if i == 0 else None)
        ax.plot([i + (j - 0.5) * 0.28 - 0.1, i + (j - 0.5) * 0.28 + 0.1], [np.median(g)] * 2, color='black', lw=1.2)
ax.axhline(0.5, color='#999999', lw=0.7, ls=':')
ax.set_xticks(range(len(xc)))
ax.set_xticklabels([b for _, b in xc])
ax.set_ylabel('Test AUPRC')
ax.set_title('(b) Reverse experiment with XGBoost', loc='left')
ax.legend(frameon=False, loc='center left', fontsize=6.5)
fig.tight_layout()
save(fig, 'Figure_8')

# ------------------------------------------------------------ Figure 9
cc = pd.read_csv(os.path.join(args.results, 'experiment_21_cc', 'runs.csv'))
bf = pd.read_csv(os.path.join(args.results, 'experiment_21_baf', 'runs.csv'))
bf['MV'] = bf[[f'AUPRC_Variant_{v}' for v in ['I', 'II', 'III', 'IV', 'V']]].mean(axis=1)
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.9))
ax = axs[0]
sets = [('lgbm', 'orig'), ('lgbm', 'bagged'), ('lgbm', 'bag_hess'), ('xgb', 'orig'), ('xgb', 'bagged')]
wins = ['O1', 'O2', 'O3']
w = 0.16
for k, st in enumerate(sets):
    vals = [100 * (cc[(cc.Origin == o) & (cc.Learner == st[0]) & (cc.Regime == st[1])].AUPRC_test < 0.5).mean() for o in wins]
    ax.bar(np.arange(3) + (k - 2) * w, vals, w, color=COL[st], label=SET_LABEL[st])
ax.set_xticks(range(3)); ax.set_xticklabels(['Window 1', 'Window 2', 'Window 3'])
ax.set_ylabel('Collapsed runs (%)')
ax.set_title('(a) CreditCard collapses', loc='left')


def paired(df, metric, st, o):
    g = df[(df.Origin == o) & (df.Learner == st[0]) & (df.Regime == st[1])].pivot_table(index='Replicate', columns='Config', values=metric)
    d = (g['M2_pct'] - g['noise_normal']).dropna() * 100
    return d


for ax, df, metric, setl, origins, title in (
        (axs[1], cc, 'AUPRC_test', [('lgbm', 'bag_hess'), ('xgb', 'bagged')], wins, '(b) CreditCard'),
        (axs[2], bf, 'MV', [('lgbm', 'orig'), ('lgbm', 'bagged'), ('xgb', 'orig'), ('xgb', 'bagged')], ['O1', 'O2', 'O3', 'O4'], '(c) BAF, mean variant')):
    n = len(setl)
    offs = np.linspace(-0.22, 0.22, n) if n > 1 else [0]
    for k, st in enumerate(setl):
        for i, o in enumerate(origins):
            d = paired(df, metric, st, o)
            q1, md, q3 = np.percentile(d, [25, 50, 75])
            ax.plot([i + offs[k]] * 2, [q1, q3], color=COL[st], lw=1.6)
            ax.plot(i + offs[k], md, 'o', color=COL[st], ms=3.5, label=SET_LABEL[st] if i == 0 else None)
    ax.axhline(0, color='black', lw=0.7)
    ax.set_xticks(range(len(origins)))
    ax.set_xticklabels([f'W{i + 1}' for i in range(len(origins))])
    ax.set_ylabel('Consensus − noise (pp)')
    ax.set_title(title, loc='left')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=7)
fig.tight_layout()
save(fig, 'Figure_9')
print('figures written to', args.out)
