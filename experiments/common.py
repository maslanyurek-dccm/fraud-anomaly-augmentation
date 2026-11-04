"""Shared code for experiments 19-22.

Data loading for CreditCard and BAF, the anomaly detectors (Isolation Forest,
Mahalanobis distance, LOF, Extended Isolation Forest, One-Class SVM,
autoencoder), the learners, and a weighted average precision that can be
recomputed quickly for many bootstrap weight vectors.
"""
import os
import gc
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

N_JOBS = -1

# Small utilities
def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def append_row(row, path):
    pd.DataFrame([row]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)


def done_keys(path, cols):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path, usecols=cols)
    return set(map(tuple, d.astype(str).values))


def save_settings(out, extra):
    import sys, platform, sklearn
    s = dict(extra, python=sys.version, platform=platform.platform(), lightgbm=lgb.__version__,
             xgboost=xgb.__version__, sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__)
    with open(os.path.join(out, 'settings.json'), 'w') as fh:
        json.dump(s, fh, indent=2, default=str)


def train_percentile(train_scores, scores):
    st = np.sort(np.asarray(train_scores, np.float64))
    return (np.searchsorted(st, np.asarray(scores, np.float64), side='right') / len(st)).astype(np.float32)


# CreditCard
def load_creditcard(path, cuts=(0.60, 0.80, 1.00)):
    """Chronological split. cuts = (end of train, end of val, end of test) as fractions."""
    df = pd.read_csv(path).sort_values('Time', kind='stable').reset_index(drop=True)
    n = len(df)
    a, b, c = (int(n * f) for f in cuts)
    feats = [x for x in df.columns if x not in ('Class', 'Time')]
    X = df[feats].astype(np.float32).values
    y = df['Class'].astype(np.int8).values
    sl = {'train': slice(0, a), 'val': slice(a, b), 'test': slice(b, c)}
    XR = {k: X[s] for k, s in sl.items()}
    Y = {k: y[s] for k, s in sl.items()}
    sc = StandardScaler().fit(XR['train'])
    XS = {k: sc.transform(v).astype(np.float32) for k, v in XR.items()}
    return XR, XS, Y, ['test']


# BAF  (all six files are encoded once and cached on disk)
BAF_FILES = {'Base': 'Base.csv', 'Variant I': 'Variant I.csv', 'Variant II': 'Variant II.csv',
             'Variant III': 'Variant III.csv', 'Variant IV': 'Variant IV.csv', 'Variant V': 'Variant V.csv'}
VARIANTS = ['Variant I', 'Variant II', 'Variant III', 'Variant IV', 'Variant V']


def baf_store(data_dir, cache_dir, fit_months=(0, 1, 2, 3, 4, 5)):
    """Returns dict name -> (E float32 [n,30], month int8, y int8). The ordinal encoder is fitted on
    Base rows of fit_months only. Cached as .npz so later scripts load in seconds."""
    os.makedirs(cache_dir, exist_ok=True)
    tag = 'baf_enc_' + ''.join(map(str, fit_months)) + '.npz'
    cpath = os.path.join(cache_dir, tag)
    if os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        return {k: (z[k + '_E'], z[k + '_m'], z[k + '_y']) for k in BAF_FILES}, list(z['cols'])
    log('Encoding BAF files (first run only, cached afterwards)...')
    base = pd.read_csv(os.path.join(data_dir, BAF_FILES['Base']))
    cols = [c for c in base.columns if c not in ('fraud_bool', 'month')]
    cat = base[cols].select_dtypes(include=['object', 'category']).columns.tolist()
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.float32)
    enc.fit(base.loc[base['month'].isin(fit_months), cat].astype(str))
    out, arrays = {}, {}
    for name, fn in BAF_FILES.items():
        d = base if name == 'Base' else pd.read_csv(os.path.join(data_dir, fn))
        E = np.zeros((len(d), len(cols)), np.float32)
        cv = enc.transform(d[cat].astype(str))
        for j, c in enumerate(cols):
            E[:, j] = cv[:, cat.index(c)] if c in cat else pd.to_numeric(d[c], errors='coerce').fillna(0).astype(np.float32).values
        out[name] = (E, d['month'].astype(np.int8).values, d['fraud_bool'].astype(np.int8).values)
        arrays.update({name + '_E': E, name + '_m': out[name][1], name + '_y': out[name][2]})
        if name != 'Base':
            del d
        gc.collect()
    np.savez(cpath, cols=np.array(cols, dtype=object), **arrays)
    return out, cols


def baf_split(store, train_months, val_month, test_month):
    E, m, y = store['Base']
    tr = np.isin(m, train_months)
    XR = {'train': E[tr], 'val': E[m == val_month]}
    Y = {'train': y[tr], 'val': y[m == val_month]}
    for name in BAF_FILES:
        Ei, mi, yi = store[name]
        XR[name] = Ei[mi == test_month]
        Y[name] = yi[mi == test_month]
    sc = StandardScaler().fit(XR['train'])
    XS = {k: sc.transform(v).astype(np.float32) for k, v in XR.items()}
    return XR, XS, Y, list(BAF_FILES)


# Anomaly scores (higher = more anomalous), always on scaled data
def if_scores(XS, seed):
    iso = IsolationForest(n_estimators=200, contamination='auto', n_jobs=N_JOBS, random_state=seed).fit(XS['train'])
    return {k: (-iso.decision_function(v)).astype(np.float32) for k, v in XS.items()}


def mahalanobis_scores(XS):
    Xt = XS['train'].astype(np.float64)
    mu, prec = Xt.mean(0), np.linalg.pinv(np.cov(Xt, rowvar=False))
    out = {}
    for k, v in XS.items():
        r = np.empty(len(v), np.float32)
        for i in range(0, len(v), 200_000):
            D = v[i:i + 200_000].astype(np.float64) - mu
            r[i:i + 200_000] = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', D, prec, D), 0))
        out[k] = r
    return out


def _subsample(n, m, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, min(n, m), replace=False))


def lof_scores(XS, fit_n=30_000, seed=0):
    idx = _subsample(len(XS['train']), fit_n, seed)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=N_JOBS).fit(XS['train'][idx])
    return {k: (-lof.score_samples(v)).astype(np.float32) for k, v in XS.items()}


def ocsvm_scores(XS, seed, fit_n=50_000):
    idx = _subsample(len(XS['train']), fit_n, seed)
    d = XS['train'].shape[1]
    ny = Nystroem(gamma=1.0 / d, n_components=300, random_state=seed).fit(XS['train'][idx])
    oc = SGDOneClassSVM(nu=0.05, random_state=seed).fit(ny.transform(XS['train'][idx]))
    out = {}
    for k, v in XS.items():
        r = np.empty(len(v), np.float32)
        for i in range(0, len(v), 200_000):
            r[i:i + 200_000] = -oc.decision_function(ny.transform(v[i:i + 200_000]))
        out[k] = r
    return out


def ae_scores(XS, seed, fit_n=100_000):
    idx = _subsample(len(XS['train']), fit_n, seed)
    d = XS['train'].shape[1]
    ae = MLPRegressor(hidden_layer_sizes=(max(8, d // 2), max(4, d // 6), max(8, d // 2)), activation='relu',
                      batch_size=512, max_iter=40, learning_rate_init=1e-3, random_state=seed)
    Xf = XS['train'][idx]
    ae.fit(Xf, Xf)
    out = {}
    for k, v in XS.items():
        r = np.empty(len(v), np.float32)
        for i in range(0, len(v), 200_000):
            blk = v[i:i + 200_000]
            r[i:i + 200_000] = ((ae.predict(blk) - blk) ** 2).mean(1)
        out[k] = r
    return out


class ExtendedIF:
    """Extended Isolation Forest (Hariri et al., 2021), fully extended random hyperplanes."""

    def __init__(self, n_trees=100, sample_size=256, seed=0):
        self.n_trees, self.psi, self.rng = n_trees, sample_size, np.random.default_rng(seed)

    @staticmethod
    def _c(n):
        if n > 2:
            return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
        return 1.0 if n == 2 else 0.0

    def fit(self, X):
        self.limit = int(np.ceil(np.log2(self.psi)))
        self.trees = []
        for _ in range(self.n_trees):
            S = X[self.rng.choice(len(X), min(self.psi, len(X)), replace=False)]
            nodes = []  # (normal, point, left, right, size, depth)

            def build(D, depth):
                i = len(nodes)
                nodes.append(None)
                if depth >= self.limit or len(D) <= 1:
                    nodes[i] = (None, None, -1, -1, len(D), depth)
                    return i
                nrm = self.rng.standard_normal(D.shape[1]).astype(np.float32)
                p = self.rng.uniform(D.min(0), D.max(0)).astype(np.float32)
                go_left = (D - p) @ nrm <= 0
                l = build(D[go_left], depth + 1)
                r = build(D[~go_left], depth + 1)
                nodes[i] = (nrm, p, l, r, len(D), depth)
                return i
            build(S, 0)
            k = len(nodes)
            d = X.shape[1]
            T = dict(nrm=np.zeros((k, d), np.float32), p=np.zeros((k, d), np.float32), l=np.full(k, -1), r=np.full(k, -1),
                     leaf=np.zeros(k, bool), hl=np.zeros(k, np.float32))
            for j, (nrm, p, l, r, size, depth) in enumerate(nodes):
                if nrm is None:
                    T['leaf'][j] = True
                    T['hl'][j] = depth + self._c(size)
                else:
                    T['nrm'][j], T['p'][j], T['l'][j], T['r'][j] = nrm, p, l, r
            self.trees.append(T)
        return self

    def score(self, X, chunk=100_000):
        out = np.empty(len(X), np.float32)
        for s in range(0, len(X), chunk):
            B = X[s:s + chunk]
            h = np.zeros(len(B), np.float64)
            for T in self.trees:
                node = np.zeros(len(B), np.int64)
                for _ in range(self.limit + 1):
                    act = ~T['leaf'][node]
                    if not act.any():
                        break
                    a = np.where(act)[0]
                    nd = node[a]
                    proj = np.einsum('ij,ij->i', B[a] - T['p'][nd], T['nrm'][nd])
                    node[a] = np.where(proj <= 0, T['l'][nd], T['r'][nd])
                h += T['hl'][node]
            out[s:s + chunk] = -(h / len(self.trees))   # shorter paths = more anomalous
        return out


def eif_scores(XS, seed, fit_n=100_000):
    idx = _subsample(len(XS['train']), fit_n, seed)
    m = ExtendedIF(n_trees=100, seed=seed).fit(XS['train'][idx])
    return {k: m.score(v) for k, v in XS.items()}


# Features of one replicate
NOISE_KINDS = ['noise_normal', 'noise_uniform', 'noise_lognormal', 'noise_t3', 'noise_corr', 'noise5']


def replicate_features(XS, Y, r, need, cache=None):
    """Returns dict feature_name -> {split: array}. `need` = set of feature names to compute.
    cache: dict for deterministic features shared across replicates (Mahalanobis, LOF)."""
    keys = list(XS)
    F = {}
    seeds = [100 * r + k for k in range(1, 6)]
    if need & {'if_pct', 'cons_pct', 'disagree', 'cons_perm', 'cons_distmatch'}:
        raws = [if_scores(XS, s) for s in seeds]
        pcts = [{k: train_percentile(sc['train'], sc[k]) for k in keys} for sc in raws]
        F['if_pct'] = pcts[0]
        F['cons_pct'] = {k: np.mean([p[k] for p in pcts], 0).astype(np.float32) for k in keys}
        F['disagree'] = {k: np.std([p[k] for p in pcts], 0, ddof=1).astype(np.float32) for k in keys}
    rng = np.random.default_rng(10_000 + r)
    n = {k: len(XS[k]) for k in keys}
    if 'noise_normal' in need:
        F['noise_normal'] = {k: rng.standard_normal(n[k]).astype(np.float32) for k in keys}
    if 'noise_uniform' in need:
        F['noise_uniform'] = {k: rng.uniform(0, 1, n[k]).astype(np.float32) for k in keys}
    if 'noise_lognormal' in need:
        F['noise_lognormal'] = {k: rng.lognormal(0, 1, n[k]).astype(np.float32) for k in keys}
    if 'noise_t3' in need:
        F['noise_t3'] = {k: rng.standard_t(3, n[k]).astype(np.float32) for k in keys}
    if 'noise_corr' in need:   # correlated (rho = 0.7) with a randomly chosen existing feature, independent of y
        j = int(rng.integers(0, XS['train'].shape[1]))
        F['noise_corr'] = {k: (0.7 * XS[k][:, j] + np.sqrt(1 - 0.49) * rng.standard_normal(n[k])).astype(np.float32) for k in keys}
        F['noise_corr_feature'] = j
    if 'noise5' in need:
        F['noise5'] = {k: rng.standard_normal((n[k], 5)).astype(np.float32) for k in keys}
    if 'cons_perm' in need:
        F['cons_perm'] = {k: rng.permutation(F['cons_pct'][k]) for k in keys}
    if 'cons_distmatch' in need:   # independent draws from the training marginal of the consensus
        tr = np.sort(F['cons_pct']['train'])
        F['cons_distmatch'] = {k: np.quantile(tr, rng.uniform(0, 1, n[k])).astype(np.float32) for k in keys}
    if cache is not None and 'maha_pct' in need:
        if 'maha' not in cache:
            cache['maha'] = mahalanobis_scores(XS)
        F['maha_pct'] = {k: train_percentile(cache['maha']['train'], cache['maha'][k]) for k in keys}
    if cache is not None and 'lof_pct' in need:
        if 'lof' not in cache:
            log('  fitting LOF (deterministic, once)')
            cache['lof'] = lof_scores(XS)
        F['lof_pct'] = {k: train_percentile(cache['lof']['train'], cache['lof'][k]) for k in keys}
    for name, fn in (('eif_pct', eif_scores), ('ocsvm_pct', ocsvm_scores), ('ae_pct', ae_scores)):
        if name in need:
            sc = fn(XS, seed=seeds[0])
            F[name] = {k: train_percentile(sc['train'], sc[k]) for k in keys}
    return F


CONFIGS = {
    'M0': [], 'M1_pct': ['if_pct'], 'M2_pct': ['cons_pct'], 'M3': ['cons_pct', 'disagree'],
    'noise_normal': ['noise_normal'], 'noise_uniform': ['noise_uniform'], 'noise_lognormal': ['noise_lognormal'],
    'noise_t3': ['noise_t3'], 'noise_corr': ['noise_corr'], 'noise5': ['noise5'],
    'cons_perm': ['cons_perm'], 'cons_distmatch': ['cons_distmatch'],
    'maha_pct': ['maha_pct'], 'lof_pct': ['lof_pct'], 'eif_pct': ['eif_pct'], 'ocsvm_pct': ['ocsvm_pct'], 'ae_pct': ['ae_pct'],
}
DETERMINISTIC_CONFIGS = {'M0', 'maha_pct', 'lof_pct'}


def build_X(base, F, config):
    extra = CONFIGS[config]
    if not extra:
        return base
    return {k: np.column_stack([base[k]] + [F[e][k].reshape(len(base[k]), -1) for e in extra]).astype(np.float32) for k in base}


# Learners
# deterministic=True + a fixed histogram strategy make LightGBM results reproducible across runs and machines.
# (With the default automatic col-/row-wise choice, the unstable LightGBM configurations on CreditCard can land
#  on different trajectories in different runs; experiment 20 part D measures this directly.)
LGBM_BASE = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=N_JOBS, verbosity=-1, random_state=42,
                 deterministic=True, force_col_wise=True)
XGB_BASE = dict(n_estimators=300, learning_rate=0.05, max_depth=6, tree_method='hist', n_jobs=N_JOBS,
                random_state=42, eval_metric='logloss', verbosity=0)


def learner_params(learner, regime, r, override=None):
    if learner == 'lgbm':
        p = dict(LGBM_BASE)
        if regime != 'orig':      # every other LightGBM regime is built on bagging
            p.update(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=int(r))
        if regime == 'bag_hess':
            p.update(min_child_weight=1.0)          # = min_sum_hessian_in_leaf
        elif regime == 'bag_mds':
            p.update(max_delta_step=1.0)
        elif regime == 'bag_l2':
            p.update(reg_lambda=1.0)
    else:
        p = dict(XGB_BASE)
        if regime != 'orig':
            p.update(subsample=0.8, colsample_bytree=0.8, random_state=int(r))
        if regime in ('xrev_both', 'xrev_leafwise'):
            p.update(min_child_weight=1e-3, reg_lambda=0.0)     # LightGBM defaults
        elif regime == 'xrev_mcw':
            p.update(min_child_weight=1e-3)
        elif regime == 'xrev_l2':
            p.update(reg_lambda=0.0)
        if regime == 'xrev_leafwise':
            p.update(grow_policy='lossguide', max_leaves=31, max_depth=0)
    if override:
        p.update(override)
    return p


def fit_predict(learner, params, X, Y, eval_keys):
    t0 = time.perf_counter()
    m = (lgb.LGBMClassifier if learner == 'lgbm' else xgb.XGBClassifier)(**params)
    m.fit(X['train'], Y['train'])
    fit_sec = time.perf_counter() - t0
    preds = {k: m.predict_proba(X[k])[:, 1].astype(np.float32) for k in ['val'] + eval_keys}
    return m, preds, fit_sec


def run_metrics(Y, preds, eval_keys):
    pv = preds['val']
    thr = float(np.quantile(pv[Y['val'] == 0], 0.95))
    row = dict(Val_AUPRC=average_precision_score(Y['val'], pv), Log10_Thr5=float(np.log10(max(thr, 1e-300))))
    for k in eval_keys:
        p, y = preds[k], Y[k]
        tag = k.replace(' ', '_')
        row[f'AUPRC_{tag}'] = average_precision_score(y, p)
        row[f'ROC_{tag}'] = roc_auc_score(y, p)
        for f in (0.01, 0.05):
            t = float(np.quantile(pv[Y['val'] == 0], 1 - f))
            row[f'Recall{int(f * 100)}_{tag}'] = float((p[y == 1] >= t).mean())
    return row


def shap_rank_added(learner, model, X_eval, n_base, regime=None):
    if X_eval.shape[1] == n_base:
        return np.nan
    if learner == 'lgbm':
        contrib = model.booster_.predict(X_eval, pred_contrib=True)[:, :-1]
    else:
        contrib = model.get_booster().predict(xgb.DMatrix(X_eval), pred_contribs=True)[:, :-1]
    mabs = np.abs(contrib).mean(0)
    return int((-mabs).argsort().argsort()[n_base] + 1)


# Fast weighted average precision (identical to sklearn's step-wise ap)
class APIndex:
    """Pre-sorts one score vector so that AP can be recomputed for many bootstrap weight vectors."""

    def __init__(self, scores, y):
        o = np.argsort(-scores, kind='mergesort')
        s = scores[o]
        self.order = o
        self.y = y[o].astype(np.float64)
        self.last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]   # last index of each tie group

    def ap(self, w):
        ws = w[self.order]
        tp = np.cumsum(ws * self.y)[self.last]
        fp = np.cumsum(ws * (1 - self.y))[self.last]
        P = tp[-1]
        if P <= 0:
            return np.nan
        prec = tp / np.maximum(tp + fp, 1e-12)
        rec = tp / P
        return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def stratified_weights(y, rng):
    w = np.zeros(len(y), np.float64)
    for cls in (0, 1):
        idx = np.nonzero(y == cls)[0]
        w[idx] = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx))
    return w
