# ============================================================
# EXPERIMENT 16
# VARIANCE, CONTROL AND MECHANISM EXPERIMENTS ON CREDITCARD
#
# Purpose: decide between two explanations of the collapses
#   H1: over-reliance on the anomaly feature
#   H2: general LightGBM training instability (underspecification)
#
# Design (all on ONE code path, same split as experiments 12/15):
#   pipelines : "raw"    -> LightGBM sees unscaled predictors (= exp 12/14)
#               "scaled" -> LightGBM sees train-fitted StandardScaler output (= exp 15)
#               Isolation Forests are ALWAYS fitted on scaled predictors (as before).
#   regimes   : "orig"   -> original LightGBM params (deterministic)
#               "bagged" -> bagging/feature subsampling, LightGBM seed = replicate
#               "reg"    -> orig + min_child_samples=100 + early stopping on val AUPRC
#   mechanism regimes (LightGBM only; "bagged" + one change each, tests the
#   hessian / leaf-value divergence hypothesis):
#               "bag_hess" -> min_sum_hessian_in_leaf=1 (XGBoost min_child_weight default)
#               "bag_mds"  -> max_delta_step=1 (caps leaf output)
#               "bag_l2"   -> reg_lambda=1 (L2 on leaf values)
#   learners  : "lgbm"   -> LightGBM (as in all earlier experiments)
#               "xgb"    -> XGBoost hist (is the collapse LightGBM-specific?)
#   configs   : M0        no extra feature
#               M0_noise  + N(0,1) noise column              (control for "any extra column")
#               M2_perm   + consensus, row-permuted          (control: same marginal, no signal)
#               M1_raw    + single raw IF score              (= original M1)
#               M1_pct    + single IF train-percentile       (2x2 design)
#               M2_raw    + mean of 5 raw IF scores          (2x2 design)
#               M2_pct    + consensus percentile             (= original M2)
#               M3        + consensus + disagreement         (= original M3)
#               M4_raw    + Mahalanobis distance (DETERMINISTIC, no seed)
#               M4_pct    + Mahalanobis train-percentile (DETERMINISTIC)
#               -> M4 removes the IF randomness entirely: any spread left in
#                  M4 comes from the supervised learner, not from the anomaly score.
#   replicates: r = 1..R. IF seeds of replicate r = [100r+1, ..., 100r+5]
#               -> replicates 1..10 reproduce the seed sets of experiment 12.
#
# Output: results/experiment_16/runs.csv (one row per fit, resumable)
#         results/experiment_16/representation.csv
#         results/experiment_16/summary.csv
#
# Usage:
#   python experiment_16_variance_controls.py                  # full run (R=20)
#   python experiment_16_variance_controls.py --quick          # smoke test (R=2)
#   python experiment_16_variance_controls.py --replicates 30
#   python experiment_16_variance_controls.py --pipelines raw --regimes orig
#   python experiment_16_variance_controls.py --learners lgbm      # LightGBM only
# ============================================================

import os
import sys
import time
import json
import argparse
import platform
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="creditcard.csv")
ap.add_argument("--out", default=os.path.join("results", "experiment_16"))
ap.add_argument("--replicates", type=int, default=20)
ap.add_argument("--pipelines", nargs="+", default=["raw", "scaled"])
ap.add_argument("--regimes", nargs="+", default=["orig", "bagged", "reg"])
ap.add_argument("--learners", nargs="+", default=["lgbm", "xgb"])
ap.add_argument("--configs", nargs="+", default=[
    "M0", "M0_noise", "M2_perm", "M1_raw", "M1_pct", "M2_raw", "M2_pct", "M3", "M4_raw", "M4_pct"])
ap.add_argument("--quick", action="store_true", help="2 replicates, smoke test")
ap.add_argument("--n_jobs", type=int, default=-1)
args = ap.parse_args()
if args.quick:
    args.replicates = 2

os.makedirs(args.out, exist_ok=True)
RUNS_FILE = os.path.join(args.out, "runs.csv")
REPR_FILE = os.path.join(args.out, "representation.csv")
SUMMARY_FILE = os.path.join(args.out, "summary.csv")

TARGET, TIME_COL = "Class", "Time"
FPR_LEVELS = [0.001, 0.005, 0.01, 0.05]
TOP_K = [50, 100, 500]
COLLAPSE_AUPRC = 0.5          # pre-specified collapse threshold
IF_PARAMS = dict(n_estimators=200, contamination="auto", n_jobs=args.n_jobs)
SHAP_NORMALS = 5000

BASE_LGBM = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                 n_jobs=args.n_jobs, verbosity=-1, random_state=42)


def lgbm_params(regime, replicate):
    p = dict(BASE_LGBM)
    if regime == "bagged" or regime.startswith("bag_"):
        p.update(subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                 random_state=int(replicate))
        if regime == "bag_hess":
            p.update(min_child_weight=1.0)      # = min_sum_hessian_in_leaf
        elif regime == "bag_mds":
            p.update(max_delta_step=1.0)
        elif regime == "bag_l2":
            p.update(reg_lambda=1.0)
    elif regime == "reg":
        # metric restricted to average_precision so early stopping monitors AUPRC only
        # (v1 of this script also monitored binary_logloss and stopped at iteration 1)
        p.update(n_estimators=1000, min_child_samples=100, metric="average_precision")
    return p


# XGBoost mirrors the LightGBM regimes as closely as possible.
# orig  : deterministic (no row/column subsampling), 300 trees, lr 0.05, depth 6
# bagged: subsample/colsample 0.8 (= experiment 08 XGBoost setting), seed = replicate
# reg   : orig + min_child_weight=5 + early stopping on validation AUPRC
#         (min_child_weight is hessian-based, so it is only an analogue of min_child_samples)
LGBM_ONLY_REGIMES = {"bag_hess", "bag_mds", "bag_l2"}
BASE_XGB = dict(n_estimators=300, learning_rate=0.05, max_depth=6, tree_method="hist",
                n_jobs=args.n_jobs, random_state=42, eval_metric="logloss", verbosity=0)


def xgb_params(regime, replicate):
    p = dict(BASE_XGB)
    if regime == "bagged":
        p.update(subsample=0.8, colsample_bytree=0.8, random_state=int(replicate))
    elif regime == "reg":
        p.update(n_estimators=1000, min_child_weight=5, eval_metric="aucpr", early_stopping_rounds=50)
    return p


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def train_percentile(sorted_train, scores):
    r = np.searchsorted(sorted_train, np.asarray(scores, np.float64), side="right")
    return (r / len(sorted_train)).astype(np.float32)


def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-6, None), np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))


def threshold_at_fpr(y, prob, fpr):
    return float(np.quantile(prob[y == 0], 1.0 - fpr))


def operating_metrics(y_val, p_val, y_test, p_test):
    out = {}
    for f in FPR_LEVELS:
        t = threshold_at_fpr(y_val, p_val, f)
        pred = p_test >= t
        tp = int(np.sum(pred & (y_test == 1)))
        fp = int(np.sum(pred & (y_test == 0)))
        tag = f"{f * 100:g}pct".replace(".", "_")
        out[f"Recall_at_{tag}FPR"] = tp / max(1, int(y_test.sum()))
        out[f"TestFPR_at_{tag}FPR"] = fp / max(1, int((y_test == 0).sum()))
        out[f"TP_at_{tag}FPR"] = tp
        out[f"FP_at_{tag}FPR"] = fp
    order = np.argsort(-p_test, kind="stable")
    for k in TOP_K:
        out[f"Precision_at_{k}"] = float(y_test[order[:k]].mean())
    return out


def load_done():
    if not os.path.exists(RUNS_FILE):
        return set()
    d = pd.read_csv(RUNS_FILE, usecols=["Learner", "Pipeline", "Regime", "Config", "Replicate"])
    return set(map(tuple, d.astype(str).values))


def append_row(row, path):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


# ------------------------------------------------------------
# DATA (identical protocol to experiments 12 and 15)
# ------------------------------------------------------------
print("=" * 80)
print("EXPERIMENT 16 - VARIANCE, CONTROL AND MECHANISM EXPERIMENTS")
print("=" * 80)
df = pd.read_csv(args.data).sort_values(TIME_COL, kind="stable").reset_index(drop=True)
n = len(df)
tr_end, va_end = int(n * 0.60), int(n * 0.80)
FEATURES = [c for c in df.columns if c not in (TARGET, TIME_COL)]
X_raw = df[FEATURES].astype(np.float32).values
y = df[TARGET].astype(np.int8).values
idx = {"train": slice(0, tr_end), "val": slice(tr_end, va_end), "test": slice(va_end, n)}
Y = {k: y[s] for k, s in idx.items()}
for k in idx:
    print(f"{k:6s}: {len(Y[k]):,} rows | fraud={int(Y[k].sum())}")

scaler = StandardScaler()
X_scaled_tr = scaler.fit_transform(X_raw[idx["train"]]).astype(np.float32)
XS = {"train": X_scaled_tr,
      "val": scaler.transform(X_raw[idx["val"]]).astype(np.float32),
      "test": scaler.transform(X_raw[idx["test"]]).astype(np.float32)}
XR = {k: X_raw[s] for k, s in idx.items()}
BASE_X = {"raw": XR, "scaled": XS}

# ------------------------------------------------------------
# DETERMINISTIC ANOMALY SCORE: Mahalanobis distance (train mean / covariance, scaled space)
# ------------------------------------------------------------
_mu = XS["train"].astype(np.float64).mean(axis=0)
_prec = np.linalg.pinv(np.cov(XS["train"].astype(np.float64), rowvar=False))


def mahalanobis(X):
    D = X.astype(np.float64) - _mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", D, _prec, D), 0)).astype(np.float32)


MAHA_RAW = {k: mahalanobis(XS[k]) for k in XS}
_st = np.sort(MAHA_RAW["train"].astype(np.float64))
MAHA_PCT = {k: train_percentile(_st, MAHA_RAW[k]) for k in XS}

# fixed SHAP sample: all test frauds + fixed random normals
rng_shap = np.random.default_rng(42)
te_pos = np.where(Y["test"] == 1)[0]
te_neg = np.where(Y["test"] == 0)[0]
SHAP_IDX = np.concatenate([te_pos, rng_shap.choice(te_neg, min(SHAP_NORMALS, len(te_neg)), replace=False)])

# ------------------------------------------------------------
# ISOLATION FOREST CACHE (IF always on scaled predictors)
# ------------------------------------------------------------
_if_cache = {}


def if_scores(seed):
    if seed not in _if_cache:
        iso = IsolationForest(**IF_PARAMS, random_state=seed).fit(XS["train"])
        _if_cache[seed] = {k: (-iso.decision_function(XS[k])).astype(np.float32) for k in XS}
    return _if_cache[seed]


def replicate_features(r):
    seeds = [100 * r + k for k in range(1, 6)]
    raws = [if_scores(s) for s in seeds]
    pcts = []
    for s in raws:
        st = np.sort(s["train"].astype(np.float64))
        pcts.append({k: train_percentile(st, s[k]) for k in s})
    F = {}
    F["single_raw"] = raws[0]
    F["single_pct"] = pcts[0]
    F["mean_raw"] = {k: np.mean([s[k] for s in raws], axis=0).astype(np.float32) for k in XS}
    F["cons_pct"] = {k: np.mean([p[k] for p in pcts], axis=0).astype(np.float32) for k in XS}
    F["disagree"] = {k: np.std([p[k] for p in pcts], axis=0, ddof=1).astype(np.float32) for k in XS}
    rng = np.random.default_rng(10_000 + r)
    F["noise"] = {k: rng.standard_normal(len(Y[k])).astype(np.float32) for k in XS}
    F["cons_perm"] = {k: rng.permutation(F["cons_pct"][k]) for k in XS}
    F["maha_raw"] = MAHA_RAW
    F["maha_pct"] = MAHA_PCT
    return seeds, F


CONFIG_FEATURES = {
    "M0": [], "M0_noise": ["noise"], "M2_perm": ["cons_perm"],
    "M1_raw": ["single_raw"], "M1_pct": ["single_pct"],
    "M2_raw": ["mean_raw"], "M2_pct": ["cons_pct"], "M3": ["cons_pct", "disagree"],
    "M4_raw": ["maha_raw"], "M4_pct": ["maha_pct"],
}
DETERMINISTIC_CONFIGS = {"M0", "M4_raw", "M4_pct"}


def is_stochastic(config, regime):
    # deterministic inputs + deterministic regime -> same model for every replicate
    return not (config in DETERMINISTIC_CONFIGS and regime in ("orig", "reg"))


# ------------------------------------------------------------
# ONE FIT
# ------------------------------------------------------------
def run_one(learner, pipeline, regime, config, r, F):
    base = BASE_X[pipeline]
    extra = CONFIG_FEATURES[config]
    X = {k: np.column_stack([base[k]] + [F[e][k] for e in extra]).astype(np.float32) if extra else base[k]
         for k in base}
    t0 = time.perf_counter()
    if learner == "lgbm":
        params = lgbm_params(regime, r)
        model = lgb.LGBMClassifier(**params)
        if regime == "reg":
            model.fit(X["train"], Y["train"], eval_set=[(X["val"], Y["val"])],
                      eval_metric="average_precision",
                      callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)])
            best_iter = int(model.best_iteration_ or params["n_estimators"])
        else:
            model.fit(X["train"], Y["train"])
            best_iter = params["n_estimators"]
    else:
        params = xgb_params(regime, r)
        model = xgb.XGBClassifier(**params)
        if regime == "reg":
            model.fit(X["train"], Y["train"], eval_set=[(X["val"], Y["val"])], verbose=False)
            best_iter = int(model.best_iteration) + 1
        else:
            model.fit(X["train"], Y["train"])
            best_iter = params["n_estimators"]
    fit_sec = time.perf_counter() - t0

    p_val = model.predict_proba(X["val"])[:, 1]
    p_te = model.predict_proba(X["test"])[:, 1]
    row = dict(Learner=learner, Pipeline=pipeline, Regime=regime, Config=config, Replicate=r,
               Val_AUPRC=average_precision_score(Y["val"], p_val),
               Test_AUPRC=average_precision_score(Y["test"], p_te),
               Test_ROC_AUC=roc_auc_score(Y["test"], p_te),
               Best_Iteration=best_iter, Fit_sec=fit_sec,
               Log10_ValNeg_q95=float(np.log10(max(threshold_at_fpr(Y["val"], p_val, 0.05), 1e-300))),
               Frac_Test_Prob_below_1e6=float(np.mean(p_te < 1e-6)))
    row.update(operating_metrics(Y["val"], p_val, Y["test"], p_te))

    # reliance on the added feature(s): gain rank and mean |SHAP| rank (LightGBM pred_contrib)
    # (columns always present so that the resumable CSV keeps a fixed schema)
    row.update(Added_Gain_Rank=np.nan, Added_Gain_Share=np.nan, Added_SHAP_Rank=np.nan,
               Added_MeanAbsSHAP=np.nan, Added_SHAP_Share=np.nan, N_Features=int(X["train"].shape[1]))
    if extra:
        n_base = base["train"].shape[1]
        n_all = X["train"].shape[1]
        if learner == "lgbm":
            gain = model.booster_.feature_importance("gain").astype(float)
            contrib = model.booster_.predict(X["test"][SHAP_IDX], pred_contrib=True)[:, :-1]
        else:
            bst = model.get_booster()
            g = bst.get_score(importance_type="total_gain")
            gain = np.array([g.get(f"f{i}", 0.0) for i in range(n_all)], dtype=float)
            it = (0, best_iter) if regime == "reg" else (0, 0)
            contrib = bst.predict(xgb.DMatrix(X["test"][SHAP_IDX]), pred_contribs=True,
                                  iteration_range=it)[:, :-1]
        gain_rank = (-gain).argsort().argsort() + 1
        mabs = np.abs(contrib).mean(axis=0)
        shap_rank = (-mabs).argsort().argsort() + 1
        j = n_base  # first added feature
        row.update(Added_Gain_Rank=int(gain_rank[j]), Added_Gain_Share=float(gain[j] / max(gain.sum(), 1e-12)),
                   Added_SHAP_Rank=int(shap_rank[j]), Added_MeanAbsSHAP=float(mabs[j]),
                   Added_SHAP_Share=float(mabs[j] / max(mabs.sum(), 1e-12)), N_Features=int(X["train"].shape[1]))
    return row


# ------------------------------------------------------------
# MAIN LOOP (resumable)
# ------------------------------------------------------------
done = load_done()
repr_done = set(pd.read_csv(REPR_FILE)["Replicate"].astype(int)) if os.path.exists(REPR_FILE) else set()
total = len(args.learners) * len(args.pipelines) * len(args.regimes) * len(args.configs) * args.replicates
count, t_start = 0, time.perf_counter()

for r in range(1, args.replicates + 1):
    seeds, F = replicate_features(r)

    if r not in repr_done:  # representation diagnostics, once per replicate
        for name in ["single_raw", "single_pct", "mean_raw", "cons_pct"] + (["maha_raw", "maha_pct"] if r == 1 else []):
            f = F[name]
            yt = Y["test"]
            append_row(dict(Replicate=r, IF_Seeds=";".join(map(str, seeds)), Feature=name,
                            Test_Standalone_AUPRC=average_precision_score(yt, f["test"]),
                            Test_Standalone_ROC_AUC=roc_auc_score(yt, f["test"]),
                            Train_Standalone_ROC_AUC=roc_auc_score(Y["train"], f["train"]),
                            PSI_train_to_val=psi(f["train"], f["val"]),
                            PSI_train_to_test=psi(f["train"], f["test"]),
                            PSI_train_to_test_normals=psi(f["train"][Y["train"] == 0], f["test"][yt == 0]),
                            Mean_Fraud_train=float(f["train"][Y["train"] == 1].mean()),
                            Mean_Fraud_test=float(f["test"][yt == 1].mean()),
                            Mean_Normal_train=float(f["train"][Y["train"] == 0].mean()),
                            Mean_Normal_test=float(f["test"][yt == 0].mean())), REPR_FILE)

    for learner in args.learners:
      for pipeline in args.pipelines:
        for regime in args.regimes:
            for config in args.configs:
                count += 1
                if learner == "xgb" and regime in LGBM_ONLY_REGIMES:
                    continue
                if not is_stochastic(config, regime) and r > 1:
                    continue
                key = (learner, pipeline, regime, config, str(r))
                if key in done:
                    continue
                row = run_one(learner, pipeline, regime, config, r, F)
                row["IF_Seeds"] = ";".join(map(str, seeds)) if config not in DETERMINISTIC_CONFIGS else ""
                append_row(row, RUNS_FILE)
                el = time.perf_counter() - t_start
                print(f"[{count}/{total}] r={r:2d} {learner:4s} {pipeline:6s} {regime:6s} {config:9s} "
                      f"valAP={row['Val_AUPRC']:.3f} testAP={row['Test_AUPRC']:.3f} "
                      f"R@1%={row['Recall_at_1pctFPR']:.3f}  ({el / 60:.1f} min)", flush=True)
    _if_cache.clear()  # free memory between replicates

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------
runs = pd.read_csv(RUNS_FILE)


def q(x, p):
    return float(np.quantile(x, p))


summary = (runs.groupby(["Learner", "Pipeline", "Regime", "Config"])
           .agg(N=("Test_AUPRC", "size"),
                Median_AUPRC=("Test_AUPRC", "median"),
                Q1_AUPRC=("Test_AUPRC", lambda x: q(x, .25)),
                Q3_AUPRC=("Test_AUPRC", lambda x: q(x, .75)),
                Mean_AUPRC=("Test_AUPRC", "mean"),
                SD_AUPRC=("Test_AUPRC", "std"),
                Min_AUPRC=("Test_AUPRC", "min"),
                Max_AUPRC=("Test_AUPRC", "max"),
                Collapse_Rate=("Test_AUPRC", lambda x: float(np.mean(x < COLLAPSE_AUPRC))),
                Median_Val_AUPRC=("Val_AUPRC", "median"),
                Median_Recall_1pctFPR=("Recall_at_1pctFPR", "median"),
                Median_Precision_at_100=("Precision_at_100", "median"),
                Median_Added_SHAP_Rank=("Added_SHAP_Rank", "median"))
           .reset_index())
summary.to_csv(SUMMARY_FILE, index=False)

with open(os.path.join(args.out, "settings.json"), "w") as fh:
    json.dump(dict(args=vars(args), base_lgbm=BASE_LGBM, if_params=IF_PARAMS, fpr_levels=FPR_LEVELS,
                   base_xgb=BASE_XGB, collapse_threshold=COLLAPSE_AUPRC, xgboost=xgb.__version__, python=sys.version, platform=platform.platform(),
                   lightgbm=lgb.__version__, sklearn=sklearn.__version__, numpy=np.__version__,
                   pandas=pd.__version__), fh, indent=2)

pd.set_option("display.width", 200)
print("\n" + "=" * 80)
print("SUMMARY (Test AUPRC)")
print("=" * 80)
print(summary[["Learner", "Pipeline", "Regime", "Config", "N", "Median_AUPRC", "Q1_AUPRC", "Q3_AUPRC",
               "Min_AUPRC", "Collapse_Rate", "Median_Recall_1pctFPR"]].round(3).to_string(index=False))
print(f"\nSaved: {RUNS_FILE}\n       {REPR_FILE}\n       {SUMMARY_FILE}")
