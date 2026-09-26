# ============================================================
# EXPERIMENT 18
# BAF: MULTI-SEED, CONTROL AND LEARNER-STABILITY EXPERIMENT
#
# Same logic as experiment 16 (CreditCard), on the BAF protocol of
# experiment 05B:
#   Train      = Base Month 0-5
#   Validation = Base Month 6   (thresholds, early stopping)
#   Evaluation = Base Month 7 and Variant I-V Month 7 (time-aligned)
#   month is NOT a predictor; Variant extra columns (x1/x2) are ignored.
#   Frozen model + frozen validation thresholds on every evaluation set.
#
# Questions:
#   1. Does the small consensus gain (Table 3, single seed) survive
#      multi-seed evaluation and uninformative control columns?
#   2. Is LightGBM unstable on BAF as it is on CreditCard? (XGBoost as reference)
#
# pipelines : "enc"    -> learner sees ordinal-encoded, UNSCALED predictors
#             "scaled" -> learner sees train-fitted StandardScaler output (= exp 05B)
#             Anomaly scores are ALWAYS computed on scaled predictors.
# regimes   : orig (deterministic), bagged (seed = replicate), reg (early stopping)
# configs   : M0, M0_noise, M2_perm, M1_raw, M1_pct, M2_raw, M2_pct, M3, M4_raw, M4_pct
#             (defaults below use a reduced set to keep run time reasonable)
# replicates: IF seeds of replicate r = [100r+1, ..., 100r+5]
#
# Output (resumable): results/experiment_18/runs.csv   (one row per fit x evaluation set)
#                     results/experiment_18/summary.csv
#                     results/experiment_18/paired.csv  (differences vs M0)
#                     results/experiment_18/settings.json
#
# Usage:
#   python experiment_18_baf_variance_controls.py --quick        # smoke test
#   python experiment_18_baf_variance_controls.py                # default run
#   python experiment_18_baf_variance_controls.py --pipelines scaled   # ~half the time
# ============================================================

import os
import sys
import gc
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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default=".")
ap.add_argument("--out", default=os.path.join("results", "experiment_18"))
ap.add_argument("--replicates", type=int, default=10)
ap.add_argument("--learners", nargs="+", default=["lgbm", "xgb"])
ap.add_argument("--pipelines", nargs="+", default=["enc", "scaled"])
ap.add_argument("--regimes", nargs="+", default=["orig", "bagged"])
ap.add_argument("--configs", nargs="+", default=[
    "M0", "M0_noise", "M2_perm", "M1_raw", "M2_pct", "M3", "M4_pct"])
ap.add_argument("--quick", action="store_true", help="2 replicates, smoke test")
ap.add_argument("--n_jobs", type=int, default=-1)
args = ap.parse_args()
if args.quick:
    args.replicates = 2

os.makedirs(args.out, exist_ok=True)
RUNS_FILE = os.path.join(args.out, "runs.csv")
SUMMARY_FILE = os.path.join(args.out, "summary.csv")
PAIRED_FILE = os.path.join(args.out, "paired.csv")

TARGET, TIME_COL = "fraud_bool", "month"
TRAIN_MONTHS, VAL_MONTH, TEST_MONTH = [0, 1, 2, 3, 4, 5], 6, 7
BASE_FILE = "Base.csv"
VARIANTS = {"Variant I": "Variant I.csv", "Variant II": "Variant II.csv",
            "Variant III": "Variant III.csv", "Variant IV": "Variant IV.csv",
            "Variant V": "Variant V.csv"}
FPR_LEVELS = [0.01, 0.05]
SHAP_NORMALS = 5000
IF_PARAMS = dict(n_estimators=200, contamination="auto", n_jobs=args.n_jobs)

BASE_LGBM = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                 n_jobs=args.n_jobs, verbosity=-1, random_state=42)
BASE_XGB = dict(n_estimators=300, learning_rate=0.05, max_depth=6, tree_method="hist",
                n_jobs=args.n_jobs, random_state=42, eval_metric="logloss", verbosity=0)


def lgbm_params(regime, r):
    p = dict(BASE_LGBM)
    if regime == "bagged":
        p.update(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=int(r))
    elif regime == "reg":
        p.update(n_estimators=1000, min_child_samples=100, metric="average_precision")
    return p


def xgb_params(regime, r):
    p = dict(BASE_XGB)
    if regime == "bagged":
        p.update(subsample=0.8, colsample_bytree=0.8, random_state=int(r))
    elif regime == "reg":
        p.update(n_estimators=1000, min_child_weight=5, eval_metric="aucpr", early_stopping_rounds=50)
    return p


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def train_percentile(sorted_train, scores):
    r = np.searchsorted(sorted_train, np.asarray(scores, np.float64), side="right")
    return (r / len(sorted_train)).astype(np.float32)


def thr_at_fpr(y, p, f):
    return float(np.quantile(p[y == 0], 1.0 - f))


def append_row(row, path):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def load_done():
    if not os.path.exists(RUNS_FILE):
        return set()
    d = pd.read_csv(RUNS_FILE, usecols=["Learner", "Pipeline", "Regime", "Config", "Replicate"])
    return set(map(tuple, d.astype(str).values))


class BAFEncoder:
    """Ordinal-encode categoricals (fit on train), keep numerics; optional scaling."""

    def fit(self, X):
        self.cols = list(X.columns)
        self.cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                  dtype=np.float32).fit(X[self.cat].astype(str)) if self.cat else None
        self.scaler = StandardScaler().fit(self.encode(X))
        return self

    def encode(self, X):
        miss = [c for c in self.cols if c not in X.columns]
        if miss:
            raise ValueError(f"Missing Base features: {miss}")
        X = X[self.cols]
        out = np.zeros((len(X), len(self.cols)), dtype=np.float32)
        cat_vals = self.enc.transform(X[self.cat].astype(str)) if self.cat else None
        for j, c in enumerate(self.cols):
            if c in self.cat:
                out[:, j] = cat_vals[:, self.cat.index(c)]
            else:
                out[:, j] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(np.float32).values
        return out

    def scale(self, E):
        return self.scaler.transform(E).astype(np.float32)


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------
print("=" * 80)
print("EXPERIMENT 18 - BAF MULTI-SEED / CONTROL / LEARNER-STABILITY")
print("=" * 80)
t0 = time.time()
base = pd.read_csv(os.path.join(args.data_dir, BASE_FILE))
parts = {"train": base[base[TIME_COL].isin(TRAIN_MONTHS)],
         "val": base[base[TIME_COL] == VAL_MONTH],
         "Base": base[base[TIME_COL] == TEST_MONTH]}
DROP = [TARGET, TIME_COL]
encoder = BAFEncoder().fit(parts["train"].drop(columns=DROP))
ENC, SCL, Y = {}, {}, {}
for k, d in parts.items():
    ENC[k] = encoder.encode(d.drop(columns=DROP))
    SCL[k] = encoder.scale(ENC[k])
    Y[k] = d[TARGET].astype(np.int8).values
del base, parts
gc.collect()
for name, fn in VARIANTS.items():
    d = pd.read_csv(os.path.join(args.data_dir, fn))
    d = d[d[TIME_COL] == TEST_MONTH]
    ENC[name] = encoder.encode(d.drop(columns=DROP, errors="ignore"))
    SCL[name] = encoder.scale(ENC[name])
    Y[name] = d[TARGET].astype(np.int8).values
    del d
    gc.collect()
EVAL_SETS = ["Base"] + list(VARIANTS)
for k in ["train", "val"] + EVAL_SETS:
    print(f"{k:12s}: {len(Y[k]):>9,} rows | fraud={int(Y[k].sum()):>6,} ({Y[k].mean() * 100:.3f}%)")
print(f"Data prepared in {time.time() - t0:.1f} s | features={len(encoder.cols)}")
BASE_X = {"enc": ENC, "scaled": SCL}
ALL_KEYS = ["train", "val"] + EVAL_SETS

rng_shap = np.random.default_rng(42)
_pos = np.where(Y["Base"] == 1)[0]
_neg = np.where(Y["Base"] == 0)[0]
SHAP_IDX = np.concatenate([_pos, rng_shap.choice(_neg, min(SHAP_NORMALS, len(_neg)), replace=False)])

# ------------------------------------------------------------
# ANOMALY FEATURES
# ------------------------------------------------------------
_mu = SCL["train"].astype(np.float64).mean(axis=0)
_prec = np.linalg.pinv(np.cov(SCL["train"].astype(np.float64), rowvar=False))


def mahalanobis(Z):
    out = np.empty(len(Z), dtype=np.float32)
    for i in range(0, len(Z), 200_000):  # chunked to limit memory
        D = Z[i:i + 200_000].astype(np.float64) - _mu
        out[i:i + 200_000] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", D, _prec, D), 0))
    return out


MAHA_RAW = {k: mahalanobis(SCL[k]) for k in ALL_KEYS}
_st = np.sort(MAHA_RAW["train"].astype(np.float64))
MAHA_PCT = {k: train_percentile(_st, MAHA_RAW[k]) for k in ALL_KEYS}


def replicate_features(r):
    seeds = [100 * r + k for k in range(1, 6)]
    raws, pcts = [], []
    for s in seeds:
        iso = IsolationForest(**IF_PARAMS, random_state=s).fit(SCL["train"])
        sc = {k: (-iso.decision_function(SCL[k])).astype(np.float32) for k in ALL_KEYS}
        st = np.sort(sc["train"].astype(np.float64))
        raws.append(sc)
        pcts.append({k: train_percentile(st, sc[k]) for k in ALL_KEYS})
        del iso
    F = {"single_raw": raws[0], "single_pct": pcts[0],
         "mean_raw": {k: np.mean([s[k] for s in raws], axis=0).astype(np.float32) for k in ALL_KEYS},
         "cons_pct": {k: np.mean([p[k] for p in pcts], axis=0).astype(np.float32) for k in ALL_KEYS},
         "disagree": {k: np.std([p[k] for p in pcts], axis=0, ddof=1).astype(np.float32) for k in ALL_KEYS},
         "maha_raw": MAHA_RAW, "maha_pct": MAHA_PCT}
    rng = np.random.default_rng(10_000 + r)
    F["noise"] = {k: rng.standard_normal(len(Y[k])).astype(np.float32) for k in ALL_KEYS}
    F["cons_perm"] = {k: rng.permutation(F["cons_pct"][k]) for k in ALL_KEYS}
    return seeds, F


CONFIG_FEATURES = {"M0": [], "M0_noise": ["noise"], "M2_perm": ["cons_perm"],
                   "M1_raw": ["single_raw"], "M1_pct": ["single_pct"], "M2_raw": ["mean_raw"],
                   "M2_pct": ["cons_pct"], "M3": ["cons_pct", "disagree"],
                   "M4_raw": ["maha_raw"], "M4_pct": ["maha_pct"]}
DETERMINISTIC = {"M0", "M4_raw", "M4_pct"}


def is_stochastic(config, regime):
    return not (config in DETERMINISTIC and regime in ("orig", "reg"))


# ------------------------------------------------------------
# ONE FIT -> one row per evaluation set
# ------------------------------------------------------------
def run_one(learner, pipeline, regime, config, r, F):
    base = BASE_X[pipeline]
    extra = CONFIG_FEATURES[config]
    X = {k: (np.column_stack([base[k]] + [F[e][k] for e in extra]).astype(np.float32) if extra else base[k])
         for k in ALL_KEYS}
    t = time.perf_counter()
    if learner == "lgbm":
        p = lgbm_params(regime, r)
        m = lgb.LGBMClassifier(**p)
        if regime == "reg":
            m.fit(X["train"], Y["train"], eval_set=[(X["val"], Y["val"])], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)])
            best = int(m.best_iteration_ or p["n_estimators"])
        else:
            m.fit(X["train"], Y["train"])
            best = p["n_estimators"]
    else:
        p = xgb_params(regime, r)
        m = xgb.XGBClassifier(**p)
        if regime == "reg":
            m.fit(X["train"], Y["train"], eval_set=[(X["val"], Y["val"])], verbose=False)
            best = int(m.best_iteration) + 1
        else:
            m.fit(X["train"], Y["train"])
            best = p["n_estimators"]
    fit_sec = time.perf_counter() - t

    pv = m.predict_proba(X["val"])[:, 1]
    thr = {f: thr_at_fpr(Y["val"], pv, f) for f in FPR_LEVELS}
    common = dict(Learner=learner, Pipeline=pipeline, Regime=regime, Config=config, Replicate=r,
                  Val_AUPRC=average_precision_score(Y["val"], pv), Best_Iteration=best, Fit_sec=fit_sec,
                  Log10_ValNeg_q95=float(np.log10(max(thr[0.05], 1e-300))),
                  Added_SHAP_Rank=np.nan, Added_SHAP_Share=np.nan, Added_Gain_Rank=np.nan)
    if extra:
        n_base, n_all = base["train"].shape[1], X["train"].shape[1]
        if learner == "lgbm":
            gain = m.booster_.feature_importance("gain").astype(float)
            contrib = m.booster_.predict(X["Base"][SHAP_IDX], pred_contrib=True)[:, :-1]
        else:
            bst = m.get_booster()
            g = bst.get_score(importance_type="total_gain")
            gain = np.array([g.get(f"f{i}", 0.0) for i in range(n_all)])
            it = (0, best) if regime == "reg" else (0, 0)
            contrib = bst.predict(xgb.DMatrix(X["Base"][SHAP_IDX]), pred_contribs=True, iteration_range=it)[:, :-1]
        mabs = np.abs(contrib).mean(axis=0)
        common.update(Added_SHAP_Rank=int((-mabs).argsort().argsort()[n_base] + 1),
                      Added_SHAP_Share=float(mabs[n_base] / max(mabs.sum(), 1e-12)),
                      Added_Gain_Rank=int((-gain).argsort().argsort()[n_base] + 1))
    rows = []
    for es in EVAL_SETS:
        pe = m.predict_proba(X[es])[:, 1]
        ye = Y[es]
        row = dict(common, EvalSet=es, AUPRC=average_precision_score(ye, pe), ROC_AUC=roc_auc_score(ye, pe))
        for f in FPR_LEVELS:
            pred = pe >= thr[f]
            tag = f"{f * 100:g}pct"
            row[f"Recall_at_{tag}FPR"] = float(pred[ye == 1].mean())
            row[f"TestFPR_at_{tag}FPR"] = float(pred[ye == 0].mean())
        rows.append(row)
    return rows


# ------------------------------------------------------------
# MAIN LOOP (resumable)
# ------------------------------------------------------------
done = load_done()
t_start = time.perf_counter()
for r in range(1, args.replicates + 1):
    seeds, F = replicate_features(r)
    for learner in args.learners:
        for pipeline in args.pipelines:
            for regime in args.regimes:
                for config in args.configs:
                    if not is_stochastic(config, regime) and r > 1:
                        continue
                    if (learner, pipeline, regime, config, str(r)) in done:
                        continue
                    rows = run_one(learner, pipeline, regime, config, r, F)
                    for row in rows:
                        row["IF_Seeds"] = "" if config in DETERMINISTIC else ";".join(map(str, seeds))
                        append_row(row, RUNS_FILE)
                    b = rows[0]["AUPRC"]
                    mv = np.mean([x["AUPRC"] for x in rows[1:]])
                    print(f"r={r:2d} {learner:4s} {pipeline:6s} {regime:6s} {config:9s} "
                          f"valAP={rows[0]['Val_AUPRC']:.4f} baseAP={b:.4f} meanVarAP={mv:.4f} "
                          f"({(time.perf_counter() - t_start) / 60:.1f} min)", flush=True)
    del F
    gc.collect()

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------
runs = pd.read_csv(RUNS_FILE)
keys = ["Learner", "Pipeline", "Regime", "Config", "Replicate"]
wide = runs.pivot_table(index=keys, columns="EvalSet", values="AUPRC").reset_index()
var_cols = [v for v in VARIANTS if v in wide.columns]
wide["MeanVariant"] = wide[var_cols].mean(axis=1)
wide["WorstVariant"] = wide[var_cols].min(axis=1)
fpr = runs[runs.EvalSet != "Base"].groupby(keys)["TestFPR_at_5pctFPR"].mean().rename("MeanVariant_FPR5").reset_index()
wide = wide.merge(fpr, on=keys)


def q(x, p):
    return float(np.quantile(x, p))


grp = ["Learner", "Pipeline", "Regime", "Config"]
summary = wide.groupby(grp).agg(
    N=("Base", "size"),
    Base_Median=("Base", "median"), Base_Q1=("Base", lambda x: q(x, .25)), Base_Q3=("Base", lambda x: q(x, .75)),
    Base_Min=("Base", "min"), Base_Max=("Base", "max"),
    MeanVar_Median=("MeanVariant", "median"), MeanVar_Q1=("MeanVariant", lambda x: q(x, .25)),
    MeanVar_Q3=("MeanVariant", lambda x: q(x, .75)), MeanVar_Min=("MeanVariant", "min"),
    MeanVar_Max=("MeanVariant", "max"), Worst_Median=("WorstVariant", "median"),
    MeanVar_FPR5_Median=("MeanVariant_FPR5", "median")).reset_index()
summary.to_csv(SUMMARY_FILE, index=False)

# paired differences vs M0 (same replicate = same learner seed in "bagged";
# deterministic M0 as reference in "orig")
prow = []
for (L, P, R), g in wide.groupby(["Learner", "Pipeline", "Regime"]):
    m0 = g[g.Config == "M0"].set_index("Replicate")
    if m0.empty:
        continue
    for c in g.Config.unique():
        if c == "M0":
            continue
        gc_ = g[g.Config == c].set_index("Replicate")
        for metric in ["Base", "MeanVariant"]:
            if len(m0) == 1:
                d = gc_[metric] - float(m0[metric].iloc[0])
            else:
                idx = gc_.index.intersection(m0.index)
                d = gc_.loc[idx, metric] - m0.loc[idx, metric]
            d = d.dropna()
            p = float(wilcoxon(d)[1]) if len(d) >= 5 and np.any(d != 0) else np.nan
            prow.append(dict(Learner=L, Pipeline=P, Regime=R, Config=c, Metric=metric, N=len(d),
                             Median_Diff=float(d.median()) if len(d) else np.nan,
                             Mean_Diff=float(d.mean()) if len(d) else np.nan,
                             Frac_Positive=float((d > 0).mean()) if len(d) else np.nan,
                             Wilcoxon_p=p))
pd.DataFrame(prow).to_csv(PAIRED_FILE, index=False)

with open(os.path.join(args.out, "settings.json"), "w") as fh:
    json.dump(dict(args=vars(args), base_lgbm=BASE_LGBM, base_xgb=BASE_XGB, if_params=IF_PARAMS,
                   features=encoder.cols, categorical=encoder.cat, python=sys.version,
                   platform=platform.platform(), lightgbm=lgb.__version__, xgboost=xgb.__version__,
                   sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__), fh, indent=2)

pd.set_option("display.width", 220)
print("\n" + "=" * 80)
print("SUMMARY (AUPRC medians)")
print("=" * 80)
print(summary[grp + ["N", "Base_Median", "Base_Min", "Base_Max", "MeanVar_Median", "MeanVar_Min",
                     "MeanVar_Max"]].round(4).to_string(index=False))
print(f"\nSaved: {RUNS_FILE}\n       {SUMMARY_FILE}\n       {PAIRED_FILE}")
