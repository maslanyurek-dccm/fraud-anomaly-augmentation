"""Experiment 17: shift of the anomaly features on CreditCard.

Compares the PSI of the anomaly features computed with in-sample training
scores and with two-fold cross-fitted training scores, and measures the shift
between the two halves of the training period.
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join("data", "creditcard.csv"))
ap.add_argument("--out", default=os.path.join("results", "experiment_17"))
ap.add_argument("--replicates", type=int, default=5)
ap.add_argument("--n_jobs", type=int, default=-1)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

TARGET, TIME_COL = "Class", "Time"
IF_PARAMS = dict(n_estimators=200, contamination="auto", n_jobs=args.n_jobs)


def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-6, None), np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))


# Data (same protocol as experiments 12-16)
df = pd.read_csv(args.data).sort_values(TIME_COL, kind="stable").reset_index(drop=True)
n = len(df)
tr_end, va_end = int(n * 0.60), int(n * 0.80)
FEATURES = [c for c in df.columns if c not in (TARGET, TIME_COL)]
X = df[FEATURES].astype(np.float32).values
y = df[TARGET].values
sc = StandardScaler().fit(X[:tr_end])
XS = {"train": sc.transform(X[:tr_end]).astype(np.float32),
      "val": sc.transform(X[tr_end:va_end]).astype(np.float32),
      "test": sc.transform(X[va_end:]).astype(np.float32)}
Y = {"train": y[:tr_end], "val": y[tr_end:va_end], "test": y[va_end:]}
half = len(Y["train"]) // 2
folds = list(KFold(n_splits=2, shuffle=True, random_state=0).split(XS["train"]))


# Scorers
def fit_if(Xfit, seed):
    m = IsolationForest(**IF_PARAMS, random_state=seed).fit(Xfit)
    return lambda Z: -m.decision_function(Z)


def fit_maha(Xfit):
    Xf = Xfit.astype(np.float64)
    mu, prec = Xf.mean(0), np.linalg.pinv(np.cov(Xf, rowvar=False))

    def score(Z):
        D = Z.astype(np.float64) - mu
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", D, prec, D), 0))
    return score


def evaluate(name, make_scorer):
    full = make_scorer(XS["train"])
    s_in = full(XS["train"])
    s_cf = np.empty(len(s_in))
    for a, b in folds:
        s_cf[b] = make_scorer(XS["train"][a])(XS["train"][b])
    s_val, s_te = full(XS["val"]), full(XS["test"])
    nrm = {k: Y[k] == 0 for k in Y}
    row = dict(Scorer=name)
    for tag, s_tr in [("insample", s_in), ("crossfit", s_cf)]:
        row[f"PSI_train_to_val_{tag}"] = psi(s_tr, s_val)
        row[f"PSI_train_to_test_{tag}"] = psi(s_tr, s_te)
        row[f"PSI_train_to_test_normals_{tag}"] = psi(s_tr[nrm["train"]], s_te[nrm["test"]])
        row[f"Median_train_normal_{tag}"] = float(np.median(s_tr[nrm["train"]]))
    row["Median_test_normal"] = float(np.median(s_te[nrm["test"]]))
    row["PSI_trainhalf1_to_trainhalf2_crossfit"] = psi(s_cf[:half], s_cf[half:])
    row["Mean_abs_diff_insample_vs_crossfit"] = float(np.mean(np.abs(s_in - s_cf)))
    return row


rows = []
for r in range(1, args.replicates + 1):
    seed = 100 * r + 1
    rows.append(evaluate(f"IF_seed_{seed}", lambda Xf, s=seed: fit_if(Xf, s)))
    print(f"IF seed {seed} done", flush=True)
rows.append(evaluate("Mahalanobis", fit_maha))

res = pd.DataFrame(rows)
path = os.path.join(args.out, "crossfit_psi.csv")
res.to_csv(path, index=False)
pd.set_option("display.width", 200)
cols = ["Scorer", "PSI_train_to_val_insample", "PSI_train_to_val_crossfit",
        "PSI_train_to_test_insample", "PSI_train_to_test_crossfit",
        "PSI_trainhalf1_to_trainhalf2_crossfit"]
print(res[cols].round(3).to_string(index=False))
print(f"\nSaved: {path}")
