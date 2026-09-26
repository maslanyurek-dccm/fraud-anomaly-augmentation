import os
import gc
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# EXPERIMENT 02
# Isolation Forest Diagnostic Analysis
#
# Goal:
# Determine whether the unsupervised IF anomaly score itself
# contains useful fraud-discrimination information.
#
# Protocol:
# Base Month 0-5 -> IF training
# Base Month 6   -> Validation diagnostic
# Base Month 7   -> Test diagnostic
# Variant I-V    -> Distribution-shift diagnostic
#
# IMPORTANT:
# IF is fitted ONLY ONCE on Base Month 0-5.
# ============================================================

SEED = 42
TARGET = "fraud_bool"
TIME_COL = "month"

BASE_FILE = "Base.csv"

VARIANTS = {
    "Variant I": "Variant I.csv",
    "Variant II": "Variant II.csv",
    "Variant III": "Variant III.csv",
    "Variant IV": "Variant IV.csv",
    "Variant V": "Variant V.csv",
}

OUTPUT_DIR = os.path.join("results", "experiment_02")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IF_PARAMS = {
    "n_estimators": 200,
    "contamination": "auto",
    "random_state": SEED,
    "n_jobs": -1
}

TOP_PERCENTAGES = [0.01, 0.02, 0.05, 0.10]


# ============================================================
# HELPERS
# ============================================================

def header(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def align_columns(df, feature_columns):

    missing = [
        c for c in feature_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing Base features: {missing}"
        )

    # Extra Variant columns (e.g. x1/x2) are ignored
    return df[feature_columns].copy()


# ============================================================
# PREPROCESSOR
# ============================================================

class BAFPreprocessor:

    def __init__(self):

        self.feature_columns = None
        self.categorical_columns = None
        self.encoder = None
        self.scaler = None

    def fit(self, X):

        self.feature_columns = list(X.columns)

        self.categorical_columns = (
            X.select_dtypes(
                include=["object", "category"]
            )
            .columns
            .tolist()
        )

        if self.categorical_columns:

            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float32
            )

            self.encoder.fit(
                X[self.categorical_columns].astype(str)
            )

        encoded = self._encode(X)

        self.scaler = StandardScaler()
        self.scaler.fit(encoded)

        del encoded
        gc.collect()

        return self

    def _encode(self, X):

        X = align_columns(
            X,
            self.feature_columns
        )

        output = np.zeros(
            (len(X), len(self.feature_columns)),
            dtype=np.float32
        )

        categorical_set = set(
            self.categorical_columns
        )

        if self.categorical_columns:

            encoded_cat = self.encoder.transform(
                X[self.categorical_columns].astype(str)
            )

            cat_positions = {
                col: i
                for i, col in
                enumerate(self.categorical_columns)
            }

        for j, col in enumerate(self.feature_columns):

            if col in categorical_set:

                output[:, j] = encoded_cat[
                    :, cat_positions[col]
                ]

            else:

                output[:, j] = pd.to_numeric(
                    X[col],
                    errors="coerce"
                ).fillna(0).astype(np.float32).values

        return output

    def transform(self, X):

        encoded = self._encode(X)

        scaled = self.scaler.transform(
            encoded
        ).astype(np.float32)

        del encoded
        gc.collect()

        return scaled


# ============================================================
# DIAGNOSTIC FUNCTION
# ============================================================

def analyze_scores(dataset_name, y, scores):

    y = np.asarray(y)
    scores = np.asarray(scores)

    fraud_scores = scores[y == 1]
    normal_scores = scores[y == 0]

    prevalence = y.mean()

    # --------------------------------------------
    # Discrimination metrics
    # --------------------------------------------

    roc_auc = roc_auc_score(
        y,
        scores
    )

    auprc = average_precision_score(
        y,
        scores
    )

    # --------------------------------------------
    # Score statistics
    # --------------------------------------------

    summary = {
        "Dataset": dataset_name,
        "N": len(y),
        "Fraud_Count": int(y.sum()),
        "Fraud_Prevalence": prevalence,

        "IF_ROC_AUC": roc_auc,
        "IF_AUPRC": auprc,

        "Overall_Score_Mean":
            float(np.mean(scores)),

        "Normal_Score_Mean":
            float(np.mean(normal_scores)),

        "Fraud_Score_Mean":
            float(np.mean(fraud_scores)),

        "Normal_Score_Median":
            float(np.median(normal_scores)),

        "Fraud_Score_Median":
            float(np.median(fraud_scores)),

        "Mean_Score_Difference":
            float(
                np.mean(fraud_scores)
                - np.mean(normal_scores)
            ),

        "Median_Score_Difference":
            float(
                np.median(fraud_scores)
                - np.median(normal_scores)
            )
    }

    enrichment_rows = []

    # --------------------------------------------
    # Top anomaly enrichment
    # --------------------------------------------

    for p in TOP_PERCENTAGES:

        threshold = np.quantile(
            scores,
            1.0 - p
        )

        mask = scores >= threshold

        selected_n = int(mask.sum())

        selected_fraud = int(
            y[mask].sum()
        )

        selected_rate = (
            y[mask].mean()
            if selected_n > 0
            else np.nan
        )

        enrichment = (
            selected_rate / prevalence
            if prevalence > 0
            else np.nan
        )

        fraud_capture = (
            selected_fraud / y.sum()
            if y.sum() > 0
            else np.nan
        )

        enrichment_rows.append({
            "Dataset": dataset_name,
            "Top_Percent": p * 100,
            "Threshold": threshold,
            "Selected_N": selected_n,
            "Fraud_in_Selected":
                selected_fraud,
            "Dataset_Fraud_Rate":
                prevalence,
            "Selected_Fraud_Rate":
                selected_rate,
            "Enrichment_Factor":
                enrichment,
            "Fraud_Capture_Rate":
                fraud_capture
        })

    return summary, enrichment_rows


# ============================================================
# LOAD BASE
# ============================================================

header("EXPERIMENT 02 - IF DIAGNOSTIC")

print("Loading Base.csv...")

start = time.time()

base = pd.read_csv(BASE_FILE)

print(
    f"Loaded in {time.time()-start:.2f} sec"
)

# Temporal partitions
train = base[
    base[TIME_COL].isin([0, 1, 2, 3, 4, 5])
].copy()

validation = base[
    base[TIME_COL] == 6
].copy()

test = base[
    base[TIME_COL] == 7
].copy()

print(f"Train      : {len(train):,}")
print(f"Validation : {len(validation):,}")
print(f"Test       : {len(test):,}")


# ============================================================
# PREPROCESSING
# ============================================================

header("PREPROCESSING")

X_train_raw = train.drop(
    columns=[TARGET]
)

y_train = train[TARGET].astype(np.int8).values

X_val_raw = validation.drop(
    columns=[TARGET]
)

y_val = validation[TARGET].astype(np.int8).values

X_test_raw = test.drop(
    columns=[TARGET]
)

y_test = test[TARGET].astype(np.int8).values

preprocessor = BAFPreprocessor()

print(
    "Fitting preprocessing on Base Month 0-5 only..."
)

preprocessor.fit(
    X_train_raw
)

feature_columns = preprocessor.feature_columns

print("Transforming Train...")
X_train = preprocessor.transform(
    X_train_raw
)

print("Transforming Validation...")
X_val = preprocessor.transform(
    X_val_raw
)

print("Transforming Test...")
X_test = preprocessor.transform(
    X_test_raw
)

del base
del train
del validation
del test

del X_train_raw
del X_val_raw
del X_test_raw

gc.collect()


# ============================================================
# FIT ISOLATION FOREST
# ============================================================

header("FITTING ISOLATION FOREST")

iso = IsolationForest(
    **IF_PARAMS
)

start = time.time()

iso.fit(X_train)

fit_time = time.time() - start

print(
    f"Isolation Forest fit time: "
    f"{fit_time:.2f} sec"
)

# IMPORTANT:
# sklearn decision_function:
# lower = more anomalous.
#
# We multiply by -1:
# higher = more anomalous.

train_scores = (
    -iso.decision_function(X_train)
).astype(np.float32)

val_scores = (
    -iso.decision_function(X_val)
).astype(np.float32)

test_scores = (
    -iso.decision_function(X_test)
).astype(np.float32)


# ============================================================
# BASE DIAGNOSTICS
# ============================================================

summary_rows = []
enrichment_rows = []

for name, y, scores in [

    (
        "Base Train (Month 0-5)",
        y_train,
        train_scores
    ),

    (
        "Base Validation (Month 6)",
        y_val,
        val_scores
    ),

    (
        "Base Test (Month 7)",
        y_test,
        test_scores
    )
]:

    summary, enrichment = analyze_scores(
        name,
        y,
        scores
    )

    summary_rows.append(summary)
    enrichment_rows.extend(enrichment)


# We no longer need these large matrices
del X_val
del X_test
del val_scores
del test_scores

gc.collect()


# ============================================================
# VARIANT DIAGNOSTICS
# ============================================================

header("VARIANT DIAGNOSTICS")

for variant_name, file_name in VARIANTS.items():

    print(
        f"\nProcessing {variant_name}..."
    )

    start = time.time()

    df = pd.read_csv(file_name)

    print(
        f"Loaded {len(df):,} rows "
        f"in {time.time()-start:.2f} sec"
    )

    y = (
        df[TARGET]
        .astype(np.int8)
        .values
    )

    X_raw = df.drop(
        columns=[TARGET]
    )

    # Only Base-known features
    X_raw = align_columns(
        X_raw,
        feature_columns
    )

    X = preprocessor.transform(
        X_raw
    )

    scores = (
        -iso.decision_function(X)
    ).astype(np.float32)

    summary, enrichment = analyze_scores(
        variant_name,
        y,
        scores
    )

    summary_rows.append(summary)
    enrichment_rows.extend(enrichment)

    print(
        f"IF ROC-AUC = "
        f"{summary['IF_ROC_AUC']:.4f}"
    )

    print(
        f"IF AUPRC   = "
        f"{summary['IF_AUPRC']:.4f}"
    )

    print(
        f"Fraud mean score  = "
        f"{summary['Fraud_Score_Mean']:.6f}"
    )

    print(
        f"Normal mean score = "
        f"{summary['Normal_Score_Mean']:.6f}"
    )

    del df
    del y
    del X_raw
    del X
    del scores

    gc.collect()


# ============================================================
# SAVE RESULTS
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

enrichment_df = pd.DataFrame(
    enrichment_rows
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "if_score_summary.csv"
)

enrichment_path = os.path.join(
    OUTPUT_DIR,
    "if_enrichment_analysis.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)

enrichment_df.to_csv(
    enrichment_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# CREATE COMPACT ENRICHMENT TABLE
# ============================================================

enrichment_pivot = enrichment_df.pivot(
    index="Dataset",
    columns="Top_Percent",
    values="Enrichment_Factor"
)

enrichment_pivot.columns = [
    f"Top_{int(c)}pct_Enrichment"
    for c in enrichment_pivot.columns
]

enrichment_pivot = (
    enrichment_pivot
    .reset_index()
)

pivot_path = os.path.join(
    OUTPUT_DIR,
    "if_enrichment_table.csv"
)

enrichment_pivot.to_csv(
    pivot_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PRINT RESULTS
# ============================================================

header("IF SCORE SUMMARY")

display_columns = [
    "Dataset",
    "Fraud_Prevalence",
    "IF_ROC_AUC",
    "IF_AUPRC",
    "Normal_Score_Mean",
    "Fraud_Score_Mean",
    "Mean_Score_Difference"
]

print(
    summary_df[
        display_columns
    ].round(6).to_string(index=False)
)


header("ANOMALY ENRICHMENT")

print(
    enrichment_pivot
    .round(3)
    .to_string(index=False)
)


header("EXPERIMENT 02 COMPLETE")

print(
    f"IF fit time: "
    f"{fit_time:.2f} sec"
)

print("\nCreated files:")

print(
    f"  {summary_path}"
)

print(
    f"  {enrichment_path}"
)

print(
    f"  {pivot_path}"
)

print(
    "\nSend these three CSV files for analysis."
)