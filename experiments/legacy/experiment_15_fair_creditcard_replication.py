import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# EXPERIMENT 15
# FAIR PREPROCESSING-CONTROLLED CREDITCARD REPLICATION
#
# All supervised models receive the SAME scaled predictors.
#
# M0 = Scaled LightGBM
# M1 = Scaled LightGBM + Single raw IF score (seed 42)
# M2 = Scaled LightGBM + IF Consensus
# M3 = Scaled LightGBM + Consensus + Disagreement
#
# NO hyperparameter search
# NO model selection
# NO aggregation search
#
# This experiment supersedes earlier CreditCard comparisons
# where preprocessing differed between pipelines.
# ============================================================


DATA_FILE = "creditcard.csv"

OUTPUT_DIR = os.path.join(
    "results",
    "experiment_15"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

TARGET = "Class"
TIME_COL = "Time"

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20

TARGET_FPR = 0.05

IF_SEEDS = [
    42,
    52,
    62,
    72,
    82
]

SINGLE_IF_SEED = 42


LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 42
}


IF_PARAMS = {
    "n_estimators": 200,
    "contamination": "auto",
    "n_jobs": -1
}


# ============================================================
# HELPERS
# ============================================================

def header(text):

    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)


def train_percentile(
    sorted_train_scores,
    scores
):

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    ranks = np.searchsorted(
        sorted_train_scores,
        scores,
        side="right"
    )

    return (
        ranks
        / len(sorted_train_scores)
    ).astype(np.float32)


def threshold_at_target_fpr(
    y_true,
    probabilities,
    target_fpr=0.05
):

    y_true = np.asarray(y_true)

    probabilities = np.asarray(
        probabilities
    )

    negative_scores = probabilities[
        y_true == 0
    ]

    threshold = np.quantile(
        negative_scores,
        1.0 - target_fpr
    )

    return float(threshold)


def calculate_metrics(
    y_true,
    probabilities,
    threshold
):

    y_true = np.asarray(y_true)

    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

    return {

        "AUPRC":
            average_precision_score(
                y_true,
                probabilities
            ),

        "ROC_AUC":
            roc_auc_score(
                y_true,
                probabilities
            ),

        "Precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "MCC":
            matthews_corrcoef(
                y_true,
                predictions
            ),

        "Balanced_Accuracy":
            balanced_accuracy_score(
                y_true,
                predictions
            ),

        "FPR":
            fpr,

        "Threshold":
            threshold,

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp)
    }


def create_lgbm():

    return lgb.LGBMClassifier(
        **LGBM_PARAMS
    )


def evaluate_model(
    name,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):

    print(f"\nTraining {name}...")

    model = create_lgbm()

    t0 = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    fit_sec = (
        time.perf_counter() - t0
    )


    val_prob = model.predict_proba(
        X_val
    )[:, 1]


    threshold = threshold_at_target_fpr(
        y_val,
        val_prob,
        TARGET_FPR
    )


    t0 = time.perf_counter()

    test_prob = model.predict_proba(
        X_test
    )[:, 1]

    inference_sec = (
        time.perf_counter() - t0
    )


    metrics = calculate_metrics(
        y_test,
        test_prob,
        threshold
    )


    print(
        f"  AUPRC={metrics['AUPRC']:.6f} | "
        f"ROC-AUC={metrics['ROC_AUC']:.6f} | "
        f"Recall={metrics['Recall']:.6f} | "
        f"MCC={metrics['MCC']:.6f} | "
        f"FPR={metrics['FPR']:.6f}"
    )


    return (
        model,
        test_prob,
        metrics,
        fit_sec,
        inference_sec
    )


# ============================================================
# LOAD DATA
# ============================================================

header(
    "EXPERIMENT 15 - FAIR CREDITCARD REPLICATION"
)

df = pd.read_csv(
    DATA_FILE
)


print(
    f"Rows   : {len(df):,}"
)

print(
    f"Frauds : {int(df[TARGET].sum())}"
)


# Stable chronological ordering
df = (
    df.sort_values(
        TIME_COL,
        kind="stable"
    )
    .reset_index(drop=True)
)


# ============================================================
# CHRONOLOGICAL 60 / 20 / 20
# ============================================================

n = len(df)

train_end = int(
    n * TRAIN_RATIO
)

val_end = int(
    n * (
        TRAIN_RATIO
        + VALIDATION_RATIO
    )
)


train_df = df.iloc[
    :train_end
].copy()

val_df = df.iloc[
    train_end:val_end
].copy()

test_df = df.iloc[
    val_end:
].copy()


FEATURES = [
    c
    for c in df.columns
    if c not in [
        TARGET,
        TIME_COL
    ]
]


y_train = (
    train_df[TARGET]
    .astype(np.int8)
    .values
)

y_val = (
    val_df[TARGET]
    .astype(np.int8)
    .values
)

y_test = (
    test_df[TARGET]
    .astype(np.int8)
    .values
)


print("\nSPLITS")

print(
    f"Train      : {len(y_train):,} | "
    f"Fraud={int(y_train.sum())}"
)

print(
    f"Validation : {len(y_val):,} | "
    f"Fraud={int(y_val.sum())}"
)

print(
    f"Test       : {len(y_test):,} | "
    f"Fraud={int(y_test.sum())}"
)

print(
    f"Predictors : {len(FEATURES)}"
)


# ============================================================
# RAW FEATURES
# ============================================================

X_train_raw = (
    train_df[FEATURES]
    .astype(np.float32)
    .values
)

X_val_raw = (
    val_df[FEATURES]
    .astype(np.float32)
    .values
)

X_test_raw = (
    test_df[FEATURES]
    .astype(np.float32)
    .values
)


# ============================================================
# COMMON PREPROCESSING
#
# IMPORTANT:
# SAME scaler output goes to:
# - Plain LightGBM
# - Single IF-LGBM
# - Consensus-LGBM
# - EAC-LGBM
#
# Scaler is fitted ONLY on training data.
# ============================================================

header(
    "COMMON STANDARDIZATION"
)

scaler = StandardScaler()

X_train = (
    scaler
    .fit_transform(
        X_train_raw
    )
    .astype(np.float32)
)

X_val = (
    scaler
    .transform(
        X_val_raw
    )
    .astype(np.float32)
)

X_test = (
    scaler
    .transform(
        X_test_raw
    )
    .astype(np.float32)
)


# ============================================================
# ISOLATION FORESTS
# ============================================================

header(
    "ISOLATION FOREST REPRESENTATIONS"
)


train_percentiles = []
val_percentiles = []
test_percentiles = []

single_train_raw = None
single_val_raw = None
single_test_raw = None

if_timing_rows = []


for seed in IF_SEEDS:

    print(
        f"Isolation Forest seed={seed}"
    )

    iso = IsolationForest(
        **IF_PARAMS,
        random_state=seed
    )


    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    t0 = time.perf_counter()

    iso.fit(
        X_train
    )

    fit_sec = (
        time.perf_counter() - t0
    )


    # --------------------------------------------------------
    # TRAIN SCORE
    # --------------------------------------------------------

    t0 = time.perf_counter()

    train_raw = (
        -iso.decision_function(
            X_train
        )
    ).astype(np.float32)

    train_score_sec = (
        time.perf_counter() - t0
    )


    # --------------------------------------------------------
    # VALIDATION SCORE
    # --------------------------------------------------------

    t0 = time.perf_counter()

    val_raw = (
        -iso.decision_function(
            X_val
        )
    ).astype(np.float32)

    val_score_sec = (
        time.perf_counter() - t0
    )


    # --------------------------------------------------------
    # TEST SCORE
    # --------------------------------------------------------

    t0 = time.perf_counter()

    test_raw = (
        -iso.decision_function(
            X_test
        )
    ).astype(np.float32)

    test_score_sec = (
        time.perf_counter() - t0
    )


    # --------------------------------------------------------
    # SINGLE IF
    # --------------------------------------------------------

    if seed == SINGLE_IF_SEED:

        single_train_raw = (
            train_raw.copy()
        )

        single_val_raw = (
            val_raw.copy()
        )

        single_test_raw = (
            test_raw.copy()
        )


    # --------------------------------------------------------
    # TRAIN-REFERENCED PERCENTILES
    # --------------------------------------------------------

    sorted_train = np.sort(
        train_raw.astype(
            np.float64
        )
    )


    train_pct = train_percentile(
        sorted_train,
        train_raw
    )

    val_pct = train_percentile(
        sorted_train,
        val_raw
    )

    test_pct = train_percentile(
        sorted_train,
        test_raw
    )


    train_percentiles.append(
        train_pct
    )

    val_percentiles.append(
        val_pct
    )

    test_percentiles.append(
        test_pct
    )


    if_timing_rows.append({

        "Seed":
            seed,

        "IF_Fit_sec":
            fit_sec,

        "Train_Score_sec":
            train_score_sec,

        "Validation_Score_sec":
            val_score_sec,

        "Test_Score_sec":
            test_score_sec
    })


# ============================================================
# CONSENSUS AND DISAGREEMENT
# ============================================================

P_train = np.column_stack(
    train_percentiles
)

P_val = np.column_stack(
    val_percentiles
)

P_test = np.column_stack(
    test_percentiles
)


consensus_train = np.mean(
    P_train,
    axis=1
).astype(np.float32)

consensus_val = np.mean(
    P_val,
    axis=1
).astype(np.float32)

consensus_test = np.mean(
    P_test,
    axis=1
).astype(np.float32)


disagreement_train = np.std(
    P_train,
    axis=1,
    ddof=1
).astype(np.float32)

disagreement_val = np.std(
    P_val,
    axis=1,
    ddof=1
).astype(np.float32)

disagreement_test = np.std(
    P_test,
    axis=1,
    ddof=1
).astype(np.float32)


# ============================================================
# MODEL MATRICES
# ============================================================

# M0
X0_train = X_train
X0_val = X_val
X0_test = X_test


# M1
X1_train = np.column_stack([
    X_train,
    single_train_raw
]).astype(np.float32)

X1_val = np.column_stack([
    X_val,
    single_val_raw
]).astype(np.float32)

X1_test = np.column_stack([
    X_test,
    single_test_raw
]).astype(np.float32)


# M2
X2_train = np.column_stack([
    X_train,
    consensus_train
]).astype(np.float32)

X2_val = np.column_stack([
    X_val,
    consensus_val
]).astype(np.float32)

X2_test = np.column_stack([
    X_test,
    consensus_test
]).astype(np.float32)


# M3
X3_train = np.column_stack([
    X_train,
    consensus_train,
    disagreement_train
]).astype(np.float32)

X3_val = np.column_stack([
    X_val,
    consensus_val,
    disagreement_val
]).astype(np.float32)

X3_test = np.column_stack([
    X_test,
    consensus_test,
    disagreement_test
]).astype(np.float32)


# ============================================================
# TRAIN MODELS
# ============================================================

header(
    "SUPERVISED MODELS"
)


models_to_run = [

    (
        "M0_Scaled_LGBM",
        X0_train,
        X0_val,
        X0_test
    ),

    (
        "M1_Scaled_SingleIF_LGBM",
        X1_train,
        X1_val,
        X1_test
    ),

    (
        "M2_Scaled_Consensus_LGBM",
        X2_train,
        X2_val,
        X2_test
    ),

    (
        "M3_Scaled_EAC_LGBM",
        X3_train,
        X3_val,
        X3_test
    )
]


result_rows = []
prediction_df = pd.DataFrame({

    "Class":
        y_test

})

model_timing_rows = []


for (
    model_name,
    Xt,
    Xv,
    Xte

) in models_to_run:

    (
        model,
        test_prob,
        metrics,
        fit_sec,
        infer_sec

    ) = evaluate_model(

        model_name,
        Xt,
        Xv,
        Xte,
        y_train,
        y_val,
        y_test
    )


    result_rows.append({

        "Model":
            model_name,

        **metrics
    })


    prediction_df[
        model_name
    ] = test_prob


    model_timing_rows.append({

        "Model":
            model_name,

        "LightGBM_Fit_sec":
            fit_sec,

        "LightGBM_Test_Inference_sec":
            infer_sec
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    result_rows
)


# ============================================================
# DELTA VS FAIR BASELINE
# ============================================================

baseline = results_df[
    results_df["Model"]
    == "M0_Scaled_LGBM"
].iloc[0]


comparison_rows = []


for _, row in results_df.iterrows():

    if (
        row["Model"]
        == "M0_Scaled_LGBM"
    ):
        continue


    comparison_rows.append({

        "Model":
            row["Model"],

        "Delta_AUPRC":
            (
                row["AUPRC"]
                - baseline["AUPRC"]
            ),

        "Relative_AUPRC_Change_pct":
            (
                (
                    row["AUPRC"]
                    - baseline["AUPRC"]
                )
                / baseline["AUPRC"]
                * 100
            ),

        "Delta_ROC_AUC":
            (
                row["ROC_AUC"]
                - baseline["ROC_AUC"]
            ),

        "Delta_Recall":
            (
                row["Recall"]
                - baseline["Recall"]
            ),

        "Delta_F1":
            (
                row["F1"]
                - baseline["F1"]
            ),

        "Delta_MCC":
            (
                row["MCC"]
                - baseline["MCC"]
            ),

        "Delta_FPR":
            (
                row["FPR"]
                - baseline["FPR"]
            )
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# ANOMALY REPRESENTATION DIAGNOSTICS
# ============================================================

representation_df = pd.DataFrame([

    {
        "Representation":
            "Single_IF_Raw_seed42",

        "Standalone_AUPRC":
            average_precision_score(
                y_test,
                single_test_raw
            ),

        "Standalone_ROC_AUC":
            roc_auc_score(
                y_test,
                single_test_raw
            ),

        "Fraud_Mean":
            float(
                np.mean(
                    single_test_raw[
                        y_test == 1
                    ]
                )
            ),

        "Normal_Mean":
            float(
                np.mean(
                    single_test_raw[
                        y_test == 0
                    ]
                )
            )
    },

    {
        "Representation":
            "IF_Consensus",

        "Standalone_AUPRC":
            average_precision_score(
                y_test,
                consensus_test
            ),

        "Standalone_ROC_AUC":
            roc_auc_score(
                y_test,
                consensus_test
            ),

        "Fraud_Mean":
            float(
                np.mean(
                    consensus_test[
                        y_test == 1
                    ]
                )
            ),

        "Normal_Mean":
            float(
                np.mean(
                    consensus_test[
                        y_test == 0
                    ]
                )
            )
    },

    {
        "Representation":
            "IF_Disagreement",

        "Standalone_AUPRC":
            average_precision_score(
                y_test,
                disagreement_test
            ),

        "Standalone_ROC_AUC":
            roc_auc_score(
                y_test,
                disagreement_test
            ),

        "Fraud_Mean":
            float(
                np.mean(
                    disagreement_test[
                        y_test == 1
                    ]
                )
            ),

        "Normal_Mean":
            float(
                np.mean(
                    disagreement_test[
                        y_test == 0
                    ]
                )
            )
    }

])


# ============================================================
# SANITY CHECK
# ============================================================

historical_exp08 = 0.687363

observed_baseline = float(
    baseline["AUPRC"]
)

difference_from_exp08 = (
    observed_baseline
    - historical_exp08
)


header(
    "SANITY CHECK"
)

print(
    f"Historical Exp08 scaled LGBM : "
    f"{historical_exp08:.6f}"
)

print(
    f"Experiment15 scaled LGBM     : "
    f"{observed_baseline:.6f}"
)

print(
    f"Difference                   : "
    f"{difference_from_exp08:+.9f}"
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "main_results.csv"
    ),
    index=False
)


comparison_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "comparison_vs_scaled_lgbm.csv"
    ),
    index=False
)


prediction_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_predictions.csv"
    ),
    index=False
)


representation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "anomaly_representation_diagnostics.csv"
    ),
    index=False
)


pd.DataFrame(
    if_timing_rows
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "if_timing.csv"
    ),
    index=False
)


pd.DataFrame(
    model_timing_rows
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_timing.csv"
    ),
    index=False
)


# ============================================================
# SETTINGS
# ============================================================

settings = {

    "experiment":
        (
            "Experiment 15 - "
            "Fair Preprocessing-Controlled "
            "CreditCard Replication"
        ),

    "dataset":
        DATA_FILE,

    "purpose":
        (
            "Re-evaluate anomaly augmentation on "
            "CreditCard under identical preprocessing "
            "for all LightGBM models."
        ),

    "supersedes":
        (
            "Earlier CreditCard performance comparisons "
            "that used raw predictors for LightGBM."
        ),

    "split":
        "Chronological 60/20/20",

    "sorting":
        "Stable sort by Time",

    "time_used_as_predictor":
        False,

    "preprocessing":
        (
            "StandardScaler fitted on training predictors "
            "and applied identically to every supervised "
            "model."
        ),

    "models": {

        "M0":
            "Scaled LightGBM",

        "M1":
            (
                "Scaled LightGBM + "
                "single raw IF anomaly score"
            ),

        "M2":
            (
                "Scaled LightGBM + "
                "five-IF consensus"
            ),

        "M3":
            (
                "Scaled LightGBM + "
                "five-IF consensus + disagreement"
            )
    },

    "IF_seeds":
        IF_SEEDS,

    "single_IF_seed":
        SINGLE_IF_SEED,

    "IF_parameters":
        IF_PARAMS,

    "LightGBM_parameters":
        LGBM_PARAMS,

    "consensus":
        (
            "Arithmetic mean of five "
            "train-referenced IF anomaly percentiles"
        ),

    "disagreement":
        (
            "Sample standard deviation of five "
            "train-referenced IF anomaly percentiles"
        ),

    "threshold":
        (
            "Model-specific threshold selected on "
            "validation negatives for approximately "
            "5% FPR and frozen on test."
        ),

    "primary_metric":
        "AUPRC",

    "SMOTE":
        False,

    "hyperparameter_search":
        False,

    "model_selection":
        False,

    "aggregation_search":
        False
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "experiment_settings.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        settings,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

header(
    "EXPERIMENT 15 RESULTS"
)

print(
    "\nMAIN RESULTS\n"
)

print(
    results_df
    .round(6)
    .to_string(index=False)
)


print(
    "\nCOMPARISON VS FAIR SCALED BASELINE\n"
)

print(
    comparison_df
    .round(6)
    .to_string(index=False)
)


print(
    "\nANOMALY REPRESENTATION DIAGNOSTICS\n"
)

print(
    representation_df
    .round(6)
    .to_string(index=False)
)


print(
    "\nResults saved to:"
)

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)

print(
    "\nEXPERIMENT 15 COMPLETE."
)