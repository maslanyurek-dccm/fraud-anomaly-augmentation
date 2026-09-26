import os
import gc
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
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

import lightgbm as lgb
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# EXPERIMENT 08
# EXTERNAL VALIDATION ON CREDITCARD.CSV
#
# Independent dataset validation.
#
# IMPORTANT:
# BAF-trained models are NOT transferred to creditcard.csv.
#
# The methodological idea is independently evaluated:
#
# LightGBM
# versus
# Isolation Forest anomaly score + LightGBM
#
# Reference models are also included.
#
# Temporal split based on Time:
#   First 60%  -> Train
#   Next 20%   -> Validation
#   Last 20%   -> Test
#
# Threshold:
# Validation-derived ~5% FPR
#
# No SMOTE.
# No hyperparameter search.
# No hypothesis testing.
# ============================================================


DATA_FILE = "creditcard.csv"

TARGET = "Class"
TIME_COL = "Time"

SEEDS = [42, 52, 62, 72, 82]

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

TARGET_FPR = 0.05

OUTPUT_DIR = os.path.join(
    "results",
    "experiment_08"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FIXED PARAMETERS
# ============================================================

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_jobs": -1,
    "verbosity": -1
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


def find_threshold_at_fpr(
    y_true,
    probabilities,
    target_fpr=0.05
):

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

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
    probabilities = np.asarray(probabilities)

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


def create_models(seed):

    return {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=seed,
                n_jobs=-1
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced"
            ),

        "Extra Trees":
            ExtraTreesClassifier(
                n_estimators=300,
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced"
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1
            ),

        "LightGBM":
            lgb.LGBMClassifier(
                **LGBM_PARAMS,
                random_state=seed
            )
    }


# ============================================================
# LOAD DATA
# ============================================================

header(
    "EXPERIMENT 08 - EXTERNAL VALIDATION"
)

print(
    "Loading creditcard.csv..."
)

start = time.time()

df = pd.read_csv(
    DATA_FILE
)

print(
    f"Loaded in "
    f"{time.time()-start:.2f} sec"
)

print(
    f"Rows    : {len(df):,}"
)

print(
    f"Columns : {len(df.columns)}"
)

print(
    f"Frauds  : "
    f"{int(df[TARGET].sum()):,}"
)

print(
    f"Fraud rate: "
    f"{df[TARGET].mean()*100:.6f}%"
)


# ============================================================
# SORT BY TIME
# ============================================================

df = (
    df.sort_values(
        TIME_COL
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# TEMPORAL SPLIT
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


header(
    "TEMPORAL SPLIT"
)

print(
    f"Train      : "
    f"{len(train_df):,}"
)

print(
    f"Validation : "
    f"{len(val_df):,}"
)

print(
    f"Test       : "
    f"{len(test_df):,}"
)

print()

print(
    f"Train fraud rate : "
    f"{train_df[TARGET].mean()*100:.6f}%"
)

print(
    f"Val fraud rate   : "
    f"{val_df[TARGET].mean()*100:.6f}%"
)

print(
    f"Test fraud rate  : "
    f"{test_df[TARGET].mean()*100:.6f}%"
)

print()

print(
    f"Train Time range : "
    f"{train_df[TIME_COL].min()} "
    f"- "
    f"{train_df[TIME_COL].max()}"
)

print(
    f"Val Time range   : "
    f"{val_df[TIME_COL].min()} "
    f"- "
    f"{val_df[TIME_COL].max()}"
)

print(
    f"Test Time range  : "
    f"{test_df[TIME_COL].min()} "
    f"- "
    f"{test_df[TIME_COL].max()}"
)


# ============================================================
# FEATURES
#
# Time is used ONLY for temporal splitting.
# It is not supplied to the models.
# ============================================================

FEATURE_COLUMNS = [
    col for col in df.columns
    if col not in [
        TARGET,
        TIME_COL
    ]
]


X_train_raw = (
    train_df[
        FEATURE_COLUMNS
    ]
    .astype(np.float32)
    .values
)

X_val_raw = (
    val_df[
        FEATURE_COLUMNS
    ]
    .astype(np.float32)
    .values
)

X_test_raw = (
    test_df[
        FEATURE_COLUMNS
    ]
    .astype(np.float32)
    .values
)


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


# ============================================================
# STANDARDIZATION
#
# Fit ONLY on training data.
# ============================================================

header(
    "PREPROCESSING"
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

print(
    f"Predictors: "
    f"{len(FEATURE_COLUMNS)}"
)

del df
del train_df
del val_df
del test_df

del X_train_raw
del X_val_raw
del X_test_raw

gc.collect()


# ============================================================
# DATASET INFO
# ============================================================

dataset_info = pd.DataFrame([

    {
        "Split":
            "Train",

        "Rows":
            len(y_train),

        "Fraud_Count":
            int(y_train.sum()),

        "Fraud_Rate":
            float(y_train.mean())
    },

    {
        "Split":
            "Validation",

        "Rows":
            len(y_val),

        "Fraud_Count":
            int(y_val.sum()),

        "Fraud_Rate":
            float(y_val.mean())
    },

    {
        "Split":
            "Test",

        "Rows":
            len(y_test),

        "Fraud_Count":
            int(y_test.sum()),

        "Fraud_Rate":
            float(y_test.mean())
    }

])

dataset_info.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dataset_info.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# EXPERIMENT
# ============================================================

all_results = []
timing_results = []


for seed_index, seed in enumerate(
    SEEDS,
    start=1
):

    header(
        f"SEED {seed} "
        f"({seed_index}/{len(SEEDS)})"
    )


    # ========================================================
    # STANDARD MODELS
    # ========================================================

    models = create_models(
        seed
    )

    for model_name, model in (
        models.items()
    ):

        print(
            f"\nTraining "
            f"{model_name}..."
        )

        start = time.time()

        model.fit(
            X_train,
            y_train
        )

        train_time = (
            time.time()
            - start
        )


        # ====================================================
        # VALIDATION THRESHOLD
        # ====================================================

        val_prob = (
            model.predict_proba(
                X_val
            )[:, 1]
        )

        threshold = (
            find_threshold_at_fpr(
                y_val,
                val_prob,
                TARGET_FPR
            )
        )


        # ====================================================
        # TEST
        # ====================================================

        start = time.time()

        test_prob = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        inference_time = (
            time.time()
            - start
        )

        metrics = (
            calculate_metrics(
                y_test,
                test_prob,
                threshold
            )
        )

        all_results.append({

            "Seed":
                seed,

            "Model":
                model_name,

            "Dataset":
                "CreditCard Test",

            **metrics
        })

        timing_results.append({

            "Seed":
                seed,

            "Model":
                model_name,

            "Model_Training_sec":
                train_time,

            "IF_Fit_sec":
                0.0,

            "Total_Training_Cost_sec":
                train_time,

            "Test_Inference_sec":
                inference_time
        })

        print(
            f"  AUPRC="
            f"{metrics['AUPRC']:.6f} | "
            f"ROC-AUC="
            f"{metrics['ROC_AUC']:.6f} | "
            f"Recall="
            f"{metrics['Recall']:.6f} | "
            f"F1="
            f"{metrics['F1']:.6f} | "
            f"FPR="
            f"{metrics['FPR']:.6f}"
        )

        del model
        gc.collect()


    # ========================================================
    # PROPOSED IF-LGBM
    # ========================================================

    print(
        "\nTraining Proposed IF-LGBM..."
    )


    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    iso = IsolationForest(
        **IF_PARAMS,
        random_state=seed
    )

    start = time.time()

    iso.fit(
        X_train
    )

    if_fit_time = (
        time.time()
        - start
    )


    # ========================================================
    # CREATE ANOMALY SCORE
    # ========================================================

    train_if = (
        -iso.decision_function(
            X_train
        )
    ).astype(np.float32)

    val_if = (
        -iso.decision_function(
            X_val
        )
    ).astype(np.float32)


    X_train_if = (
        np.column_stack(
            [
                X_train,
                train_if
            ]
        )
        .astype(np.float32)
    )

    X_val_if = (
        np.column_stack(
            [
                X_val,
                val_if
            ]
        )
        .astype(np.float32)
    )


    # ========================================================
    # LIGHTGBM
    # ========================================================

    proposed = (
        lgb.LGBMClassifier(
            **LGBM_PARAMS,
            random_state=seed
        )
    )

    start = time.time()

    proposed.fit(
        X_train_if,
        y_train
    )

    model_train_time = (
        time.time()
        - start
    )


    # ========================================================
    # VALIDATION THRESHOLD
    # ========================================================

    val_prob = (
        proposed.predict_proba(
            X_val_if
        )[:, 1]
    )

    threshold = (
        find_threshold_at_fpr(
            y_val,
            val_prob,
            TARGET_FPR
        )
    )


    # ========================================================
    # TEST - END TO END INFERENCE
    #
    # Includes:
    # Isolation Forest score generation
    # +
    # LightGBM prediction
    # ========================================================

    start = time.time()

    test_if = (
        -iso.decision_function(
            X_test
        )
    ).astype(np.float32)

    X_test_if = (
        np.column_stack(
            [
                X_test,
                test_if
            ]
        )
        .astype(np.float32)
    )

    test_prob = (
        proposed.predict_proba(
            X_test_if
        )[:, 1]
    )

    inference_time = (
        time.time()
        - start
    )


    metrics = (
        calculate_metrics(
            y_test,
            test_prob,
            threshold
        )
    )


    all_results.append({

        "Seed":
            seed,

        "Model":
            "Proposed IF-LGBM",

        "Dataset":
            "CreditCard Test",

        **metrics
    })


    timing_results.append({

        "Seed":
            seed,

        "Model":
            "Proposed IF-LGBM",

        "Model_Training_sec":
            model_train_time,

        "IF_Fit_sec":
            if_fit_time,

        "Total_Training_Cost_sec":
            (
                model_train_time
                + if_fit_time
            ),

        "Test_Inference_sec":
            inference_time
    })


    print(
        f"  AUPRC="
        f"{metrics['AUPRC']:.6f} | "
        f"ROC-AUC="
        f"{metrics['ROC_AUC']:.6f} | "
        f"Recall="
        f"{metrics['Recall']:.6f} | "
        f"F1="
        f"{metrics['F1']:.6f} | "
        f"FPR="
        f"{metrics['FPR']:.6f}"
    )


    del iso
    del proposed

    del train_if
    del val_if
    del test_if

    del X_train_if
    del X_val_if
    del X_test_if

    gc.collect()


# ============================================================
# RESULTS
# ============================================================

header(
    "CREATING RESULTS"
)

results_df = pd.DataFrame(
    all_results
)

timing_df = pd.DataFrame(
    timing_results
)


results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_results.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

timing_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "timing.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

METRICS = [
    "AUPRC",
    "ROC_AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
    "Balanced_Accuracy",
    "FPR"
]


summary = (
    results_df
    .groupby("Model")[
        METRICS
    ]
    .agg(
        ["mean", "std"]
    )
    .reset_index()
)

summary.columns = [
    "_".join(
        [
            str(x)
            for x in col
            if str(x) != ""
        ]
    )
    if isinstance(col, tuple)
    else col

    for col in summary.columns
]

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# TIMING SUMMARY
# ============================================================

timing_summary = (
    timing_df
    .groupby("Model")[
        [
            "Model_Training_sec",
            "IF_Fit_sec",
            "Total_Training_Cost_sec",
            "Test_Inference_sec"
        ]
    ]
    .agg(
        ["mean", "std"]
    )
    .reset_index()
)

timing_summary.columns = [
    "_".join(
        [
            str(x)
            for x in col
            if str(x) != ""
        ]
    )
    if isinstance(col, tuple)
    else col

    for col in timing_summary.columns
]

timing_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "timing_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# IF-LGBM VS LIGHTGBM
#
# This is the key external validation comparison.
# ============================================================

comparison_rows = []


for seed in SEEDS:

    lgbm_row = results_df[
        (results_df["Seed"] == seed)
        &
        (results_df["Model"] == "LightGBM")
    ].iloc[0]

    if_row = results_df[
        (results_df["Seed"] == seed)
        &
        (
            results_df["Model"]
            == "Proposed IF-LGBM"
        )
    ].iloc[0]


    comparison_rows.append({

        "Seed":
            seed,

        "LightGBM_AUPRC":
            lgbm_row["AUPRC"],

        "IF_LGBM_AUPRC":
            if_row["AUPRC"],

        "Delta_AUPRC":
            (
                if_row["AUPRC"]
                - lgbm_row["AUPRC"]
            ),

        "LightGBM_ROC_AUC":
            lgbm_row["ROC_AUC"],

        "IF_LGBM_ROC_AUC":
            if_row["ROC_AUC"],

        "Delta_ROC_AUC":
            (
                if_row["ROC_AUC"]
                - lgbm_row["ROC_AUC"]
            ),

        "LightGBM_Recall":
            lgbm_row["Recall"],

        "IF_LGBM_Recall":
            if_row["Recall"],

        "Delta_Recall":
            (
                if_row["Recall"]
                - lgbm_row["Recall"]
            ),

        "LightGBM_F1":
            lgbm_row["F1"],

        "IF_LGBM_F1":
            if_row["F1"],

        "Delta_F1":
            (
                if_row["F1"]
                - lgbm_row["F1"]
            ),

        "LightGBM_MCC":
            lgbm_row["MCC"],

        "IF_LGBM_MCC":
            if_row["MCC"],

        "Delta_MCC":
            (
                if_row["MCC"]
                - lgbm_row["MCC"]
            ),

        "LightGBM_FPR":
            lgbm_row["FPR"],

        "IF_LGBM_FPR":
            if_row["FPR"],

        "Delta_FPR":
            (
                if_row["FPR"]
                - lgbm_row["FPR"]
            )
    })


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "if_lgbm_vs_lightgbm.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# DESCRIPTIVE COMPARISON SUMMARY
# ============================================================

comparison_summary = pd.DataFrame([{

    "Mean_Delta_AUPRC":
        comparison_df[
            "Delta_AUPRC"
        ].mean(),

    "Positive_AUPRC_Seeds":
        int(
            (
                comparison_df[
                    "Delta_AUPRC"
                ] > 0
            ).sum()
        ),

    "Mean_Delta_ROC_AUC":
        comparison_df[
            "Delta_ROC_AUC"
        ].mean(),

    "Positive_ROC_AUC_Seeds":
        int(
            (
                comparison_df[
                    "Delta_ROC_AUC"
                ] > 0
            ).sum()
        ),

    "Mean_Delta_Recall":
        comparison_df[
            "Delta_Recall"
        ].mean(),

    "Positive_Recall_Seeds":
        int(
            (
                comparison_df[
                    "Delta_Recall"
                ] > 0
            ).sum()
        ),

    "Mean_Delta_F1":
        comparison_df[
            "Delta_F1"
        ].mean(),

    "Positive_F1_Seeds":
        int(
            (
                comparison_df[
                    "Delta_F1"
                ] > 0
            ).sum()
        ),

    "Mean_Delta_MCC":
        comparison_df[
            "Delta_MCC"
        ].mean(),

    "Positive_MCC_Seeds":
        int(
            (
                comparison_df[
                    "Delta_MCC"
                ] > 0
            ).sum()
        )
}])


comparison_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "if_lgbm_comparison_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SETTINGS
# ============================================================

settings = {

    "experiment":
        "Experiment 08 - External Validation",

    "dataset":
        "creditcard.csv",

    "purpose":
        (
            "Independent evaluation of the "
            "Isolation Forest anomaly-score "
            "augmentation strategy on a second "
            "fraud dataset."
        ),

    "important_note":
        (
            "Models trained on BAF are not "
            "transferred to creditcard.csv. "
            "The methodological strategy is "
            "independently retrained and evaluated."
        ),

    "seeds":
        SEEDS,

    "split_protocol":
        "Chronological 60/20/20",

    "train_ratio":
        TRAIN_RATIO,

    "validation_ratio":
        VALIDATION_RATIO,

    "test_ratio":
        TEST_RATIO,

    "time_column":
        TIME_COL,

    "time_used_as_predictor":
        False,

    "target":
        TARGET,

    "target_fpr":
        TARGET_FPR,

    "threshold_protocol":
        (
            "Threshold selected on validation "
            "negative examples to target "
            "approximately 5% FPR and frozen "
            "for the chronological test set."
        ),

    "primary_metric":
        "AUPRC",

    "models": [
        "Logistic Regression",
        "Random Forest",
        "Extra Trees",
        "XGBoost",
        "LightGBM",
        "Proposed IF-LGBM"
    ],

    "proposed_method":
        "Isolation Forest raw anomaly score + LightGBM",

    "SMOTE":
        False,

    "hyperparameter_search":
        False,

    "hypothesis_testing":
        False,

    "IF_parameters":
        IF_PARAMS,

    "LightGBM_parameters":
        LGBM_PARAMS
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
# FINAL SCREEN OUTPUT
# ============================================================

header(
    "EXPERIMENT 08 COMPLETE"
)

print(
    "\nDATASET SPLITS\n"
)

print(
    dataset_info
    .round(6)
    .to_string(
        index=False
    )
)

print(
    "\nMODEL SUMMARY\n"
)

print(
    summary
    .round(6)
    .to_string(
        index=False
    )
)

print(
    "\nIF-LGBM VS LIGHTGBM\n"
)

print(
    comparison_df
    .round(6)
    .to_string(
        index=False
    )
)

print(
    "\nDESCRIPTIVE EXTERNAL VALIDATION SUMMARY\n"
)

print(
    comparison_summary
    .round(6)
    .to_string(
        index=False
    )
)

print(
    "\nFiles saved to:"
)

print(
    OUTPUT_DIR
)