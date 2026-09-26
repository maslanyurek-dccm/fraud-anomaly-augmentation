import os
import gc
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
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

import lightgbm as lgb

warnings.filterwarnings("ignore")


# ============================================================
# EXPERIMENT 04
# Multi-Seed Confirmation
#
# M1 = LightGBM
# M2 = LightGBM + Raw Isolation Forest Score
#
# Seeds:
# 42, 52, 62, 72, 82
#
# Temporal protocol:
# Month 0-5 -> Train
# Month 6   -> Validation
# Month 7   -> Test
#
# month is NOT used as predictor.
#
# Threshold:
# Learned on validation negatives at ~5% FPR
# and frozen for Test + Variants.
# ============================================================


SEEDS = [
    42,
    52,
    62,
    72,
    82
]

TARGET = "fraud_bool"
TIME_COL = "month"
TARGET_FPR = 0.05

BASE_FILE = "Base.csv"

VARIANTS = {
    "Variant I": "Variant I.csv",
    "Variant II": "Variant II.csv",
    "Variant III": "Variant III.csv",
    "Variant IV": "Variant IV.csv",
    "Variant V": "Variant V.csv"
}

OUTPUT_DIR = os.path.join(
    "results",
    "experiment_04"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FIXED PARAMETERS
# ============================================================

LGBM_BASE_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_jobs": -1,
    "verbosity": -1
}

IF_BASE_PARAMS = {
    "n_estimators": 200,
    "contamination": "auto",
    "n_jobs": -1
}


# ============================================================
# HELPERS
# ============================================================

def header(text):

    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def align_columns(
    df,
    feature_columns
):

    missing = [
        c for c in feature_columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing Base features: {missing}"
        )

    return df[
        feature_columns
    ].copy()


def find_threshold_at_fpr(
    y_true,
    y_prob,
    target_fpr=0.05
):

    y_true = np.asarray(
        y_true
    )

    y_prob = np.asarray(
        y_prob
    )

    negative_scores = y_prob[
        y_true == 0
    ]

    threshold = np.quantile(
        negative_scores,
        1.0 - target_fpr
    )

    return float(
        threshold
    )


def calculate_metrics(
    y_true,
    y_prob,
    threshold
):

    y_true = np.asarray(
        y_true
    )

    y_prob = np.asarray(
        y_prob
    )

    y_pred = (
        y_prob >= threshold
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
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
                y_prob
            ),

        "ROC_AUC":
            roc_auc_score(
                y_true,
                y_prob
            ),

        "Threshold":
            threshold,

        "FPR":
            fpr,

        "Precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "MCC":
            matthews_corrcoef(
                y_true,
                y_pred
            ),

        "Balanced_Accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


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

        self.feature_columns = list(
            X.columns
        )

        self.categorical_columns = (
            X.select_dtypes(
                include=[
                    "object",
                    "category"
                ]
            )
            .columns
            .tolist()
        )

        if self.categorical_columns:

            self.encoder = OrdinalEncoder(
                handle_unknown=
                    "use_encoded_value",
                unknown_value=-1,
                dtype=np.float32
            )

            self.encoder.fit(
                X[
                    self.categorical_columns
                ].astype(str)
            )

        encoded = self._encode(
            X
        )

        self.scaler = StandardScaler()

        self.scaler.fit(
            encoded
        )

        del encoded
        gc.collect()

        return self

    def _encode(self, X):

        X = align_columns(
            X,
            self.feature_columns
        )

        output = np.zeros(
            (
                len(X),
                len(self.feature_columns)
            ),
            dtype=np.float32
        )

        categorical_set = set(
            self.categorical_columns
        )

        if self.categorical_columns:

            encoded_cat = (
                self.encoder.transform(
                    X[
                        self.categorical_columns
                    ].astype(str)
                )
            )

            cat_positions = {
                col: i
                for i, col
                in enumerate(
                    self.categorical_columns
                )
            }

        for j, col in enumerate(
            self.feature_columns
        ):

            if col in categorical_set:

                output[:, j] = (
                    encoded_cat[
                        :,
                        cat_positions[col]
                    ]
                )

            else:

                output[:, j] = (
                    pd.to_numeric(
                        X[col],
                        errors="coerce"
                    )
                    .fillna(0)
                    .astype(np.float32)
                    .values
                )

        return output

    def transform(self, X):

        encoded = self._encode(
            X
        )

        scaled = (
            self.scaler.transform(
                encoded
            )
            .astype(np.float32)
        )

        del encoded
        gc.collect()

        return scaled


# ============================================================
# LOAD BASE ONCE
# ============================================================

header(
    "EXPERIMENT 04 - MULTI-SEED CONFIRMATION"
)

print("Loading Base.csv...")

start = time.time()

base = pd.read_csv(
    BASE_FILE
)

print(
    f"Loaded in "
    f"{time.time()-start:.2f} sec"
)


# ============================================================
# TEMPORAL SPLIT
# ============================================================

train_df = base[
    base[TIME_COL].isin(
        [0, 1, 2, 3, 4, 5]
    )
].copy()

val_df = base[
    base[TIME_COL] == 6
].copy()

test_df = base[
    base[TIME_COL] == 7
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

print(
    f"Train fraud: "
    f"{train_df[TARGET].mean()*100:.4f}%"
)

print(
    f"Val fraud  : "
    f"{val_df[TARGET].mean()*100:.4f}%"
)

print(
    f"Test fraud : "
    f"{test_df[TARGET].mean()*100:.4f}%"
)


# ============================================================
# REMOVE TARGET + MONTH
# ============================================================

DROP_COLUMNS = [
    TARGET,
    TIME_COL
]

X_train_raw = train_df.drop(
    columns=DROP_COLUMNS
)

y_train = (
    train_df[TARGET]
    .astype(np.int8)
    .values
)

X_val_raw = val_df.drop(
    columns=DROP_COLUMNS
)

y_val = (
    val_df[TARGET]
    .astype(np.int8)
    .values
)

X_test_raw = test_df.drop(
    columns=DROP_COLUMNS
)

y_test = (
    test_df[TARGET]
    .astype(np.int8)
    .values
)

del base
del train_df
del val_df
del test_df

gc.collect()


# ============================================================
# PREPROCESSING - FIT ONCE
# ============================================================

header(
    "PREPROCESSING"
)

preprocessor = BAFPreprocessor()

preprocessor.fit(
    X_train_raw
)

feature_columns = (
    preprocessor.feature_columns
)

print(
    f"Predictors: "
    f"{len(feature_columns)}"
)

print(
    "Transforming Base Train..."
)

X_train = preprocessor.transform(
    X_train_raw
)

print(
    "Transforming Base Validation..."
)

X_val = preprocessor.transform(
    X_val_raw
)

print(
    "Transforming Base Test..."
)

X_test = preprocessor.transform(
    X_test_raw
)

del X_train_raw
del X_val_raw
del X_test_raw

gc.collect()


# ============================================================
# LOAD + PREPROCESS VARIANTS ONCE
#
# They are read one at a time, transformed,
# then stored as float32 arrays.
# ============================================================

header(
    "PREPROCESSING VARIANTS"
)

variant_data = {}

for variant_name, file_name in (
    VARIANTS.items()
):

    print(
        f"Preparing {variant_name}..."
    )

    start = time.time()

    df = pd.read_csv(
        file_name
    )

    y = (
        df[TARGET]
        .astype(np.int8)
        .values
    )

    X_raw = df.drop(
        columns=[
            TARGET,
            TIME_COL
        ]
    )

    X_raw = align_columns(
        X_raw,
        feature_columns
    )

    X = preprocessor.transform(
        X_raw
    )

    variant_data[
        variant_name
    ] = {
        "X": X,
        "y": y
    }

    print(
        f"  {len(y):,} rows "
        f"prepared in "
        f"{time.time()-start:.2f} sec"
    )

    del df
    del X_raw

    gc.collect()


# ============================================================
# MAIN MULTI-SEED LOOP
# ============================================================

all_results = []
timing_rows = []

header(
    "RUNNING FIVE SEEDS"
)


for seed_index, seed in enumerate(
    SEEDS,
    start=1
):

    print(
        "\n" + "#" * 80
    )

    print(
        f"SEED {seed} "
        f"({seed_index}/{len(SEEDS)})"
    )

    print(
        "#" * 80
    )


    # ========================================================
    # M1 - PURE LGBM
    # ========================================================

    print(
        "\n[M1] Training LGBM..."
    )

    lgbm_params = {
        **LGBM_BASE_PARAMS,
        "random_state": seed
    }

    model_m1 = lgb.LGBMClassifier(
        **lgbm_params
    )

    start = time.time()

    model_m1.fit(
        X_train,
        y_train
    )

    m1_train_time = (
        time.time() - start
    )

    # Validation threshold
    val_prob_m1 = (
        model_m1.predict_proba(
            X_val
        )[:, 1]
    )

    threshold_m1 = (
        find_threshold_at_fpr(
            y_val,
            val_prob_m1,
            TARGET_FPR
        )
    )

    # Base Test
    start = time.time()

    test_prob_m1 = (
        model_m1.predict_proba(
            X_test
        )[:, 1]
    )

    m1_test_time = (
        time.time() - start
    )

    metrics = calculate_metrics(
        y_test,
        test_prob_m1,
        threshold_m1
    )

    all_results.append({
        "Seed": seed,
        "Model": "M1_LGBM",
        "Dataset":
            "Base Test (Month 7)",
        **metrics
    })


    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    print(
        "[IF] Fitting Isolation Forest..."
    )

    if_params = {
        **IF_BASE_PARAMS,
        "random_state": seed
    }

    iso = IsolationForest(
        **if_params
    )

    start = time.time()

    iso.fit(
        X_train
    )

    if_fit_time = (
        time.time() - start
    )

    print(
        f"     IF fit: "
        f"{if_fit_time:.2f} sec"
    )

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

    test_if = (
        -iso.decision_function(
            X_test
        )
    ).astype(np.float32)


    # ========================================================
    # M2 - LGBM + RAW IF
    # ========================================================

    print(
        "[M2] Training LGBM + Raw IF..."
    )

    X_train_if = np.column_stack(
        [
            X_train,
            train_if
        ]
    ).astype(np.float32)

    X_val_if = np.column_stack(
        [
            X_val,
            val_if
        ]
    ).astype(np.float32)

    X_test_if = np.column_stack(
        [
            X_test,
            test_if
        ]
    ).astype(np.float32)

    model_m2 = lgb.LGBMClassifier(
        **lgbm_params
    )

    start = time.time()

    model_m2.fit(
        X_train_if,
        y_train
    )

    m2_train_time = (
        time.time() - start
    )

    val_prob_m2 = (
        model_m2.predict_proba(
            X_val_if
        )[:, 1]
    )

    threshold_m2 = (
        find_threshold_at_fpr(
            y_val,
            val_prob_m2,
            TARGET_FPR
        )
    )

    start = time.time()

    test_prob_m2 = (
        model_m2.predict_proba(
            X_test_if
        )[:, 1]
    )

    m2_test_time = (
        time.time() - start
    )

    metrics = calculate_metrics(
        y_test,
        test_prob_m2,
        threshold_m2
    )

    all_results.append({
        "Seed": seed,
        "Model": "M2_LGBM_RAW_IF",
        "Dataset":
            "Base Test (Month 7)",
        **metrics
    })


    # ========================================================
    # PRINT BASE COMPARISON
    # ========================================================

    base_m1 = all_results[-2]
    base_m2 = all_results[-1]

    print(
        "\nBASE TEST:"
    )

    print(
        f"M1 AUPRC = "
        f"{base_m1['AUPRC']:.6f}"
    )

    print(
        f"M2 AUPRC = "
        f"{base_m2['AUPRC']:.6f}"
    )

    print(
        f"Delta    = "
        f"{base_m2['AUPRC'] - base_m1['AUPRC']:+.6f}"
    )


    # ========================================================
    # VARIANT TESTING
    # ========================================================

    for variant_name in VARIANTS:

        X_variant = (
            variant_data[
                variant_name
            ]["X"]
        )

        y_variant = (
            variant_data[
                variant_name
            ]["y"]
        )

        # ------------------------
        # M1
        # ------------------------

        prob_m1 = (
            model_m1.predict_proba(
                X_variant
            )[:, 1]
        )

        metrics_m1 = (
            calculate_metrics(
                y_variant,
                prob_m1,
                threshold_m1
            )
        )

        all_results.append({
            "Seed": seed,
            "Model": "M1_LGBM",
            "Dataset": variant_name,
            **metrics_m1
        })


        # ------------------------
        # M2 IF score
        # ------------------------

        variant_if = (
            -iso.decision_function(
                X_variant
            )
        ).astype(np.float32)

        X_variant_if = (
            np.column_stack(
                [
                    X_variant,
                    variant_if
                ]
            )
            .astype(np.float32)
        )

        prob_m2 = (
            model_m2.predict_proba(
                X_variant_if
            )[:, 1]
        )

        metrics_m2 = (
            calculate_metrics(
                y_variant,
                prob_m2,
                threshold_m2
            )
        )

        all_results.append({
            "Seed": seed,
            "Model":
                "M2_LGBM_RAW_IF",
            "Dataset":
                variant_name,
            **metrics_m2
        })

        print(
            f"{variant_name:12s} | "
            f"AUPRC "
            f"{metrics_m1['AUPRC']:.4f}"
            f" -> "
            f"{metrics_m2['AUPRC']:.4f} | "
            f"Delta="
            f"{metrics_m2['AUPRC'] - metrics_m1['AUPRC']:+.5f}"
        )

        del variant_if
        del X_variant_if

        gc.collect()


    # ========================================================
    # TIMING
    # ========================================================

    timing_rows.append({

        "Seed":
            seed,

        "M1_LGBM_Train_sec":
            m1_train_time,

        "IF_Fit_sec":
            if_fit_time,

        "M2_LGBM_Train_sec":
            m2_train_time,

        "M1_Base_Inference_sec":
            m1_test_time,

        "M2_Base_Inference_sec":
            m2_test_time,

        "M1_Threshold":
            threshold_m1,

        "M2_Threshold":
            threshold_m2
    })


    # ========================================================
    # CLEAN SEED-SPECIFIC OBJECTS
    # ========================================================

    del model_m1
    del model_m2
    del iso

    del train_if
    del val_if
    del test_if

    del X_train_if
    del X_val_if
    del X_test_if

    del val_prob_m1
    del val_prob_m2
    del test_prob_m1
    del test_prob_m2

    gc.collect()


# ============================================================
# RAW RESULTS
# ============================================================

header(
    "CREATING SUMMARY TABLES"
)

results_df = pd.DataFrame(
    all_results
)

timing_df = pd.DataFrame(
    timing_rows
)

raw_path = os.path.join(
    OUTPUT_DIR,
    "all_seed_results.csv"
)

results_df.to_csv(
    raw_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# MEAN +/- SD
# ============================================================

METRICS = [
    "AUPRC",
    "ROC_AUC",
    "FPR",
    "Precision",
    "Recall",
    "F1",
    "MCC",
    "Balanced_Accuracy"
]

summary_df = (
    results_df
    .groupby(
        [
            "Model",
            "Dataset"
        ]
    )[METRICS]
    .agg(
        ["mean", "std"]
    )
    .reset_index()
)

# Flatten MultiIndex columns
summary_df.columns = [
    "_".join(
        [
            str(x)
            for x in col
            if str(x) != ""
        ]
    )
    if isinstance(col, tuple)
    else col

    for col in summary_df.columns
]

summary_path = os.path.join(
    OUTPUT_DIR,
    "mean_sd_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PAIRED DELTAS
# ============================================================

m1 = results_df[
    results_df["Model"]
    == "M1_LGBM"
].copy()

m2 = results_df[
    results_df["Model"]
    == "M2_LGBM_RAW_IF"
].copy()

paired = m1.merge(
    m2,
    on=[
        "Seed",
        "Dataset"
    ],
    suffixes=(
        "_M1",
        "_M2"
    )
)

for metric in METRICS:

    paired[
        f"Delta_{metric}"
    ] = (
        paired[
            f"{metric}_M2"
        ]
        -
        paired[
            f"{metric}_M1"
        ]
    )


paired_path = os.path.join(
    OUTPUT_DIR,
    "paired_deltas.csv"
)

paired.to_csv(
    paired_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# DELTA SUMMARY PER DATASET
# ============================================================

delta_columns = [
    f"Delta_{m}"
    for m in METRICS
]

delta_summary = (
    paired
    .groupby(
        "Dataset"
    )[delta_columns]
    .agg(
        ["mean", "std", "median"]
    )
    .reset_index()
)

delta_summary.columns = [
    "_".join(
        [
            str(x)
            for x in col
            if str(x) != ""
        ]
    )
    if isinstance(col, tuple)
    else col

    for col in delta_summary.columns
]

delta_summary_path = os.path.join(
    OUTPUT_DIR,
    "delta_summary_by_dataset.csv"
)

delta_summary.to_csv(
    delta_summary_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# OVERALL POSITIVE EFFECT COUNTS
# ============================================================

effect_rows = []

for metric in [
    "AUPRC",
    "ROC_AUC",
    "Recall",
    "F1",
    "MCC",
    "Balanced_Accuracy"
]:

    delta_col = (
        f"Delta_{metric}"
    )

    values = paired[
        delta_col
    ]

    effect_rows.append({

        "Metric":
            metric,

        "Comparisons":
            len(values),

        "Positive":
            int(
                (values > 0).sum()
            ),

        "Negative":
            int(
                (values < 0).sum()
            ),

        "Equal":
            int(
                (values == 0).sum()
            ),

        "Mean_Delta":
            float(
                values.mean()
            ),

        "Median_Delta":
            float(
                values.median()
            ),

        "SD_Delta":
            float(
                values.std()
            )
    })

effect_df = pd.DataFrame(
    effect_rows
)

effect_path = os.path.join(
    OUTPUT_DIR,
    "effect_direction_summary.csv"
)

effect_df.to_csv(
    effect_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# VARIANT-ONLY SUMMARY
# ============================================================

variant_names = list(
    VARIANTS.keys()
)

variant_paired = paired[
    paired["Dataset"].isin(
        variant_names
    )
].copy()

variant_summary_rows = []

for seed in SEEDS:

    temp = variant_paired[
        variant_paired["Seed"]
        == seed
    ]

    variant_summary_rows.append({

        "Seed":
            seed,

        "Mean_Delta_AUPRC":
            temp[
                "Delta_AUPRC"
            ].mean(),

        "Mean_Delta_ROC_AUC":
            temp[
                "Delta_ROC_AUC"
            ].mean(),

        "Mean_Delta_Recall":
            temp[
                "Delta_Recall"
            ].mean(),

        "Mean_Delta_F1":
            temp[
                "Delta_F1"
            ].mean(),

        "Mean_Delta_MCC":
            temp[
                "Delta_MCC"
            ].mean(),

        "Positive_AUPRC_Variants":
            int(
                (
                    temp[
                        "Delta_AUPRC"
                    ] > 0
                ).sum()
            ),

        "Positive_Recall_Variants":
            int(
                (
                    temp[
                        "Delta_Recall"
                    ] > 0
                ).sum()
            )
    })


variant_summary_df = pd.DataFrame(
    variant_summary_rows
)

variant_summary_path = os.path.join(
    OUTPUT_DIR,
    "variant_seed_summary.csv"
)

variant_summary_df.to_csv(
    variant_summary_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# TIMING
# ============================================================

timing_path = os.path.join(
    OUTPUT_DIR,
    "timing.csv"
)

timing_df.to_csv(
    timing_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SETTINGS
# ============================================================

settings = {

    "experiment":
        "Experiment 04 - Multi-Seed Confirmation",

    "seeds":
        SEEDS,

    "models": [
        "M1_LGBM",
        "M2_LGBM_RAW_IF"
    ],

    "train_months":
        [0, 1, 2, 3, 4, 5],

    "validation_month":
        6,

    "test_month":
        7,

    "month_used_as_predictor":
        False,

    "target_fpr":
        TARGET_FPR,

    "lightgbm_parameters":
        LGBM_BASE_PARAMS,

    "isolation_forest_parameters":
        IF_BASE_PARAMS,

    "primary_metric":
        "AUPRC",

    "secondary_operating_metric":
        "Recall at validation-derived ~5% FPR",

    "design_note":
        (
            "Experiment 04 confirms only the two "
            "preselected models from Experiment 03. "
            "No additional IF representation or "
            "hyperparameter search is performed."
        )
}

settings_path = os.path.join(
    OUTPUT_DIR,
    "experiment_settings.json"
)

with open(
    settings_path,
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
# FINAL SCREEN SUMMARY
# ============================================================

header(
    "EXPERIMENT 04 COMPLETE"
)

print(
    "\nEFFECT DIRECTION SUMMARY\n"
)

print(
    effect_df
    .round(6)
    .to_string(
        index=False
    )
)

print(
    "\nVARIANT-ONLY SEED SUMMARY\n"
)

print(
    variant_summary_df
    .round(6)
    .to_string(
        index=False
    )
)

print(
    "\nFiles created:"
)

for path in [
    raw_path,
    summary_path,
    paired_path,
    delta_summary_path,
    effect_path,
    variant_summary_path,
    timing_path,
    settings_path
]:

    print(
        f"  {path}"
    )