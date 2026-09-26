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
# EXPERIMENT 03
# Anomaly Representation Ablation
#
# M1 = LGBM
# M2 = LGBM + raw IF score
# M3 = LGBM + IF percentile
# M4 = LGBM + multi-level IF representation
#
# Temporal protocol:
# Month 0-5 -> Train
# Month 6   -> Validation
# Month 7   -> Test
#
# IMPORTANT:
# month is used ONLY for temporal splitting.
# It is NOT used as a predictor.
#
# All IF transformations / thresholds are learned
# ONLY from Base Train.
# ============================================================


SEED = 42

TARGET = "fraud_bool"
TIME_COL = "month"

TARGET_FPR = 0.05

BASE_FILE = "Base.csv"

VARIANTS = {
    "Variant I": "Variant I.csv",
    "Variant II": "Variant II.csv",
    "Variant III": "Variant III.csv",
    "Variant IV": "Variant IV.csv",
    "Variant V": "Variant V.csv",
}

OUTPUT_DIR = os.path.join(
    "results",
    "experiment_03"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1
}

IF_PARAMS = {
    "n_estimators": 200,
    "contamination": "auto",
    "random_state": SEED,
    "n_jobs": -1
}

ANOMALY_LEVELS = [
    0.01,
    0.02,
    0.05,
    0.10
]

MODELS = [
    "M1_LGBM",
    "M2_LGBM_RAW_IF",
    "M3_LGBM_IF_PERCENTILE",
    "M4_LGBM_MULTI_IF"
]


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

    return df[
        feature_columns
    ].copy()


def find_threshold_at_fpr(
    y_true,
    y_prob,
    target_fpr=0.05
):

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    negative_scores = y_prob[
        y_true == 0
    ]

    if len(negative_scores) == 0:

        raise ValueError(
            "No negative samples found."
        )

    threshold = np.quantile(
        negative_scores,
        1.0 - target_fpr
    )

    return float(threshold)


def calculate_metrics(
    y_true,
    y_prob,
    threshold
):

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

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

        "Recall_at_5pctFPR":
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

        encoded = self._encode(X)

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

        encoded = self._encode(X)

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
# EMPIRICAL PERCENTILE TRANSFORMER
# ============================================================

class IFRepresentation:

    """
    Learns anomaly representation ONLY from
    Base training anomaly scores.
    """

    def __init__(self):

        self.sorted_train_scores = None
        self.anomaly_thresholds = {}

    def fit(self, train_scores):

        train_scores = np.asarray(
            train_scores,
            dtype=np.float32
        )

        # Save sorted training distribution
        self.sorted_train_scores = np.sort(
            train_scores
        )

        # Learn anomaly cutoffs from TRAIN only
        for p in ANOMALY_LEVELS:

            threshold = np.quantile(
                train_scores,
                1.0 - p
            )

            self.anomaly_thresholds[p] = (
                float(threshold)
            )

        return self

    def percentile(self, scores):

        """
        Maps score to its empirical percentile
        relative to Base Train distribution.

        High percentile = more anomalous.
        """

        scores = np.asarray(
            scores,
            dtype=np.float32
        )

        ranks = np.searchsorted(
            self.sorted_train_scores,
            scores,
            side="right"
        )

        percentile = (
            ranks
            / len(
                self.sorted_train_scores
            )
        )

        return percentile.astype(
            np.float32
        )

    def binary_indicators(self, scores):

        scores = np.asarray(
            scores,
            dtype=np.float32
        )

        indicators = []

        for p in ANOMALY_LEVELS:

            threshold = (
                self.anomaly_thresholds[p]
            )

            indicator = (
                scores >= threshold
            ).astype(np.float32)

            indicators.append(
                indicator
            )

        return np.column_stack(
            indicators
        ).astype(np.float32)


# ============================================================
# CREATE MODEL REPRESENTATION
# ============================================================

def build_representation(
    model_name,
    X,
    raw_if_score,
    if_representation
):

    raw_if_score = np.asarray(
        raw_if_score,
        dtype=np.float32
    )

    if model_name == "M1_LGBM":

        return X


    elif model_name == "M2_LGBM_RAW_IF":

        return np.column_stack(
            [
                X,
                raw_if_score
            ]
        ).astype(np.float32)


    elif model_name == "M3_LGBM_IF_PERCENTILE":

        percentile = (
            if_representation.percentile(
                raw_if_score
            )
        )

        return np.column_stack(
            [
                X,
                percentile
            ]
        ).astype(np.float32)


    elif model_name == "M4_LGBM_MULTI_IF":

        percentile = (
            if_representation.percentile(
                raw_if_score
            )
        )

        indicators = (
            if_representation
            .binary_indicators(
                raw_if_score
            )
        )

        return np.column_stack(
            [
                X,
                raw_if_score,
                percentile,
                indicators
            ]
        ).astype(np.float32)


    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


# ============================================================
# LOAD BASE
# ============================================================

header(
    "EXPERIMENT 03 - ANOMALY REPRESENTATION"
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

train = base[
    base[TIME_COL].isin(
        [0, 1, 2, 3, 4, 5]
    )
].copy()

validation = base[
    base[TIME_COL] == 6
].copy()

test = base[
    base[TIME_COL] == 7
].copy()


header("TEMPORAL SPLIT")

print(
    f"Train      : {len(train):,}"
)

print(
    f"Validation : {len(validation):,}"
)

print(
    f"Test       : {len(test):,}"
)

print(
    f"Train fraud rate: "
    f"{train[TARGET].mean()*100:.4f}%"
)

print(
    f"Val fraud rate  : "
    f"{validation[TARGET].mean()*100:.4f}%"
)

print(
    f"Test fraud rate : "
    f"{test[TARGET].mean()*100:.4f}%"
)


# ============================================================
# REMOVE TARGET + MONTH
# ============================================================

# Critical difference from Experiment 01:
# month is NOT a predictor.

DROP_COLUMNS = [
    TARGET,
    TIME_COL
]

X_train_raw = train.drop(
    columns=DROP_COLUMNS
)

y_train = (
    train[TARGET]
    .astype(np.int8)
    .values
)

X_val_raw = validation.drop(
    columns=DROP_COLUMNS
)

y_val = (
    validation[TARGET]
    .astype(np.int8)
    .values
)

X_test_raw = test.drop(
    columns=DROP_COLUMNS
)

y_test = (
    test[TARGET]
    .astype(np.int8)
    .values
)

del base
del train
del validation
del test

gc.collect()


# ============================================================
# PREPROCESSING
# ============================================================

header("PREPROCESSING")

preprocessor = BAFPreprocessor()

print(
    "Fitting preprocessing on "
    "Base Train only..."
)

preprocessor.fit(
    X_train_raw
)

feature_columns = (
    preprocessor.feature_columns
)

print(
    f"Predictor count: "
    f"{len(feature_columns)}"
)

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

del X_train_raw
del X_val_raw
del X_test_raw

gc.collect()


# ============================================================
# FIT ISOLATION FOREST
# ============================================================

header("ISOLATION FOREST")

iso = IsolationForest(
    **IF_PARAMS
)

start = time.time()

iso.fit(
    X_train
)

if_fit_time = (
    time.time() - start
)

print(
    f"IF fit time: "
    f"{if_fit_time:.2f} sec"
)

# Higher score = more anomalous

train_if_score = (
    -iso.decision_function(
        X_train
    )
).astype(np.float32)

val_if_score = (
    -iso.decision_function(
        X_val
    )
).astype(np.float32)

test_if_score = (
    -iso.decision_function(
        X_test
    )
).astype(np.float32)


# ============================================================
# FIT IF REPRESENTATION
# ============================================================

header(
    "LEARNING ANOMALY REPRESENTATION"
)

if_rep = IFRepresentation()

if_rep.fit(
    train_if_score
)

print(
    "Train-derived anomaly thresholds:"
)

for p, threshold in (
    if_rep.anomaly_thresholds.items()
):

    print(
        f"Top {int(p*100):2d}% -> "
        f"{threshold:.6f}"
    )


# ============================================================
# TRAIN MODELS
# ============================================================

header(
    "TRAINING FOUR REPRESENTATIONS"
)

trained_models = {}
thresholds = {}
results = []
timings = []


for model_name in MODELS:

    print(
        "\n" + "-" * 80
    )

    print(
        f"TRAINING: {model_name}"
    )

    print(
        "-" * 80
    )

    train_X = build_representation(
        model_name,
        X_train,
        train_if_score,
        if_rep
    )

    val_X = build_representation(
        model_name,
        X_val,
        val_if_score,
        if_rep
    )

    test_X = build_representation(
        model_name,
        X_test,
        test_if_score,
        if_rep
    )

    print(
        f"Feature count: "
        f"{train_X.shape[1]}"
    )

    model = lgb.LGBMClassifier(
        **LGBM_PARAMS
    )

    start = time.time()

    model.fit(
        train_X,
        y_train
    )

    train_time = (
        time.time() - start
    )

    # --------------------------------------------
    # VALIDATION
    # --------------------------------------------

    val_prob = (
        model.predict_proba(
            val_X
        )[:, 1]
    )

    threshold = (
        find_threshold_at_fpr(
            y_val,
            val_prob,
            TARGET_FPR
        )
    )

    val_metrics = (
        calculate_metrics(
            y_val,
            val_prob,
            threshold
        )
    )

    # --------------------------------------------
    # TEST
    # --------------------------------------------

    start = time.time()

    test_prob = (
        model.predict_proba(
            test_X
        )[:, 1]
    )

    inference_time = (
        time.time() - start
    )

    test_metrics = (
        calculate_metrics(
            y_test,
            test_prob,
            threshold
        )
    )

    print(
        f"Threshold : "
        f"{threshold:.6f}"
    )

    print(
        f"Val FPR   : "
        f"{val_metrics['FPR']:.4f}"
    )

    print(
        f"Test AUPRC: "
        f"{test_metrics['AUPRC']:.4f}"
    )

    print(
        f"Test Recall: "
        f"{test_metrics['Recall_at_5pctFPR']:.4f}"
    )

    print(
        f"Test FPR: "
        f"{test_metrics['FPR']:.4f}"
    )

    results.append({

        "Model":
            model_name,

        "Dataset":
            "Base Test (Month 7)",

        **test_metrics,

        "Train_Time_sec":
            train_time,

        "Inference_Time_sec":
            inference_time
    })

    timings.append({

        "Model":
            model_name,

        "Feature_Count":
            train_X.shape[1],

        "Train_Time_sec":
            train_time,

        "Base_Test_Inference_Time_sec":
            inference_time,

        "Threshold":
            threshold
    })

    trained_models[
        model_name
    ] = model

    thresholds[
        model_name
    ] = threshold

    # M1 returns original X arrays.
    # Do NOT delete original arrays.

    if model_name != "M1_LGBM":

        del train_X
        del val_X
        del test_X

    gc.collect()


# ============================================================
# VARIANT EVALUATION
# ============================================================

header(
    "DISTRIBUTION SHIFT EVALUATION"
)


for variant_name, file_name in (
    VARIANTS.items()
):

    print(
        "\n" + "=" * 80
    )

    print(
        f"PROCESSING {variant_name}"
    )

    print(
        "=" * 80
    )

    start = time.time()

    df = pd.read_csv(
        file_name
    )

    print(
        f"Loaded {len(df):,} rows "
        f"in {time.time()-start:.2f} sec"
    )

    y_variant = (
        df[TARGET]
        .astype(np.int8)
        .values
    )

    # month excluded here as well
    X_raw = df.drop(
        columns=[
            TARGET,
            TIME_COL
        ]
    )

    # x1/x2 etc. automatically removed
    X_raw = align_columns(
        X_raw,
        feature_columns
    )

    X_variant = (
        preprocessor.transform(
            X_raw
        )
    )

    variant_if_score = (
        -iso.decision_function(
            X_variant
        )
    ).astype(np.float32)

    for model_name in MODELS:

        eval_X = (
            build_representation(
                model_name,
                X_variant,
                variant_if_score,
                if_rep
            )
        )

        model = trained_models[
            model_name
        ]

        threshold = thresholds[
            model_name
        ]

        start = time.time()

        prob = (
            model.predict_proba(
                eval_X
            )[:, 1]
        )

        inference_time = (
            time.time() - start
        )

        metrics = (
            calculate_metrics(
                y_variant,
                prob,
                threshold
            )
        )

        results.append({

            "Model":
                model_name,

            "Dataset":
                variant_name,

            **metrics,

            "Train_Time_sec":
                np.nan,

            "Inference_Time_sec":
                inference_time
        })

        print(
            f"{model_name:28s} | "
            f"AUPRC="
            f"{metrics['AUPRC']:.4f} | "
            f"Recall="
            f"{metrics['Recall_at_5pctFPR']:.4f} | "
            f"FPR="
            f"{metrics['FPR']:.4f}"
        )

        if model_name != "M1_LGBM":

            del eval_X

    del df
    del X_raw
    del X_variant
    del variant_if_score
    del y_variant

    gc.collect()


# ============================================================
# RESULTS
# ============================================================

header("SAVING RESULTS")

results_df = pd.DataFrame(
    results
)

timing_df = pd.DataFrame(
    timings
)


# ------------------------------------------------------------
# Full results
# ------------------------------------------------------------

results_path = os.path.join(
    OUTPUT_DIR,
    "all_results.csv"
)

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# AUPRC
# ------------------------------------------------------------

auprc_table = results_df.pivot(
    index="Model",
    columns="Dataset",
    values="AUPRC"
)

auprc_path = os.path.join(
    OUTPUT_DIR,
    "auprc_table.csv"
)

auprc_table.to_csv(
    auprc_path,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# Recall
# ------------------------------------------------------------

recall_table = results_df.pivot(
    index="Model",
    columns="Dataset",
    values="Recall_at_5pctFPR"
)

recall_path = os.path.join(
    OUTPUT_DIR,
    "recall_table.csv"
)

recall_table.to_csv(
    recall_path,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# FPR drift
# ------------------------------------------------------------

fpr_table = results_df.pivot(
    index="Model",
    columns="Dataset",
    values="FPR"
)

fpr_path = os.path.join(
    OUTPUT_DIR,
    "fpr_table.csv"
)

fpr_table.to_csv(
    fpr_path,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# Timing
# ------------------------------------------------------------

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
# ROBUSTNESS SUMMARY
# ============================================================

robustness_rows = []

variant_names = list(
    VARIANTS.keys()
)

for model_name in MODELS:

    model_df = results_df[
        results_df["Model"]
        == model_name
    ]

    base_row = model_df[
        model_df["Dataset"]
        == "Base Test (Month 7)"
    ]

    variant_rows = model_df[
        model_df["Dataset"].isin(
            variant_names
        )
    ]

    base_auprc = float(
        base_row[
            "AUPRC"
        ].iloc[0]
    )

    base_recall = float(
        base_row[
            "Recall_at_5pctFPR"
        ].iloc[0]
    )

    mean_variant_auprc = float(
        variant_rows[
            "AUPRC"
        ].mean()
    )

    mean_variant_recall = float(
        variant_rows[
            "Recall_at_5pctFPR"
        ].mean()
    )

    mean_variant_fpr = float(
        variant_rows[
            "FPR"
        ].mean()
    )

    robustness_rows.append({

        "Model":
            model_name,

        "Base_AUPRC":
            base_auprc,

        "Mean_Variant_AUPRC":
            mean_variant_auprc,

        "AUPRC_Absolute_Drop":
            (
                base_auprc
                - mean_variant_auprc
            ),

        "AUPRC_Retention_Ratio":
            (
                mean_variant_auprc
                / base_auprc
            ),

        "Base_Recall":
            base_recall,

        "Mean_Variant_Recall":
            mean_variant_recall,

        "Recall_Absolute_Drop":
            (
                base_recall
                - mean_variant_recall
            ),

        "Recall_Retention_Ratio":
            (
                mean_variant_recall
                / base_recall
            ),

        "Mean_Variant_FPR":
            mean_variant_fpr,

        "Mean_FPR_Deviation_From_5pct":
            (
                mean_variant_fpr
                - TARGET_FPR
            )
    })


robustness_df = pd.DataFrame(
    robustness_rows
)

robustness_path = os.path.join(
    OUTPUT_DIR,
    "robustness_summary.csv"
)

robustness_df.to_csv(
    robustness_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SAVE SETTINGS
# ============================================================

settings = {

    "experiment":
        "Experiment 03 - Anomaly Representation",

    "seed":
        SEED,

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

    "models":
        MODELS,

    "anomaly_levels":
        ANOMALY_LEVELS,

    "lightgbm_parameters":
        LGBM_PARAMS,

    "isolation_forest_parameters":
        IF_PARAMS,

    "important_note":
        (
            "All anomaly percentiles and binary "
            "anomaly thresholds were learned only "
            "from Base training anomaly scores. "
            "No variant-specific thresholding was used."
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
# FINAL OUTPUT
# ============================================================

header(
    "EXPERIMENT 03 COMPLETE"
)

print("\nAUPRC:\n")

print(
    auprc_table
    .round(4)
    .to_string()
)

print("\nRECALL:\n")

print(
    recall_table
    .round(4)
    .to_string()
)

print("\nFPR:\n")

print(
    fpr_table
    .round(4)
    .to_string()
)

print("\nROBUSTNESS:\n")

print(
    robustness_df
    .round(4)
    .to_string(index=False)
)

print("\nCreated files:")

print(
    f"  {results_path}"
)

print(
    f"  {auprc_path}"
)

print(
    f"  {recall_path}"
)

print(
    f"  {fpr_path}"
)

print(
    f"  {robustness_path}"
)

print(
    f"  {timing_path}"
)

print(
    f"  {settings_path}"
)