import os
import gc
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
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
# EXPERIMENT 05B
# FINAL TIME-ALIGNED ML BENCHMARK
#
# IMPORTANT CORRECTION:
#
# Train      = Base Month 0-5
# Validation = Base Month 6
# Test       = Base Month 7
#
# Shift evaluation:
# Variant I  Month 7
# Variant II Month 7
# Variant III Month 7
# Variant IV Month 7
# Variant V  Month 7
#
# Thus all final test datasets belong to the SAME month.
#
# month is NOT used as predictor.
#
# Threshold:
# Selected on Base Month 6 to target ~5% FPR,
# then frozen for Base Month 7 and all Variant Month 7 sets.
#
# No statistical hypothesis testing.
# ============================================================


SEEDS = [42, 52, 62, 72, 82]

TARGET = "fraud_bool"
TIME_COL = "month"

TRAIN_MONTHS = [0, 1, 2, 3, 4, 5]
VALIDATION_MONTH = 6
TEST_MONTH = 7

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
    "experiment_05b"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FIXED MODEL PARAMETERS
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


def align_columns(df, feature_columns):

    missing = [
        col for col in feature_columns
        if col not in df.columns
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
                handle_unknown="use_encoded_value",
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

            positions = {
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
                        positions[col]
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

        transformed = (
            self.scaler
            .transform(encoded)
            .astype(np.float32)
        )

        del encoded
        gc.collect()

        return transformed


# ============================================================
# MODEL FACTORY
# ============================================================

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
# LOAD BASE
# ============================================================

header(
    "EXPERIMENT 05B - FINAL TIME-ALIGNED BENCHMARK"
)

print("Loading Base.csv...")

start = time.time()

base = pd.read_csv(
    BASE_FILE
)

print(
    f"Loaded in {time.time()-start:.2f} sec"
)


# ============================================================
# TEMPORAL SPLIT
# ============================================================

train_df = base[
    base[TIME_COL].isin(
        TRAIN_MONTHS
    )
].copy()

val_df = base[
    base[TIME_COL]
    == VALIDATION_MONTH
].copy()

test_df = base[
    base[TIME_COL]
    == TEST_MONTH
].copy()


header("TEMPORAL SPLIT")

print(
    f"Train      : {len(train_df):,}"
)

print(
    f"Validation : {len(val_df):,}"
)

print(
    f"Base Test  : {len(test_df):,}"
)

print(
    f"Train fraud rate : "
    f"{train_df[TARGET].mean()*100:.4f}%"
)

print(
    f"Val fraud rate   : "
    f"{val_df[TARGET].mean()*100:.4f}%"
)

print(
    f"Test fraud rate  : "
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
# PREPROCESSING
# ============================================================

header("PREPROCESSING")

preprocessor = BAFPreprocessor()

preprocessor.fit(
    X_train_raw
)

feature_columns = (
    preprocessor.feature_columns
)

X_train = preprocessor.transform(
    X_train_raw
)

X_val = preprocessor.transform(
    X_val_raw
)

X_test = preprocessor.transform(
    X_test_raw
)

print(
    f"Predictors: {len(feature_columns)}"
)

del X_train_raw
del X_val_raw
del X_test_raw

gc.collect()


# ============================================================
# LOAD VARIANTS
#
# CRITICAL:
# ONLY MONTH 7 IS RETAINED.
# ============================================================

header(
    "PREPARING TIME-ALIGNED VARIANTS"
)

variant_data = {}

for name, filename in VARIANTS.items():

    print(
        f"\nLoading {name}..."
    )

    start = time.time()

    df = pd.read_csv(
        filename
    )

    original_rows = len(df)

    # ========================================================
    # THE IMPORTANT 05B CORRECTION
    # ========================================================

    df = df[
        df[TIME_COL]
        == TEST_MONTH
    ].copy()

    # ========================================================

    print(
        f"  Original rows : "
        f"{original_rows:,}"
    )

    print(
        f"  Month 7 rows  : "
        f"{len(df):,}"
    )

    print(
        f"  Fraud rate    : "
        f"{df[TARGET].mean()*100:.4f}%"
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

    # Variant III/V extra columns such as x1/x2
    # are intentionally excluded.
    X_raw = align_columns(
        X_raw,
        feature_columns
    )

    X = preprocessor.transform(
        X_raw
    )

    variant_data[name] = {
        "X": X,
        "y": y,
        "rows": len(df),
        "fraud_rate":
            float(df[TARGET].mean())
    }

    print(
        f"  Prepared in "
        f"{time.time()-start:.2f} sec"
    )

    del df
    del X_raw

    gc.collect()


# ============================================================
# SANITY CHECK
# ============================================================

header(
    "TIME-ALIGNMENT CHECK"
)

print(
    f"Base Test = Month {TEST_MONTH}"
)

for name in VARIANTS:

    print(
        f"{name:12s} = "
        f"Month {TEST_MONTH}, "
        f"{variant_data[name]['rows']:,} rows, "
        f"fraud={variant_data[name]['fraud_rate']*100:.4f}%"
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

    models = create_models(
        seed
    )


    # ========================================================
    # STANDARD MODELS
    # ========================================================

    for model_name, model in models.items():

        print(
            f"\nTraining {model_name}..."
        )

        start = time.time()

        model.fit(
            X_train,
            y_train
        )

        train_time = (
            time.time() - start
        )


        # ====================================================
        # VALIDATION-DERIVED THRESHOLD
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
        # BASE MONTH 7
        # ====================================================

        start = time.time()

        test_prob = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        base_inference_time = (
            time.time() - start
        )

        metrics = calculate_metrics(
            y_test,
            test_prob,
            threshold
        )

        all_results.append({

            "Seed":
                seed,

            "Model":
                model_name,

            "Dataset":
                "Base Test (Month 7)",

            "Evaluation_Month":
                TEST_MONTH,

            **metrics
        })

        print(
            f"  Base M7 | "
            f"AUPRC={metrics['AUPRC']:.5f} | "
            f"Recall={metrics['Recall']:.5f} | "
            f"FPR={metrics['FPR']:.5f}"
        )


        # ====================================================
        # VARIANT MONTH 7
        # ====================================================

        variant_inference_total = 0.0

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

            start = time.time()

            probabilities = (
                model.predict_proba(
                    X_variant
                )[:, 1]
            )

            inference_time = (
                time.time() - start
            )

            variant_inference_total += (
                inference_time
            )

            variant_metrics = (
                calculate_metrics(
                    y_variant,
                    probabilities,
                    threshold
                )
            )

            all_results.append({

                "Seed":
                    seed,

                "Model":
                    model_name,

                "Dataset":
                    variant_name,

                "Evaluation_Month":
                    TEST_MONTH,

                **variant_metrics
            })

            print(
                f"  {variant_name:12s} | "
                f"AUPRC="
                f"{variant_metrics['AUPRC']:.5f} | "
                f"Recall="
                f"{variant_metrics['Recall']:.5f} | "
                f"FPR="
                f"{variant_metrics['FPR']:.5f}"
            )


        timing_results.append({

            "Seed":
                seed,

            "Model":
                model_name,

            "Train_Time_sec":
                train_time,

            "IF_Fit_Time_sec":
                0.0,

            "Total_Training_Cost_sec":
                train_time,

            "Base_Test_Inference_sec":
                base_inference_time,

            "Five_Variants_Inference_sec":
                variant_inference_total
        })

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
        time.time() - start
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

    X_test_if = (
        np.column_stack(
            [
                X_test,
                test_if
            ]
        )
        .astype(np.float32)
    )


    # ========================================================
    # LIGHTGBM
    # ========================================================

    proposed = lgb.LGBMClassifier(
        **LGBM_PARAMS,
        random_state=seed
    )

    start = time.time()

    proposed.fit(
        X_train_if,
        y_train
    )

    lgbm_train_time = (
        time.time() - start
    )


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
    # PROPOSED - BASE MONTH 7
    # ========================================================

    start = time.time()

    test_prob = (
        proposed.predict_proba(
            X_test_if
        )[:, 1]
    )

    base_inference_time = (
        time.time() - start
    )

    metrics = calculate_metrics(
        y_test,
        test_prob,
        threshold
    )

    all_results.append({

        "Seed":
            seed,

        "Model":
            "Proposed IF-LGBM",

        "Dataset":
            "Base Test (Month 7)",

        "Evaluation_Month":
            TEST_MONTH,

        **metrics
    })

    print(
        f"  Base M7 | "
        f"AUPRC={metrics['AUPRC']:.5f} | "
        f"Recall={metrics['Recall']:.5f} | "
        f"FPR={metrics['FPR']:.5f}"
    )


    # ========================================================
    # PROPOSED - VARIANT MONTH 7
    # ========================================================

    variant_inference_total = 0.0

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

        start = time.time()

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

        probabilities = (
            proposed.predict_proba(
                X_variant_if
            )[:, 1]
        )

        inference_time = (
            time.time() - start
        )

        variant_inference_total += (
            inference_time
        )

        variant_metrics = (
            calculate_metrics(
                y_variant,
                probabilities,
                threshold
            )
        )

        all_results.append({

            "Seed":
                seed,

            "Model":
                "Proposed IF-LGBM",

            "Dataset":
                variant_name,

            "Evaluation_Month":
                TEST_MONTH,

            **variant_metrics
        })

        print(
            f"  {variant_name:12s} | "
            f"AUPRC="
            f"{variant_metrics['AUPRC']:.5f} | "
            f"Recall="
            f"{variant_metrics['Recall']:.5f} | "
            f"FPR="
            f"{variant_metrics['FPR']:.5f}"
        )

        del variant_if
        del X_variant_if
        gc.collect()


    timing_results.append({

        "Seed":
            seed,

        "Model":
            "Proposed IF-LGBM",

        "Train_Time_sec":
            lgbm_train_time,

        "IF_Fit_Time_sec":
            if_fit_time,

        "Total_Training_Cost_sec":
            (
                lgbm_train_time
                + if_fit_time
            ),

        "Base_Test_Inference_sec":
            base_inference_time,

        "Five_Variants_Inference_sec":
            variant_inference_total
    })


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
# SAVE RAW RESULTS
# ============================================================

header(
    "CREATING FINAL TABLES"
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
# MEAN / SD RESULTS
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

mean_sd = (
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

mean_sd.columns = [
    "_".join(
        [
            str(x)
            for x in col
            if str(x) != ""
        ]
    )
    if isinstance(col, tuple)
    else col

    for col in mean_sd.columns
]

mean_sd.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "mean_sd_results.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# ROBUSTNESS BY SEED
# ============================================================

variant_names = list(
    VARIANTS.keys()
)

robustness_rows = []


for model_name in (
    results_df["Model"].unique()
):

    model_df = results_df[
        results_df["Model"]
        == model_name
    ]

    for seed in SEEDS:

        seed_df = model_df[
            model_df["Seed"]
            == seed
        ]

        base_row = seed_df[
            seed_df["Dataset"]
            == "Base Test (Month 7)"
        ].iloc[0]

        variants = seed_df[
            seed_df["Dataset"]
            .isin(variant_names)
        ]

        mean_variant_auprc = (
            variants["AUPRC"].mean()
        )

        worst_variant_auprc = (
            variants["AUPRC"].min()
        )

        mean_variant_recall = (
            variants["Recall"].mean()
        )

        worst_variant_recall = (
            variants["Recall"].min()
        )

        mean_fpr = (
            variants["FPR"].mean()
        )

        mean_abs_fpr_drift = (
            np.abs(
                variants["FPR"]
                - TARGET_FPR
            )
            .mean()
        )

        max_abs_fpr_drift = (
            np.abs(
                variants["FPR"]
                - TARGET_FPR
            )
            .max()
        )

        robustness_rows.append({

            "Seed":
                seed,

            "Model":
                model_name,

            "Base_AUPRC":
                base_row["AUPRC"],

            "Mean_Variant_AUPRC":
                mean_variant_auprc,

            "Worst_Variant_AUPRC":
                worst_variant_auprc,

            "AUPRC_Absolute_Drop":
                (
                    base_row["AUPRC"]
                    - mean_variant_auprc
                ),

            "AUPRC_Retention_Ratio":
                (
                    mean_variant_auprc
                    / base_row["AUPRC"]
                ),

            "Base_Recall":
                base_row["Recall"],

            "Mean_Variant_Recall":
                mean_variant_recall,

            "Worst_Variant_Recall":
                worst_variant_recall,

            "Recall_Absolute_Drop":
                (
                    base_row["Recall"]
                    - mean_variant_recall
                ),

            "Recall_Retention_Ratio":
                (
                    mean_variant_recall
                    / base_row["Recall"]
                ),

            "Mean_Variant_FPR":
                mean_fpr,

            "Mean_Absolute_FPR_Drift":
                mean_abs_fpr_drift,

            "Max_Absolute_FPR_Drift":
                max_abs_fpr_drift
        })


robustness_df = pd.DataFrame(
    robustness_rows
)

robustness_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "robustness_by_seed.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# ROBUSTNESS SUMMARY
# ============================================================

ROBUSTNESS_METRICS = [
    "Base_AUPRC",
    "Mean_Variant_AUPRC",
    "Worst_Variant_AUPRC",
    "AUPRC_Absolute_Drop",
    "AUPRC_Retention_Ratio",
    "Base_Recall",
    "Mean_Variant_Recall",
    "Worst_Variant_Recall",
    "Recall_Absolute_Drop",
    "Recall_Retention_Ratio",
    "Mean_Variant_FPR",
    "Mean_Absolute_FPR_Drift",
    "Max_Absolute_FPR_Drift"
]

robustness_summary = (
    robustness_df
    .groupby("Model")[
        ROBUSTNESS_METRICS
    ]
    .agg(
        ["mean", "std"]
    )
    .reset_index()
)

robustness_summary.columns = [
    "_".join(
        [
            str(x)
            for x in col
            if str(x) != ""
        ]
    )
    if isinstance(col, tuple)
    else col

    for col in robustness_summary.columns
]

robustness_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "robustness_summary.csv"
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
            "Train_Time_sec",
            "IF_Fit_Time_sec",
            "Total_Training_Cost_sec",
            "Base_Test_Inference_sec",
            "Five_Variants_Inference_sec"
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
# MAIN BENCHMARK TABLE
# ============================================================

main_rows = []


for model_name in (
    results_df["Model"].unique()
):

    model_results = results_df[
        results_df["Model"]
        == model_name
    ]

    base_results = model_results[
        model_results["Dataset"]
        == "Base Test (Month 7)"
    ]

    variant_results = model_results[
        model_results["Dataset"]
        .isin(variant_names)
    ]

    robust = robustness_df[
        robustness_df["Model"]
        == model_name
    ]

    timing_model = timing_df[
        timing_df["Model"]
        == model_name
    ]

    main_rows.append({

        "Model":
            model_name,

        "Base_AUPRC":
            base_results[
                "AUPRC"
            ].mean(),

        "Base_ROC_AUC":
            base_results[
                "ROC_AUC"
            ].mean(),

        "Base_Recall":
            base_results[
                "Recall"
            ].mean(),

        "Base_F1":
            base_results[
                "F1"
            ].mean(),

        "Base_MCC":
            base_results[
                "MCC"
            ].mean(),

        "Base_FPR":
            base_results[
                "FPR"
            ].mean(),

        "Mean_Variant_AUPRC":
            variant_results[
                "AUPRC"
            ].mean(),

        "Worst_Variant_AUPRC":
            robust[
                "Worst_Variant_AUPRC"
            ].mean(),

        "AUPRC_Absolute_Drop":
            robust[
                "AUPRC_Absolute_Drop"
            ].mean(),

        "AUPRC_Retention":
            robust[
                "AUPRC_Retention_Ratio"
            ].mean(),

        "Mean_Variant_Recall":
            variant_results[
                "Recall"
            ].mean(),

        "Worst_Variant_Recall":
            robust[
                "Worst_Variant_Recall"
            ].mean(),

        "Mean_Variant_FPR":
            robust[
                "Mean_Variant_FPR"
            ].mean(),

        "Mean_Absolute_FPR_Drift":
            robust[
                "Mean_Absolute_FPR_Drift"
            ].mean(),

        "Training_Cost_sec":
            timing_model[
                "Total_Training_Cost_sec"
            ].mean(),

        "Variant_Inference_sec":
            timing_model[
                "Five_Variants_Inference_sec"
            ].mean()
    })


main_table = pd.DataFrame(
    main_rows
)

main_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "main_benchmark_table.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# DATASET INFORMATION
# ============================================================

dataset_rows = [{

    "Dataset":
        "Base Test (Month 7)",

    "Month":
        TEST_MONTH,

    "Rows":
        len(y_test),

    "Fraud_Count":
        int(y_test.sum()),

    "Fraud_Rate":
        float(y_test.mean())
}]


for name in VARIANTS:

    y = variant_data[name]["y"]

    dataset_rows.append({

        "Dataset":
            name,

        "Month":
            TEST_MONTH,

        "Rows":
            len(y),

        "Fraud_Count":
            int(y.sum()),

        "Fraud_Rate":
            float(y.mean())
    })


dataset_info = pd.DataFrame(
    dataset_rows
)

dataset_info.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dataset_info.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SETTINGS
# ============================================================

settings = {

    "experiment":
        "Experiment 05B - Final Time-Aligned ML Benchmark",

    "seeds":
        SEEDS,

    "models": [
        "Logistic Regression",
        "Random Forest",
        "Extra Trees",
        "XGBoost",
        "LightGBM",
        "Proposed IF-LGBM"
    ],

    "train_dataset":
        "Base",

    "train_months":
        TRAIN_MONTHS,

    "validation_dataset":
        "Base",

    "validation_month":
        VALIDATION_MONTH,

    "base_test_month":
        TEST_MONTH,

    "variant_test_month":
        TEST_MONTH,

    "time_aligned_evaluation":
        True,

    "month_used_as_predictor":
        False,

    "target_fpr":
        TARGET_FPR,

    "primary_metric":
        "AUPRC",

    "statistical_hypothesis_tests":
        False,

    "threshold_protocol":
        (
            "Threshold selected independently "
            "for each trained model on Base Month 6 "
            "negative examples to target approximately "
            "5% FPR; threshold then frozen for Base "
            "Month 7 and Variant I-V Month 7."
        ),

    "proposed_method":
        "Isolation Forest raw anomaly score + LightGBM",

    "IF_parameters":
        IF_PARAMS,

    "LightGBM_parameters":
        LGBM_PARAMS,

    "important_note":
        (
            "This experiment replaces Experiment 05 "
            "for final benchmark reporting because "
            "Base and Variant evaluation datasets are "
            "time-aligned to Month 7."
        )
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
# FINISH
# ============================================================

header(
    "EXPERIMENT 05B COMPLETE"
)

print(
    "\nDATASET INFORMATION\n"
)

print(
    dataset_info
    .round(6)
    .to_string(
        index=False
    )
)

print(
    "\nFINAL TIME-ALIGNED BENCHMARK\n"
)

print(
    main_table
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