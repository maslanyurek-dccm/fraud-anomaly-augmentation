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
# EXPERIMENT 05
# FINAL ML BENCHMARK
#
# Models:
# Logistic Regression
# Random Forest
# Extra Trees
# XGBoost
# LightGBM
# Proposed IF-LGBM
#
# 5 seeds
#
# Train      : Month 0-5
# Validation : Month 6
# Test       : Month 7
#
# Threshold is selected on validation negatives
# targeting 5% FPR and then frozen.
#
# month is NOT a predictor.
# ============================================================


SEEDS = [42, 52, 62, 72, 82]

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
    "experiment_05"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


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

    return float(
        np.quantile(
            negative_scores,
            1.0 - target_fpr
        )
    )


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

    models = {

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
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                random_state=seed,
                n_jobs=-1,
                verbosity=-1
            )
    }

    return models


# ============================================================
# LOAD BASE
# ============================================================

header(
    "EXPERIMENT 05 - FINAL ML BENCHMARK"
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


header("TEMPORAL SPLIT")

print(
    f"Train      : {len(train_df):,}"
)

print(
    f"Validation : {len(val_df):,}"
)

print(
    f"Test       : {len(test_df):,}"
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
    f"Number of predictors: "
    f"{len(feature_columns)}"
)

del X_train_raw
del X_val_raw
del X_test_raw

gc.collect()


# ============================================================
# LOAD VARIANTS
# ============================================================

header(
    "PREPROCESSING VARIANTS"
)

variant_data = {}

for name, filename in VARIANTS.items():

    print(
        f"Preparing {name}..."
    )

    start = time.time()

    df = pd.read_csv(
        filename
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

    # Removes x1/x2 in variants where present.
    X_raw = align_columns(
        X_raw,
        feature_columns
    )

    X = preprocessor.transform(
        X_raw
    )

    variant_data[name] = {
        "X": X,
        "y": y
    }

    print(
        f"  {len(y):,} rows prepared "
        f"in {time.time()-start:.2f} sec"
    )

    del df
    del X_raw

    gc.collect()


# ============================================================
# EXPERIMENT
# ============================================================

all_results = []
timing_results = []


for seed_no, seed in enumerate(
    SEEDS,
    start=1
):

    header(
        f"SEED {seed} "
        f"({seed_no}/{len(SEEDS)})"
    )

    models = create_models(
        seed
    )


    # ========================================================
    # STANDARD MODELS
    # ========================================================

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
            time.time() - start
        )

        # -------------------------
        # Validation threshold
        # -------------------------

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

        # -------------------------
        # Base Test
        # -------------------------

        start = time.time()

        test_prob = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        inference_time = (
            time.time() - start
        )

        metrics = calculate_metrics(
            y_test,
            test_prob,
            threshold
        )

        all_results.append({

            "Seed": seed,
            "Model": model_name,
            "Dataset":
                "Base Test (Month 7)",

            **metrics
        })

        print(
            f"  Base AUPRC="
            f"{metrics['AUPRC']:.5f} | "
            f"Recall="
            f"{metrics['Recall']:.5f} | "
            f"FPR="
            f"{metrics['FPR']:.5f}"
        )


        # -------------------------
        # Variants
        # -------------------------

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

            this_inference = (
                time.time() - start
            )

            variant_inference_total += (
                this_inference
            )

            metrics_variant = (
                calculate_metrics(
                    y_variant,
                    probabilities,
                    threshold
                )
            )

            all_results.append({

                "Seed": seed,
                "Model": model_name,
                "Dataset":
                    variant_name,

                **metrics_variant
            })

        timing_results.append({

            "Seed":
                seed,

            "Model":
                model_name,

            "Train_Time_sec":
                train_time,

            "Base_Test_Inference_sec":
                inference_time,

            "Five_Variants_Inference_sec":
                variant_inference_total,

            "IF_Fit_Time_sec":
                0.0,

            "Total_Training_Cost_sec":
                train_time
        })

        del model
        gc.collect()


    # ========================================================
    # PROPOSED IF-LGBM
    # ========================================================

    print(
        "\nTraining Proposed IF-LGBM..."
    )


    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=seed,
        n_jobs=-1
    )

    start = time.time()

    iso.fit(
        X_train
    )

    if_fit_time = (
        time.time() - start
    )


    # Higher = more anomalous
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


    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    proposed = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1
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


    # --------------------------------------------------------
    # Base Test
    # --------------------------------------------------------

    start = time.time()

    test_prob = (
        proposed.predict_proba(
            X_test_if
        )[:, 1]
    )

    base_inference = (
        time.time() - start
    )

    metrics = calculate_metrics(
        y_test,
        test_prob,
        threshold
    )

    all_results.append({

        "Seed": seed,
        "Model":
            "Proposed IF-LGBM",

        "Dataset":
            "Base Test (Month 7)",

        **metrics
    })

    print(
        f"  Base AUPRC="
        f"{metrics['AUPRC']:.5f} | "
        f"Recall="
        f"{metrics['Recall']:.5f} | "
        f"FPR="
        f"{metrics['FPR']:.5f}"
    )


    # --------------------------------------------------------
    # Variants
    # --------------------------------------------------------

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

        this_inference = (
            time.time() - start
        )

        variant_inference_total += (
            this_inference
        )

        metrics_variant = (
            calculate_metrics(
                y_variant,
                probabilities,
                threshold
            )
        )

        all_results.append({

            "Seed": seed,

            "Model":
                "Proposed IF-LGBM",

            "Dataset":
                variant_name,

            **metrics_variant
        })

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

        "Base_Test_Inference_sec":
            base_inference,

        "Five_Variants_Inference_sec":
            variant_inference_total,

        "IF_Fit_Time_sec":
            if_fit_time,

        "Total_Training_Cost_sec":
            (
                if_fit_time
                + lgbm_train_time
            )
    })


    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

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
# MEAN / SD
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
# ROBUSTNESS SUMMARY
# ============================================================

variant_names = list(
    VARIANTS.keys()
)

robustness_rows = []

for model_name in (
    results_df["Model"]
    .unique()
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

        base = seed_df[
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
                base["AUPRC"],

            "Mean_Variant_AUPRC":
                mean_variant_auprc,

            "Worst_Variant_AUPRC":
                worst_variant_auprc,

            "AUPRC_Absolute_Drop":
                (
                    base["AUPRC"]
                    - mean_variant_auprc
                ),

            "AUPRC_Retention_Ratio":
                (
                    mean_variant_auprc
                    / base["AUPRC"]
                ),

            "Base_Recall":
                base["Recall"],

            "Mean_Variant_Recall":
                mean_variant_recall,

            "Worst_Variant_Recall":
                worst_variant_recall,

            "Recall_Absolute_Drop":
                (
                    base["Recall"]
                    - mean_variant_recall
                ),

            "Recall_Retention_Ratio":
                (
                    mean_variant_recall
                    / base["Recall"]
                ),

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
# ROBUSTNESS MEAN / SD
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
# COMPACT MAIN TABLE
# ============================================================

main_rows = []

for model_name in (
    results_df["Model"]
    .unique()
):

    model_results = results_df[
        results_df["Model"]
        == model_name
    ]

    base = model_results[
        model_results["Dataset"]
        == "Base Test (Month 7)"
    ]

    variants = model_results[
        model_results["Dataset"]
        .isin(variant_names)
    ]

    model_robustness = (
        robustness_df[
            robustness_df["Model"]
            == model_name
        ]
    )

    model_timing = (
        timing_df[
            timing_df["Model"]
            == model_name
        ]
    )

    main_rows.append({

        "Model":
            model_name,

        "Base_AUPRC":
            base["AUPRC"].mean(),

        "Base_ROC_AUC":
            base["ROC_AUC"].mean(),

        "Base_Recall":
            base["Recall"].mean(),

        "Base_F1":
            base["F1"].mean(),

        "Base_MCC":
            base["MCC"].mean(),

        "Mean_Variant_AUPRC":
            variants["AUPRC"].mean(),

        "Worst_Variant_AUPRC":
            (
                model_robustness[
                    "Worst_Variant_AUPRC"
                ].mean()
            ),

        "AUPRC_Retention":
            (
                model_robustness[
                    "AUPRC_Retention_Ratio"
                ].mean()
            ),

        "Mean_Variant_Recall":
            variants["Recall"].mean(),

        "Mean_Absolute_FPR_Drift":
            (
                model_robustness[
                    "Mean_Absolute_FPR_Drift"
                ].mean()
            ),

        "Training_Cost_sec":
            (
                model_timing[
                    "Total_Training_Cost_sec"
                ].mean()
            )
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
# SETTINGS
# ============================================================

settings = {

    "experiment":
        "Experiment 05 - Final ML Benchmark",

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

    "primary_metric":
        "AUPRC",

    "secondary_metrics": [
        "ROC-AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "Balanced Accuracy",
        "FPR"
    ],

    "robustness_metrics": [
        "Mean Variant AUPRC",
        "Worst Variant AUPRC",
        "AUPRC Absolute Drop",
        "AUPRC Retention Ratio",
        "Mean Variant Recall",
        "Worst Variant Recall",
        "Mean Absolute FPR Drift",
        "Maximum Absolute FPR Drift"
    ],

    "statistical_hypothesis_tests":
        False,

    "proposed_method":
        "Isolation Forest anomaly score + LightGBM",

    "IF_parameters": {
        "n_estimators": 200,
        "contamination": "auto"
    },

    "LightGBM_parameters": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31
    }
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
    "EXPERIMENT 05 COMPLETE"
)

print(
    "\nMAIN BENCHMARK TABLE\n"
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