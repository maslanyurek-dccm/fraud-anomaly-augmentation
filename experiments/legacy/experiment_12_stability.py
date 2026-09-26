import os
import json
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)

from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


# ============================================================
# EXPERIMENT 12
# STOCHASTIC STABILITY ANALYSIS
#
# Research question:
# Does an ensemble consensus of multiple Isolation Forests
# reduce stochastic seed sensitivity compared with using
# one Isolation Forest representation?
#
# IMPORTANT:
# - No hyperparameter search
# - No model selection
# - No SMOTE
# - Same chronological protocol
# - Same LightGBM configuration
# - Each consensus uses exactly 5 IFs
# ============================================================


DATA_FILE = "creditcard.csv"

OUTPUT_DIR = os.path.join(
    "results",
    "experiment_12"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


TARGET = "Class"
TIME_COL = "Time"

TARGET_FPR = 0.05

LGBM_SEED = 42

IF_PER_ENSEMBLE = 5


# ============================================================
# PRE-SPECIFIED REPLICATE SEEDS
#
# 10 ensembles x 5 IFs = 50 unique IF seeds
# ============================================================

ENSEMBLE_SEEDS = {

    1:  [101, 102, 103, 104, 105],

    2:  [201, 202, 203, 204, 205],

    3:  [301, 302, 303, 304, 305],

    4:  [401, 402, 403, 404, 405],

    5:  [501, 502, 503, 504, 505],

    6:  [601, 602, 603, 604, 605],

    7:  [701, 702, 703, 704, 705],

    8:  [801, 802, 803, 804, 805],

    9:  [901, 902, 903, 904, 905],

    10: [1001, 1002, 1003, 1004, 1005]

}


# ============================================================
# HELPERS
# ============================================================

def percentile_from_train(
    sorted_train_scores,
    scores
):

    ranks = np.searchsorted(
        sorted_train_scores,
        np.asarray(
            scores,
            dtype=np.float64
        ),
        side="right"
    )

    return (
        ranks /
        len(sorted_train_scores)
    ).astype(
        np.float32
    )


def select_threshold(
    y_true,
    probabilities,
    target_fpr=0.05
):

    negative_probabilities = (
        probabilities[
            y_true == 0
        ]
    )

    return float(
        np.quantile(
            negative_probabilities,
            1.0 - target_fpr
        )
    )


def evaluate_probability_metrics(
    y_true,
    probability
):

    return {

        "AUPRC":
            average_precision_score(
                y_true,
                probability
            ),

        "ROC_AUC":
            roc_auc_score(
                y_true,
                probability
            )

    }


def train_lgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test
):

    model = LGBMClassifier(

        n_estimators=300,

        learning_rate=0.05,

        num_leaves=31,

        random_state=LGBM_SEED,

        n_jobs=-1,

        verbosity=-1

    )


    start = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    fit_time = (
        time.perf_counter()
        - start
    )


    val_prob = (
        model.predict_proba(
            X_val
        )[:, 1]
    )


    threshold = select_threshold(

        y_val,

        val_prob,

        TARGET_FPR

    )


    start = time.perf_counter()

    test_prob = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    inference_time = (
        time.perf_counter()
        - start
    )


    return (
        model,
        test_prob,
        threshold,
        fit_time,
        inference_time
    )


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "EXPERIMENT 12"
)

print(
    "STOCHASTIC STABILITY ANALYSIS"
)

print(
    "=" * 80
)


df = pd.read_csv(
    DATA_FILE
)


df = (
    df
    .sort_values(
        TIME_COL,
        kind="stable"
    )
    .reset_index(
        drop=True
    )
)


n = len(df)

train_end = int(
    n * 0.60
)

val_end = int(
    n * 0.80
)


train_df = (
    df.iloc[
        :train_end
    ]
    .copy()
)

val_df = (
    df.iloc[
        train_end:val_end
    ]
    .copy()
)

test_df = (
    df.iloc[
        val_end:
    ]
    .copy()
)


feature_cols = [

    c

    for c in df.columns

    if c not in [
        TARGET,
        TIME_COL
    ]

]


X_train = (
    train_df[
        feature_cols
    ]
    .astype(
        np.float32
    )
    .values
)

X_val = (
    val_df[
        feature_cols
    ]
    .astype(
        np.float32
    )
    .values
)

X_test = (
    test_df[
        feature_cols
    ]
    .astype(
        np.float32
    )
    .values
)


y_train = (
    train_df[
        TARGET
    ]
    .astype(int)
    .values
)

y_val = (
    val_df[
        TARGET
    ]
    .astype(int)
    .values
)

y_test = (
    test_df[
        TARGET
    ]
    .astype(int)
    .values
)


print(
    f"\nTrain      : {len(y_train)} "
    f"| fraud={y_train.sum()}"
)

print(
    f"Validation : {len(y_val)} "
    f"| fraud={y_val.sum()}"
)

print(
    f"Test       : {len(y_test)} "
    f"| fraud={y_test.sum()}"
)

print(
    f"Features   : {len(feature_cols)}"
)


# ============================================================
# STANDARDIZATION FOR IF
#
# TRAIN ONLY
# ============================================================

print(
    "\nScaling predictors for Isolation Forest..."
)

scaler = StandardScaler()

X_train_scaled = (
    scaler.fit_transform(
        X_train
    )
)

X_val_scaled = (
    scaler.transform(
        X_val
    )
)

X_test_scaled = (
    scaler.transform(
        X_test
    )
)


# ============================================================
# BASELINE LIGHTGBM
#
# Run once.
# ============================================================

print(
    "\nTraining baseline LightGBM..."
)


(
    baseline_model,
    baseline_prob,
    baseline_threshold,
    baseline_fit_time,
    baseline_inference_time

) = train_lgbm(

    X_train,
    y_train,

    X_val,
    y_val,

    X_test

)


baseline_metrics = (
    evaluate_probability_metrics(
        y_test,
        baseline_prob
    )
)


BASELINE_AUPRC = (
    baseline_metrics[
        "AUPRC"
    ]
)

BASELINE_ROC = (
    baseline_metrics[
        "ROC_AUC"
    ]
)


print(
    f"Baseline AUPRC  : "
    f"{BASELINE_AUPRC:.6f}"
)

print(
    f"Baseline ROC-AUC: "
    f"{BASELINE_ROC:.6f}"
)


# ============================================================
# RESULTS
# ============================================================

single_results = []

consensus_results = []

if_timing_rows = []

model_timing_rows = []


# ============================================================
# REPLICATES
# ============================================================

for replicate_id, seeds in (
    ENSEMBLE_SEEDS.items()
):

    print(
        "\n" + "=" * 80
    )

    print(
        f"ENSEMBLE REPLICATE "
        f"{replicate_id}/"
        f"{len(ENSEMBLE_SEEDS)}"
    )

    print(
        "Seeds:",
        seeds
    )

    print(
        "=" * 80
    )


    train_percentiles = []

    val_percentiles = []

    test_percentiles = []


    # ========================================================
    # FIVE IFs IN THIS REPLICATE
    # ========================================================

    for seed in seeds:

        print(
            f"\nTraining IF seed={seed}..."
        )


        iso = IsolationForest(

            n_estimators=200,

            contamination="auto",

            random_state=seed,

            n_jobs=-1

        )


        start = time.perf_counter()

        iso.fit(
            X_train_scaled
        )

        if_fit_time = (
            time.perf_counter()
            - start
        )


        start = time.perf_counter()

        train_raw = (
            -iso.decision_function(
                X_train_scaled
            )
        )

        val_raw = (
            -iso.decision_function(
                X_val_scaled
            )
        )

        test_raw = (
            -iso.decision_function(
                X_test_scaled
            )
        )

        if_score_time = (
            time.perf_counter()
            - start
        )


        sorted_train = np.sort(
            np.asarray(
                train_raw,
                dtype=np.float64
            )
        )


        train_pct = (
            percentile_from_train(
                sorted_train,
                train_raw
            )
        )

        val_pct = (
            percentile_from_train(
                sorted_train,
                val_raw
            )
        )

        test_pct = (
            percentile_from_train(
                sorted_train,
                test_raw
            )
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


        # ====================================================
        # SINGLE IF-LGBM
        #
        # IMPORTANT:
        # Uses raw anomaly score, preserving the previous
        # single-IF definition.
        # ====================================================

        Xtr_single = np.column_stack(
            [
                X_train,
                train_raw.astype(
                    np.float32
                )
            ]
        )

        Xv_single = np.column_stack(
            [
                X_val,
                val_raw.astype(
                    np.float32
                )
            ]
        )

        Xt_single = np.column_stack(
            [
                X_test,
                test_raw.astype(
                    np.float32
                )
            ]
        )


        (
            single_model,
            single_prob,
            single_threshold,
            single_fit_time,
            single_inference_time

        ) = train_lgbm(

            Xtr_single,
            y_train,

            Xv_single,
            y_val,

            Xt_single

        )


        single_metrics = (
            evaluate_probability_metrics(
                y_test,
                single_prob
            )
        )


        single_results.append({

            "Replicate":
                replicate_id,

            "IF_Seed":
                seed,

            "AUPRC":
                single_metrics[
                    "AUPRC"
                ],

            "ROC_AUC":
                single_metrics[
                    "ROC_AUC"
                ],

            "Delta_AUPRC_vs_LGBM":
                (
                    single_metrics[
                        "AUPRC"
                    ]
                    - BASELINE_AUPRC
                ),

            "Threshold":
                single_threshold

        })


        if_timing_rows.append({

            "Replicate":
                replicate_id,

            "IF_Seed":
                seed,

            "IF_Fit_sec":
                if_fit_time,

            "IF_Score_All_Splits_sec":
                if_score_time

        })


        model_timing_rows.append({

            "Model_Type":
                "Single_IF_LGBM",

            "Replicate":
                replicate_id,

            "IF_Seed":
                seed,

            "LGBM_Fit_sec":
                single_fit_time,

            "LGBM_Test_Inference_sec":
                single_inference_time

        })


        print(
            f"Single IF AUPRC="
            f"{single_metrics['AUPRC']:.6f}"
        )


    # ========================================================
    # CONSENSUS FOR THIS REPLICATE
    # ========================================================

    P_train = np.column_stack(
        train_percentiles
    )

    P_val = np.column_stack(
        val_percentiles
    )

    P_test = np.column_stack(
        test_percentiles
    )


    consensus_train = (
        np.mean(
            P_train,
            axis=1
        )
        .astype(
            np.float32
        )
    )

    consensus_val = (
        np.mean(
            P_val,
            axis=1
        )
        .astype(
            np.float32
        )
    )

    consensus_test = (
        np.mean(
            P_test,
            axis=1
        )
        .astype(
            np.float32
        )
    )


    Xtr_consensus = np.column_stack(
        [
            X_train,
            consensus_train
        ]
    )

    Xv_consensus = np.column_stack(
        [
            X_val,
            consensus_val
        ]
    )

    Xt_consensus = np.column_stack(
        [
            X_test,
            consensus_test
        ]
    )


    (
        consensus_model,
        consensus_prob,
        consensus_threshold,
        consensus_fit_time,
        consensus_inference_time

    ) = train_lgbm(

        Xtr_consensus,
        y_train,

        Xv_consensus,
        y_val,

        Xt_consensus

    )


    consensus_metrics = (
        evaluate_probability_metrics(
            y_test,
            consensus_prob
        )
    )


    # ========================================================
    # WITHIN-ENSEMBLE SINGLE IF STATISTICS
    # ========================================================

    current_single_values = [

        row["AUPRC"]

        for row in single_results

        if row["Replicate"]
        == replicate_id

    ]


    consensus_results.append({

        "Replicate":
            replicate_id,

        "Seeds":
            ",".join(
                map(
                    str,
                    seeds
                )
            ),

        "Consensus_AUPRC":
            consensus_metrics[
                "AUPRC"
            ],

        "Consensus_ROC_AUC":
            consensus_metrics[
                "ROC_AUC"
            ],

        "Delta_AUPRC_vs_LGBM":
            (
                consensus_metrics[
                    "AUPRC"
                ]
                - BASELINE_AUPRC
            ),

        "Single_IF_Mean_AUPRC":
            np.mean(
                current_single_values
            ),

        "Single_IF_SD_AUPRC":
            np.std(
                current_single_values,
                ddof=1
            ),

        "Single_IF_Min_AUPRC":
            np.min(
                current_single_values
            ),

        "Single_IF_Max_AUPRC":
            np.max(
                current_single_values
            ),

        "Consensus_minus_SingleMean":
            (
                consensus_metrics[
                    "AUPRC"
                ]
                - np.mean(
                    current_single_values
                )
            ),

        "Consensus_Threshold":
            consensus_threshold

    })


    model_timing_rows.append({

        "Model_Type":
            "Consensus_LGBM",

        "Replicate":
            replicate_id,

        "IF_Seed":
            np.nan,

        "LGBM_Fit_sec":
            consensus_fit_time,

        "LGBM_Test_Inference_sec":
            consensus_inference_time

    })


    print(
        "\nConsensus AUPRC = "
        f"{consensus_metrics['AUPRC']:.6f}"
    )

    print(
        "Single IF mean = "
        f"{np.mean(current_single_values):.6f}"
    )

    print(
        "Single IF SD   = "
        f"{np.std(current_single_values, ddof=1):.6f}"
    )


# ============================================================
# DATAFRAMES
# ============================================================

single_df = pd.DataFrame(
    single_results
)

consensus_df = pd.DataFrame(
    consensus_results
)

if_timing_df = pd.DataFrame(
    if_timing_rows
)

model_timing_df = pd.DataFrame(
    model_timing_rows
)


# ============================================================
# GLOBAL STABILITY SUMMARY
# ============================================================

single_values = (
    single_df[
        "AUPRC"
    ]
    .values
)

consensus_values = (
    consensus_df[
        "Consensus_AUPRC"
    ]
    .values
)


def descriptive_summary(
    values,
    method
):

    values = np.asarray(
        values,
        dtype=float
    )

    mean = np.mean(
        values
    )

    sd = np.std(
        values,
        ddof=1
    )

    minimum = np.min(
        values
    )

    maximum = np.max(
        values
    )

    return {

        "Method":
            method,

        "N":
            len(values),

        "Mean_AUPRC":
            mean,

        "SD_AUPRC":
            sd,

        "Min_AUPRC":
            minimum,

        "Max_AUPRC":
            maximum,

        "Range_AUPRC":
            maximum - minimum,

        "CV_percent":
            (
                (sd / mean) * 100
                if mean != 0
                else np.nan
            ),

        "Mean_Delta_vs_LGBM":
            mean - BASELINE_AUPRC

    }


stability_summary = pd.DataFrame(
    [

        descriptive_summary(
            single_values,
            "Single_IF_LGBM"
        ),

        descriptive_summary(
            consensus_values,
            "Consensus_LGBM"
        )

    ]
)


# ============================================================
# ADD STABILITY RATIOS
# ============================================================

single_sd = (
    stability_summary.loc[
        stability_summary[
            "Method"
        ]
        == "Single_IF_LGBM",
        "SD_AUPRC"
    ]
    .iloc[0]
)

consensus_sd = (
    stability_summary.loc[
        stability_summary[
            "Method"
        ]
        == "Consensus_LGBM",
        "SD_AUPRC"
    ]
    .iloc[0]
)


single_range = (
    np.ptp(
        single_values
    )
)

consensus_range = (
    np.ptp(
        consensus_values
    )
)


stability_comparison = pd.DataFrame(
    [
        {

            "Baseline_LGBM_AUPRC":
                BASELINE_AUPRC,

            "Single_IF_SD":
                single_sd,

            "Consensus_SD":
                consensus_sd,

            "SD_Ratio_Consensus_over_Single":
                (
                    consensus_sd
                    / single_sd
                ),

            "SD_Reduction_percent":
                (
                    (
                        1
                        - consensus_sd
                        / single_sd
                    )
                    * 100
                ),

            "Single_IF_Range":
                single_range,

            "Consensus_Range":
                consensus_range,

            "Range_Reduction_percent":
                (
                    (
                        1
                        - consensus_range
                        / single_range
                    )
                    * 100
                ),

            "Single_IF_Below_Baseline_Count":
                int(
                    np.sum(
                        single_values
                        < BASELINE_AUPRC
                    )
                ),

            "Consensus_Below_Baseline_Count":
                int(
                    np.sum(
                        consensus_values
                        < BASELINE_AUPRC
                    )
                ),

            "Single_IF_Positive_Delta_Rate":
                float(
                    np.mean(
                        single_values
                        > BASELINE_AUPRC
                    )
                ),

            "Consensus_Positive_Delta_Rate":
                float(
                    np.mean(
                        consensus_values
                        > BASELINE_AUPRC
                    )
                )

        }
    ]
)


# ============================================================
# CONSENSUS VS WITHIN-REPLICATE SINGLE IF
# ============================================================

consensus_df[
    "Consensus_Better_Than_SingleMean"
] = (

    consensus_df[
        "Consensus_AUPRC"
    ]

    >

    consensus_df[
        "Single_IF_Mean_AUPRC"
    ]

)


consensus_df[
    "Consensus_Better_Than_SingleMax"
] = (

    consensus_df[
        "Consensus_AUPRC"
    ]

    >

    consensus_df[
        "Single_IF_Max_AUPRC"
    ]

)


# ============================================================
# SAVE FILES
# ============================================================

single_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "single_if_50_seeds.csv"
    ),
    index=False
)


consensus_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "consensus_10_replicates.csv"
    ),
    index=False
)


stability_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "stability_summary.csv"
    ),
    index=False
)


stability_comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "stability_comparison.csv"
    ),
    index=False
)


if_timing_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "if_timing.csv"
    ),
    index=False
)


model_timing_df.to_csv(
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
        "Experiment 12 - Stochastic Stability Analysis",

    "dataset":
        DATA_FILE,

    "research_question":
        (
            "Does five-model Isolation Forest consensus "
            "reduce stochastic anomaly-representation "
            "variability relative to single Isolation Forest "
            "augmentation?"
        ),

    "split":
        "Chronological 60/20/20",

    "time_used_as_predictor":
        False,

    "ensemble_replicates":
        len(
            ENSEMBLE_SEEDS
        ),

    "if_per_ensemble":
        IF_PER_ENSEMBLE,

    "total_unique_if_seeds":
        int(
            sum(
                len(x)
                for x in
                ENSEMBLE_SEEDS.values()
            )
        ),

    "ensemble_seeds":
        ENSEMBLE_SEEDS,

    "if_parameters": {

        "n_estimators":
            200,

        "contamination":
            "auto"

    },

    "consensus_definition":
        (
            "Arithmetic mean of five train-referenced "
            "Isolation Forest anomaly percentiles."
        ),

    "single_if_definition":
        (
            "Raw negative IsolationForest decision_function."
        ),

    "lightgbm_parameters": {

        "n_estimators":
            300,

        "learning_rate":
            0.05,

        "num_leaves":
            31,

        "random_state":
            LGBM_SEED

    },

    "target_fpr":
        TARGET_FPR,

    "primary_metric":
        "AUPRC",

    "statistics":
        (
            "Descriptive stability analysis only; "
            "no inferential hypothesis test."
        ),

    "hyperparameter_search":
        False,

    "purpose":
        (
            "Quantify whether consensus reduces the seed "
            "sensitivity observed for single-IF anomaly "
            "augmentation."
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
# PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STABILITY SUMMARY"
)

print(
    "=" * 80
)


print(
    stability_summary
    .round(6)
    .to_string(
        index=False
    )
)


print(
    "\n" + "=" * 80
)

print(
    "STABILITY COMPARISON"
)

print(
    "=" * 80
)


print(
    stability_comparison
    .round(6)
    .to_string(
        index=False
    )
)


print(
    "\n" + "=" * 80
)

print(
    "CONSENSUS REPLICATES"
)

print(
    "=" * 80
)


print(

    consensus_df[
        [
            "Replicate",
            "Consensus_AUPRC",
            "Single_IF_Mean_AUPRC",
            "Single_IF_SD_AUPRC",
            "Single_IF_Min_AUPRC",
            "Single_IF_Max_AUPRC",
            "Consensus_minus_SingleMean",
            "Consensus_Better_Than_SingleMean",
            "Consensus_Better_Than_SingleMax"
        ]
    ]

    .round(6)

    .to_string(
        index=False
    )

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
    "\nEXPERIMENT 12 COMPLETE."
)