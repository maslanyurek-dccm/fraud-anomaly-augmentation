"""Earlier SHAP analysis of the consensus feature on CreditCard.

Fits the consensus configuration for the replicates of experiment 12 and
records the SHAP rank of the added feature. creditcard.csv is read from the
working directory; results go to results/experiment_14.
"""
import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")


DATA_FILE = "creditcard.csv"

OUTPUT_DIR = os.path.join(
    "results",
    "experiment_14"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "Class"
TIME_COL = "Time"

TARGET_FPR = 0.05

REPLICATES = {
    2: [201, 202, 203, 204, 205],
    8: [801, 802, 803, 804, 805]
}

CONSENSUS_NAME = "IF_Consensus"

# SHAP on the full 56k test set is unnecessary.
# Fixed sample, with all fraud cases retained.
SHAP_NORMAL_SAMPLE = 5000
SHAP_RANDOM_STATE = 42


# Helpers

def percentile_from_train(sorted_train_scores, scores):

    ranks = np.searchsorted(
        sorted_train_scores,
        np.asarray(scores, dtype=np.float64),
        side="right"
    )

    return (
        ranks / len(sorted_train_scores)
    ).astype(np.float32)


def threshold_for_target_fpr(y_val, prob_val, target_fpr=0.05):

    negative_scores = prob_val[y_val == 0]

    threshold = np.quantile(
        negative_scores,
        1.0 - target_fpr
    )

    return float(threshold)


def create_lgbm():

    return lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )


def calculate_consensus(
    X_train_scaled,
    X_val_scaled,
    X_test_scaled,
    seeds
):

    train_percentiles = []
    val_percentiles = []
    test_percentiles = []

    timing_rows = []

    for seed in seeds:

        print(f"    Isolation Forest seed={seed}")

        iso = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=seed,
            n_jobs=-1
        )

        t0 = time.perf_counter()

        iso.fit(X_train_scaled)

        fit_sec = time.perf_counter() - t0


        t0 = time.perf_counter()

        train_raw = -iso.decision_function(
            X_train_scaled
        )

        val_raw = -iso.decision_function(
            X_val_scaled
        )

        test_raw = -iso.decision_function(
            X_test_scaled
        )

        score_sec = time.perf_counter() - t0


        sorted_train = np.sort(
            np.asarray(train_raw, dtype=np.float64)
        )


        train_pct = percentile_from_train(
            sorted_train,
            train_raw
        )

        val_pct = percentile_from_train(
            sorted_train,
            val_raw
        )

        test_pct = percentile_from_train(
            sorted_train,
            test_raw
        )


        train_percentiles.append(train_pct)
        val_percentiles.append(val_pct)
        test_percentiles.append(test_pct)


        timing_rows.append({
            "IF_Seed": seed,
            "IF_Fit_sec": fit_sec,
            "IF_Score_AllSplits_sec": score_sec
        })


    train_consensus = np.mean(
        np.column_stack(train_percentiles),
        axis=1
    ).astype(np.float32)

    val_consensus = np.mean(
        np.column_stack(val_percentiles),
        axis=1
    ).astype(np.float32)

    test_consensus = np.mean(
        np.column_stack(test_percentiles),
        axis=1
    ).astype(np.float32)


    return (
        train_consensus,
        val_consensus,
        test_consensus,
        timing_rows
    )


def get_shap_values(model, X):

    explainer = shap.TreeExplainer(model)

    values = explainer.shap_values(X)

    # Compatibility with different SHAP versions
    if isinstance(values, list):
        if len(values) == 2:
            values = values[1]
        else:
            values = values[-1]

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, -1]

    return values


# Load data

print("\n" + "=" * 80)
print("EXPERIMENT 14 - SHAP CONSENSUS ANALYSIS")
print("=" * 80)

df = pd.read_csv(DATA_FILE)

df = (
    df.sort_values(
        TIME_COL,
        kind="stable"
    )
    .reset_index(drop=True)
)


# Chronological 60 / 20 / 20

n = len(df)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()


feature_cols = [
    c for c in df.columns
    if c not in [TARGET, TIME_COL]
]


X_train_df = (
    train_df[feature_cols]
    .astype(np.float32)
)

X_val_df = (
    val_df[feature_cols]
    .astype(np.float32)
)

X_test_df = (
    test_df[feature_cols]
    .astype(np.float32)
)


y_train = (
    train_df[TARGET]
    .astype(int)
    .values
)

y_val = (
    val_df[TARGET]
    .astype(int)
    .values
)

y_test = (
    test_df[TARGET]
    .astype(int)
    .values
)


print(
    f"Train      : {len(y_train)} | fraud={y_train.sum()}"
)

print(
    f"Validation : {len(y_val)} | fraud={y_val.sum()}"
)

print(
    f"Test       : {len(y_test)} | fraud={y_test.sum()}"
)

print(
    f"Original features: {len(feature_cols)}"
)


# Standardization for isolation forest only

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_df.values
)

X_val_scaled = scaler.transform(
    X_val_df.values
)

X_test_scaled = scaler.transform(
    X_test_df.values
)


# SHAP sample
# Keep ALL fraud observations.
# Add fixed random sample of normal observations.
# Same rows used for all three models.

fraud_indices = np.where(
    y_test == 1
)[0]

normal_indices = np.where(
    y_test == 0
)[0]


rng = np.random.default_rng(
    SHAP_RANDOM_STATE
)

sample_size = min(
    SHAP_NORMAL_SAMPLE,
    len(normal_indices)
)

sampled_normal_indices = rng.choice(
    normal_indices,
    size=sample_size,
    replace=False
)


shap_indices = np.concatenate([
    fraud_indices,
    sampled_normal_indices
])

shap_indices = np.sort(shap_indices)

y_shap = y_test[shap_indices]


print(
    f"\nSHAP sample: {len(shap_indices)}"
)

print(
    f"  Fraud : {(y_shap == 1).sum()}"
)

print(
    f"  Normal: {(y_shap == 0).sum()}"
)


# Result containers

performance_rows = []
importance_rows = []
consensus_rows = []
class_shap_rows = []
timing_rows = []


# M0 - plain lightgbm

print("\nTraining Plain LightGBM...")

plain_model = create_lgbm()

t0 = time.perf_counter()

plain_model.fit(
    X_train_df,
    y_train
)

plain_fit_sec = time.perf_counter() - t0


val_prob = plain_model.predict_proba(
    X_val_df
)[:, 1]

test_prob = plain_model.predict_proba(
    X_test_df
)[:, 1]


plain_threshold = threshold_for_target_fpr(
    y_val,
    val_prob,
    TARGET_FPR
)


plain_auprc = average_precision_score(
    y_test,
    test_prob
)

plain_roc = roc_auc_score(
    y_test,
    test_prob
)


performance_rows.append({
    "Model": "Plain_LGBM",
    "Replicate": 0,
    "AUPRC": plain_auprc,
    "ROC_AUC": plain_roc,
    "Validation_Threshold": plain_threshold
})


# SHAP - plain model

print("Calculating SHAP for Plain LightGBM...")

X_plain_shap = X_test_df.iloc[
    shap_indices
].copy()


t0 = time.perf_counter()

plain_shap = get_shap_values(
    plain_model,
    X_plain_shap
)

plain_shap_sec = time.perf_counter() - t0


plain_mean_abs = np.mean(
    np.abs(plain_shap),
    axis=0
)


for feature, value in zip(
    feature_cols,
    plain_mean_abs
):

    importance_rows.append({
        "Model": "Plain_LGBM",
        "Replicate": 0,
        "Feature": feature,
        "Mean_Abs_SHAP": value
    })


timing_rows.append({
    "Model": "Plain_LGBM",
    "Component": "LightGBM_Fit",
    "Seconds": plain_fit_sec
})

timing_rows.append({
    "Model": "Plain_LGBM",
    "Component": "SHAP",
    "Seconds": plain_shap_sec
})


# Consensus replicates 2 and 8

for replicate_id, seeds in REPLICATES.items():

    print("\n" + "=" * 80)
    print(
        f"Consensus Replicate {replicate_id}"
    )
    print(
        f"Seeds: {seeds}"
    )
    print("=" * 80)


    (
        train_consensus,
        val_consensus,
        test_consensus,
        if_times

    ) = calculate_consensus(

        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        seeds
    )


    # Augment original feature space

    X_train_aug = X_train_df.copy()
    X_val_aug = X_val_df.copy()
    X_test_aug = X_test_df.copy()

    X_train_aug[CONSENSUS_NAME] = (
        train_consensus
    )

    X_val_aug[CONSENSUS_NAME] = (
        val_consensus
    )

    X_test_aug[CONSENSUS_NAME] = (
        test_consensus
    )


    # Train exact same LightGBM configuration

    model = create_lgbm()

    print("  Training Consensus-LGBM...")

    t0 = time.perf_counter()

    model.fit(
        X_train_aug,
        y_train
    )

    fit_sec = time.perf_counter() - t0


    val_prob = model.predict_proba(
        X_val_aug
    )[:, 1]

    test_prob = model.predict_proba(
        X_test_aug
    )[:, 1]


    threshold = threshold_for_target_fpr(
        y_val,
        val_prob,
        TARGET_FPR
    )


    auprc = average_precision_score(
        y_test,
        test_prob
    )

    roc = roc_auc_score(
        y_test,
        test_prob
    )


    performance_rows.append({
        "Model": "Consensus_LGBM",
        "Replicate": replicate_id,
        "AUPRC": auprc,
        "ROC_AUC": roc,
        "Validation_Threshold": threshold
    })


    # SHAP

    print("  Calculating SHAP...")

    X_aug_shap = X_test_aug.iloc[
        shap_indices
    ].copy()


    t0 = time.perf_counter()

    shap_values = get_shap_values(
        model,
        X_aug_shap
    )

    shap_sec = time.perf_counter() - t0


    augmented_features = (
        feature_cols
        + [CONSENSUS_NAME]
    )


    mean_abs = np.mean(
        np.abs(shap_values),
        axis=0
    )


    # Global importance

    for feature, value in zip(
        augmented_features,
        mean_abs
    ):

        importance_rows.append({
            "Model": "Consensus_LGBM",
            "Replicate": replicate_id,
            "Feature": feature,
            "Mean_Abs_SHAP": value
        })


    # Consensus feature position

    consensus_idx = augmented_features.index(
        CONSENSUS_NAME
    )

    consensus_shap = shap_values[
        :,
        consensus_idx
    ]

    consensus_feature_values = (
        X_aug_shap[
            CONSENSUS_NAME
        ].values
    )


    consensus_global_importance = float(
        np.mean(
            np.abs(
                consensus_shap
            )
        )
    )


    order = np.argsort(
        -mean_abs
    )

    consensus_rank = int(
        np.where(
            order == consensus_idx
        )[0][0]
        + 1
    )


    # Fraud vs normal SHAP behavior

    fraud_mask = (
        y_shap == 1
    )

    normal_mask = (
        y_shap == 0
    )


    fraud_mean_abs_shap = float(
        np.mean(
            np.abs(
                consensus_shap[
                    fraud_mask
                ]
            )
        )
    )

    normal_mean_abs_shap = float(
        np.mean(
            np.abs(
                consensus_shap[
                    normal_mask
                ]
            )
        )
    )


    fraud_mean_shap = float(
        np.mean(
            consensus_shap[
                fraud_mask
            ]
        )
    )

    normal_mean_shap = float(
        np.mean(
            consensus_shap[
                normal_mask
            ]
        )
    )


    fraud_positive_rate = float(
        np.mean(
            consensus_shap[
                fraud_mask
            ] > 0
        )
    )

    normal_positive_rate = float(
        np.mean(
            consensus_shap[
                normal_mask
            ] > 0
        )
    )


    consensus_rows.append({

        "Replicate":
            replicate_id,

        "AUPRC":
            auprc,

        "ROC_AUC":
            roc,

        "Consensus_MeanAbsSHAP":
            consensus_global_importance,

        "Consensus_SHAP_Rank":
            consensus_rank,

        "Fraud_MeanAbsSHAP":
            fraud_mean_abs_shap,

        "Normal_MeanAbsSHAP":
            normal_mean_abs_shap,

        "Fraud_MeanSHAP":
            fraud_mean_shap,

        "Normal_MeanSHAP":
            normal_mean_shap,

        "Fraud_PositiveSHAP_Rate":
            fraud_positive_rate,

        "Normal_PositiveSHAP_Rate":
            normal_positive_rate

    })


    # Save per-observation consensus SHAP

    temp = pd.DataFrame({

        "Replicate":
            replicate_id,

        "Test_Index":
            shap_indices,

        "Class":
            y_shap,

        "Consensus_Value":
            consensus_feature_values,

        "Consensus_SHAP":
            consensus_shap,

        "Abs_Consensus_SHAP":
            np.abs(
                consensus_shap
            )

    })


    class_shap_rows.append(
        temp
    )


    # Timing

    timing_rows.append({
        "Model":
            f"Consensus_LGBM_R{replicate_id}",

        "Component":
            "LightGBM_Fit",

        "Seconds":
            fit_sec
    })

    timing_rows.append({
        "Model":
            f"Consensus_LGBM_R{replicate_id}",

        "Component":
            "SHAP",

        "Seconds":
            shap_sec
    })


    for row in if_times:

        timing_rows.append({

            "Model":
                f"Consensus_LGBM_R{replicate_id}",

            "Component":
                f"IF_seed_{row['IF_Seed']}",

            "Seconds":
                (
                    row["IF_Fit_sec"]
                    +
                    row[
                        "IF_Score_AllSplits_sec"
                    ]
                )

        })


# Dataframes

performance_df = pd.DataFrame(
    performance_rows
)

importance_df = pd.DataFrame(
    importance_rows
)

consensus_df = pd.DataFrame(
    consensus_rows
)

class_shap_df = pd.concat(
    class_shap_rows,
    ignore_index=True
)

timing_df = pd.DataFrame(
    timing_rows
)


# Feature ranks

importance_df["SHAP_Rank"] = (

    importance_df

    .groupby(
        ["Model", "Replicate"]
    )["Mean_Abs_SHAP"]

    .rank(
        ascending=False,
        method="min"
    )

    .astype(int)

)


importance_df = importance_df.sort_values(
    [
        "Model",
        "Replicate",
        "SHAP_Rank"
    ]
)


# Top-10 features

top10_df = (

    importance_df[
        importance_df[
            "SHAP_Rank"
        ] <= 10
    ]

    .copy()

)


# Save

performance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_performance.csv"
    ),
    index=False
)


importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "shap_feature_importance.csv"
    ),
    index=False
)


top10_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "shap_top10_features.csv"
    ),
    index=False
)


consensus_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "consensus_shap_summary.csv"
    ),
    index=False
)


class_shap_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "consensus_shap_observations.csv"
    ),
    index=False
)


timing_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "timing.csv"
    ),
    index=False
)


# Settings

settings = {

    "experiment":
        "Experiment 14 - SHAP Analysis",

    "dataset":
        DATA_FILE,

    "split":
        "Chronological 60/20/20",

    "time_used_as_predictor":
        False,

    "models": [
        "Plain LightGBM",
        "Consensus-LGBM Replicate 2",
        "Consensus-LGBM Replicate 8"
    ],

    "replicate_2_seeds":
        REPLICATES[2],

    "replicate_8_seeds":
        REPLICATES[8],

    "consensus_definition":
        (
            "Arithmetic mean of five "
            "train-referenced IF anomaly percentiles"
        ),

    "lightgbm_parameters": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42
    },

    "if_parameters": {
        "n_estimators": 200,
        "contamination": "auto"
    },

    "shap_normal_sample":
        SHAP_NORMAL_SAMPLE,

    "all_test_frauds_included":
        True,

    "new_model_selection":
        False,

    "hyperparameter_search":
        False,

    "purpose":
        (
            "Explain how the IF consensus feature is used "
            "by LightGBM in a successful and a failure-case "
            "ensemble replicate."
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


# Print results

print("\n" + "=" * 80)
print("MODEL PERFORMANCE")
print("=" * 80)

print(
    performance_df
    .round(6)
    .to_string(index=False)
)


print("\n" + "=" * 80)
print("CONSENSUS SHAP SUMMARY")
print("=" * 80)

print(
    consensus_df
    .round(6)
    .to_string(index=False)
)


print("\n" + "=" * 80)
print("TOP-10 SHAP FEATURES")
print("=" * 80)

print(
    top10_df[
        [
            "Model",
            "Replicate",
            "SHAP_Rank",
            "Feature",
            "Mean_Abs_SHAP"
        ]
    ]
    .round(6)
    .to_string(index=False)
)


print("\nResults saved to:")
print(
    os.path.abspath(
        OUTPUT_DIR
    )
)

print("\nEXPERIMENT 14 COMPLETE.")
