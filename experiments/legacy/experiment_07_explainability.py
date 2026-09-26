import os
import gc
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import IsolationForest

import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")


# ============================================================
# EXPERIMENT 07
# EXPLAINABILITY AND ANOMALY-SCORE CONTRIBUTION
#
# Models:
#   1. LightGBM
#   2. IF-LGBM
#
# Protocol frozen from Experiment 05B:
#   Train = Base Month 0-5
#   Explain = Base Month 7
#   month is NOT a predictor
#
# Seed = 42
#
# Goals:
#   - Global feature importance
#   - SHAP feature importance
#   - Rank of IF anomaly score
#   - Contribution of anomaly score
#
# No model selection.
# No hyperparameter search.
# No hypothesis testing.
# ============================================================


SEED = 42

TARGET = "fraud_bool"
TIME_COL = "month"

TRAIN_MONTHS = [0, 1, 2, 3, 4, 5]
TEST_MONTH = 7

BASE_FILE = "Base.csv"

SHAP_SAMPLE_SIZE = 20_000

OUTPUT_DIR = os.path.join(
    "results",
    "experiment_07"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS - FROZEN FROM EXPERIMENT 05B
# ============================================================

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": SEED
}

IF_PARAMS = {
    "n_estimators": 200,
    "contamination": "auto",
    "n_jobs": -1,
    "random_state": SEED
}


# ============================================================
# HELPERS
# ============================================================

def header(text):
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)


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

        X = X[
            self.feature_columns
        ].copy()

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

            encoded_cat = self.encoder.transform(
                X[
                    self.categorical_columns
                ].astype(str)
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

        transformed = (
            self.scaler
            .transform(encoded)
            .astype(np.float32)
        )

        del encoded
        gc.collect()

        return transformed


def save_importance_plot(
    df,
    value_column,
    title,
    filename,
    top_n=15
):

    plot_df = (
        df
        .sort_values(
            value_column,
            ascending=False
        )
        .head(top_n)
        .sort_values(
            value_column,
            ascending=True
        )
    )

    plt.figure(
        figsize=(9, 7)
    )

    plt.barh(
        plot_df["Feature"],
        plot_df[value_column]
    )

    plt.xlabel(
        value_column.replace("_", " ")
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def calculate_shap_importance(
    model,
    X,
    feature_names
):

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X
    )

    # Compatibility across SHAP versions
    if isinstance(
        shap_values,
        list
    ):

        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[-1]

    shap_values = np.asarray(
        shap_values
    )

    # Some SHAP versions can return
    # (samples, features, classes)
    if shap_values.ndim == 3:

        shap_values = (
            shap_values[:, :, -1]
        )

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0
    )

    importance = pd.DataFrame({

        "Feature":
            feature_names,

        "Mean_Absolute_SHAP":
            mean_abs_shap
    })

    importance = (
        importance
        .sort_values(
            "Mean_Absolute_SHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance[
        "SHAP_Rank"
    ] = np.arange(
        1,
        len(importance) + 1
    )

    return (
        importance,
        shap_values
    )


# ============================================================
# LOAD DATA
# ============================================================

header(
    "EXPERIMENT 07 - EXPLAINABILITY"
)

print(
    "Loading Base.csv..."
)

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
        TRAIN_MONTHS
    )
].copy()

test_df = base[
    base[TIME_COL]
    == TEST_MONTH
].copy()

print(
    f"Train rows : "
    f"{len(train_df):,}"
)

print(
    f"Test rows  : "
    f"{len(test_df):,}"
)


# ============================================================
# REMOVE TARGET + MONTH
# ============================================================

X_train_raw = train_df.drop(
    columns=[
        TARGET,
        TIME_COL
    ]
)

y_train = (
    train_df[TARGET]
    .astype(np.int8)
    .values
)

X_test_raw = test_df.drop(
    columns=[
        TARGET,
        TIME_COL
    ]
)

y_test = (
    test_df[TARGET]
    .astype(np.int8)
    .values
)

del base
del train_df
del test_df

gc.collect()


# ============================================================
# PREPROCESSING
# ============================================================

header(
    "PREPROCESSING"
)

preprocessor = BAFPreprocessor()

preprocessor.fit(
    X_train_raw
)

feature_names = (
    preprocessor.feature_columns
)

X_train = preprocessor.transform(
    X_train_raw
)

X_test = preprocessor.transform(
    X_test_raw
)

print(
    f"Number of predictors: "
    f"{len(feature_names)}"
)

del X_train_raw
del X_test_raw

gc.collect()


# ============================================================
# SHAP SAMPLE
#
# Random sample from Base Month 7.
# Same rows used for both models.
# ============================================================

rng = np.random.default_rng(
    SEED
)

sample_size = min(
    SHAP_SAMPLE_SIZE,
    len(X_test)
)

sample_indices = rng.choice(
    len(X_test),
    size=sample_size,
    replace=False
)

X_shap = X_test[
    sample_indices
]

y_shap = y_test[
    sample_indices
]

print(
    f"SHAP sample size: "
    f"{len(X_shap):,}"
)

print(
    f"SHAP sample fraud rate: "
    f"{y_shap.mean()*100:.4f}%"
)


# ============================================================
# MODEL 1 - LIGHTGBM
# ============================================================

header(
    "LIGHTGBM"
)

lightgbm_model = (
    lgb.LGBMClassifier(
        **LGBM_PARAMS
    )
)

start = time.time()

lightgbm_model.fit(
    X_train,
    y_train
)

lgbm_train_time = (
    time.time() - start
)

print(
    f"Training time: "
    f"{lgbm_train_time:.3f} sec"
)


# ============================================================
# LIGHTGBM BUILT-IN FEATURE IMPORTANCE
# ============================================================

lgbm_builtin = pd.DataFrame({

    "Feature":
        feature_names,

    "Gain_Importance":
        lightgbm_model.booster_
        .feature_importance(
            importance_type="gain"
        ),

    "Split_Importance":
        lightgbm_model.booster_
        .feature_importance(
            importance_type="split"
        )
})

lgbm_builtin[
    "Gain_Rank"
] = (
    lgbm_builtin[
        "Gain_Importance"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

lgbm_builtin = (
    lgbm_builtin
    .sort_values(
        "Gain_Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

lgbm_builtin.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lightgbm_builtin_importance.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# LIGHTGBM SHAP
# ============================================================

print(
    "Calculating LightGBM SHAP..."
)

start = time.time()

(
    lgbm_shap_importance,
    lgbm_shap_values
) = calculate_shap_importance(
    lightgbm_model,
    X_shap,
    feature_names
)

lgbm_shap_time = (
    time.time() - start
)

print(
    f"SHAP completed in "
    f"{lgbm_shap_time:.2f} sec"
)

lgbm_shap_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lightgbm_shap_importance.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# MODEL 2 - ISOLATION FOREST
# ============================================================

header(
    "ISOLATION FOREST"
)

isolation_forest = (
    IsolationForest(
        **IF_PARAMS
    )
)

start = time.time()

isolation_forest.fit(
    X_train
)

if_fit_time = (
    time.time() - start
)

print(
    f"IF fit time: "
    f"{if_fit_time:.3f} sec"
)


# ============================================================
# CREATE ANOMALY FEATURES
# ============================================================

print(
    "Creating anomaly scores..."
)

train_anomaly = (
    -isolation_forest
    .decision_function(
        X_train
    )
    .astype(np.float32)
)

shap_anomaly = (
    -isolation_forest
    .decision_function(
        X_shap
    )
    .astype(np.float32)
)

X_train_if = np.column_stack(
    [
        X_train,
        train_anomaly
    ]
).astype(np.float32)

X_shap_if = np.column_stack(
    [
        X_shap,
        shap_anomaly
    ]
).astype(np.float32)

if_feature_names = (
    feature_names
    + ["IF_anomaly_score"]
)


# ============================================================
# MODEL 2 - IF-LGBM
# ============================================================

header(
    "IF-LGBM"
)

if_lgbm_model = (
    lgb.LGBMClassifier(
        **LGBM_PARAMS
    )
)

start = time.time()

if_lgbm_model.fit(
    X_train_if,
    y_train
)

if_lgbm_train_time = (
    time.time() - start
)

print(
    f"LightGBM fit time: "
    f"{if_lgbm_train_time:.3f} sec"
)

print(
    f"IF + LightGBM cost: "
    f"{if_fit_time + if_lgbm_train_time:.3f} sec"
)


# ============================================================
# IF-LGBM BUILT-IN IMPORTANCE
# ============================================================

if_lgbm_builtin = pd.DataFrame({

    "Feature":
        if_feature_names,

    "Gain_Importance":
        if_lgbm_model.booster_
        .feature_importance(
            importance_type="gain"
        ),

    "Split_Importance":
        if_lgbm_model.booster_
        .feature_importance(
            importance_type="split"
        )
})

if_lgbm_builtin[
    "Gain_Rank"
] = (
    if_lgbm_builtin[
        "Gain_Importance"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

if_lgbm_builtin = (
    if_lgbm_builtin
    .sort_values(
        "Gain_Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

if_lgbm_builtin.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "if_lgbm_builtin_importance.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# IF-LGBM SHAP
# ============================================================

print(
    "Calculating IF-LGBM SHAP..."
)

start = time.time()

(
    if_lgbm_shap_importance,
    if_lgbm_shap_values
) = calculate_shap_importance(
    if_lgbm_model,
    X_shap_if,
    if_feature_names
)

if_lgbm_shap_time = (
    time.time() - start
)

print(
    f"SHAP completed in "
    f"{if_lgbm_shap_time:.2f} sec"
)

if_lgbm_shap_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "if_lgbm_shap_importance.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# ANOMALY SCORE ANALYSIS
# ============================================================

header(
    "ANOMALY SCORE CONTRIBUTION"
)

if_shap_row = (
    if_lgbm_shap_importance[
        if_lgbm_shap_importance[
            "Feature"
        ] == "IF_anomaly_score"
    ]
    .iloc[0]
)

if_builtin_row = (
    if_lgbm_builtin[
        if_lgbm_builtin[
            "Feature"
        ] == "IF_anomaly_score"
    ]
    .iloc[0]
)

fraud_anomaly = shap_anomaly[
    y_shap == 1
]

normal_anomaly = shap_anomaly[
    y_shap == 0
]

anomaly_summary = pd.DataFrame([{

    "Feature":
        "IF_anomaly_score",

    "SHAP_Rank":
        int(
            if_shap_row[
                "SHAP_Rank"
            ]
        ),

    "Mean_Absolute_SHAP":
        float(
            if_shap_row[
                "Mean_Absolute_SHAP"
            ]
        ),

    "Gain_Rank":
        int(
            if_builtin_row[
                "Gain_Rank"
            ]
        ),

    "Gain_Importance":
        float(
            if_builtin_row[
                "Gain_Importance"
            ]
        ),

    "Split_Importance":
        float(
            if_builtin_row[
                "Split_Importance"
            ]
        ),

    "Mean_Anomaly_Score_Fraud":
        float(
            np.mean(
                fraud_anomaly
            )
        ),

    "Mean_Anomaly_Score_NonFraud":
        float(
            np.mean(
                normal_anomaly
            )
        ),

    "Median_Anomaly_Score_Fraud":
        float(
            np.median(
                fraud_anomaly
            )
        ),

    "Median_Anomaly_Score_NonFraud":
        float(
            np.median(
                normal_anomaly
            )
        )
}])

anomaly_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "anomaly_score_contribution.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print(
    anomaly_summary
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# FEATURE-RANK COMPARISON
# ============================================================

lgbm_rank = (
    lgbm_shap_importance[
        [
            "Feature",
            "Mean_Absolute_SHAP",
            "SHAP_Rank"
        ]
    ]
    .rename(
        columns={
            "Mean_Absolute_SHAP":
                "LightGBM_SHAP",

            "SHAP_Rank":
                "LightGBM_Rank"
        }
    )
)

if_rank = (
    if_lgbm_shap_importance[
        [
            "Feature",
            "Mean_Absolute_SHAP",
            "SHAP_Rank"
        ]
    ]
    .rename(
        columns={
            "Mean_Absolute_SHAP":
                "IF_LGBM_SHAP",

            "SHAP_Rank":
                "IF_LGBM_Rank"
        }
    )
)

rank_comparison = (
    pd.merge(
        lgbm_rank,
        if_rank,
        on="Feature",
        how="outer"
    )
)

rank_comparison[
    "Rank_Change"
] = (
    rank_comparison[
        "LightGBM_Rank"
    ]
    -
    rank_comparison[
        "IF_LGBM_Rank"
    ]
)

rank_comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "shap_rank_comparison.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PLOTS
# ============================================================

save_importance_plot(
    lgbm_shap_importance,
    "Mean_Absolute_SHAP",
    "LightGBM - Global SHAP Feature Importance",
    "lightgbm_shap_importance.png"
)

save_importance_plot(
    if_lgbm_shap_importance,
    "Mean_Absolute_SHAP",
    "IF-LGBM - Global SHAP Feature Importance",
    "if_lgbm_shap_importance.png"
)

save_importance_plot(
    lgbm_builtin,
    "Gain_Importance",
    "LightGBM - Gain-Based Feature Importance",
    "lightgbm_gain_importance.png"
)

save_importance_plot(
    if_lgbm_builtin,
    "Gain_Importance",
    "IF-LGBM - Gain-Based Feature Importance",
    "if_lgbm_gain_importance.png"
)


# ============================================================
# SHAP SUMMARY PLOTS
# ============================================================

plt.figure()

shap.summary_plot(
    lgbm_shap_values,
    X_shap,
    feature_names=feature_names,
    show=False,
    max_display=15
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "lightgbm_shap_summary.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure()

shap.summary_plot(
    if_lgbm_shap_values,
    X_shap_if,
    feature_names=if_feature_names,
    show=False,
    max_display=15
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "if_lgbm_shap_summary.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SETTINGS
# ============================================================

settings = {

    "experiment":
        "Experiment 07 - Explainability and Anomaly-Score Contribution",

    "seed":
        SEED,

    "train_dataset":
        "Base",

    "train_months":
        TRAIN_MONTHS,

    "explanation_dataset":
        "Base Month 7",

    "month_used_as_predictor":
        False,

    "models": [
        "LightGBM",
        "IF-LGBM"
    ],

    "SHAP_sample_size":
        SHAP_SAMPLE_SIZE,

    "SHAP_sampling":
        "Random sample from Base Month 7",

    "SHAP_method":
        "TreeExplainer",

    "importance_methods": [
        "Mean absolute SHAP value",
        "LightGBM gain importance",
        "LightGBM split importance"
    ],

    "IF_anomaly_score":
        "-IsolationForest.decision_function",

    "IF_parameters":
        IF_PARAMS,

    "LightGBM_parameters":
        LGBM_PARAMS,

    "model_selection":
        False,

    "hyperparameter_search":
        False,

    "hypothesis_testing":
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
# FINISH
# ============================================================

header(
    "EXPERIMENT 07 COMPLETE"
)

print(
    "\nTOP 15 LIGHTGBM SHAP FEATURES\n"
)

print(
    lgbm_shap_importance
    .head(15)
    .round(6)
    .to_string(
        index=False
    )
)

print(
    "\nTOP 15 IF-LGBM SHAP FEATURES\n"
)

print(
    if_lgbm_shap_importance
    .head(15)
    .round(6)
    .to_string(
        index=False
    )
)

print(
    "\nIF ANOMALY SCORE CONTRIBUTION\n"
)

print(
    anomaly_summary
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