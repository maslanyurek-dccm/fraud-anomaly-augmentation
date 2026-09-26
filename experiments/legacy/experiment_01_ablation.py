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
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ============================================================
# EXPERIMENT 01
# Ablation Study:
# LGBM
# LGBM + SMOTE
# LGBM + IF
# LGBM + IF + SMOTE
#
# Temporal protocol:
# Month 0-5 -> Train
# Month 6   -> Validation
# Month 7   -> Test
#
# Threshold:
# Selected ONLY on validation set at approximately 5% FPR
#
# Distribution shift:
# Frozen model + frozen threshold -> Variant I-V
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

OUTPUT_DIR = os.path.join("results", "experiment_01")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# MODEL SETTINGS
# ------------------------------------------------------------

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

SMOTE_SAMPLING_STRATEGY = 0.20

MODEL_CONFIGS = {
    "M1_LGBM": {
        "use_if": False,
        "use_smote": False
    },
    "M2_LGBM_SMOTE": {
        "use_if": False,
        "use_smote": True
    },
    "M3_LGBM_IF": {
        "use_if": True,
        "use_smote": False
    },
    "M4_LGBM_IF_SMOTE": {
        "use_if": True,
        "use_smote": True
    }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def memory_mb(df):
    return df.memory_usage(deep=True).sum() / (1024 ** 2)


def optimize_numeric_dtypes(df):
    """
    Reduce memory usage without changing logical values.
    """
    df = df.copy()

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    for col in df.select_dtypes(include=["int64"]).columns:
        if col == TARGET:
            df[col] = df[col].astype("int8")
        elif col == TIME_COL:
            df[col] = df[col].astype("int8")

    return df


def find_threshold_at_fpr(y_true, y_prob, target_fpr=0.05):
    """
    Finds a threshold using ONLY the negative class scores.

    Threshold is chosen so that approximately target_fpr proportion
    of negative validation examples are classified positive.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    negative_scores = y_prob[y_true == 0]

    if len(negative_scores) == 0:
        raise ValueError("No negative samples available for FPR threshold.")

    threshold = np.quantile(
        negative_scores,
        1.0 - target_fpr
    )

    return float(threshold)


def calculate_metrics(y_true, y_prob, threshold):
    """
    Calculate threshold-independent and threshold-dependent metrics.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    auprc = average_precision_score(
        y_true,
        y_prob
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    return {
        "AUPRC": auprc,
        "ROC_AUC": roc_auc,
        "Threshold": threshold,
        "FPR": fpr,
        "Precision": precision,
        "Recall_at_5pctFPR": recall,
        "F1": f1,
        "MCC": mcc,
        "Balanced_Accuracy": balanced_acc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


def align_columns(df, feature_columns):
    """
    Keep ONLY Base training features.
    Extra columns in variants (e.g. x1/x2) are ignored.
    Missing Base features raise an error.
    """

    missing = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required Base features missing: {missing}"
        )

    return df[feature_columns].copy()


# ============================================================
# PREPROCESSOR
# ============================================================

class BAFPreprocessor:

    def __init__(self):
        self.feature_columns = None
        self.categorical_columns = None
        self.numeric_columns = None

        self.encoder = None
        self.scaler = None

    def fit(self, X_train):

        self.feature_columns = list(X_train.columns)

        self.categorical_columns = (
            X_train
            .select_dtypes(include=["object", "category"])
            .columns
            .tolist()
        )

        self.numeric_columns = [
            c for c in self.feature_columns
            if c not in self.categorical_columns
        ]

        print(
            f"Categorical features: "
            f"{len(self.categorical_columns)}"
        )

        print(
            f"Numeric features: "
            f"{len(self.numeric_columns)}"
        )

        # ----------------------------
        # Categorical encoder
        # ----------------------------

        if self.categorical_columns:

            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float32
            )

            self.encoder.fit(
                X_train[self.categorical_columns]
                .astype(str)
            )

        # ----------------------------
        # Scaler
        # ----------------------------

        train_numeric = self._encode_only(X_train)

        self.scaler = StandardScaler()

        self.scaler.fit(train_numeric)

        del train_numeric
        gc.collect()

        return self

    def _encode_only(self, X):

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

        # Encode categoricals in one operation
        encoded_categorical = None

        if self.categorical_columns:

            encoded_categorical = self.encoder.transform(
                X[self.categorical_columns].astype(str)
            )

            cat_position = {
                col: i
                for i, col
                in enumerate(self.categorical_columns)
            }

        for j, col in enumerate(self.feature_columns):

            if col in categorical_set:

                output[:, j] = encoded_categorical[
                    :, cat_position[col]
                ]

            else:

                output[:, j] = pd.to_numeric(
                    X[col],
                    errors="coerce"
                ).fillna(0).astype(np.float32).values

        return output

    def transform(self, X):

        encoded = self._encode_only(X)

        scaled = self.scaler.transform(
            encoded
        ).astype(np.float32)

        del encoded
        gc.collect()

        return scaled


# ============================================================
# LOAD BASE
# ============================================================

print_header("EXPERIMENT 01 - ABLATION STUDY")

print("Loading Base.csv...")

start = time.time()

base_df = pd.read_csv(BASE_FILE)
base_df = optimize_numeric_dtypes(base_df)

print(
    f"Base loaded in "
    f"{time.time() - start:.2f} sec"
)

print(
    f"Shape: {base_df.shape}"
)

print(
    f"Memory: {memory_mb(base_df):.2f} MB"
)


# ============================================================
# TEMPORAL SPLIT
# ============================================================

if TIME_COL not in base_df.columns:
    raise ValueError(
        f"Temporal column '{TIME_COL}' not found."
    )

months = sorted(
    base_df[TIME_COL]
    .dropna()
    .unique()
    .tolist()
)

print(f"Detected months: {months}")

required_months = set(range(8))

if not required_months.issubset(set(months)):
    raise ValueError(
        "Expected months 0-7 were not all found."
    )

train_df = base_df[
    base_df[TIME_COL].isin([0, 1, 2, 3, 4, 5])
].copy()

val_df = base_df[
    base_df[TIME_COL] == 6
].copy()

test_df = base_df[
    base_df[TIME_COL] == 7
].copy()

print_header("TEMPORAL SPLIT")

print(f"Train      : {len(train_df):,}")
print(f"Validation : {len(val_df):,}")
print(f"Test       : {len(test_df):,}")

print(
    f"Train fraud rate: "
    f"{train_df[TARGET].mean()*100:.4f}%"
)

print(
    f"Val fraud rate  : "
    f"{val_df[TARGET].mean()*100:.4f}%"
)

print(
    f"Test fraud rate : "
    f"{test_df[TARGET].mean()*100:.4f}%"
)


# ============================================================
# X / y
# ============================================================

DROP_COLUMNS = [TARGET]

# IMPORTANT:
# month is retained as a feature for this pilot because it exists
# in the original BAF feature space.
# Temporal splitting itself is based on month.

X_train_raw = train_df.drop(
    columns=DROP_COLUMNS
)

y_train = train_df[TARGET].astype(np.int8).values

X_val_raw = val_df.drop(
    columns=DROP_COLUMNS
)

y_val = val_df[TARGET].astype(np.int8).values

X_test_raw = test_df.drop(
    columns=DROP_COLUMNS
)

y_test = test_df[TARGET].astype(np.int8).values

# We no longer need full Base
del base_df
del train_df
del val_df
del test_df
gc.collect()


# ============================================================
# PREPROCESSING
# ============================================================

print_header("PREPROCESSING")

preprocessor = BAFPreprocessor()

print("Fitting preprocessing ONLY on training data...")

preprocessor.fit(X_train_raw)

feature_columns = preprocessor.feature_columns

print("Transforming Train...")
X_train = preprocessor.transform(X_train_raw)

print("Transforming Validation...")
X_val = preprocessor.transform(X_val_raw)

print("Transforming Test...")
X_test = preprocessor.transform(X_test_raw)

# Raw train/val/test no longer needed
del X_train_raw
del X_val_raw
del X_test_raw
gc.collect()

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape  : {X_val.shape}")
print(f"X_test shape : {X_test.shape}")


# ============================================================
# ISOLATION FOREST
# ============================================================

print_header("ISOLATION FOREST")

if_start = time.time()

iso = IsolationForest(
    **IF_PARAMS
)

print("Fitting IF ONLY on Base training data...")

iso.fit(X_train)

if_fit_time = time.time() - if_start

print(
    f"Isolation Forest fit time: "
    f"{if_fit_time:.2f} sec"
)

print("Generating anomaly scores...")

# sklearn decision_function:
# lower values = more abnormal.
# Multiply by -1 so HIGHER score = more anomalous.

if_train_score = (
    -iso.decision_function(X_train)
).astype(np.float32)

if_val_score = (
    -iso.decision_function(X_val)
).astype(np.float32)

if_test_score = (
    -iso.decision_function(X_test)
).astype(np.float32)

X_train_if = np.column_stack(
    [X_train, if_train_score]
).astype(np.float32)

X_val_if = np.column_stack(
    [X_val, if_val_score]
).astype(np.float32)

X_test_if = np.column_stack(
    [X_test, if_test_score]
).astype(np.float32)

print(
    f"Original features : {X_train.shape[1]}"
)

print(
    f"With IF score     : {X_train_if.shape[1]}"
)


# ============================================================
# TRAIN FOUR ABLATION MODELS
# ============================================================

print_header("TRAINING ABLATION MODELS")

trained_models = {}
model_thresholds = {}
training_times = []
all_results = []


for model_name, config in MODEL_CONFIGS.items():

    print("\n" + "-" * 80)
    print(f"TRAINING: {model_name}")
    print("-" * 80)

    use_if = config["use_if"]
    use_smote = config["use_smote"]

    if use_if:
        train_X = X_train_if
        val_X = X_val_if
        test_X = X_test_if
    else:
        train_X = X_train
        val_X = X_val
        test_X = X_test

    train_y = y_train

    smote_time = 0.0

    # --------------------------------------------------------
    # SMOTE
    # --------------------------------------------------------

    if use_smote:

        print("Applying SMOTE to TRAIN only...")

        smote_start = time.time()

        smote = SMOTE(
            sampling_strategy=SMOTE_SAMPLING_STRATEGY,
            random_state=SEED
        )

        fit_X, fit_y = smote.fit_resample(
            train_X,
            train_y
        )

        smote_time = time.time() - smote_start

        print(
            f"After SMOTE: "
            f"{len(fit_y):,} samples"
        )

        print(
            f"SMOTE time: "
            f"{smote_time:.2f} sec"
        )

    else:

        fit_X = train_X
        fit_y = train_y

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    model = lgb.LGBMClassifier(
        **LGBM_PARAMS
    )

    fit_start = time.time()

    model.fit(
        fit_X,
        fit_y
    )

    model_fit_time = time.time() - fit_start

    print(
        f"LGBM fit time: "
        f"{model_fit_time:.2f} sec"
    )

    # --------------------------------------------------------
    # VALIDATION -> threshold selection
    # --------------------------------------------------------

    val_start = time.time()

    val_prob = model.predict_proba(
        val_X
    )[:, 1]

    val_inference_time = (
        time.time() - val_start
    )

    threshold = find_threshold_at_fpr(
        y_val,
        val_prob,
        TARGET_FPR
    )

    model_thresholds[model_name] = threshold

    val_metrics = calculate_metrics(
        y_val,
        val_prob,
        threshold
    )

    print(
        f"Validation threshold @ ~5% FPR: "
        f"{threshold:.6f}"
    )

    print(
        f"Validation actual FPR: "
        f"{val_metrics['FPR']:.4f}"
    )

    print(
        f"Validation Recall: "
        f"{val_metrics['Recall_at_5pctFPR']:.4f}"
    )

    print(
        f"Validation AUPRC: "
        f"{val_metrics['AUPRC']:.4f}"
    )

    # --------------------------------------------------------
    # BASE TEST
    # --------------------------------------------------------

    test_start = time.time()

    test_prob = model.predict_proba(
        test_X
    )[:, 1]

    test_inference_time = (
        time.time() - test_start
    )

    test_metrics = calculate_metrics(
        y_test,
        test_prob,
        threshold
    )

    row = {
        "Model": model_name,
        "Dataset": "Base Test (Month 7)",
        **test_metrics,
        "Train_Time_sec": model_fit_time,
        "SMOTE_Time_sec": smote_time,
        "IF_Fit_Time_sec":
            if_fit_time if use_if else 0.0,
        "Inference_Time_sec":
            test_inference_time
    }

    all_results.append(row)

    print("\nBASE TEST RESULT")
    print(
        f"AUPRC             : "
        f"{test_metrics['AUPRC']:.4f}"
    )
    print(
        f"ROC-AUC           : "
        f"{test_metrics['ROC_AUC']:.4f}"
    )
    print(
        f"Recall @ threshold: "
        f"{test_metrics['Recall_at_5pctFPR']:.4f}"
    )
    print(
        f"Actual Test FPR   : "
        f"{test_metrics['FPR']:.4f}"
    )
    print(
        f"F1                : "
        f"{test_metrics['F1']:.4f}"
    )
    print(
        f"MCC               : "
        f"{test_metrics['MCC']:.4f}"
    )

    trained_models[model_name] = model

    training_times.append({
        "Model": model_name,
        "LGBM_Train_Time_sec":
            model_fit_time,
        "SMOTE_Time_sec":
            smote_time,
        "IF_Fit_Time_sec":
            if_fit_time if use_if else 0.0,
        "Validation_Inference_Time_sec":
            val_inference_time,
        "Base_Test_Inference_Time_sec":
            test_inference_time,
        "Threshold":
            threshold
    })

    if use_smote:
        del fit_X
        del fit_y
        gc.collect()


# ============================================================
# FREEZE MODELS + THRESHOLDS
# TEST ON VARIANTS
# ============================================================

print_header(
    "DISTRIBUTION SHIFT EVALUATION"
)

for variant_name, variant_file in VARIANTS.items():

    print("\n" + "=" * 80)
    print(f"TESTING: {variant_name}")
    print("=" * 80)

    if not os.path.exists(variant_file):
        print(
            f"[WARNING] Missing file: "
            f"{variant_file}"
        )
        continue

    load_start = time.time()

    variant_df = pd.read_csv(
        variant_file
    )

    variant_df = optimize_numeric_dtypes(
        variant_df
    )

    print(
        f"Loaded {len(variant_df):,} rows "
        f"in {time.time()-load_start:.2f} sec"
    )

    # ---------------------------------------------
    # Important:
    # Base model knows ONLY Base features.
    # x1/x2 etc. are automatically ignored.
    # ---------------------------------------------

    y_variant = (
        variant_df[TARGET]
        .astype(np.int8)
        .values
    )

    X_variant_raw = variant_df.drop(
        columns=[TARGET]
    )

    # Explicit alignment with Base
    X_variant_raw = align_columns(
        X_variant_raw,
        feature_columns
    )

    print("Applying Base-trained preprocessing...")

    X_variant = preprocessor.transform(
        X_variant_raw
    )

    # IF score from Base-trained IF
    print(
        "Generating scores with "
        "Base-trained Isolation Forest..."
    )

    variant_if_score = (
        -iso.decision_function(X_variant)
    ).astype(np.float32)

    X_variant_if = np.column_stack(
        [X_variant, variant_if_score]
    ).astype(np.float32)

    # ---------------------------------------------
    # Frozen models
    # ---------------------------------------------

    for model_name, model in trained_models.items():

        config = MODEL_CONFIGS[
            model_name
        ]

        eval_X = (
            X_variant_if
            if config["use_if"]
            else X_variant
        )

        threshold = model_thresholds[
            model_name
        ]

        infer_start = time.time()

        prob = model.predict_proba(
            eval_X
        )[:, 1]

        inference_time = (
            time.time() - infer_start
        )

        metrics = calculate_metrics(
            y_variant,
            prob,
            threshold
        )

        all_results.append({
            "Model": model_name,
            "Dataset": variant_name,
            **metrics,
            "Train_Time_sec": np.nan,
            "SMOTE_Time_sec": np.nan,
            "IF_Fit_Time_sec": np.nan,
            "Inference_Time_sec":
                inference_time
        })

        print(
            f"{model_name:25s} | "
            f"AUPRC={metrics['AUPRC']:.4f} | "
            f"Recall={metrics['Recall_at_5pctFPR']:.4f} | "
            f"FPR={metrics['FPR']:.4f}"
        )

    # Free variant memory
    del variant_df
    del X_variant_raw
    del X_variant
    del X_variant_if
    del variant_if_score
    del y_variant

    gc.collect()


# ============================================================
# SAVE RAW RESULTS
# ============================================================

print_header("SAVING RESULTS")

results_df = pd.DataFrame(
    all_results
)

results_path = os.path.join(
    OUTPUT_DIR,
    "all_results.csv"
)

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig"
)


timing_df = pd.DataFrame(
    training_times
)

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
# CREATE AUPRC TABLE
# ============================================================

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


# ============================================================
# CREATE RECALL TABLE
# ============================================================

recall_table = results_df.pivot(
    index="Model",
    columns="Dataset",
    values="Recall_at_5pctFPR"
)

recall_path = os.path.join(
    OUTPUT_DIR,
    "recall_at_5fpr_table.csv"
)

recall_table.to_csv(
    recall_path,
    encoding="utf-8-sig"
)


# ============================================================
# ROBUSTNESS SUMMARY
# ============================================================

variant_names = list(
    VARIANTS.keys()
)

robustness_rows = []

for model_name in MODEL_CONFIGS:

    model_rows = results_df[
        results_df["Model"] == model_name
    ]

    base_row = model_rows[
        model_rows["Dataset"]
        == "Base Test (Month 7)"
    ]

    variant_rows = model_rows[
        model_rows["Dataset"]
        .isin(variant_names)
    ]

    if (
        len(base_row) == 1
        and len(variant_rows) > 0
    ):

        base_auprc = float(
            base_row["AUPRC"].iloc[0]
        )

        mean_variant_auprc = float(
            variant_rows["AUPRC"].mean()
        )

        base_recall = float(
            base_row[
                "Recall_at_5pctFPR"
            ].iloc[0]
        )

        mean_variant_recall = float(
            variant_rows[
                "Recall_at_5pctFPR"
            ].mean()
        )

        auprc_drop = (
            base_auprc
            - mean_variant_auprc
        )

        recall_drop = (
            base_recall
            - mean_variant_recall
        )

        auprc_retention = (
            mean_variant_auprc
            / base_auprc
            if base_auprc > 0
            else np.nan
        )

        recall_retention = (
            mean_variant_recall
            / base_recall
            if base_recall > 0
            else np.nan
        )

        robustness_rows.append({
            "Model":
                model_name,

            "Base_AUPRC":
                base_auprc,

            "Mean_Variant_AUPRC":
                mean_variant_auprc,

            "AUPRC_Absolute_Drop":
                auprc_drop,

            "AUPRC_Retention_Ratio":
                auprc_retention,

            "Base_Recall_at_5pctFPR":
                base_recall,

            "Mean_Variant_Recall":
                mean_variant_recall,

            "Recall_Absolute_Drop":
                recall_drop,

            "Recall_Retention_Ratio":
                recall_retention
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
# SAVE EXPERIMENT SETTINGS
# ============================================================

settings = {
    "seed": SEED,
    "target": TARGET,
    "time_column": TIME_COL,
    "train_months": [0, 1, 2, 3, 4, 5],
    "validation_month": 6,
    "test_month": 7,
    "target_fpr": TARGET_FPR,
    "smote_sampling_strategy":
        SMOTE_SAMPLING_STRATEGY,
    "lightgbm_parameters":
        LGBM_PARAMS,
    "isolation_forest_parameters":
        IF_PARAMS,
    "models":
        MODEL_CONFIGS,
    "note":
        (
            "All preprocessing, Isolation Forest and "
            "SMOTE fitting use Base training data only. "
            "Thresholds are selected using Base month 6 "
            "validation data. Models and thresholds are "
            "then frozen for Base month 7 and Variant I-V."
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

print_header("EXPERIMENT 01 COMPLETE")

print("\nAUPRC TABLE:\n")
print(
    auprc_table.round(4)
)

print("\nRECALL @ VALIDATION-DERIVED 5% FPR THRESHOLD:\n")
print(
    recall_table.round(4)
)

print("\nROBUSTNESS SUMMARY:\n")
print(
    robustness_df.round(4)
)

print("\nFiles created:")

print(f"  {results_path}")
print(f"  {auprc_path}")
print(f"  {recall_path}")
print(f"  {robustness_path}")
print(f"  {timing_path}")
print(f"  {settings_path}")

print(
    "\nIMPORTANT: Do not interpret the results "
    "before checking all output tables."
)