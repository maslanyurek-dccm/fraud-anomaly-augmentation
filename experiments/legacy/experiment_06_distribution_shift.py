"""Distribution-shift diagnostics for the BAF variants (Table 2, Figure 2).

Compares month 7 of the Base file with month 7 of each variant using PSI, the
Jensen-Shannon distance and the Kolmogorov-Smirnov statistic. No model is
trained. Base.csv and the Variant *.csv files are read from the working
directory; results are written to results/experiment_06.
"""
import os
import gc
import json
import warnings

import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")


TARGET = "fraud_bool"
TIME_COL = "month"

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
    "experiment_06"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# Settings

N_BINS = 10
EPS = 1e-6

# To keep KS computation efficient.
KS_SAMPLE_SIZE = 100_000

RANDOM_STATE = 42


# Helpers

def header(text):

    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)


def clean_numeric(series):

    return (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
        .values
    )


# PSI

def calculate_psi(
    reference,
    comparison,
    bins=10
):

    reference = clean_numeric(reference)
    comparison = clean_numeric(comparison)

    if (
        len(reference) == 0
        or len(comparison) == 0
    ):
        return np.nan

    if np.unique(reference).size < 2:
        return 0.0

    quantiles = np.linspace(
        0,
        1,
        bins + 1
    )

    breakpoints = np.quantile(
        reference,
        quantiles
    )

    breakpoints = np.unique(
        breakpoints
    )

    if len(breakpoints) < 2:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(
        reference,
        bins=breakpoints
    )

    comp_counts, _ = np.histogram(
        comparison,
        bins=breakpoints
    )

    ref_pct = (
        ref_counts
        / max(ref_counts.sum(), 1)
    )

    comp_pct = (
        comp_counts
        / max(comp_counts.sum(), 1)
    )

    ref_pct = np.clip(
        ref_pct,
        EPS,
        None
    )

    comp_pct = np.clip(
        comp_pct,
        EPS,
        None
    )

    psi = np.sum(
        (
            comp_pct - ref_pct
        )
        *
        np.log(
            comp_pct / ref_pct
        )
    )

    return float(psi)


# Jensen-shannon distance

def calculate_js_distance(
    reference,
    comparison,
    bins=20
):

    reference = clean_numeric(reference)
    comparison = clean_numeric(comparison)

    if (
        len(reference) == 0
        or len(comparison) == 0
    ):
        return np.nan

    combined = np.concatenate(
        [
            reference,
            comparison
        ]
    )

    if np.min(combined) == np.max(combined):
        return 0.0

    edges = np.histogram_bin_edges(
        combined,
        bins=bins
    )

    ref_counts, _ = np.histogram(
        reference,
        bins=edges
    )

    comp_counts, _ = np.histogram(
        comparison,
        bins=edges
    )

    ref_prob = (
        ref_counts.astype(float)
        + EPS
    )

    comp_prob = (
        comp_counts.astype(float)
        + EPS
    )

    ref_prob /= ref_prob.sum()
    comp_prob /= comp_prob.sum()

    return float(
        jensenshannon(
            ref_prob,
            comp_prob,
            base=2
        )
    )


# KS distance
# Only KS statistic is retained.
# p-value is intentionally ignored.

def calculate_ks_distance(
    reference,
    comparison
):

    reference = clean_numeric(reference)
    comparison = clean_numeric(comparison)

    if (
        len(reference) == 0
        or len(comparison) == 0
    ):
        return np.nan

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    if len(reference) > KS_SAMPLE_SIZE:

        reference = rng.choice(
            reference,
            size=KS_SAMPLE_SIZE,
            replace=False
        )

    if len(comparison) > KS_SAMPLE_SIZE:

        comparison = rng.choice(
            comparison,
            size=KS_SAMPLE_SIZE,
            replace=False
        )

    statistic = ks_2samp(
        reference,
        comparison
    ).statistic

    return float(statistic)


# Categorical JS distance

def categorical_js_distance(
    reference,
    comparison
):

    ref = (
        reference
        .astype(str)
        .fillna("__MISSING__")
    )

    comp = (
        comparison
        .astype(str)
        .fillna("__MISSING__")
    )

    categories = sorted(
        set(ref.unique())
        |
        set(comp.unique())
    )

    ref_counts = (
        ref.value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0
        )
        .values
        .astype(float)
    )

    comp_counts = (
        comp.value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0
        )
        .values
        .astype(float)
    )

    ref_counts = (
        ref_counts + EPS
    )

    comp_counts = (
        comp_counts + EPS
    )

    ref_counts /= ref_counts.sum()
    comp_counts /= comp_counts.sum()

    return float(
        jensenshannon(
            ref_counts,
            comp_counts,
            base=2
        )
    )


# Fraud / non-fraud separation
# Numeric:
# standardized mean difference
# Absolute value used because we care about
# strength of separation, not direction.

def numeric_class_separation(
    df,
    feature
):

    fraud = clean_numeric(
        df.loc[
            df[TARGET] == 1,
            feature
        ]
    )

    normal = clean_numeric(
        df.loc[
            df[TARGET] == 0,
            feature
        ]
    )

    if (
        len(fraud) < 2
        or len(normal) < 2
    ):
        return np.nan

    mean_fraud = np.mean(fraud)
    mean_normal = np.mean(normal)

    var_fraud = np.var(
        fraud,
        ddof=1
    )

    var_normal = np.var(
        normal,
        ddof=1
    )

    pooled_sd = np.sqrt(
        (
            var_fraud
            + var_normal
        ) / 2
    )

    if pooled_sd == 0:
        return 0.0

    return float(
        abs(
            mean_fraud
            - mean_normal
        )
        / pooled_sd
    )


# Categorical class separation
# Total Variation Distance between
# category distributions of fraud/non-fraud.

def categorical_class_separation(
    df,
    feature
):

    fraud = (
        df.loc[
            df[TARGET] == 1,
            feature
        ]
        .astype(str)
    )

    normal = (
        df.loc[
            df[TARGET] == 0,
            feature
        ]
        .astype(str)
    )

    categories = sorted(
        set(fraud.unique())
        |
        set(normal.unique())
    )

    p = (
        fraud.value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0
        )
        .values
    )

    q = (
        normal.value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0
        )
        .values
    )

    return float(
        0.5
        * np.abs(
            p - q
        ).sum()
    )


# Load reference

header(
    "EXPERIMENT 06 - DISTRIBUTION SHIFT DIAGNOSTIC"
)

print(
    "Loading Base.csv..."
)

base = pd.read_csv(
    BASE_FILE
)

reference = base[
    base[TIME_COL] == 7
].copy()

del base
gc.collect()

print(
    f"Reference = Base Month 7"
)

print(
    f"Rows      = {len(reference):,}"
)

print(
    f"Fraud rate= "
    f"{reference[TARGET].mean():.6f}"
)


# Feature types

excluded = {
    TARGET,
    TIME_COL
}

features = [
    c for c in reference.columns
    if c not in excluded
]

categorical_features = (
    reference[
        features
    ]
    .select_dtypes(
        include=[
            "object",
            "category"
        ]
    )
    .columns
    .tolist()
)

numeric_features = [
    c for c in features
    if c not in categorical_features
]

print(
    f"Numeric features     : "
    f"{len(numeric_features)}"
)

print(
    f"Categorical features : "
    f"{len(categorical_features)}"
)


# Reference class separation

header(
    "REFERENCE CLASS SEPARATION"
)

reference_separation = {}

for feature in numeric_features:

    reference_separation[
        feature
    ] = numeric_class_separation(
        reference,
        feature
    )

for feature in categorical_features:

    reference_separation[
        feature
    ] = categorical_class_separation(
        reference,
        feature
    )


# Main loop

dataset_summary_rows = []
feature_shift_rows = []
separation_rows = []


# Base reference row
dataset_summary_rows.append({

    "Dataset":
        "Base Test (Month 7)",

    "Rows":
        len(reference),

    "Fraud_Count":
        int(reference[TARGET].sum()),

    "Fraud_Rate":
        reference[TARGET].mean(),

    "Fraud_Rate_Ratio_vs_Base":
        1.0,

    "Mean_PSI":
        0.0,

    "Max_PSI":
        0.0,

    "Mean_JS_Distance":
        0.0,

    "Max_JS_Distance":
        0.0,

    "Mean_KS_Numeric":
        0.0,

    "Max_KS_Numeric":
        0.0,

    "Mean_Separation_Change":
        0.0
})


for variant_name, filename in (
    VARIANTS.items()
):

    header(
        variant_name
    )

    print(
        f"Loading {filename}..."
    )

    variant = pd.read_csv(
        filename
    )

    # Compare Month 7 to Month 7

    variant = variant[
        variant[TIME_COL] == 7
    ].copy()

    print(
        f"Rows       : "
        f"{len(variant):,}"
    )

    print(
        f"Fraud rate : "
        f"{variant[TARGET].mean():.6f}"
    )


    numeric_psi_values = []
    js_values = []
    ks_values = []
    separation_changes = []


    # Numeric features

    for feature in numeric_features:

        if feature not in variant.columns:
            continue

        psi = calculate_psi(
            reference[feature],
            variant[feature],
            N_BINS
        )

        js = calculate_js_distance(
            reference[feature],
            variant[feature]
        )

        ks = calculate_ks_distance(
            reference[feature],
            variant[feature]
        )

        current_sep = (
            numeric_class_separation(
                variant,
                feature
            )
        )

        base_sep = (
            reference_separation[
                feature
            ]
        )

        sep_change = (
            current_sep - base_sep
            if (
                not np.isnan(current_sep)
                and not np.isnan(base_sep)
            )
            else np.nan
        )

        feature_shift_rows.append({

            "Dataset":
                variant_name,

            "Feature":
                feature,

            "Feature_Type":
                "Numeric",

            "PSI":
                psi,

            "JS_Distance":
                js,

            "KS_Distance":
                ks,

            "Base_Class_Separation":
                base_sep,

            "Variant_Class_Separation":
                current_sep,

            "Separation_Change":
                sep_change
        })

        numeric_psi_values.append(
            psi
        )

        js_values.append(
            js
        )

        ks_values.append(
            ks
        )

        separation_changes.append(
            sep_change
        )


    # Categorical features

    for feature in categorical_features:

        if feature not in variant.columns:
            continue

        js = categorical_js_distance(
            reference[feature],
            variant[feature]
        )

        current_sep = (
            categorical_class_separation(
                variant,
                feature
            )
        )

        base_sep = (
            reference_separation[
                feature
            ]
        )

        sep_change = (
            current_sep - base_sep
            if (
                not np.isnan(current_sep)
                and not np.isnan(base_sep)
            )
            else np.nan
        )

        feature_shift_rows.append({

            "Dataset":
                variant_name,

            "Feature":
                feature,

            "Feature_Type":
                "Categorical",

            "PSI":
                np.nan,

            "JS_Distance":
                js,

            "KS_Distance":
                np.nan,

            "Base_Class_Separation":
                base_sep,

            "Variant_Class_Separation":
                current_sep,

            "Separation_Change":
                sep_change
        })

        js_values.append(
            js
        )

        separation_changes.append(
            sep_change
        )


    # Dataset summary

    numeric_psi_values = np.array(
        numeric_psi_values,
        dtype=float
    )

    js_values = np.array(
        js_values,
        dtype=float
    )

    ks_values = np.array(
        ks_values,
        dtype=float
    )

    separation_changes = np.array(
        separation_changes,
        dtype=float
    )


    dataset_summary_rows.append({

        "Dataset":
            variant_name,

        "Rows":
            len(variant),

        "Fraud_Count":
            int(
                variant[TARGET].sum()
            ),

        "Fraud_Rate":
            variant[TARGET].mean(),

        "Fraud_Rate_Ratio_vs_Base":
            (
                variant[TARGET].mean()
                /
                reference[TARGET].mean()
            ),

        "Mean_PSI":
            np.nanmean(
                numeric_psi_values
            ),

        "Max_PSI":
            np.nanmax(
                numeric_psi_values
            ),

        "Mean_JS_Distance":
            np.nanmean(
                js_values
            ),

        "Max_JS_Distance":
            np.nanmax(
                js_values
            ),

        "Mean_KS_Numeric":
            np.nanmean(
                ks_values
            ),

        "Max_KS_Numeric":
            np.nanmax(
                ks_values
            ),

        "Mean_Separation_Change":
            np.nanmean(
                separation_changes
            )
    })


    del variant
    gc.collect()


# Dataframes

dataset_summary = pd.DataFrame(
    dataset_summary_rows
)

feature_shift = pd.DataFrame(
    feature_shift_rows
)


# Top shifted features

top_shifted = (
    feature_shift
    .sort_values(
        [
            "Dataset",
            "JS_Distance"
        ],
        ascending=[
            True,
            False
        ]
    )
    .groupby(
        "Dataset",
        group_keys=False
    )
    .head(10)
)


# Top separation changes

separation_table = (
    feature_shift[
        [
            "Dataset",
            "Feature",
            "Feature_Type",
            "Base_Class_Separation",
            "Variant_Class_Separation",
            "Separation_Change"
        ]
    ]
    .copy()
)

separation_table[
    "Absolute_Separation_Change"
] = np.abs(
    separation_table[
        "Separation_Change"
    ]
)

top_separation_changes = (
    separation_table
    .sort_values(
        [
            "Dataset",
            "Absolute_Separation_Change"
        ],
        ascending=[
            True,
            False
        ]
    )
    .groupby(
        "Dataset",
        group_keys=False
    )
    .head(10)
)


# Save

dataset_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dataset_shift_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

feature_shift.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_shift_analysis.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

top_shifted.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top_shifted_features.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

separation_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "class_separation_analysis.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

top_separation_changes.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top_separation_changes.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# Settings

settings = {

    "experiment":
        "Experiment 06 - Distribution Shift Diagnostic",

    "reference_dataset":
        "Base Month 7",

    "comparison_datasets": [
        "Variant I Month 7",
        "Variant II Month 7",
        "Variant III Month 7",
        "Variant IV Month 7",
        "Variant V Month 7"
    ],

    "model_training":
        False,

    "hypothesis_testing":
        False,

    "distribution_metrics": [
        "Population Stability Index",
        "Jensen-Shannon Distance",
        "Kolmogorov-Smirnov Distance"
    ],

    "ks_p_value_used":
        False,

    "class_separation": {
        "numeric":
            "Absolute standardized mean difference",

        "categorical":
            "Total variation distance"
    },

    "PSI_bins":
        N_BINS,

    "KS_sample_size":
        KS_SAMPLE_SIZE,

    "random_state":
        RANDOM_STATE
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


# Screen output

header(
    "EXPERIMENT 06 COMPLETE"
)

print(
    "\nDATASET SHIFT SUMMARY\n"
)

print(
    dataset_summary
    .round(6)
    .to_string(
        index=False
    )
)

print(
    "\nTOP 10 SHIFTED FEATURES PER VARIANT\n"
)

print(
    top_shifted[
        [
            "Dataset",
            "Feature",
            "Feature_Type",
            "PSI",
            "JS_Distance",
            "KS_Distance"
        ]
    ]
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
