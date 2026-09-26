import os
import time
import pandas as pd
import numpy as np

# ============================================================
# EXPERIMENT 00 - DATASET AUDIT
# Fraud Detection under Distribution Shift
# ============================================================

DATASETS = {
    "Base": "Base.csv",
    "Variant I": "Variant I.csv",
    "Variant II": "Variant II.csv",
    "Variant III": "Variant III.csv",
    "Variant IV": "Variant IV.csv",
    "Variant V": "Variant V.csv",
}

TARGET = "fraud_bool"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Yardımcı fonksiyon
# ------------------------------------------------------------
def mb(value):
    """Byte -> MB"""
    return value / (1024 ** 2)


# ------------------------------------------------------------
# Sonuç listeleri
# ------------------------------------------------------------
dataset_overview = []
column_summary = []
missing_summary = []
variant_comparison = []

dataset_columns = {}
fraud_rates = {}
dataset_shapes = {}

print("=" * 75)
print("EXPERIMENT 00 - BAF DATASET AUDIT")
print("=" * 75)


# ============================================================
# 1. VERİ SETLERİNİ İNCELE
# ============================================================

for dataset_name, file_name in DATASETS.items():

    print("\n" + "=" * 75)
    print(f"DATASET: {dataset_name}")
    print(f"FILE   : {file_name}")
    print("=" * 75)

    if not os.path.exists(file_name):
        print(f"[ERROR] Dosya bulunamadı: {file_name}")
        continue

    start = time.time()

    print("Veri okunuyor...")
    df = pd.read_csv(file_name)

    elapsed = time.time() - start

    rows, cols = df.shape
    dataset_shapes[dataset_name] = (rows, cols)
    dataset_columns[dataset_name] = list(df.columns)

    memory_mb = mb(df.memory_usage(deep=True).sum())

    # --------------------------------------------------------
    # Target kontrolü
    # --------------------------------------------------------

    if TARGET not in df.columns:
        print(f"[WARNING] Target sütunu bulunamadı: {TARGET}")

        fraud_count = np.nan
        nonfraud_count = np.nan
        fraud_rate = np.nan

    else:
        fraud_count = int((df[TARGET] == 1).sum())
        nonfraud_count = int((df[TARGET] == 0).sum())

        fraud_rate = (
            fraud_count / len(df) * 100
            if len(df) > 0
            else np.nan
        )

        fraud_rates[dataset_name] = fraud_rate

    # --------------------------------------------------------
    # Eksik / duplicate
    # --------------------------------------------------------

    total_missing = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    # --------------------------------------------------------
    # Veri tipleri
    # --------------------------------------------------------

    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    # target'ı feature sayısından çıkar
    feature_numeric_cols = [
        c for c in numeric_cols if c != TARGET
    ]

    # --------------------------------------------------------
    # Dataset overview
    # --------------------------------------------------------

    dataset_overview.append({
        "Dataset": dataset_name,
        "Rows": rows,
        "Columns": cols,
        "Fraud_Count": fraud_count,
        "NonFraud_Count": nonfraud_count,
        "Fraud_Rate_Percent": fraud_rate,
        "Numeric_Features": len(feature_numeric_cols),
        "Categorical_Features": len(categorical_cols),
        "Missing_Values": total_missing,
        "Duplicate_Rows": duplicate_rows,
        "Memory_MB": memory_mb,
        "Load_Time_sec": elapsed
    })

    # --------------------------------------------------------
    # Column-level bilgiler
    # --------------------------------------------------------

    for col in df.columns:

        missing_count = int(df[col].isnull().sum())

        try:
            unique_count = int(df[col].nunique(dropna=True))
        except Exception:
            unique_count = np.nan

        column_summary.append({
            "Dataset": dataset_name,
            "Column": col,
            "Data_Type": str(df[col].dtype),
            "Unique_Values": unique_count,
            "Missing_Count": missing_count,
            "Missing_Percent":
                (missing_count / rows * 100) if rows else np.nan
        })

        if missing_count > 0:
            missing_summary.append({
                "Dataset": dataset_name,
                "Column": col,
                "Missing_Count": missing_count,
                "Missing_Percent":
                    (missing_count / rows * 100) if rows else np.nan
            })

    # --------------------------------------------------------
    # Ekrana kısa özet
    # --------------------------------------------------------

    print(f"Rows              : {rows:,}")
    print(f"Columns           : {cols}")

    if TARGET in df.columns:
        print(f"Fraud             : {fraud_count:,}")
        print(f"Non-Fraud         : {nonfraud_count:,}")
        print(f"Fraud Rate        : {fraud_rate:.4f}%")

    print(f"Numeric Features  : {len(feature_numeric_cols)}")
    print(f"Categorical       : {len(categorical_cols)}")
    print(f"Missing Values    : {total_missing:,}")
    print(f"Duplicate Rows    : {duplicate_rows:,}")
    print(f"Memory Usage      : {memory_mb:.2f} MB")
    print(f"Load Time         : {elapsed:.2f} sec")

    if categorical_cols:
        print("\nCategorical Columns:")
        for c in categorical_cols:
            print(f"  - {c}")

    # --------------------------------------------------------
    # month kontrolü
    # --------------------------------------------------------

    possible_time_cols = [
        c for c in df.columns
        if c.lower() in [
            "month",
            "time",
            "date",
            "timestamp",
            "day",
            "week"
        ]
    ]

    if possible_time_cols:

        print("\nPossible Temporal Columns:")

        for col in possible_time_cols:
            print(f"\n{col}:")
            print(
                df[col]
                .value_counts(dropna=False)
                .sort_index()
                .head(50)
            )

    # Belleği boşalt
    del df


# ============================================================
# 2. BASE - VARIANT SÜTUN KARŞILAŞTIRMASI
# ============================================================

print("\n")
print("=" * 75)
print("BASE vs VARIANT COLUMN COMPARISON")
print("=" * 75)

if "Base" in dataset_columns:

    base_columns = set(dataset_columns["Base"])

    for dataset_name in DATASETS:

        if dataset_name == "Base":
            continue

        if dataset_name not in dataset_columns:
            continue

        variant_columns = set(dataset_columns[dataset_name])

        missing_from_variant = sorted(
            base_columns - variant_columns
        )

        extra_in_variant = sorted(
            variant_columns - base_columns
        )

        same_columns = (
            base_columns == variant_columns
        )

        base_rate = fraud_rates.get("Base", np.nan)
        variant_rate = fraud_rates.get(dataset_name, np.nan)

        if pd.notna(base_rate) and pd.notna(variant_rate):
            fraud_rate_change = variant_rate - base_rate
        else:
            fraud_rate_change = np.nan

        variant_comparison.append({
            "Variant": dataset_name,
            "Same_Columns_As_Base": same_columns,
            "Missing_Columns_Count": len(missing_from_variant),
            "Extra_Columns_Count": len(extra_in_variant),
            "Missing_Columns":
                "; ".join(missing_from_variant),
            "Extra_Columns":
                "; ".join(extra_in_variant),
            "Base_Fraud_Rate_Percent": base_rate,
            "Variant_Fraud_Rate_Percent": variant_rate,
            "Fraud_Rate_Change_Percentage_Points":
                fraud_rate_change
        })

        print(f"\n{dataset_name}")

        print(
            "Same columns as Base:",
            same_columns
        )

        print(
            "Missing columns:",
            missing_from_variant
            if missing_from_variant
            else "None"
        )

        print(
            "Extra columns:",
            extra_in_variant
            if extra_in_variant
            else "None"
        )

        if pd.notna(fraud_rate_change):
            print(
                f"Fraud rate change: "
                f"{fraud_rate_change:+.4f} percentage points"
            )


# ============================================================
# 3. DATAFRAME'LERİ OLUŞTUR
# ============================================================

overview_df = pd.DataFrame(dataset_overview)

columns_df = pd.DataFrame(column_summary)

missing_df = pd.DataFrame(missing_summary)

variant_df = pd.DataFrame(variant_comparison)


# ============================================================
# 4. CSV ÇIKTILARI
# ============================================================

overview_path = os.path.join(
    OUTPUT_DIR,
    "dataset_overview.csv"
)

columns_path = os.path.join(
    OUTPUT_DIR,
    "column_summary.csv"
)

missing_path = os.path.join(
    OUTPUT_DIR,
    "missing_values.csv"
)

variant_path = os.path.join(
    OUTPUT_DIR,
    "variant_comparison.csv"
)


overview_df.to_csv(
    overview_path,
    index=False,
    encoding="utf-8-sig"
)

columns_df.to_csv(
    columns_path,
    index=False,
    encoding="utf-8-sig"
)

missing_df.to_csv(
    missing_path,
    index=False,
    encoding="utf-8-sig"
)

variant_df.to_csv(
    variant_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. EKRAN ÖZETİ
# ============================================================

print("\n")
print("=" * 75)
print("DATASET OVERVIEW")
print("=" * 75)

if not overview_df.empty:

    display_cols = [
        "Dataset",
        "Rows",
        "Columns",
        "Fraud_Count",
        "Fraud_Rate_Percent",
        "Missing_Values",
        "Duplicate_Rows",
        "Memory_MB"
    ]

    print(
        overview_df[display_cols]
        .to_string(index=False)
    )


print("\n")
print("=" * 75)
print("OUTPUT FILES")
print("=" * 75)

print(overview_path)
print(columns_path)
print(missing_path)
print(variant_path)

print("\nDataset audit completed successfully.")