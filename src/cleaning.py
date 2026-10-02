"""Data cleaning and validation module for Student Score Prediction.
Implements validation rules V1-V7 and missing value imputation policies.
"""

import os
import yaml
import numpy as np
import pandas as pd


ALLOWED_RANGES = {
    "attendance_pct": (0.0, 100.0),
    "study_hours_week": (0.0, 80.0),
    "assignment_avg": (0.0, 100.0),
    "midterm_score": (0.0, 100.0),
    "previous_gpa": (0.0, 4.0),
    "final_score": (0.0, 100.0),
}

CANONICAL_COLUMNS = [
    "student_id",
    "attendance_pct",
    "study_hours_week",
    "assignment_avg",
    "midterm_score",
    "previous_gpa",
    "final_score",
]


def load_config(config_path: str = "config.yaml") -> dict:
    """Load config from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_and_clean_data(
    raw_path: str = "data/raw/students_raw.csv",
    interim_path: str = "data/interim/students_clean.csv",
    log_path: str = "reports/tables/cleaning_log.csv",
    missing_table_path: str = "reports/tables/missing_values.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute validation rules V1-V7 and missing value imputation policy."""
    os.makedirs(os.path.dirname(interim_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(missing_table_path), exist_ok=True)

    df_raw = pd.read_csv(raw_path)
    initial_rows = len(df_raw)
    logs = []

    def log_action(rule_id: str, description: str, rows_affected: int, details: str = ""):
        logs.append({
            "rule_id": rule_id,
            "action": description,
            "rows_affected": rows_affected,
            "details": details
        })

    # Rule V1: Verify all required columns present
    missing_cols = [col for col in CANONICAL_COLUMNS if col not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"Rule V1 Failed: Missing required columns: {missing_cols}")
    log_action("V1", "Checked required canonical columns", 0, "All columns present")

    df = df_raw.copy()

    # Rule V7: Inconsistent text/format standardization (strip strings)
    if "student_id" in df.columns:
        df["student_id"] = df["student_id"].astype(str).str.strip()
    log_action("V7", "Standardized student_id format", len(df), "Trimmed whitespace")

    # Rule V2: Duplicate student_id removal
    dup_mask = df.duplicated(subset=["student_id"], keep="first")
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        df = df[~dup_mask].copy()
    log_action("V2", "Deduplicated on student_id", dup_count, f"Removed {dup_count} duplicate records")

    # Rule V4: Dtype coercion to float for numeric columns
    numeric_cols = [c for c in CANONICAL_COLUMNS if c != "student_id"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    log_action("V4", "Coerced numeric columns to float", len(df), "Converted bad characters to NaN")

    # Rule V3: Check range boundaries and set out-of-range to NaN
    out_of_bounds_count = 0
    for col, (low, high) in ALLOWED_RANGES.items():
        oob_mask = (df[col] < low) | (df[col] > high)
        count = int(oob_mask.sum())
        if count > 0:
            out_of_bounds_count += count
            df.loc[oob_mask, col] = np.nan
    log_action("V3", "Checked numeric range boundaries", out_of_bounds_count, f"Set {out_of_bounds_count} invalid entries to NaN")

    # Rule V5: Missing target (final_score) must be dropped
    missing_target_mask = df["final_score"].isna()
    missing_target_count = int(missing_target_mask.sum())
    if missing_target_count > 0:
        df = df[~missing_target_mask].copy()
    log_action("V5", "Dropped records missing target (final_score)", missing_target_count, f"Removed {missing_target_count} rows")

    # Rule V6: Drop rows missing > 50% of feature columns
    feature_cols = [c for c in numeric_cols if c != "final_score"]
    excessive_missing_mask = df[feature_cols].isna().sum(axis=1) > (len(feature_cols) / 2)
    excessive_count = int(excessive_missing_mask.sum())
    if excessive_count > 0:
        df = df[~excessive_missing_mask].copy()
    log_action("V6", "Dropped rows missing > 50% of features", excessive_count, f"Removed {excessive_count} rows")

    # Build Missingness Table BEFORE Imputation (A4 requirement)
    missing_stats = []
    for col in CANONICAL_COLUMNS:
        n_missing = int(df[col].isna().sum())
        pct_missing = round((n_missing / len(df)) * 100, 2)
        if col in feature_cols:
            skew = round(float(df[col].skew(skipna=True)), 3)
            method = "Mean (skew between -0.5 and 0.5)" if abs(skew) <= 0.5 else "Median (skewed distribution)"
        elif col == "final_score":
            skew = round(float(df[col].skew(skipna=True)), 3)
            method = "None (rows dropped per V5)"
        else:
            skew = None
            method = "None (ID column)"

        missing_stats.append({
            "column": col,
            "missing_count": n_missing,
            "missing_pct": pct_missing,
            "skewness": skew,
            "imputation_strategy": method
        })
    df_missing = pd.DataFrame(missing_stats)
    df_missing.to_csv(missing_table_path, index=False)

    # Impute missing feature values based on skewness policy
    for col in feature_cols:
        if df[col].isna().sum() > 0:
            skew = df[col].skew(skipna=True)
            if abs(skew) <= 0.5:
                fill_val = round(float(df[col].mean()), 2)
                strategy = "mean"
            else:
                fill_val = round(float(df[col].median()), 2)
                strategy = "median"
            count = int(df[col].isna().sum())
            df[col] = df[col].fillna(fill_val)
            log_action("A4", f"Imputed {col} using {strategy}", count, f"Filled with value: {fill_val}")

    # Final assertions
    assert df[feature_cols].isna().sum().sum() == 0, "Error: Features still contain missing values"
    assert df["final_score"].isna().sum() == 0, "Error: Target contains missing values"

    # Save cleaned interim dataset
    df.to_csv(interim_path, index=False)
    log_action("DONE", "Saved canonical cleaned dataset", len(df), f"Rows: {len(df)} (Original: {initial_rows})")

    df_log = pd.DataFrame(logs)
    df_log.to_csv(log_path, index=False)

    print(f"Cleaning complete. Retained {len(df)} / {initial_rows} rows.")
    print(f"Clean data saved to {interim_path}")
    print(f"Cleaning log saved to {log_path}")
    print(f"Missing values report saved to {missing_table_path}")

    return df, df_log, df_missing


if __name__ == "__main__":
    validate_and_clean_data()
