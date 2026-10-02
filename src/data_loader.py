"""Data ingestion and dataset generation utility for Student Score Prediction.
"""

import os
import yaml
import numpy as np
import pandas as pd


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration yaml file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_benchmark_raw_dataset(
    output_path: str = "data/raw/students_raw.csv",
    n_samples: int = 400,
    seed: int = 42
) -> pd.DataFrame:
    """Generates realistic student academic data matching the canonical specification.
    
    Includes realistic academic variations, correlated features, a few outliers,
    and a small percentage of realistic missing values / duplicate records
    to evaluate data cleaning validation rules (V1-V7).
    """
    np.random.seed(seed)
    
    # Latent student aptitude / engagement factor
    latent_factor = np.random.normal(0, 1, n_samples)
    
    # 1. Attendance percentage (0 - 100%)
    attendance = 78 + 10 * latent_factor + np.random.normal(0, 6, n_samples)
    attendance = np.clip(attendance, 35, 100)
    
    # 2. Weekly study hours (0 - 80 hours)
    study_hours = 16 + 5 * latent_factor + np.random.gamma(shape=2.5, scale=2.0, size=n_samples)
    study_hours = np.clip(study_hours, 2, 50)
    
    # 3. Assignment average (0 - 100)
    assignment_avg = 74 + 11 * latent_factor + 0.15 * attendance + np.random.normal(0, 5, n_samples)
    assignment_avg = np.clip(assignment_avg, 30, 100)
    
    # 4. Midterm score (0 - 100)
    midterm_score = 70 + 12 * latent_factor + 0.3 * (study_hours * 1.5) + np.random.normal(0, 6, n_samples)
    midterm_score = np.clip(midterm_score, 25, 100)
    
    # 5. Previous GPA (0.0 - 4.0)
    gpa_raw = 3.0 + 0.45 * latent_factor + np.random.normal(0, 0.25, n_samples)
    previous_gpa = np.clip(gpa_raw, 1.8, 4.0)
    
    # 6. Final Score (Target: 0 - 100)
    # Realistic weighted academic composition with non-linear interaction and residual variation
    base_final = (
        0.32 * midterm_score +
        0.24 * assignment_avg +
        0.18 * attendance +
        0.45 * study_hours +
        4.5 * previous_gpa +
        np.random.normal(0, 4.0, n_samples)
    )
    # Slight scaling adjustment to fit 0-100 realistic span
    final_score = np.clip(base_final * 0.92 + 2.0, 30, 100)
    
    df = pd.DataFrame({
        "student_id": [f"STU_{i+1:04d}" for i in range(n_samples)],
        "attendance_pct": np.round(attendance, 1),
        "study_hours_week": np.round(study_hours, 1),
        "assignment_avg": np.round(assignment_avg, 1),
        "midterm_score": np.round(midterm_score, 1),
        "previous_gpa": np.round(previous_gpa, 2),
        "final_score": np.round(final_score, 1)
    })
    
    # Inject deliberate validation challenges to test Cleaner Agent (V1-V7 rules):
    # 1. Two duplicate rows
    df = pd.concat([df, df.iloc[[12, 45]]], ignore_index=True)
    
    # 2. A few missing values (~2-3% random in feature columns)
    missing_indices_study = np.random.choice(range(n_samples), size=8, replace=False)
    df.loc[missing_indices_study, "study_hours_week"] = np.nan
    
    missing_indices_assign = np.random.choice(range(n_samples), size=6, replace=False)
    df.loc[missing_indices_assign, "assignment_avg"] = np.nan
    
    missing_indices_gpa = np.random.choice(range(n_samples), size=5, replace=False)
    df.loc[missing_indices_gpa, "previous_gpa"] = np.nan
    
    # 3. Two missing final_score rows (to verify V5 drop rule)
    df.loc[10, "final_score"] = np.nan
    df.loc[75, "final_score"] = np.nan
    
    # 4. One invalid out-of-range value (e.g. attendance 125% entry error)
    df.loc[22, "attendance_pct"] = 125.0
    
    # Ensure target directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated raw dataset with {len(df)} rows at: {output_path}")
    return df


def load_raw_data(filepath: str = "data/raw/students_raw.csv") -> pd.DataFrame:
    """Reads raw CSV dataset without modifications."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at {filepath}")
    return pd.read_csv(filepath)


if __name__ == "__main__":
    generate_benchmark_raw_dataset()
