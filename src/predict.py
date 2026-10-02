"""Prediction inference module for Student Score Prediction.
Supports single student predictions and evaluation of custom scenarios.
"""

import os
import yaml
import numpy as np
import pandas as pd
import joblib


def load_config(config_path: str = "config.yaml") -> dict:
    """Load config yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict_custom_students(
    model_path: str = "models/best_model.joblib",
    output_path: str = "reports/tables/custom_predictions.csv"
) -> pd.DataFrame:
    """Evaluates at least 3 defined academic scenarios using the champion model."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline = joblib.load(model_path)

    scenarios = [
        {
            "scenario": "High Attendance, Low Midterm Benchmark",
            "student_id": "SIM_HIGH_ATT_LOW_MID",
            "attendance_pct": 98.0,
            "study_hours_week": 14.0,
            "assignment_avg": 78.0,
            "midterm_score": 45.0,
            "previous_gpa": 2.70,
            "context": "Consistent lecture attendance but struggled on timed midterm exam"
        },
        {
            "scenario": "Low Attendance, High Prior GPA (Fast Learner)",
            "student_id": "SIM_LOW_ATT_HIGH_GPA",
            "attendance_pct": 55.0,
            "study_hours_week": 26.0,
            "assignment_avg": 92.0,
            "midterm_score": 88.0,
            "previous_gpa": 3.85,
            "context": "Low class presence compensated by heavy self-study and high prior foundation"
        },
        {
            "scenario": "Average Student on All Dimensions",
            "student_id": "SIM_MEDIAN_STUDENT",
            "attendance_pct": 78.0,
            "study_hours_week": 21.0,
            "assignment_avg": 85.0,
            "midterm_score": 79.0,
            "previous_gpa": 3.00,
            "context": "Benchmark student matching cohort median characteristics exactly"
        },
        {
            "scenario": "High Effort, Low Assignments (Struggling with Homework)",
            "student_id": "SIM_HIGH_EFFORT_LOW_HW",
            "attendance_pct": 95.0,
            "study_hours_week": 34.0,
            "assignment_avg": 62.0,
            "midterm_score": 72.0,
            "previous_gpa": 2.90,
            "context": "High study investment but lower formative assignment return"
        }
    ]

    df_scenarios = pd.DataFrame(scenarios)
    feature_cols = [
        "attendance_pct",
        "study_hours_week",
        "assignment_avg",
        "midterm_score",
        "previous_gpa"
    ]

    predictions = pipeline.predict(df_scenarios[feature_cols])
    # Clip predictions to physiological exam range 0-100
    df_scenarios["predicted_final_score"] = np.round(np.clip(predictions, 0, 100), 1)

    df_scenarios.to_csv(output_path, index=False)
    print(f"Generated custom scenario predictions at: {output_path}")
    return df_scenarios


if __name__ == "__main__":
    predict_custom_students()
