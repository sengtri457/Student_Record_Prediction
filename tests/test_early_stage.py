"""Unit tests for Early-Stage Pre-Midterm Model and Midterm Anomaly Detector."""

import os
import pytest
import joblib
import pandas as pd
from src.early_stage_model import MidtermAnomalyDetector, train_early_stage_model


def test_early_stage_model_artifact():
    """Verify that early stage model exists and predicts reasonably."""
    model_path = "models/early_stage_model.joblib"
    assert os.path.exists(model_path), "Missing early stage model artifact"

    pipe = joblib.load(model_path)
    sample_df = pd.DataFrame([{
        "attendance_pct": 80.0,
        "study_hours_week": 20.0,
        "assignment_avg": 82.0,
        "previous_gpa": 3.1
    }])
    pred = pipe.predict(sample_df)
    assert len(pred) == 1
    assert 60.0 <= pred[0] <= 95.0, f"Unexpected predicted score: {pred[0]}"


def test_early_stage_comparison_metrics():
    """Verify pre-midterm metrics table."""
    table_path = "reports/tables/early_stage_comparison.csv"
    assert os.path.exists(table_path), "Missing early stage comparison table"
    df = pd.read_csv(table_path)
    assert len(df) == 1
    assert df.loc[0, "test_r2"] > 0.85, "Pre-midterm model test R2 must exceed 0.85"


def test_midterm_anomaly_detector():
    """Verify that acute exam divergence is correctly identified."""
    detector = MidtermAnomalyDetector()

    # Case A: Normal student (consistent marks)
    res_normal = detector.detect_anomaly(
        actual_midterm=82.0,
        assignment_avg=85.0,
        study_hours_week=22.0,
        previous_gpa=3.2
    )
    assert not res_normal["is_anomaly"], "False positive anomaly detected for normal student"

    # Case B: High coursework, severe midterm drop (anxiety / illness)
    res_anomaly = detector.detect_anomaly(
        actual_midterm=42.0,
        assignment_avg=92.0,
        study_hours_week=28.0,
        previous_gpa=3.75
    )
    assert res_anomaly["is_anomaly"], "Failed to detect acute midterm anomaly"
    assert res_anomaly["point_deficit"] >= 20.0
