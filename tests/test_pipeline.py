"""End-to-end integration and anti-leakage verification tests."""

import os
import joblib
import pandas as pd
import numpy as np
import yaml
from sklearn.pipeline import Pipeline


def test_config_validity():
    """Verify config.yaml parameters."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config["seed"] == 42, "Seed must be fixed to 42"
    assert config["target"] == "final_score", "Target must be final_score"
    assert "final_score" not in config["features"], "Target must not be in features"
    assert len(config["features"]) == 5, "Expected 5 canonical features"


def test_split_and_scaler_isolation():
    """Verify 80/20 train/test split and scaler isolation."""
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"
    assert os.path.exists(train_path), "Train split missing"
    assert os.path.exists(test_path), "Test split missing"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    total = len(train_df) + len(test_df)

    test_ratio = len(test_df) / total
    assert 0.18 <= test_ratio <= 0.22, f"Test ratio {test_ratio} deviates from 20%"

    # Verify no student overlap
    train_ids = set(train_df["student_id"])
    test_ids = set(test_df["student_id"])
    assert len(train_ids.intersection(test_ids)) == 0, "Train and test ID overlap detected (Leakage)"


def test_models_and_champion_selection():
    """Verify trained model artifacts and champion model pipeline."""
    for model_name in ["linear_regression", "random_forest", "best_model"]:
        path = f"models/{model_name}.joblib"
        assert os.path.exists(path), f"Missing model artifact: {path}"
        pipe = joblib.load(path)
        assert isinstance(pipe, Pipeline), f"Model {model_name} is not a Pipeline"

    # Test inference on single synthetic input
    best_pipe = joblib.load("models/best_model.joblib")
    sample_input = pd.DataFrame([{
        "attendance_pct": 85.0,
        "study_hours_week": 20.0,
        "assignment_avg": 80.0,
        "midterm_score": 75.0,
        "previous_gpa": 3.2
    }])
    pred = best_pipe.predict(sample_input)
    assert len(pred) == 1
    assert 0.0 <= pred[0] <= 100.0, f"Predicted score {pred[0]} out of range [0, 100]"


def test_figures_and_tables_exist():
    """Verify all 10 figures and key CSV tables exist."""
    required_figures = [
        "fig01_final_score_hist.png",
        "fig02_corr_heatmap.png",
        "fig03_midterm_vs_final.png",
        "fig04_attendance_box.png",
        "fig05_study_vs_final.png",
        "fig09_residuals.png",
        "fig10_actual_vs_pred.png"
    ]
    for fig in required_figures:
        p = os.path.join("reports/figures", fig)
        assert os.path.exists(p), f"Missing figure: {p}"
        assert os.path.getsize(p) > 1000, f"Figure {p} is empty or corrupted"

    required_tables = [
        "reports/tables/descriptive_stats.csv",
        "reports/tables/model_comparison.csv",
        "reports/tables/metrics.csv",
        "reports/tables/coefficients.csv",
        "reports/tables/feature_importance.csv",
        "reports/tables/test_predictions.csv",
        "reports/tables/custom_predictions.csv"
    ]
    for tab in required_tables:
        assert os.path.exists(tab), f"Missing table: {tab}"


def test_final_report_exists():
    """Verify final report markdown file exists and has content."""
    report_path = "reports/final_report.md"
    assert os.path.exists(report_path), "Final report missing"
    assert os.path.getsize(report_path) > 3000, "Final report appears incomplete"
