"""Unit tests for Early Warning, Prescription Engine, and Phase 2 visualizer."""

import os
import pytest
import pandas as pd
from src.early_warning import EarlyWarningSystem


def test_early_warning_risk_classification():
    """Verify risk classification tiers."""
    ews = EarlyWarningSystem()
    
    tier, badge, color = ews.classify_risk(88.0)
    assert tier == "LOW_RISK"
    assert "Low" in badge
    
    tier, badge, color = ews.classify_risk(68.5)
    assert tier == "MODERATE_RISK"
    assert "Moderate" in badge

    tier, badge, color = ews.classify_risk(52.0)
    assert tier == "HIGH_RISK"
    assert "High" in badge


def test_early_warning_prescription_math():
    """Verify prescription calculation logic."""
    ews = EarlyWarningSystem()
    
    # Passing student (no deficit)
    rx_pass = ews.compute_prescription(
        predicted_score=82.0, current_study=20.0, current_att=90.0, current_assign=85.0, target_score=70.0
    )
    assert rx_pass["deficit_points"] == 0.0
    assert rx_pass["recommended_study_hours_add"] == 0.0

    # At-risk student (deficit of 16 points)
    rx_risk = ews.compute_prescription(
        predicted_score=54.0, current_study=15.0, current_att=70.0, current_assign=60.0, target_score=70.0
    )
    assert rx_risk["deficit_points"] == 16.0
    assert rx_risk["study_only_hours_add"] > 0
    assert rx_risk["balanced_study_add"] > 0
    assert rx_risk["balanced_attendance_add"] > 0


def test_cohort_audit_and_fig11_exist():
    """Verify cohort audit table and comparative figure exist."""
    audit_path = "reports/tables/cohort_risk_audit.csv"
    assert os.path.exists(audit_path), "Cohort risk audit table missing"
    df_audit = pd.read_csv(audit_path)
    assert len(df_audit) > 0
    assert "risk_tier" in df_audit.columns
    assert "actionable_prescription" in df_audit.columns

    fig11_path = "reports/figures/fig11_feature_importance_comparison.png"
    assert os.path.exists(fig11_path), "Fig 11 feature importance comparison missing"
    assert os.path.getsize(fig11_path) > 1000
