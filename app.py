"""Student Score Prediction & Early Warning System - Clean Streamlit Web Application.
Minimalist, high-contrast, professional academic dashboard without emojis.
Supports Dual-Stage Prediction (Weeks 1-6 Pre-Midterm vs. Weeks 7+ Post-Midterm)
and Automated Midterm Exam Anomaly Detection.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.early_stage_model import MidtermAnomalyDetector


# Page Configuration
st.set_page_config(
    page_title="Student Score Predictor | Academic Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean CSS (Slate 950 / Slate 900 Theme)
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Header typography */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    
    /* Clean Cards */
    .clean-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    /* Clean Risk Badges (No Emojis) */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-low {
        background-color: #064e3b;
        color: #6ee7b7;
        border: 1px solid #059669;
    }
    .badge-moderate {
        background-color: #78350f;
        color: #fde68a;
        border: 1px solid #d97706;
    }
    .badge-high {
        background-color: #7f1d1d;
        color: #fca5a5;
        border: 1px solid #dc2626;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sliders */
    .stSlider label {
        color: #cbd5e1 !important;
        font-weight: 500;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_assets():
    """Load serialized champion pipelines, early stage model, and detector."""
    full_model_path = "models/best_model.joblib"
    early_model_path = "models/early_stage_model.joblib"
    comp_path = "reports/tables/model_comparison.csv"
    early_comp_path = "reports/tables/early_stage_comparison.csv"
    
    full_pipe = joblib.load(full_model_path) if os.path.exists(full_model_path) else None
    early_pipe = joblib.load(early_model_path) if os.path.exists(early_model_path) else None
    comp_df = pd.read_csv(comp_path) if os.path.exists(comp_path) else None
    early_comp_df = pd.read_csv(early_comp_path) if os.path.exists(early_comp_path) else None
    detector = MidtermAnomalyDetector()
    
    return full_pipe, early_pipe, comp_df, early_comp_df, detector


full_pipe, early_pipe, comp_df, early_comp_df, detector = load_model_assets()

# Sidebar Navigation
st.sidebar.markdown("### Score Predictor")
st.sidebar.caption("Academic Early Warning System • Version 2.2")

app_view = st.sidebar.radio(
    "Navigation",
    ["Student Simulator", "Batch Roster Audit", "Model Performance"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("Topic 01: Supervised Regression Pipeline")


# VIEW 1: Individual Student Simulator
if app_view == "Student Simulator":
    st.title("Student Performance Simulator")
    st.markdown("Forecast final examination outcomes across semester milestones and detect anomalous exam variances.")

    # Timeline Selector (Dual-Mode)
    timeline_mode = st.radio(
        "Semester Milestone:",
        ["Weeks 1-6 (Early Semester / Pre-Midterm)", "Weeks 7-15 (Post-Midterm / Full Evaluation)"],
        horizontal=True
    )
    is_early_stage = "Weeks 1-6" in timeline_mode

    col_input, col_result = st.columns([1, 1.15], gap="large")

    with col_input:
        st.subheader("Academic Indicators")
        
        # Presets selector
        preset_choice = st.selectbox(
            "Load Benchmark Profile:",
            ["Custom Inputs", "Average Student (Cohort Median)", "High Performer", "At Risk Student", "Exam Anxiety Case (High HW, Low Midterm)"]
        )

        # Preset values
        if preset_choice == "High Performer":
            def_att, def_study, def_hw, def_mid, def_gpa = 96.0, 30.0, 94.0, 90.0, 3.80
        elif preset_choice == "At Risk Student":
            def_att, def_study, def_hw, def_mid, def_gpa = 55.0, 10.0, 58.0, 45.0, 2.30
        elif preset_choice == "Exam Anxiety Case (High HW, Low Midterm)":
            def_att, def_study, def_hw, def_mid, def_gpa = 92.0, 26.0, 90.0, 44.0, 3.65
        else:
            def_att, def_study, def_hw, def_mid, def_gpa = 78.0, 21.0, 85.0, 79.0, 3.00

        att = st.slider("Classroom Attendance (%)", 30.0, 100.0, float(def_att), 1.0)
        study = st.slider("Weekly Study Hours (hrs/wk)", 0.0, 60.0, float(def_study), 0.5)
        hw = st.slider("Assignment & Quiz Average (0-100)", 30.0, 100.0, float(def_hw), 1.0)
        gpa = st.slider("Prior Cumulative GPA (0.0-4.0)", 1.5, 4.0, float(def_gpa), 0.05)

        if not is_early_stage:
            mid = st.slider("Midterm Examination Score (0-100)", 20.0, 100.0, float(def_mid), 1.0)
        else:
            mid = None
            st.info("Midterm exam has not yet been administered. Evaluating student on attendance, study volume, and coursework trajectory.")

        target_goal = st.selectbox("Target Passing Goal:", [70.0, 75.0, 80.0, 85.0, 90.0], index=0)

    with col_result:
        st.subheader("Diagnostic Assessment")

        if is_early_stage and early_pipe is not None:
            active_pipe = early_pipe
            input_df = pd.DataFrame([{
                "attendance_pct": att,
                "study_hours_week": study,
                "assignment_avg": hw,
                "previous_gpa": gpa
            }])
            b_study, b_att = 0.6181, 0.2446
            active_mae = 3.0
            stage_tag = "Early-Semester Trajectory (Pre-Midterm)"
        elif not is_early_stage and full_pipe is not None:
            active_pipe = full_pipe
            input_df = pd.DataFrame([{
                "attendance_pct": att,
                "study_hours_week": study,
                "assignment_avg": hw,
                "midterm_score": mid,
                "previous_gpa": gpa
            }])
            b_study, b_att = 0.3968, 0.1907
            active_mae = 2.6
            stage_tag = "Comprehensive Post-Midterm Evaluation"
        else:
            active_pipe = None

        if active_pipe is not None:
            pred_raw = active_pipe.predict(input_df)[0]
            pred_score = float(np.clip(pred_raw, 0.0, 100.0))

            # Risk classification
            if pred_score >= 75.0:
                badge_html = '<span class="badge badge-low">LOW RISK: ON TRACK</span>'
            elif pred_score >= 60.0:
                badge_html = '<span class="badge badge-moderate">MODERATE RISK: WATCHLIST</span>'
            else:
                badge_html = '<span class="badge badge-high">HIGH RISK: CRITICAL INTERVENTION</span>'

            # Letter grade
            if pred_score >= 90.0: grade = "A"
            elif pred_score >= 80.0: grade = "B"
            elif pred_score >= 70.0: grade = "C (Pass)"
            elif pred_score >= 60.0: grade = "D"
            else: grade = "F"

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Predicted Score", f"{pred_score:.1f} / 100")
            with m2:
                st.metric("Projected Grade", grade)
            with m3:
                st.metric("Confidence Band", f"[{pred_score - active_mae:.1f} - {pred_score + active_mae:.1f}]")

            st.markdown(f"Status: {badge_html} &nbsp; <span style='font-size: 11px; color: #94a3b8;'>({stage_tag})</span>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Anomaly Detection Check (Only Post-Midterm)
            if not is_early_stage and mid is not None:
                anomaly_res = detector.detect_anomaly(
                    actual_midterm=mid,
                    assignment_avg=hw,
                    study_hours_week=study,
                    previous_gpa=gpa
                )
                if anomaly_res["is_anomaly"]:
                    st.warning(f"**Midterm Anomaly Detected:** Coursework ({hw:.0f}%) and GPA ({gpa:.2f}) projected an expected midterm of ~{anomaly_res['expected_midterm']:.0f} pts, but actual score was {mid:.0f} (-{anomaly_res['point_deficit']:.0f} deficit). This pattern indicates probable test anxiety, acute illness, or testing anomaly. Recommend counselor review.")

            # Prescription Guidance
            st.subheader("Academic Intervention Guidance")
            deficit = target_goal - pred_score

            if deficit <= 0:
                st.success(f"Student is projected to score {pred_score:.1f}, achieving target goal {target_goal:.0f}. Maintain current study routines.")
            else:
                st.warning(f"Projected score falls short of target {target_goal:.0f} by {deficit:.1f} points.")
                
                extra_study = deficit / b_study
                max_att_gain = max(0.0, 100.0 - att)
                extra_att = min(deficit / b_att, max_att_gain)
                
                dual_study = (deficit / 2.0) / b_study
                dual_att = min((deficit / 2.0) / b_att, max_att_gain)

                c_rx1, c_rx2 = st.columns(2)
                with c_rx1:
                    st.info(f"Option A: Study Allocation\n\nAdd +{extra_study:.1f} hrs/week of dedicated study outside class.")
                with c_rx2:
                    st.info(f"Option B: Dual Action\n\nAdd +{dual_study:.1f} hrs/week study AND raise attendance by +{dual_att:.1f}%.")


# VIEW 2: Batch Class Roster Audit
elif app_view == "Batch Roster Audit":
    st.title("Batch Class Roster Audit")
    st.markdown("Upload a class spreadsheet to perform cohort-wide predictions, risk classification, and recovery audits.")

    sample_template = pd.DataFrame([
        {"student_id": "STU_1001", "attendance_pct": 92.0, "study_hours_week": 26.0, "assignment_avg": 88.0, "midterm_score": 82.0, "previous_gpa": 3.45},
        {"student_id": "STU_1002", "attendance_pct": 60.0, "study_hours_week": 12.0, "assignment_avg": 65.0, "midterm_score": 52.0, "previous_gpa": 2.50},
        {"student_id": "STU_1003", "attendance_pct": 78.0, "study_hours_week": 19.0, "assignment_avg": 75.0, "midterm_score": 70.0, "previous_gpa": 2.90},
    ])

    st.download_button(
        "Download CSV Template",
        data=sample_template.to_csv(index=False),
        file_name="class_roster_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Select Class CSV File", type=["csv"])

    if uploaded_file is not None and full_pipe is not None:
        batch_df = pd.read_csv(uploaded_file)
        feature_cols = ["attendance_pct", "study_hours_week", "assignment_avg", "midterm_score", "previous_gpa"]

        missing_cols = [c for c in feature_cols if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing required columns in CSV: {missing_cols}")
        else:
            preds = full_pipe.predict(batch_df[feature_cols])
            batch_df["predicted_final_score"] = np.round(np.clip(preds, 0.0, 100.0), 1)

            def get_tier_label(score):
                if score >= 75.0: return "Low Risk"
                if score >= 60.0: return "Moderate Risk"
                return "High Risk"

            batch_df["risk_tier"] = batch_df["predicted_final_score"].apply(get_tier_label)

            st.success(f"Audit completed for {len(batch_df)} student records.")

            c1, c2, c3 = st.columns(3)
            counts = batch_df["risk_tier"].value_counts()
            c1.metric("Low Risk (On Track)", counts.get("Low Risk", 0))
            c2.metric("Moderate Risk (Watchlist)", counts.get("Moderate Risk", 0))
            c3.metric("High Risk (Critical)", counts.get("High Risk", 0))

            st.dataframe(batch_df, use_container_width=True)

            st.download_button(
                "Export Audited Roster (CSV)",
                data=batch_df.to_csv(index=False),
                file_name="audited_student_roster.csv",
                mime="text/csv"
            )


# VIEW 3: Model Performance & Drivers
else:
    st.title("Model Architecture & Performance Benchmark")
    st.markdown("Holdout testing metrics, cross-validation stability, and dual-stage model comparison.")

    tab1, tab2 = st.tabs(["Post-Midterm Model (Comprehensive)", "Pre-Midterm Early Model (Weeks 1-6)"])

    with tab1:
        if comp_df is not None:
            st.subheader("Post-Midterm Benchmark Matrix")
            st.dataframe(comp_df, use_container_width=True)

        fig11_path = "reports/figures/fig11_feature_importance_comparison.png"
        if os.path.exists(fig11_path):
            st.subheader("Feature Driver Rankings")
            st.image(fig11_path, caption="Comparative Drivers: Multiple Linear Regression (Beta) vs. Random Forest (Gini)")

    with tab2:
        if early_comp_df is not None:
            st.subheader("Pre-Midterm Performance Matrix (Weeks 1-6)")
            st.dataframe(early_comp_df, use_container_width=True)
            st.caption("Evaluates performance without midterm_score. Achieves R2 = 0.916 and RMSE = 3.69, demonstrating early intervention feasibility.")
