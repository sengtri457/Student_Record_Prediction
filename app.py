"""Student Score Prediction & Early Warning System - Streamlit Web Application.
Provides interactive student simulation, risk tier classification, actionable intervention prescriptions,
and batch class CSV scoring.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Page Config
st.set_page_config(
    page_title="Student Score Predictor & Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-green {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .badge-yellow {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .badge-red {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Load serialized models and tables."""
    model_path = "models/best_model.joblib"
    coef_path = "reports/tables/coefficients.csv"
    comp_path = "reports/tables/model_comparison.csv"
    
    pipeline = joblib.load(model_path) if os.path.exists(model_path) else None
    coef_df = pd.read_csv(coef_path) if os.path.exists(coef_path) else None
    comp_df = pd.read_csv(comp_path) if os.path.exists(comp_path) else None
    
    return pipeline, coef_df, comp_df


pipeline, coef_df, comp_df = load_artifacts()

# Sidebar Navigation
st.sidebar.title("🎓 Navigation")
app_mode = st.sidebar.radio(
    "Select System Mode:",
    ["🎯 Student Simulator & Prescription", "📁 Batch Class Scorer", "📊 Model Performance & Drivers"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Student Score Prediction Pipeline • Topic 01 • Antigravity AI")


# MODE 1: Individual Student Simulator
if app_mode == "🎯 Student Simulator & Prescription":
    st.title("🎯 Student Performance Simulator & Early Warning Engine")
    st.markdown("Forecast a student's final examination score prior to finals and derive prescriptive actions to reach their desired target grade.")

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("📝 Student Profile Input")
        attendance = st.slider("Classroom Attendance (%)", min_value=30.0, max_value=100.0, value=85.0, step=1.0)
        study_hours = st.slider("Dedicated Study Hours / Week", min_value=0.0, max_value=60.0, value=22.0, step=1.0)
        assignment_avg = st.slider("Assignment & Quiz Average (0-100)", min_value=30.0, max_value=100.0, value=80.0, step=1.0)
        midterm = st.slider("Midterm Examination Score (0-100)", min_value=20.0, max_value=100.0, value=75.0, step=1.0)
        gpa = st.slider("Prior Cumulative GPA (0.0-4.0)", min_value=1.5, max_value=4.0, value=3.20, step=0.05)

        target_goal = st.selectbox("Target Goal Score to Achieve:", [70.0, 75.0, 80.0, 85.0, 90.0], index=0)

    with col2:
        st.subheader("🔮 Predictive Diagnostic & Risk Triage")

        if pipeline is not None:
            input_df = pd.DataFrame([{
                "attendance_pct": attendance,
                "study_hours_week": study_hours,
                "assignment_avg": assignment_avg,
                "midterm_score": midterm,
                "previous_gpa": gpa
            }])
            
            raw_pred = pipeline.predict(input_df)[0]
            pred_score = float(np.clip(raw_pred, 0.0, 100.0))

            # Risk tier
            if pred_score >= 75.0:
                badge_html = '<span class="badge-green">🟢 LOW RISK (ON TRACK)</span>'
                risk_status = "LOW_RISK"
            elif pred_score >= 60.0:
                badge_html = '<span class="badge-yellow">🟡 MODERATE RISK (WATCHLIST)</span>'
                risk_status = "MODERATE_RISK"
            else:
                badge_html = '<span class="badge-red">🔴 HIGH RISK (CRITICAL INTERVENTION)</span>'
                risk_status = "HIGH_RISK"

            # Grade Letter
            grade = "A" if pred_score >= 90 else ("B" if pred_score >= 80 else ("C" if pred_score >= 70 else ("D" if pred_score >= 60 else "F")))

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Predicted Score", f"{pred_score:.1f} / 100")
            with m2:
                st.metric("Expected Letter Grade", grade)
            with m3:
                st.metric("Confidence Interval", f"[{pred_score - 2.6:.1f} - {pred_score + 2.6:.1f}]")

            st.markdown(f"**Current Status:** {badge_html}", unsafe_allow_html=True)
            st.markdown("---")

            # Prescriptions
            st.subheader("💡 Actionable Academic Prescription")
            b_study = 0.3968
            b_att = 0.1907
            deficit = target_goal - pred_score

            if deficit <= 0:
                st.success(f"🎉 Student is projected to score **{pred_score:.1f}**, exceeding the target goal of **{target_goal:.0f}**. Keep maintaining current study routines!")
            else:
                st.warning(f"⚠️ Student is projected to fall short of target **{target_goal:.0f}** by **{deficit:.1f} points**.")
                
                extra_study = deficit / b_study
                max_att_gain = max(0.0, 100.0 - attendance)
                extra_att = min(deficit / b_att, max_att_gain)
                
                half_deficit = deficit / 2.0
                dual_study = half_deficit / b_study
                dual_att = min(half_deficit / b_att, max_att_gain)

                rx1, rx2 = st.columns(2)
                with rx1:
                    st.info(f"**Option 1: Study Intervention**\n\nAdd **+{extra_study:.1f} hrs/week** of study outside class.")
                with rx2:
                    st.info(f"**Option 2: Balanced Dual Action**\n\nAdd **+{dual_study:.1f} hrs/week** study **AND** improve attendance by **+{dual_att:.1f}%**.")


# MODE 2: Batch Class CSV Scorer
elif app_mode == "📁 Batch Class Scorer":
    st.title("📁 Batch Class Roster Scorer & Risk Auditor")
    st.markdown("Upload a spreadsheet of student records to generate automated predictions, risk badges, and intervention reports for an entire course.")

    sample_csv = pd.DataFrame([
        {"student_id": "STU_1001", "attendance_pct": 92.0, "study_hours_week": 26.0, "assignment_avg": 88.0, "midterm_score": 82.0, "previous_gpa": 3.45},
        {"student_id": "STU_1002", "attendance_pct": 60.0, "study_hours_week": 12.0, "assignment_avg": 65.0, "midterm_score": 52.0, "previous_gpa": 2.50},
        {"student_id": "STU_1003", "attendance_pct": 78.0, "study_hours_week": 19.0, "assignment_avg": 75.0, "midterm_score": 70.0, "previous_gpa": 2.90},
    ])

    st.download_button(
        "📥 Download Sample CSV Template",
        data=sample_csv.to_csv(index=False),
        file_name="sample_class_roster.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Class CSV File", type=["csv"])

    if uploaded_file is not None and pipeline is not None:
        batch_df = pd.read_csv(uploaded_file)
        feature_cols = ["attendance_pct", "study_hours_week", "assignment_avg", "midterm_score", "previous_gpa"]
        
        req_missing = [c for c in feature_cols if c not in batch_df.columns]
        if req_missing:
            st.error(f"Missing required columns in uploaded CSV: {req_missing}")
        else:
            preds = pipeline.predict(batch_df[feature_cols])
            batch_df["predicted_final_score"] = np.round(np.clip(preds, 0.0, 100.0), 1)

            def get_tier(score):
                if score >= 75.0: return "Low Risk (On Track)"
                if score >= 60.0: return "Moderate Risk (Watchlist)"
                return "High Risk (Critical)"

            batch_df["risk_tier"] = batch_df["predicted_final_score"].apply(get_tier)

            st.success(f"✅ Successfully scored {len(batch_df)} students!")

            c1, c2, c3 = st.columns(3)
            counts = batch_df["risk_tier"].value_counts()
            c1.metric("🟢 Low Risk", counts.get("Low Risk (On Track)", 0))
            c2.metric("🟡 Moderate Risk", counts.get("Moderate Risk (Watchlist)", 0))
            c3.metric("🔴 High Risk", counts.get("High Risk (Critical)", 0))

            st.dataframe(batch_df, use_container_width=True)

            st.download_button(
                "📥 Download Scored Class Audit Report",
                data=batch_df.to_csv(index=False),
                file_name="scored_class_audit.csv",
                mime="text/csv"
            )


# MODE 3: Model Performance & Drivers
else:
    st.title("📊 Model Performance & Comparative Diagnostics")
    st.markdown("Detailed statistical benchmark comparing Multiple Linear Regression against Random Forest and Ridge Regression.")

    if comp_df is not None:
        st.subheader("Model Comparison Matrix")
        st.dataframe(comp_df, use_container_width=True)

    fig11_path = "reports/figures/fig11_feature_importance_comparison.png"
    if os.path.exists(fig11_path):
        st.subheader("Feature Driver Rankings")
        st.image(fig11_path, caption="Comparative Feature Drivers (Standardized Beta vs Gini Importance)")

    st.subheader("Key Educational Insights")
    st.markdown(r"""
    - **Midterm is King:** Explains over $40\%$ of tree variance and carries the highest regression slope ($\beta = +4.06$).
    - **Effort Pays Off:** Every 5 hours of weekly study is associated with $\approx +2.0$ points.
    - **Zero Data Leakage:** Models trained on an 80/20 train/test split with `StandardScaler` isolated strictly to training data.
    """)
