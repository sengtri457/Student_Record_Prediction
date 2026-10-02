"""At-Risk Early Warning and Actionable Prescription Module for Student Score Prediction.
Categorizes students into actionable risk tiers and calculates concrete intervention prescriptions.
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


class EarlyWarningSystem:
    """Early Warning and Prescription Engine."""

    def __init__(self, model_path: str = "models/best_model.joblib", coef_path: str = "reports/tables/coefficients.csv"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.pipeline = joblib.load(model_path)
        
        # Load unstandardized slopes for prescription math
        if os.path.exists(coef_path):
            coef_df = pd.read_csv(coef_path).set_index("feature")
            self.b_study = float(coef_df.loc["study_hours_week", "unstandardized_coef"])
            self.b_att = float(coef_df.loc["attendance_pct", "unstandardized_coef"])
            self.b_assign = float(coef_df.loc["assignment_avg", "unstandardized_coef"])
        else:
            # Fallback estimates from Multiple Linear Regression
            self.b_study = 0.3968
            self.b_att = 0.1907
            self.b_assign = 0.1882

    def classify_risk(self, predicted_score: float) -> tuple[str, str, str]:
        """Returns risk level, badge label, and color code."""
        if predicted_score >= 75.0:
            return "LOW_RISK", "🟢 Low Risk (On Track)", "#16a34a"
        elif predicted_score >= 60.0:
            return "MODERATE_RISK", "🟡 Moderate Risk (Watchlist)", "#d97706"
        else:
            return "HIGH_RISK", "🔴 High Risk (Critical Action Needed)", "#dc2626"

    def compute_prescription(
        self,
        predicted_score: float,
        current_study: float,
        current_att: float,
        current_assign: float,
        target_score: float = 70.0
    ) -> dict:
        """Computes concrete action steps needed to reach target_score."""
        deficit = target_score - predicted_score
        if deficit <= 0:
            return {
                "target_score": target_score,
                "deficit_points": 0.0,
                "status": "Target already achieved or exceeded",
                "recommended_study_hours_add": 0.0,
                "recommended_attendance_add": 0.0,
                "recommended_assignment_add": 0.0,
                "summary": "Student is performing on pace. Maintain current study and attendance rhythm."
            }

        # 1. Pure Study Hours Option
        extra_study = round(deficit / max(self.b_study, 0.01), 1)
        feasible_study = min(extra_study, max(0.0, 50.0 - current_study))

        # 2. Pure Attendance Option
        max_possible_att_gain = max(0.0, 100.0 - current_att)
        needed_att = deficit / max(self.b_att, 0.01)
        extra_att = round(min(needed_att, max_possible_att_gain), 1)

        # 3. Balanced Dual Action
        half_deficit = deficit / 2.0
        dual_study = round(half_deficit / max(self.b_study, 0.01), 1)
        dual_att = round(min(half_deficit / max(self.b_att, 0.01), max_possible_att_gain), 1)

        summary = (
            f"Needs +{deficit:.1f} points to reach {target_score:.0f}. "
            f"Prescription: Add +{dual_study:.1f} hrs/week study AND improve attendance by +{dual_att:.1f}%."
        )

        return {
            "target_score": target_score,
            "deficit_points": round(deficit, 2),
            "status": "Intervention Required",
            "study_only_hours_add": extra_study,
            "attendance_only_add": extra_att,
            "balanced_study_add": dual_study,
            "balanced_attendance_add": dual_att,
            "summary": summary
        }

    def audit_cohort(
        self,
        data_path: str = "data/interim/students_clean.csv",
        output_path: str = "reports/tables/cohort_risk_audit.csv"
    ) -> pd.DataFrame:
        """Audits entire student cohort, assigning risk tiers and prescriptions."""
        df = pd.read_csv(data_path)
        feature_cols = ["attendance_pct", "study_hours_week", "assignment_avg", "midterm_score", "previous_gpa"]
        
        preds = self.pipeline.predict(df[feature_cols])
        df["predicted_final_score"] = np.round(np.clip(preds, 0.0, 100.0), 1)

        risk_levels = []
        badges = []
        prescriptions = []

        for _, row in df.iterrows():
            r_code, badge, _ = self.classify_risk(row["predicted_final_score"])
            rx = self.compute_prescription(
                predicted_score=row["predicted_final_score"],
                current_study=row["study_hours_week"],
                current_att=row["attendance_pct"],
                current_assign=row["assignment_avg"],
                target_score=70.0
            )
            risk_levels.append(r_code)
            badges.append(badge)
            prescriptions.append(rx["summary"])

        df["risk_tier"] = risk_levels
        df["risk_badge"] = badges
        df["actionable_prescription"] = prescriptions

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        audit_cols = [
            "student_id", "attendance_pct", "study_hours_week", "assignment_avg",
            "midterm_score", "previous_gpa", "final_score", "predicted_final_score",
            "risk_tier", "risk_badge", "actionable_prescription"
        ]
        df_audit = df[audit_cols]
        df_audit.to_csv(output_path, index=False)

        # Summary counts
        tier_counts = df["risk_tier"].value_counts().to_dict()
        print(f"Cohort Risk Audit Complete ({len(df)} students):")
        print(f"  [LOW_RISK] Low Risk (On Track):         {tier_counts.get('LOW_RISK', 0)} ({round(tier_counts.get('LOW_RISK', 0)/len(df)*100, 1)}%)")
        print(f"  [MODERATE_RISK] Moderate Risk (Watchlist): {tier_counts.get('MODERATE_RISK', 0)} ({round(tier_counts.get('MODERATE_RISK', 0)/len(df)*100, 1)}%)")
        print(f"  [HIGH_RISK] High Risk (Critical):          {tier_counts.get('HIGH_RISK', 0)} ({round(tier_counts.get('HIGH_RISK', 0)/len(df)*100, 1)}%)")
        print(f"Audit table exported to: {output_path}")

        return df_audit


if __name__ == "__main__":
    ews = EarlyWarningSystem()
    ews.audit_cohort()
