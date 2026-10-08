"""Early-Stage (Pre-Midterm) Prediction Model & Midterm Anomaly Detector.
Allows academic forecasting in Weeks 1-6 before midterm examinations,
and detects acute exam performance anomalies (test anxiety, illness, bad day).
"""

import os
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
import joblib


def load_config(config_path: str = "config.yaml") -> dict:
    """Load config yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_early_stage_model(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
    model_output_path: str = "models/early_stage_model.joblib",
    table_output_path: str = "reports/tables/early_stage_comparison.csv"
) -> tuple[Pipeline, pd.DataFrame]:
    """Trains and evaluates the Pre-Midterm Early Prediction Model."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 4 Pre-midterm features (no midterm_score)
    early_features = ["attendance_pct", "study_hours_week", "assignment_avg", "previous_gpa"]
    target = "final_score"

    X_train = train_df[early_features]
    y_train = train_df[target]
    X_test = test_df[early_features]
    y_test = test_df[target]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # Metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_res = cross_validate(
        pipeline, X_train, y_train, cv=kf,
        scoring={"r2": "r2", "neg_rmse": "neg_root_mean_squared_error"}
    )
    cv_r2_mean = float(np.mean(cv_res["test_r2"]))
    cv_r2_std = float(np.std(cv_res["test_r2"]))
    cv_rmse_mean = float(-np.mean(cv_res["test_neg_rmse"]))

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)

    # Comparison record
    comparison = pd.DataFrame([{
        "model_stage": "Pre-Midterm (Weeks 1-6)",
        "features_used": "attendance, study_hours, assignment_avg, previous_gpa",
        "train_mae": round(train_mae, 3),
        "test_mae": round(test_mae, 3),
        "train_rmse": round(train_rmse, 3),
        "test_rmse": round(test_rmse, 3),
        "train_r2": round(train_r2, 4),
        "test_r2": round(test_r2, 4),
        "cv_r2_mean": round(cv_r2_mean, 4),
        "cv_r2_std": round(cv_r2_std, 4),
        "cv_rmse_mean": round(cv_rmse_mean, 3)
    }])
    os.makedirs(os.path.dirname(table_output_path), exist_ok=True)
    comparison.to_csv(table_output_path, index=False)

    print(f"Pre-Midterm Early Stage Model trained successfully:")
    print(f"  Test RMSE: {test_rmse:.3f} | Test MAE: {test_mae:.3f} | Test R2: {test_r2:.4f}")
    print(f"  Model saved to: {model_output_path}")
    print(f"  Comparison saved to: {table_output_path}")

    return pipeline, comparison


class MidtermAnomalyDetector:
    """Detects acute exam divergence (bad exam day, illness, test anxiety)."""

    def __init__(self, train_path: str = "data/processed/train.csv"):
        # Fit baseline midterm expectation model from coursework and GPA
        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path)
            X = train_df[["assignment_avg", "study_hours_week", "previous_gpa"]]
            y = train_df["midterm_score"]
            self.expected_midterm_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ])
            self.expected_midterm_pipe.fit(X, y)
        else:
            self.expected_midterm_pipe = None

    def detect_anomaly(
        self,
        actual_midterm: float,
        assignment_avg: float,
        study_hours_week: float,
        previous_gpa: float,
        threshold: float = 20.0
    ) -> dict:
        """Evaluates whether midterm deviates abnormally from student coursework trajectory."""
        if self.expected_midterm_pipe is not None:
            input_df = pd.DataFrame([{
                "assignment_avg": assignment_avg,
                "study_hours_week": study_hours_week,
                "previous_gpa": previous_gpa
            }])
            expected_midterm = float(self.expected_midterm_pipe.predict(input_df)[0])
        else:
            # Fallback heuristic expectation
            expected_midterm = 0.55 * assignment_avg + 0.3 * (study_hours_week * 1.5) + (previous_gpa * 12.0)

        expected_midterm = max(0.0, min(100.0, expected_midterm))
        deficit = expected_midterm - actual_midterm

        is_anomaly = (deficit >= threshold) and (assignment_avg >= 70.0)

        if is_anomaly:
            message = (
                f"Midterm Anomaly Alert: Coursework ({assignment_avg:.1f}%) and GPA ({previous_gpa:.2f}) "
                f"projected an expected midterm of ~{expected_midterm:.0f} pts, but actual score was {actual_midterm:.0f} "
                f"(-{deficit:.0f} pt deficit). Likely indicates situational test anxiety, acute illness, or testing anomaly."
            )
        else:
            message = "Midterm performance aligns consistently with observed coursework trajectory."

        return {
            "is_anomaly": is_anomaly,
            "actual_midterm": actual_midterm,
            "expected_midterm": round(expected_midterm, 1),
            "point_deficit": round(deficit, 1),
            "alert_message": message
        }


if __name__ == "__main__":
    train_early_stage_model()
    detector = MidtermAnomalyDetector()
    test_case = detector.detect_anomaly(
        actual_midterm=45.0,
        assignment_avg=88.0,
        study_hours_week=25.0,
        previous_gpa=3.65
    )
    print("Anomaly Detection Test Case:", test_case)
