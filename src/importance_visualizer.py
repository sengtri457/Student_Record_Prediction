"""Model Drivers and Feature Importance Comparison Visualizer.
Generates publication-ready comparative bar charts contrasting Linear Regression
standardized betas against Random Forest Gini feature importances.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance_comparison(
    coef_path: str = "reports/tables/coefficients.csv",
    fi_path: str = "reports/tables/feature_importance.csv",
    output_path: str = "reports/figures/fig11_feature_importance_comparison.png"
) -> str:
    """Generates comparative 2-panel feature driver visualization."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    coef_df = pd.read_csv(coef_path).sort_values("standardized_coef", ascending=True)
    fi_df = pd.read_csv(fi_path).sort_values("gini_importance", ascending=True)

    # Human-readable labels
    label_map = {
        "midterm_score": "Midterm Exam Score",
        "study_hours_week": "Weekly Study Hours",
        "previous_gpa": "Prior Cumulative GPA",
        "assignment_avg": "Assignment Average",
        "attendance_pct": "Lecture Attendance %"
    }

    coef_df["clean_feature"] = coef_df["feature"].map(label_map).fillna(coef_df["feature"])
    fi_df["clean_feature"] = fi_df["feature"].map(label_map).fillna(fi_df["feature"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    sns.set_theme(style="whitegrid")

    # Panel 1: Linear Regression Standardized Betas
    colors_lr = ["#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8"]
    bars1 = ax1.barh(coef_df["clean_feature"], coef_df["standardized_coef"], color=colors_lr, edgecolor="black")
    ax1.set_title("Multiple Linear Regression (Standardized Beta)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Standardized Coefficient (\u03b2 - Score Points per 1 SD)", fontsize=11)
    ax1.set_xlim(0, max(coef_df["standardized_coef"]) * 1.25)
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.1, bar.get_y() + bar.get_height()/2, f"{w:.2f}",
                 va="center", ha="left", fontsize=10, fontweight="bold", color="#1e293b")

    # Panel 2: Random Forest Gini Importance
    colors_rf = ["#a7f3d0", "#6ee7b7", "#34d399", "#10b981", "#059669"]
    bars2 = ax2.barh(fi_df["clean_feature"], fi_df["gini_importance"], color=colors_rf, edgecolor="black")
    ax2.set_title("Random Forest Regressor (Gini Importance)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Relative Feature Importance (MDI)", fontsize=11)
    ax2.set_xlim(0, max(fi_df["gini_importance"]) * 1.25)
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w*100:.1f}%",
                 va="center", ha="left", fontsize=10, fontweight="bold", color="#064e3b")

    fig.suptitle("Comparative Model Drivers: Parametric vs. Non-Linear Feature Rankings", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Feature importance comparison figure generated at: {output_path}")
    return output_path


if __name__ == "__main__":
    plot_feature_importance_comparison()
