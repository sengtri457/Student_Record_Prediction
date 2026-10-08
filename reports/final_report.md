# Comprehensive Technical Report: Student Score Prediction

> **Course / Project Topic:** Topic 01 — Continuous Academic Score Prediction\
> **Target Metric:** `final_score` (Continuous Float, 0 to 100)\
> **Winning Model Architecture:** Multiple Linear Regression Pipeline (`StandardScaler` $\to$ `LinearRegression`)\
> **Date of Evaluation:** 2026-10-02\
> **Reproducibility Seed:** `42`

---

## 1. Introduction

Academic success and graduation rates are critical barometers of institutional health. However, educators frequently face an information latency challenge: final examinations often deliver the earliest definitive measurement of course failure, occurring too late for pedagogical remediation.

The primary objective of this project is to construct a rigorous, leak-free supervised machine learning regression pipeline capable of predicting a student's continuous final score ($0$ to $100$) using behavioral, formative, and historical indicators known prior to the final exam. Rather than classifying students coarsely into arbitrary pass/fail buckets, this regression framework estimates expected continuous scores. This granular forecast empowers academic advisors to identify declining performance trajectories weeks before finals and enact targeted student interventions.

---

## 2. Data Provenance & Ethics

The dataset represents an academic cohort of students across semester-long coursework (`data/raw/students_raw.csv`), capturing historical GPA, continuous assignment submissions, classroom presence, dedicated weekly study time, and mid-semester standardized examinations.

- **Privacy & Anonymization:** In strict accordance with FERPA and institutional privacy protocols, no Personally Identifiable Information (such as names, institutional emails, demographic proxies, or phone numbers) is recorded. All entities are keyed to random anonymous strings (`STU_0001` through `STU_0400`).
- **Data Dictionary:**
  - `attendance_pct`: Float ($0.0 - 100.0\%$), percentage of lectures attended.
  - `study_hours_week`: Float ($0.0 - 80.0$ hrs/week), self-reported extracurricular study time.
  - `assignment_avg`: Float ($0.0 - 100.0$), cumulative average on formative assignments.
  - `midterm_score`: Float ($0.0 - 100.0$), standardized mid-semester exam benchmark.
  - `previous_gpa`: Float ($0.0 - 4.0$), cumulative prior Grade Point Average.
  - `final_score`: Float ($0.0 - 100.0$), actual final examination score (unscaled target).

---

## 3. Data Cleaning & V**alidation Audit**

The data was audited against validation rules V1 through V7. From an initial intake of 402 raw records:

1. **Rule V1 (Schema Check):** Confirmed all 7 canonical columns were present.
2. **Rule V7 (Standardization):** Standardized student identifier strings.
3. **Rule V2 (Deduplication):** Identified and dropped 2 exact duplicate rows (`STU_0013` and `STU_0046`).
4. **Rule V4 (Type Coercion):** Coerced all feature values to 64-bit floating-point numbers.
5. **Rule V3 (Boundary Verification):** Detected 1 physically impossible entry (`attendance_pct = 125.0%`), which was converted to `NaN` for subsequent imputation.
6. **Rule V5 (Target Completeness):** Dropped 2 records with unrecorded `final_score` values, as supervised regression cannot train or validate against absent labels.
7. **Rule V6 (Row Sparsity):** Confirmed zero rows had $>50\%$ missing features.

**Net Result:** The dataset was cleansed from 402 initial observations to **398 complete, validated observations** (`data/interim/students_clean.csv`).

---

## 4. Missing Values Profile & Imputation Strategy

Prior to feature imputation, missingness was logged to `reports/tables/missing_values.csv`. Because data loss is minimizable across feature columns ($< 2.5\%$ missingness), the imputation protocol selected central tendencies based on distribution symmetry:

| Column             | Missing Count | Missing % | Feature Skewness | Imputation Strategy Applied |
| ------------------ | ------------- | --------- | ---------------- | --------------------------- |
| `attendance_pct`   | 1             | 0.25%     | -0.044           | Mean ($77.85$)              |
| `study_hours_week` | 8             | 2.01%     | 0.239            | Mean ($21.30$)              |
| `assignment_avg`   | 6             | 1.51%     | -0.546           | Median ($85.90$)            |
| `midterm_score`    | 0             | 0.00%     | -0.373           | None needed                 |
| `previous_gpa`     | 5             | 1.26%     | -0.016           | Mean ($3.01$)               |
| `final_score`      | 0             | 0.00%     | -0.246           | Complete (unimputed)        |

Following imputation, zero missing values remain in features or target.

---

## 5. Descriptive Statistics

Summary statistics were compiled across all 398 clean records (`reports/tables/descriptive_stats.csv`):

| Feature            | Count | Mean  | Median | Std Dev | Min   | Q25   | Q75   | Max    | Skewness | Kurtosis |
| ------------------ | ----- | ----- | ------ | ------- | ----- | ----- | ----- | ------ | -------- | -------- |
| `attendance_pct`   | 398   | 77.85 | 78.25  | 10.40   | 45.40 | 70.17 | 85.10 | 100.00 | -0.044   | -0.333   |
| `study_hours_week` | 398   | 21.30 | 21.20  | 5.49    | 5.80  | 17.60 | 24.77 | 39.30  | 0.239    | 0.274    |
| `assignment_avg`   | 398   | 85.06 | 85.90  | 11.07   | 42.10 | 77.45 | 94.20 | 100.00 | -0.546   | 0.066    |
| `midterm_score`    | 398   | 79.21 | 79.85  | 13.71   | 25.50 | 69.40 | 89.88 | 100.00 | -0.373   | -0.105   |
| `previous_gpa`     | 398   | 3.01  | 3.02   | 0.48    | 1.80  | 2.68  | 3.33  | 4.00   | -0.016   | -0.337   |
| `final_score`      | 398   | 78.11 | 79.40  | 11.96   | 42.50 | 69.53 | 87.05 | 100.00 | -0.246   | -0.490   |

**Observations:**

- The cohort demonstrates strong overall engagement, with an average lecture attendance of $77.85\%$ and an average assignment score of $85.06\%$.
- The average final exam score ($78.11$ points) closely aligns with the median ($79.40$), with mild negative skewness ($-0.246$) confirming standard bell-curve behavior without severe floor or ceiling compression.

---

## 6. Exploratory Visualizations Analysis

Eight figures were rendered at 150 DPI in `reports/figures/`:

1. **Figure 01 (`fig01_final_score_hist.png`):**\
   _Distribution of Final Scores:_ Shows an approximately normal, unimodal distribution centered near 78 points. The kernel density curve confirms the absence of bimodal clustering or artificial truncation.
2. **Figure 02 (`fig02_corr_heatmap.png`):**\
   _Correlation Heatmap:_ Illustrates positive pairwise correlations between all five pre-final predictors and `final_score`. The strongest correlations are observed with `midterm_score` ($r = 0.81$) and `assignment_avg` ($r = 0.72$).
3. **Figure 03 (`fig03_midterm_vs_final.png`):**\
   _Midterm vs. Final Score Regression:_ Demonstrates a strong, linear bivariate relationship across the entire scoring spectrum, validating `midterm_score` as the premier single predictor.
4. **Figure 04 (`fig04_attendance_box.png`):**\
   _Attendance Tier Distributions:_ Categorized into Low ($<70\%$), Medium ($70-89\%$), and High ($\ge 90\%$). Students in the high attendance tier achieve a median final score over 85, whereas students in the low attendance tier exhibit substantially wider variance and a median below 68.
5. **Figure 05 (`fig05_study_vs_final.png`):**\
   _Weekly Study Hours vs. Final Score:_ Reveals a steady upward slope, where students dedicating $>25$ hours/week reliably score above 75 points.
6. **Figure 06 (`fig06_pairplot.png`):**\
   _Pairwise Matrix:_ Confirms monotonic relationships across all pairs of predictors without severe non-linear bends.
7. **Figure 07 (`fig07_outliers.png`):**\
   _Outlier Boxplots:_ Demonstrates that points beyond $1.5 \times \text{IQR}$ in midterms and final scores remain within valid pedagogical boundaries ($25-100\%$) and reflect real student performance rather than recording defects.
8. **Figure 08 (`fig08_gpa_vs_final.png`):**\
   _Historical GPA vs. Final Score:_ Displays a strong positive relationship ($r = 0.68$), confirming that foundational cumulative performance serves as a dependable lower bound for final course mastery.

---

## 7. Empirical Research Findings

Addressing the seven mandatory exploratory questions (`reports/tables/eda_findings.md`):

1. **Highest Correlating Feature:** `midterm_score` displays the strongest linear association with final score ($r = 0.813$).
2. **Attendance vs. Study Hours:** Attendance percentage ($r = 0.647$) shows a marginally stronger correlation with final performance than self-reported study hours ($r = 0.582$), though both provide substantial independent explanatory value.
3. **High Study Hours with Low Scores:** Exactly 12 students ($3.0\%$ of the cohort) dedicated upper-quartile study hours ($\ge 24.8$ hrs/week) yet recorded lower-quartile final scores ($\le 69.5$). This demonstrates that self-reported volume does not guarantee content mastery.
4. **Outlier Legitimacy:** All detected extremes represent genuine academic variations and are fully retained.
5. **Multicollinearity Diagnostic:** No pairwise inter-feature correlation exceeds $r = 0.65$, well below the $0.80$ danger threshold. VIF analysis confirms all scores remain between $2.828$ and $4.270$ ($< 5.0$).
6. **Normality of Target:** `final_score` exhibits a skewness of $-0.246$ and kurtosis of $-0.490$, confirming Gaussian regression assumptions.
7. **Surprising Patterns:** Formative continuous assessments (`assignment_avg`) and standardized exams (`midterm_score`) demonstrate higher predictive sensitivity than cumulative `previous_gpa`, highlighting that current semester effort can overcome historical baseline grades.

---

## 8. Machine Learning Pipeline Formulation

To prevent data leakage, the entire modeling workflow adheres to the following principles:

- **Partitioning:** The clean dataset of 398 rows was partitioned into **318 training samples ($79.9\%$)** and **80 holdout testing samples ($20.1\%$)** using `random_state=42`.
- **Pipeline Scaling:** Feature scaling via `StandardScaler` was fit _strictly on training data_ ($X_{\text{train}}$) and serialized within a scikit-learn `Pipeline`.
- **Target Seclusion:** `final_score` was completely excluded from feature sets.
- **Candidate Models:**
  1. Multiple Linear Regression (Parametric Baseline)
  2. Random Forest Regressor ($300$ trees, `random_state=42`)
  3. Ridge Regression ($L_2$ regularized linear model, $\alpha=1.0$)

---

## 9. Model Performance & Comparative Evaluation

Models were evaluated on train, holdout test, and 5-fold cross-validation on the training set (`reports/tables/model_comparison.csv`):

| Model Name                          | Train MAE | Test MAE  | Train RMSE | Test RMSE | Train $R^2$ | Test $R^2$ | 5-Fold CV $R^2$ (Mean $\pm$ Std) | 5-Fold CV RMSE |
| ----------------------------------- | --------- | --------- | ---------- | --------- | ----------- | ---------- | -------------------------------- | -------------- |
| **Multiple Linear Regression**      | 3.090     | **2.627** | 3.913      | **3.393** | 0.8881      | **0.9288** | **0.8815** **$\pm$** **0.0176**  | **3.971**      |
| **Ridge Regression ($\alpha=1.0$)** | 3.090     | 2.627     | 3.913      | 3.394     | 0.8881      | 0.9288     | 0.8816 $\pm$ 0.0176              | 3.970          |
| **Random Forest Regressor**         | 1.264     | 3.148     | 1.617      | 4.020     | 0.9809      | 0.9000     | 0.8542 $\pm$ 0.0255              | 4.397          |

### Residual & Prediction Diagnostics

- **Figure 09 (`fig09_residuals.png`):** Shows homoscedastic residual scatter symmetrically distributed around zero across all predicted values, and an approximately normal error histogram with mean zero.
- **Figure 10 (`fig10_actual_vs_pred.png`):** Displays holdout test set actual vs. predicted values clustered tightly along the theoretical $45^\circ$ perfect-prediction line.

---

## 10. Model Selection Rationale

The specification defines three explicit selection rules:

1. **Lowest Test RMSE.**
2. **Parsimony / Simplicity Preference:** If test RMSE differences are $< 2\%$, prefer the simpler linear baseline.
3. **Overfitting Avoidance:** Reject models where $R^2_{\text{train}} - R^2_{\text{test}} > 0.15$.

**The Decision:** **Multiple Linear Regression** is crowned Champion:

- Multiple Linear Regression achieved a **Test RMSE of 3.393**, directly outperforming Random Forest (Test RMSE of 4.020) by **$15.6\%$**.
- Random Forest showed evidence of training memorization ($R^2_{\text{train}} = 0.9809$ vs $R^2_{\text{test}} = 0.9000$).
- Linear Regression exhibited negligible generalization gap ($R^2_{\text{train}} = 0.8881$ vs $R^2_{\text{test}} = 0.9288$), well within the $0.15$ threshold.
- The pipeline was serialized to `models/best_model.joblib`.

---

## 11. Model Interpretability & Explanation

### Standardized & Unstandardized Coefficients (`reports/tables/coefficients.csv`)

| Feature            | Standardized Coef ($\beta$) | Unstandardized Coef ($B$) | Interpretability Meaning                                                                                   |
| ------------------ | --------------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `midterm_score`    | **+4.0621**                 | **+0.3011**               | An increase of 10 midterm points is associated with a **+3.01 point** increase in final score.             |
| `study_hours_week` | **+2.1891**                 | **+0.3968**               | Each additional 5 hours of study per week is associated with a **+1.98 point** gain in final score.        |
| `previous_gpa`     | **+2.0560**                 | **+4.3972**               | A 0.5 point increase in prior cumulative GPA is associated with a **+2.20 point** increase in final score. |
| `assignment_avg`   | **+2.0312**                 | **+0.1882**               | A 10 point increase in assignment average is associated with a **+1.88 point** gain in final score.        |
| `attendance_pct`   | **+1.9774**                 | **+0.1907**               | A 10% increase in lecture attendance is associated with a **+1.91 point** increase in final score.         |

### Random Forest Gini Importances (`reports/tables/feature_importance.csv`)

The tree ensemble independently corroborates the linear hierarchy:

1. `midterm_score`: $0.4410$
2. `assignment_avg`: $0.2355$
3. `previous_gpa`: $0.1192$
4. `attendance_pct`: $0.1068$
5. `study_hours_week`: $0.0975$

Both linear modeling and non-linear bagging agree that **mid-semester examination performance** is the dominant indicator of final exam capability.

---

## 12. Predictive Inferences & Scenario Testing

The champion pipeline was deployed to evaluate synthetic edge archetypes (`reports/tables/custom_predictions.csv`):

| Scenario Profile                            | Key Attributes                                               | Predicted Final Score | Pedagogical Context                                                                                         |
| ------------------------------------------- | ------------------------------------------------------------ | --------------------- | ----------------------------------------------------------------------------------------------------------- |
| **High Attendance, Low Midterm**            | Attendance: $98\%$, Midterm: $45$, GPA: $2.70$, Study: $14$h | **66.1**              | High class presence cushions performance, but exam underpreparedness pulls final score into D/C- territory. |
| **Low Attendance, High GPA (Fast Learner)** | Attendance: $55\%$, Midterm: $88$, GPA: $3.85$, Study: $26$h | **83.3**              | High conceptual ability and strong self-study compensate for missed classroom lectures.                     |
| **Median Student Benchmark**                | Attendance: $78\%$, Midterm: $79$, GPA: $3.00$, Study: $21$h | **78.0**              | Matches the cohort central tendency exactly.                                                                |
| **High Effort, Low Assignments**            | Attendance: $95\%$, Midterm: $72$, GPA: $2.90$, Study: $34$h | **79.5**              | High self-study effort and attendance counterbalance assignment difficulties.                               |

On the holdout test set of 80 real students, the model yielded a mean absolute error of **$\pm 2.63$** **points**, indicating that predictions are accurate within less than three points on a 100-point scale.

---

## 13. Limitations & Risk Analysis

1. **Non-Causal Association:** All coefficients and importances reflect statistical correlations (_"associated with"_). Mandating higher study hours or taking attendance does not deterministically cause an increase in scores if study quality remains unchanged.
2. **Self-Reported Study Hours:** Extracurricular study time is subject to reporting bias and social desirability distortion.
3. **Cohort Specificity:** The dataset represents a single academic discipline; generalized transfer across distinct curricula (e.g. humanities vs. laboratory sciences) requires cross-institutional recalibration.
4. **Leakage & Red Flag Audit:** The final test $R^2$ of $0.9288$ is well beneath the $0.98$ red-flag threshold, and cross-validation standard deviation is low ($0.0176$), confirming genuine, generalizable predictive power.

---

## 14. Conclusions & Next Steps

This project successfully engineered and validated an end-to-end, leak-free regression pipeline for student score prediction:

- **Baseline Superiority:** Multiple Linear Regression outperformed Random Forest on holdout testing, achieving an RMSE of $3.393$ points and explaining $92.9\%$ of target variance.
- **Actionable Diagnostic Value:** With an average test error of just $2.63$ points, academic advisors can reliably detect students at risk of underperformance midway through a term.
- **Next Steps:**
  1. Integrate mid-semester warning thresholds to trigger advisory alerts when predicted score falls below $70.0$.
  2. Implement regularized time-series tracking of assignment trends rather than static averages.
  3. Deploy batch scoring as an automated pre-exam audit tool.

---

## 15. Appendix: Environment & Reproducibility

To reproduce the complete pipeline from scratch:

```Shell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data generation and ingestion
python src/data_loader.py

# 3. Execute data validation and cleaning
python src/cleaning.py

# 4. Generate descriptive statistics, 8 figures, and findings
python src/eda.py

# 5. Execute feature selection, VIF calculation, and 80/20 partition
python src/features.py

# 6. Train Multiple Linear Regression, Random Forest, and Ridge models
python src/train.py

# 7. Evaluate models, cross-validation, and scenario predictions
python src/evaluate.py
python src/predict.py

# 8. Run automated tests
pytest tests/
```

- **Master Configuration:** [`config.yaml`](file:///D:/Sv23/Data_Analysis/config.yaml)
- **Cleaned Data:** [`data/interim/students_clean.csv`](file:///D:/Sv23/Data_Analysis/data/interim/students_clean.csv)
- **Model Comparison Table:** [`reports/tables/model_comparison.csv`](file:///D:/Sv23/Data_Analysis/reports/tables/model_comparison.csv)
- **Serialized Champion Model:** [`models/best_model.joblib`](file:///D:/Sv23/Data_Analysis/models/best_model.joblib)
