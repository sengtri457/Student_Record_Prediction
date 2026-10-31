# Part B – Machine Learning Guide

**Project:** Student Record Prediction (Academic Performance Forecasting)  
**Scope:** Part B – Machine Learning (Ten Teacher Requirements)  
**Repository:** [sengtri457/Student_Record_Prediction](https://github.com/sengtri457/Student_Record_Prediction)  
**Dataset Reference:** Cleaned academic cohort [`data/interim/students_clean.csv`](../data/interim/students_clean.csv) ($N=398$ complete student records)  
**Evidence Base:** Executed notebooks ([`notebooks/03_modeling.ipynb`](../notebooks/03_modeling.ipynb)), source scripts ([`src/train.py`](../src/train.py), [`src/evaluate.py`](../src/evaluate.py), [`src/predict.py`](../src/predict.py)), serialized models ([`models/`](../models/)), verified evaluation tables ([`reports/tables/`](../reports/tables/)), and diagnostic figures ([`reports/figures/`](../reports/figures/)).

---

## Part B Checklist & Evidence Matrix

This guide provides exhaustive, verified evidence for all ten teacher requirements of **Part B – Machine Learning**:

| Teacher Requirement | Implementation & Repository Evidence | Verified Key Result |
| :--- | :--- | :--- |
| **1. Define the Target Variable** | Section 1; [`config.yaml`](../config.yaml); [`reports/tables/feature_selection_notes.md`](../reports/tables/feature_selection_notes.md) | `final_score` (Continuous float, $0.0 - 100.0$ scale, Supervised Regression). Zero target leakage. |
| **2. Select Relevant Features** | Section 2; [`reports/tables/vif_report.csv`](../reports/tables/vif_report.csv); [`src/features.py`](../src/features.py) | 5 domain predictors (`attendance_pct`, `study_hours_week`, `assignment_avg`, `midterm_score`, `previous_gpa`). All VIFs $< 4.30$. |
| **3. Preprocess the Data** | Section 3; [`src/train.py`](../src/train.py); [`notebooks/03_modeling.ipynb`](../notebooks/03_modeling.ipynb) | Scikit-learn `Pipeline` with `StandardScaler` fit strictly on $X_{\text{train}}$ only; transformed on $X_{\text{test}}$. |
| **4. Split the Dataset (80/20)** | Section 4; [`data/processed/train.csv`](../data/processed/train.csv); [`data/processed/test.csv`](../data/processed/test.csv) | 80% Train ($N=318$), 20% Test ($N=80$). Global seed `random_state=42`. Stratified range check verified. |
| **5. Train $\ge 2$ ML Models** | Section 5; [`src/train.py`](../src/train.py); [`models/linear_regression.joblib`](../models/linear_regression.joblib), [`models/random_forest.joblib`](../models/random_forest.joblib) | Model 1: Multiple Linear Regression (OLS baseline); Model 2: Random Forest Regressor; Benchmark: Ridge Regression ($\alpha=1.0$). |
| **6. Compare the Models** | Section 6; [`reports/tables/model_comparison.csv`](../reports/tables/model_comparison.csv); [`reports/tables/cv_results.csv`](../reports/tables/cv_results.csv) | Side-by-side metric comparison table: Train/Test RMSE, MAE, $R^2$, and 5-Fold Cross-Validation ($\mu \pm \sigma$). |
| **7. Evaluate Performance** | Section 7; [`reports/tables/metrics.csv`](../reports/tables/metrics.csv); Section 7 residual checks | Holdout evaluation + 5-fold CV. Bias-variance and overfitting gap audited ($\Delta R^2$). |
| **8. Select the Model** | Section 8; [`reports/tables/selection_decision.md`](../reports/tables/selection_decision.md); [`models/best_model.joblib`](../models/best_model.joblib) | **Multiple Linear Regression** selected: Lower test RMSE ($3.393$ vs $4.020$), higher test $R^2$ ($0.9288$), Occam's razor, and full interpretability. |
| **9. Make Predictions** | Section 9; [`reports/tables/test_predictions.csv`](../reports/tables/test_predictions.csv); [`src/predict.py`](../src/predict.py) | Point predictions and absolute errors generated for all 80 holdout test students. |
| **10. Explain the Results** | Section 10; [`reports/tables/coefficients.csv`](../reports/tables/coefficients.csv); [`reports/figures/fig09_residuals.png`](../reports/figures/fig09_residuals.png) | Standardized beta coefficients, unstandardized impact, residual diagnostic verification, and causal limitation disclaimers. |

---

## 1. Define the Target Variable

### 1.1 Target Formulation
In supervised machine learning, defining the target variable dictates the mathematical task, the loss function, and the choice of evaluation metrics:

* **Variable Name:** `final_score`
* **Mathematical Data Type:** Continuous numeric variable ($\mathbb{R}$, stored as `float64`).
* **Scale & Valid Domain:** Range $[0.0, 100.0]$ points. In the cleaned dataset, observed values span from $48.20$ to $98.70$ points with a mean of $78.07$ and standard deviation of $10.45$.
* **Learning Paradigm:** **Supervised Regression**. The algorithm maps a vector of $p=5$ input features $\mathbf{x}_i \in \mathbb{R}^5$ to a continuous real-valued prediction $\hat{y}_i \in \mathbb{R}$:
  $$\hat{y}_i = f(\mathbf{x}_i)$$

### 1.2 Anti-Leakage & Problem Integrity Policy
1. **No Target Binarization:** Many preliminary student performance projects arbitrarily threshold the target into binary classes (e.g., $\ge 60 \implies \text{Pass}$, $< 60 \implies \text{Fail}$). In this project, binarization is strictly prohibited because it destroys granular grade variance, introduces arbitrary threshold bias, and degrades statistical utility for early intervention.
2. **Target Isolation:** `final_score` is never included in the explanatory matrix $\mathbf{X}$, nor is any mathematical surrogate or derivative of `final_score` retained in the training data.

---

## 2. Select Relevant Features

### 2.1 The Explanatory Feature Space
Building on Part A's exploratory data analysis, five pre-examination indicators were chosen. Each represents a distinct, domain-grounded dimension of student engagement and historical ability:

| Feature Name | Type | Unit / Scale | Pedagogical Meaning |
| :--- | :--- | :--- | :--- |
| `attendance_pct` | `float64` | $0.0 - 100.0\%$ | Physical/virtual classroom attendance rate throughout the academic term. |
| `study_hours_week` | `float64` | $0.0 - 80.0$ hrs | Independent weekly self-study and homework hours dedicated outside of class. |
| `assignment_avg` | `float64` | $0.0 - 100.0$ pts | Cumulative average grade across formative quizzes, lab work, and assignments. |
| `midterm_score` | `float64` | $0.0 - 100.0$ pts | Summative mid-semester formal examination score. |
| `previous_gpa` | `float64` | $0.0 - 4.0$ scale | Cumulative Grade Point Average prior to entering the current course. |

*Note on Identifiers:* `student_id` (e.g., `STU_0001`) is an administrative string identifier. It contains zero explanatory information and is strictly excluded from all feature matrices.

### 2.2 Multicollinearity Diagnostic: Variance Inflation Factor (VIF)
In multiple regression, high correlation between input features (collinearity) inflates the variance of coefficient estimates, making models unstable and uninterpretable. To verify feature independence, the **Variance Inflation Factor (VIF)** was computed for each candidate predictor:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
where $R_j^2$ is the coefficient of determination obtained by regressing feature $X_j$ onto all remaining four features.

```
VIF Diagnostic Table (Verified from reports/tables/vif_report.csv):
------------------------------------------------------------------
Feature Name          VIF Score    Threshold Rule    Multicollinearity Risk
------------------------------------------------------------------
attendance_pct        2.939        VIF < 5.0         Low / Acceptable
study_hours_week      2.828        VIF < 5.0         Low / Acceptable
assignment_avg        3.907        VIF < 5.0         Low / Acceptable
midterm_score         4.270        VIF < 5.0         Low / Acceptable
previous_gpa          3.281        VIF < 5.0         Low / Acceptable
------------------------------------------------------------------
```

**Diagnostic Conclusion:**
All five features exhibit VIF values well below the conservative rule-of-thumb threshold of $5.0$ (and far below the severe threshold of $10.0$). Although `assignment_avg` and `midterm_score` exhibit moderate mutual correlation ($r = 0.803$ from Part A), their VIFs ($3.91$ and $4.27$) prove they each contribute substantial independent variance. **All 5 features are retained.**

---

## 3. Preprocess the Data

### 3.1 Feature Scaling (Z-Score Standardization)
Because input features have drastically different physical units (e.g., GPA ranges $0.0 - 4.0$, study hours range $0 - 50$, and attendance ranges $0 - 100$), unscaled features would distort distance metrics, gradient descent convergence, and regression coefficient comparisons.

We apply **Standardization (Z-score normalization)**:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
where $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ are the sample mean and standard deviation computed **exclusively on the training dataset**.

### 3.2 Anti-Leakage Pipeline Architecture
A common methodological flaw in academic machine learning is computing scaling parameters across the entire dataset before splitting, which leaks test information into training. To guarantee zero data leakage:

1. **Pipeline Wrapping:** Scikit-learn's `Pipeline` object encapsulates the `StandardScaler` and the regression estimator together.
2. **Fit on Train Only:** `pipeline.fit(X_train, y_train)` computes $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ solely from the 318 training students.
3. **Out-of-Sample Transform:** When evaluating $X_{\text{test}}$, the pipeline applies the stored training parameters without recalculating test statistics.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
pipeline.fit(X_train, y_train)
```

---

## 4. Split the Dataset (80% Train / 20% Test)

### 4.1 Partitioning Specifications
The cleaned dataset of $N = 398$ student records was partitioned into an **80% training set** and a **20% testing holdout set** using `sklearn.model_selection.train_test_split`:

* **Total Cleaned Observations:** $N = 398$
* **Training Partition (80%):** $N_{\text{train}} = 318$ rows $\times$ 5 features
* **Testing Partition (20%):** $N_{\text{test}} = 80$ rows $\times$ 5 features
* **Reproducibility Guarantee:** Random generator seed locked to `random_state = 42`.
* **Serialized Split Files:**
  * Training: [`data/processed/train.csv`](../data/processed/train.csv)
  * Testing: [`data/processed/test.csv`](../data/processed/test.csv)

### 4.2 Split Distribution Verification
To confirm that the holdout test set accurately mirrors the training distribution without distributional drift:

| Metric | Training Set ($N=318$) | Testing Set ($N=80$) | Full Cleaned Set ($N=398$) |
| :--- | :---: | :---: | :---: |
| **Final Score Mean ($\mu$)** | $78.11$ pts | $77.92$ pts | $78.07$ pts |
| **Final Score Std ($\sigma$)** | $10.74$ pts | $9.28$ pts | $10.45$ pts |
| **Final Score Min - Max** | $48.20 - 98.70$ | $58.80 - 96.00$ | $48.20 - 98.70$ |
| **Midterm Score Mean** | $75.05$ pts | $75.69$ pts | $75.18$ pts |
| **Study Hours Mean** | $18.89$ hrs | $18.82$ hrs | $18.88$ hrs |

The close alignment between training and testing distributions confirms a representative, leak-free split.

---

## 5. Train at Least 2 Machine Learning Models

To satisfy the pedagogical objective of contrasting different model families, two primary models (plus one regularized benchmark) were trained:

### 5.1 Model 1: Multiple Linear Regression (Parametric Baseline)
* **Mathematical Paradigm:** Ordinary Least Squares (OLS).
* **Objective Function:** Minimize the Sum of Squared Residuals (RSS):
  $$\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} \left(y_i - \left(\beta_0 + \sum_{j=1}^p \beta_j X_{ij}\right)\right)^2$$
* **Role:** Serves as the interpretable, econometric baseline. Assumes linear additive relationships between predictors and final exam marks.
* **Pipeline:** `StandardScaler() -> LinearRegression()`
* **Saved Artifact:** [`models/linear_regression.joblib`](../models/linear_regression.joblib)

### 5.2 Model 2: Random Forest Regressor (Non-Linear Ensemble Benchmark)
* **Mathematical Paradigm:** Non-parametric bagging ensemble of decision trees.
* **Algorithm Mechanism:** Bootstraps $B=100$ random subsets of the training data and trains de-correlated decision trees with randomized feature subsets at each split:
  $$\hat{y}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})$$
* **Hyperparameters (from `config.yaml`):**
  * `n_estimators`: $100$
  * `max_depth`: $6$ (regularized to curb individual tree memorization)
  * `min_samples_split`: $5$
  * `min_samples_leaf`: $2$
  * `random_state`: $42$
* **Role:** Evaluates whether non-linear combinations or complex feature interactions yield higher predictive accuracy than linear equations.
* **Saved Artifact:** [`models/random_forest.joblib`](../models/random_forest.joblib)

### 5.3 Supplementary Benchmark: Ridge Regression ($L_2$ Regularization)
* **Mathematical Paradigm:** Tikhonov-regularized linear model minimizing $\text{RSS} + \alpha \|\boldsymbol{\beta}\|_2^2$ with $\alpha = 1.0$.
* **Role:** Verifies whether shrinkage of coefficients reduces potential collinearity variance.
* **Saved Artifact:** [`models/ridge_regression.joblib`](../models/ridge_regression.joblib)

---

## 6. Compare the Models

Model evaluation was executed on both the **Holdout Test Set ($N=80$)** and across **5-Fold Cross-Validation** ($K=5$ folds on training data).

### 6.1 Comprehensive Performance Comparison Table
*(Directly extracted and verified from [`reports/tables/model_comparison.csv`](../reports/tables/model_comparison.csv) and [`reports/tables/cv_results.csv`](../reports/tables/cv_results.csv)):*

| Model Architecture | Train $R^2$ | Test $R^2$ | Train RMSE | Test RMSE | Train MAE | Test MAE | 5-Fold CV $R^2$ ($\text{Mean} \pm \text{Std}$) | 5-Fold CV RMSE ($\text{Mean} \pm \text{Std}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multiple Linear Regression** | $0.8881$ | **$0.9288$** | $3.913$ | **$3.393$** | $3.090$ | **$2.627$** | **$0.8815 \pm 0.0176$** | **$3.971 \pm 0.271$** |
| **Random Forest Regressor** | **$0.9809$** | $0.9000$ | **$1.617$** | $4.020$ | **$1.264$** | $3.148$ | $0.8542 \pm 0.0255$ | $4.397 \pm 0.335$ |
| **Ridge Regression ($\alpha=1.0$)** | $0.8881$ | $0.9288$ | $3.913$ | $3.394$ | $3.090$ | $2.627$ | $0.8816 \pm 0.0176$ | $3.970 \pm 0.271$ |

### 6.2 5-Fold Cross-Validation Breakdown
The 5-fold cross-validation performance across each partition of the training set confirms consistent reliability:

| Fold Index | Linear Regression CV $R^2$ | Linear Regression CV RMSE | Random Forest CV $R^2$ | Random Forest CV RMSE |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | $0.8684$ | $3.922$ | $0.8268$ | $4.500$ |
| **Fold 2** | $0.8663$ | $4.391$ | $0.8330$ | $4.907$ |
| **Fold 3** | $0.9130$ | $3.632$ | $0.8977$ | $3.939$ |
| **Fold 4** | $0.8885$ | $3.763$ | $0.8656$ | $4.131$ |
| **Fold 5** | $0.8714$ | $4.146$ | $0.8481$ | $4.507$ |
| **Average ($\mu$)** | **$0.8815$** | **$3.971$** | **$0.8542$** | **$4.397$** |

---

## 7. Evaluate Model Performance

### 7.1 Metric Definitions & Interpretation
1. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
   Measures the standard deviation of unexplained residuals. Penalizes large errors heavily. Linear Regression achieved an out-of-sample RMSE of **$3.393$ points**, outperforming Random Forest (**$4.020$ points**).
2. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
   Measures average absolute magnitude of error. Linear Regression achieves **$2.627$ points**, meaning typical student predictions are within $\approx 2.6$ percentage points of the true final mark.
3. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
   Indicates the proportion of final score variance accounted for by the model. Linear Regression achieves $R^2 = 0.9288$ ($92.88\%$ explained variance) on unseen test data.

### 7.2 Overfitting & Bias-Variance Diagnosis
* **Random Forest Overfitting Signature:** Random Forest achieves a near-perfect training score ($\text{Train } R^2 = 0.9809$, $\text{RMSE} = 1.617$), but its test score drops substantially ($\text{Test } R^2 = 0.9000$, $\text{RMSE} = 4.020$). The performance gap ($\Delta R^2 = 0.0809$, error increases by $+148\%$) demonstrates that tree ensembles begin to memorize idiosyncrasies of the training sample.
* **Linear Regression Generalization:** Linear Regression exhibits balanced generalization. Its cross-validated performance ($CV\_R^2 = 0.8815$) translates cleanly to the holdout test set ($0.9288$), confirming that the underlying phenomenon in this academic cohort is inherently linear additive.

---

## 8. Select the Model Based on Evaluation Metrics

### 8.1 Selection Verdict
* **Winner & Recommended Model:** **Multiple Linear Regression**
* **Serialized Production Path:** [`models/best_model.joblib`](../models/best_model.joblib)

### 8.2 Formal Decision Rationale (The 4 Justifications)
1. **Superior Accuracy on Unseen Data:** Multiple Linear Regression achieved lower Test RMSE ($3.393$ vs $4.020$) and lower Test MAE ($2.627$ vs $3.148$) than the ensemble model.
2. **Superior Stability in Cross-Validation:** In 5-fold cross-validation, Linear Regression achieved higher average accuracy ($CV\_R^2 = 0.8815$ vs $0.8542$) and lower CV variance ($\sigma = 0.0176$ vs $0.0255$).
3. **Principle of Parsimony (Occam’s Razor):** In machine learning engineering, when a simple parametric model performs equal to or better than a complex non-linear model, the simpler model must be preferred. It requires fewer computational resources, eliminates hyperparameter tuning fragility, and cannot overfit through complex leaf splits.
4. **Pedagogical Actionability & Explainability:** A teacher cannot easily interpret the split thresholds of 100 random forest trees. In contrast, Multiple Linear Regression provides explicit slopes that advise educators exactly how many points a student stands to gain by increasing weekly study hours or attendance.

---

## 9. Make Predictions

Using the selected **Multiple Linear Regression Pipeline**, predictions were generated for all $80$ students in the unseen test set ([`reports/tables/test_predictions.csv`](../reports/tables/test_predictions.csv)).

### 9.1 Sample Test Predictions Audit

| Student ID | Actual Final Score | Predicted Score ($\hat{y}$) | Prediction Error ($y - \hat{y}$) | Absolute Error ($|e|$) | Academic Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `STU_0201` | $90.10$ | $87.56$ | $+2.54$ | $2.54$ | Excellent high-score forecast |
| `STU_0399` | $75.70$ | $76.65$ | $-0.95$ | $0.95$ | Sub-point error accuracy |
| `STU_0035` | $94.00$ | $92.65$ | $+1.35$ | $1.35$ | Accurate honors prediction |
| `STU_0211` | $89.20$ | $88.98$ | $+0.22$ | $0.22$ | Exact match within $0.22$ pts |
| `STU_0096` | $58.80$ | $60.75$ | $-1.95$ | $1.95$ | At-risk student identified early |
| `STU_0087` | $86.30$ | $85.38$ | $+0.92$ | $0.92$ | High consistency |
| `STU_0376` | $70.80$ | $70.12$ | $+0.68$ | $0.68$ | Excellent mid-tier tracking |
| `STU_0097` | $79.00$ | $79.25$ | $-0.25$ | $0.25$ | High precision |
| `STU_0225` | $66.30$ | $72.11$ | $-5.81$ | $5.81$ | Underperformed prediction |
| `STU_0373` | $96.00$ | $96.36$ | $-0.36$ | $0.36$ | Top-of-class precision |

### 9.2 Error Distribution Summary
* **Mean Error:** $\mu_e = -0.198$ points (essentially unbiased, centered at zero).
* **Median Absolute Error:** $2.03$ points.
* **$90^{\text{th}}$ Percentile Absolute Error:** $5.42$ points (9 out of 10 students are predicted within $\pm 5.4$ points of their actual grade).
* **Worst-Case Holdout Outlier:** Absolute error of $5.81$ points (`STU_0225`). No catastrophic prediction failures ($> 10$ points) occurred.

---

## 10. Explain the Results

### 10.1 Regression Coefficients & Feature Importance
The trained Linear Regression model yields the following empirical equation:

$$\begin{aligned}
\widehat{\text{final\_score}} = 78.11 &+ 0.3011 \cdot (\text{midterm\_score}) \\
&+ 0.3968 \cdot (\text{study\_hours\_week}) \\
&+ 4.3972 \cdot (\text{previous\_gpa}) \\
&+ 0.1882 \cdot (\text{assignment\_avg}) \\
&+ 0.1907 \cdot (\text{attendance\_pct})
\end{aligned}$$

To directly compare features on a normalized scale, **Standardized Coefficients ($\beta^*$)** were calculated (representing the point change in `final_score` per one standard deviation increase in that feature):

```
Standardized Feature Weight Ranking (from reports/tables/coefficients.csv):
-------------------------------------------------------------------------------------------------
Feature Rank    Feature Name        Std Coef (β*)   Unstandardized (B)   Practical Interpretation
-------------------------------------------------------------------------------------------------
Rank 1          midterm_score       +4.0621         +0.3011 pts/pt       Strongest primary determinant
Rank 2          study_hours_week    +2.1891         +0.3968 pts/hr       Highest return on effort
Rank 3          previous_gpa        +2.0560         +4.3972 pts/GPA pt   Historical ability anchor
Rank 4          assignment_avg      +2.0312         +0.1882 pts/pt       Continuous formative effort
Rank 5          attendance_pct      +1.9774         +0.1907 pts/%        Baseline instructional exposure
-------------------------------------------------------------------------------------------------
```

### 10.2 Pedagogical Explanations of Feature Impacts
1. **Midterm Examination ($\beta^* = +4.06$):** Midterm performance is by far the single most potent predictor. A one standard deviation increase in midterm marks ($+13.49$ points) adds $+4.06$ points to the final exam forecast. It is the closest intermediate proxy to the final exam format.
2. **Weekly Study Hours ($\beta^* = +2.19$):** Study time provides the second highest standardized impact. In unstandardized terms, each additional hour of independent study per week adds $+0.397$ points (or approximately $+4.0$ final points for an extra 10 hours/week).
3. **Cumulative GPA ($\beta^* = +2.06$):** Previous GPA reflects foundational academic competence and discipline. Each full $1.0$ increase in GPA adds $+4.40$ points to the final score expectation.
4. **Assignments & Attendance ($\beta^* \approx +2.0$ each):** Regular assignment completion and consistent classroom attendance provide foundational positive reinforcement, each adding $\approx 0.19$ points per unit percentage.

### 10.3 Residual Diagnostics & Model Assumptions Verification
To ensure that linear regression is scientifically valid and not misleading, the classical Gauss-Markov assumptions were verified against the diagnostic figures:

1. **Linearity & Calibration ([`fig10_actual_vs_pred.png`](../reports/figures/fig10_actual_vs_pred.png)):** Observed versus predicted values cluster tightly along the $45^\circ$ reference diagonal across the entire range ($60$ to $95$).
2. **Normality of Residuals ([`fig09_residuals.png`](../reports/figures/fig09_residuals.png)):** The Q-Q plot and error histogram display a bell-shaped Gaussian distribution centered at zero without heavy-tailed skewness.
3. **Homoscedasticity ([`fig09_residuals.png`](../reports/figures/fig09_residuals.png)):** A scatter plot of residuals against fitted values displays an even, constant band of variance across all predicted score tiers, confirming that prediction uncertainty does not expand for low or high performers.

### 10.4 Critical Academic Governance Disclaimer
* **Predictive Association $\neq$ Causal Intervention:** These regression coefficients represent statistical associations observed within this calibrated benchmark dataset. They demonstrate that students who study more hours achieve higher scores; they do not mathematically guarantee that artificially forcing an unmotivated student to sit at a desk for 10 hours will causally boost their exam mark by $4.0$ points.
* **Synthetic Cohort Scope:** Because the underlying data is a synthetic benchmark ($N=398$), external institutional claims should not be generalized to real university administration without real-world validation.

---

## How to Reproduce Part B Pipeline

To independently verify and reproduce all Part B tables, models, and predictions from the repository root:

```bash
# 1. Ensure dependencies are installed
pip install -r requirements.txt

# 2. Partition data into 80/20 train/test splits
python src/features.py

# 3. Train all models (Multiple Linear Regression, Random Forest, Ridge)
python src/train.py

# 4. Evaluate models, cross-validate, and execute model selection
python src/evaluate.py

# 5. Generate holdout predictions and residual diagnostics
python src/predict.py
```

All trained `.joblib` files are saved to `models/`, all diagnostic summaries are generated in `reports/tables/`, and all diagnostic plots are output to `reports/figures/`.
