# Part B – Step-by-Step Teacher Explanation Guide
## Point-by-Point Evidence, Numbers, and Spoken Defense (Steps 1 to 10)

**Project:** Student Record Prediction (Academic Performance Forecasting)  
**Scope:** Part B – Machine Learning (Complete 10-Requirement Point-by-Point Breakdown)  
**Target Audience:** For student use during oral presentations, defense, and grading with the teacher.  
**Companion Documents:** [Full Part B Guide](PART_B_MACHINE_LEARNING_GUIDE.md) | [Bilingual Presentation Script](PART_B_PRESENTATION_SCRIPT.md) | [Part A Guide](PART_A_DATA_ANALYSIS_GUIDE.md)

---

## Quick Navigation: The 10 Points

- [Point 1: Define the Target Variable](#point-1-define-the-target-variable)
- [Point 2: Select Relevant Features](#point-2-select-relevant-features)
- [Point 3: Preprocess the Data](#point-3-preprocess-the-data)
- [Point 4: Split the Dataset (80% Train / 20% Test)](#point-4-split-the-dataset-80-train--20-test)
- [Point 5: Train at Least 2 Machine Learning Models](#point-5-train-at-least-2-machine-learning-models)
- [Point 6: Compare the Models](#point-6-compare-the-models)
- [Point 7: Evaluate Model Performance](#point-7-evaluate-model-performance)
- [Point 8: Select the Model Based on Evaluation Metrics](#point-8-select-the-model-based-on-evaluation-metrics)
- [Point 9: Make Predictions](#point-9-make-predictions)
  - [9A: Unseen Holdout Test Predictions (80 Students)](#9a-holdout-test-predictions-80-students)
  - [9B: Custom "What-If" Student Scenarios](#9b-custom-what-if-student-scenarios)
- [Point 10: Explain the Results](#point-10-explain-the-results)
  - [10A: The Fitted Mathematical Equation](#10a-the-fitted-mathematical-equation)
  - [10B: Feature Importance Ranking (Standardized Beta Weights)](#10b-feature-importance-ranking-standardized-beta-weights)
  - [10C: Residual Assumption Diagnostics](#10c-residual-assumption-diagnostics)
  - [10D: Critical Academic Governance Disclaimer](#10d-critical-academic-governance-disclaimer)

---

## Point 1: Define the Target Variable

### 1. 📁 Where it is located in the project:
* Configuration: [`config.yaml`](../config.yaml) (Line 3: `target: final_score`)
* Documentation: [`reports/tables/feature_selection_notes.md`](../reports/tables/feature_selection_notes.md)
* Script: [`scripts/run_part_b.py`](../scripts/run_part_b.py#L32-L46)

### 2. 📊 What it contains (The Facts & Numbers):
* **Target Column Name:** `final_score`
* **Task Type:** **Supervised Regression** (continuous real-valued prediction: $\hat{y} \in \mathbb{R}$).
* **Numerical Range:** Valid theoretical domain is $[0.0, 100.0]$. In our cleaned dataset, observed values range from **$48.20$ to $98.70$ points** ($\mu = 78.07, \sigma = 10.45$).
* **Policy Decision:** We deliberately avoid binarizing the target into Pass/Fail ($\ge 60$). Continuous regression preserves granular performance variance, allowing advisors to detect a grade dropping from 88 to 65 before failure happens.
* **Leakage Protocol:** `final_score` is strictly isolated from the explanatory feature matrix $\mathbf{X}$.

### 3. 🗣️ Exactly What to Say to Your Teacher:
> **English:**  
> *"Teacher, our target variable is `final_score`, which is a continuous numeric grade from 0 to 100 points. We chose supervised regression rather than binary pass/fail classification so we can predict exact grades and detect struggling students early before they fail. To prevent data leakage, `final_score` is completely excluded from our input feature matrix."*
>
> **ភាសាខ្មែរ (Khmer):**  
> *"លោកគ្រូ/អ្នកគ្រូ អថេរគោលដៅរបស់យើងគឺ `final_score` ដែលជាពិន្ទុជាលេខបន្តពី ០ ដល់ ១០០ ពិន្ទុ។ យើងជ្រើសរើសយក Supervised Regression ជាជាងការបែងចែកជា Pass/Fail ដើម្បីទស្សន៍ទាយពិន្ទុលម្អិត និងជួយអន្តរាគមន៍សិស្សទាន់ពេលវេលា។ ដើម្បីការពារកុំឱ្យបែកធ្លាយទិន្នន័យ អថេរគោលដៅនេះត្រូវបានដកចេញដាច់ដោយឡែកពី Input Features។"*

---

## Point 2: Select Relevant Features

### 1. 📁 Where it is located in the project:
* VIF Table: [`reports/tables/vif_report.csv`](../reports/tables/vif_report.csv)
* Feature Code: [`src/features.py`](../src/features.py)
* Notes: [`reports/tables/feature_selection_notes.md`](../reports/tables/feature_selection_notes.md)

### 2. 📊 What it contains (The Facts & Numbers):
* **Selected 5 Domain Features:**
  1. `attendance_pct`: Classroom lecture attendance ($0 - 100\%$).
  2. `study_hours_week`: Self-reported independent study hours per week ($0 - 80$).
  3. `assignment_avg`: Cumulative assignment and homework score ($0 - 100$).
  4. `midterm_score`: Midterm examination score ($0 - 100$).
  5. `previous_gpa`: Cumulative Grade Point Average ($0.0 - 4.0$).
* **Excluded:** `student_id` is an administrative identifier and contains no predictive signal.
* **Variance Inflation Factor (VIF) Diagnostic Table:**
  $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

| Feature Name | VIF Score | Safety Threshold | Collinearity Risk | Decision |
| :--- | :---: | :---: | :---: | :---: |
| `attendance_pct` | **$2.939$** | $< 5.0$ | Low | **Retained** |
| `study_hours_week` | **$2.828$** | $< 5.0$ | Low | **Retained** |
| `assignment_avg` | **$3.907$** | $< 5.0$ | Low | **Retained** |
| `midterm_score` | **$4.270$** | $< 5.0$ | Low | **Retained** |
| `previous_gpa` | **$3.281$** | $< 5.0$ | Low | **Retained** |

### 3. 🗣️ Exactly What to Say to Your Teacher:
> **English:**  
> *"We selected five domain-specific academic features and dropped `student_id`. To ensure none of the features are redundant or causing multicollinearity, we calculated the Variance Inflation Factor (VIF). Every single feature scored below 4.3, well under the conservative threshold of 5.0, confirming that each predictor provides unique, independent information."*
>
> **ភាសាខ្មែរ (Khmer):**  
> *"យើងបានជ្រើសរើស Features ចំនួន ៥ ដែលឆ្លុះបញ្ចាំងពីការសិក្សា និងបានដក `student_id` ចេញ។ ដើម្បីប្រាកដថា Features ទាំងនេះមិនជាន់គ្នាខ្លាំង យើងបានគណនា VIF។ តម្លៃ VIF ទាំងអស់គឺទាបជាង ៤.៣ ដែលទាបជាងកម្រិតប្រកាសអាសន្ន ៥.០ ដូច្នេះយើងរក្សាទុក Features ទាំង ៥ នេះទាំងអស់។"*

---

## Point 3: Preprocess the Data

### 1. 📁 Where it is located in the project:
* Pipeline Code: [`src/train.py`](../src/train.py#L44-L60)
* Scaler Artifact: [`models/scaler.joblib`](../models/scaler.joblib)

### 2. 📊 What it contains (The Facts & Numbers):
* **Transformation Method:** `StandardScaler` (z-score normalization: $\mu=0, \sigma=1$).
* **Why Necessary:** Prevents features with large raw numbers (e.g., attendance $0-100$) from dominating features with small raw numbers (e.g., GPA $0.0-4.0$).
* **Anti-Leakage Architecture:** Feature scaling is encapsulated inside a scikit-learn `Pipeline([('scaler', StandardScaler()), ('regressor', Model())])`.
* **Execution Rule:** Scaler parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) are computed **strictly on the training split only**. When evaluating test data, the pipeline uses training parameters without recalculating test statistics.

### 3. 🗣️ Exactly What to Say to Your Teacher:
> **English:**  
> *"Because GPA is scaled from 0 to 4 while attendance is 0 to 100, unscaled data would distort regression weights. We standardized features using `StandardScaler`. Crucially, to prevent data leakage, we wrapped the scaler inside a scikit-learn Pipeline so it is fitted strictly on the training partition only and applied to transform the test set."*
>
> **ភាសាខ្មែរ (Khmer):**  
> *"ដោយសារ GPA គិតពី ០ ដល់ ៤ ហើយវត្តមានគិតដល់ ១០០ យើងបានធ្វើស្តង់ដារកម្មទិន្នន័យដោយប្រើ `StandardScaler`។ ដើម្បីការពារកុំឱ្យបែកធ្លាយទិន្នន័យ យើងបានដាក់ Scaler នៅក្នុង scikit-learn Pipeline ដោយគណនាតែលើ Training Set ប៉ុណ្ណោះ ហើយទើបយកទៅអនុវត្តលើ Test Set ជាក្រោយ។"*

---

## Point 4: Split the Dataset (80% Train / 20% Test)

### 1. 📁 Where it is located in the project:
* Training Split CSV: [`data/processed/train.csv`](../data/processed/train.csv) ($N=318$)
* Testing Split CSV: [`data/processed/test.csv`](../data/processed/test.csv) ($N=80$)
* Splitting Script: [`src/features.py`](../src/features.py#L36-L70)

### 2. 📊 What it contains (The Facts & Numbers):
* **Total Clean Dataset:** $N = 398$ complete student records.
* **Training Partition (80%):** $N_{\text{train}} = 318$ rows (mean final score: $78.11$, std: $10.74$).
* **Testing Partition (20%):** $N_{\text{test}} = 80$ rows (mean final score: $77.92$, std: $9.28$).
* **Reproducibility Guarantee:** Seed fixed globally to `random_state = 42`.
* **Split Balance:** Training and testing partitions have virtually identical distributions, confirming a balanced split with zero drift.

### 3. 🗣️ Exactly What to Say to Your Teacher:
> **English:**  
> *"We split our 398 clean records into an 80% training set of 318 students and a 20% holdout test set of 80 students using a fixed random seed of 42. Both sets have nearly identical average scores of 78.1 and 77.9 points, verifying an unbiased, balanced partition."*
>
> **ភាសាខ្មែរ (Khmer):**  
> *"យើងបានបែងចែកទិន្នន័យសិស្សទាំង ៣៩៨ នាក់ ជាពីរផ្នែក៖ ៨០% ជា Training Set (៣១៨ នាក់) និង ២០% ជា Testing Set (៨០ នាក់) ដោយកំណត់ Seed ថេរស្មើ ៤២។ សំណុំទិន្នន័យទាំងពីរមានមធ្យមភាគពិន្ទុប្រហាក់ប្រហែលគ្នាខ្លាំង (៧៨.១ និង ៧៧.៩) ដែលបញ្ជាក់ពីការបែងចែកប្រកបដោយតុល្យភាព។"*

---

## Point 5: Train at Least 2 Machine Learning Models

### 1. 📁 Where it is located in the project:
* Training Code: [`src/train.py`](../src/train.py)
* Serialized Model Artifacts:
  * [`models/linear_regression.joblib`](../models/linear_regression.joblib)
  * [`models/random_forest.joblib`](../models/random_forest.joblib)
  * [`models/ridge_regression.joblib`](../models/ridge_regression.joblib)

### 2. 📊 What it contains (The Facts & Numbers):
* **Model 1: Multiple Linear Regression (OLS Baseline)**
  * Parametric, interpretable hyperplane: $\hat{y} = \beta_0 + \sum_{j=1}^5 \beta_j X_j$
  * Objective: Minimize Sum of Squared Residuals (RSS).
* **Model 2: Random Forest Regressor (Non-Linear Ensemble Benchmark)**
  * Bagging ensemble of $100$ de-correlated decision trees.
  * Hyperparameters: `n_estimators=100`, `max_depth=6`, `min_samples_split=5`, `min_samples_leaf=2`, `random_state=42`.
* **Supplementary Model: Ridge Regression ($L_2$ Regularization)**
  * Penalized loss: $\text{RSS} + \alpha \|\boldsymbol{\beta}\|_2^2$ with $\alpha = 1.0$.

### 3. 🗣️ Exactly What to Say to Your Teacher:
> **English:**  
> *"To satisfy the requirement of training at least two models, we trained Multiple Linear Regression as our interpretable parametric baseline, and Random Forest Regressor with 100 trees and depth 6 to test for non-linear interactions. We also trained Ridge Regression as a regularized benchmark."*
>
> **ភាសាខ្មែរ (Khmer):**  
> *"យើងបានបង្វឹកម៉ូដែលចំនួន ២ ផ្សេងគ្នា៖ Multiple Linear Regression ជាម៉ូដែលគោលបែបលីនេអ៊ែរដែលងាយស្រួលបកស្រាយ និង Random Forest Regressor ដែលមានដើមឈើ ១០០ ដើម ដើម្បីស្វែងរកទំនាក់ទំនងមិនមែនលីនេអ៊ែរ។ យើងក៏បានបង្វឹក Ridge Regression មួយទៀតផងដែរ។"*

---

## Point 6: Compare the Models

### 1. 📁 Where it is located in the project:
* Comparison Table: [`reports/tables/model_comparison.csv`](../reports/tables/model_comparison.csv)
* Cross-Validation Table: [`reports/tables/cv_results.csv`](../reports/tables/cv_results.csv)

### 2. 📊 What it contains (The Verified Numbers):

| Model Architecture | Train $R^2$ | Test $R^2$ | Train RMSE | Test RMSE | Test MAE | 5-Fold CV $R^2$ ($\mu \pm \sigma$) | 5-Fold CV RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multiple Linear Regression** | $0.8881$ | **$0.9288$** | $3.913$ | **$3.393$** | **$2.627$** | **$0.8815 \pm 0.0176$** | **$3.971$** |
| **Random Forest Regressor** | **$0.9809$** | $0.9000$ | **$1.617$** | $4.020$ | $3.148$ | $0.8542 \pm 0.0255$ | $4.397$ |
| **Ridge Regression ($\alpha=1.0$)** | $0.8881$ | $0.9288$ | $3.913$ | $3.394$ | $2.627$ | $0.8816 \pm 0.0176$ | $3.970$ |

### 3. 🗣️ Exactly What to Say to Your Teacher:
> **English:**  
> *"Comparing our models on the holdout test set and through 5-fold cross-validation: Multiple Linear Regression outperformed Random Forest across every metric. It achieved a lower Test RMSE of 3.39 compared to 4.02, and a higher 5-fold cross-validated R-squared of 0.881 versus 0.854."*
>
> **ភាសាខ្មែរ (Khmer):**  
> *"ការប្រៀបធៀបលើ Test Set និង 5-Fold Cross-Validation បង្ហាញថា Linear Regression ទទួលបានលទ្ធផលល្អជាង Random Forest គ្រប់រង្វាស់ទាំងអស់។ វាមាន Test RMSE ទាបជាង (៣.៣៩ ធៀបនឹង ៤.០២) និងមានមធ្យមភាគ R² ក្នុងការធ្វើ Cross-Validation ខ្ពស់ជាង (០.៨៨១ ធៀបនឹង ០.៨៥៤)។"*

---

## Point 7: Evaluate Model Performance

### 1. 📁 Where it is located in the project:
* Metrics File: [`reports/tables/metrics.csv`](../reports/tables/metrics.csv)
* Residual Plot: [`reports/figures/fig09_residuals.png`](../reports/figures/fig09_residuals.png)

### 2. 📊 What it contains (The Facts & Numbers):
* **Test Accuracy:** Linear Regression achieves an MAE of **$2.627$ points**, meaning typical predictions deviate by less than $2.7$ points on a 100-point examination.
* **Variance Explained:** Linear Regression accounts for **$92.88\%$** of final score variance on unseen test data.
* **Overfitting / Bias-Variance Audit:**
  * **Random Forest Overfitting:** Train $R^2 = 0.9809$ vs Test $R^2 = 0.9000$ ($\Delta = +0.0809$). Test error expands by $+148\%$ ($1.62 \to 4.02$), showing tree memorization.
  * **Linear Regression Generalization:** Train $R^2 = 0.8881$ vs Test $R^2 = 0.9288$ ($\Delta = -0.0407$). Generalizes cleanly with zero degradation.

### 3. 🗣️ Exactly What to Say to Your Teacher:
> **English:**  
> *"Our evaluation reveals that Linear Regression has an average error of only 2.6 points and explains 92.9% of final grade variance. Furthermore, Random Forest showed signs of overfitting—scoring 0.98 on training data but dropping to 0.90 on test data—whereas Linear Regression generalized consistently across all folds."*
>
> **ភាសាខ្មែរ (Khmer):**  
> *"ការវាយតម្លៃបង្ហាញថា Linear Regression មានកំហុសជាមធ្យមត្រឹមតែ ២.៦ ពិន្ទុប៉ុណ្ណោះ និងពន្យល់ទិន្នន័យបាន ៩២.៩%។ លើសពីនេះ Random Forest បានបង្ហាញសញ្ញានៃការ Overfitting ដោយ Train R² ឡើងដល់ ០.៩៨ តែធ្លាក់មក ០.៩០ លើ Test Data ចំណែកឯ Linear Regression មានស្ថិរភាពរឹងមាំខ្លាំង។"*

---

## Point 8: Select the Model Based on Evaluation Metrics

### 1. 📁 Where it is located in the project:
* Decision Markdown: [`reports/tables/selection_decision.md`](../reports/tables/selection_decision.md)
* Champion Pipeline Artifact: [`models/best_model.joblib`](../models/best_model.joblib)

### 2. 📊 What it contains (The 4 Justifications):
* **Selected Production Model:** **Multiple Linear Regression**
* **The 4 Justifications:**
  1. **Lowest Unseen Error:** Lowest holdout RMSE ($3.393$) and MAE ($2.627$).
  2. **Cross-Validation Reliability:** Higher mean CV $R^2$ ($0.8815$) and lowest standard deviation ($\pm 0.0176$).
  3. **Occam's Razor (Principle of Parsimony):** When a simpler model matches or beats a complex ensemble, data science principles require selecting the simpler model.
  4. **Pedagogical Explainability:** Decision trees are black-boxes; linear regression gives exact point slopes that educators can communicate to students.

### 3. 🗣️ Exactly What to Say to Your Teacher:
> **English:**  
> *"We selected Multiple Linear Regression as our champion model based on four criteria: lowest test error, superior cross-validation stability, Occam's razor (preferring the simpler model when it outperforms the complex one), and full interpretability for academic advisors."*
>
> **ភាសាខ្មែរ (Khmer):**  
> *"យើងបានសម្រេចជ្រើសរើស Multiple Linear Regression ជាម៉ូដែលផ្លូវការ ដោយសារលក្ខណៈវិនិច្ឆ័យ ៤ យ៉ាង៖ កំហុសទាបជាងគេ, ស្ថិរភាព Cross-Validation ខ្ពស់ជាង, គោលការណ៍ Occam's Razor (ជ្រើសរើសម៉ូដែលសាមញ្ញពេលវាផ្តល់លទ្ធផលល្អជាង) និងភាពងាយស្រួលបកស្រាយសម្រាប់លោកគ្រូអ្នកគ្រូ។"*

---

## Point 9: Make Predictions

Point 9 covers **both holdout test evaluation** and **scenario-based forecasting**.

### 9A: Holdout Test Predictions (80 Students)
* 📁 **File:** [`reports/tables/test_predictions.csv`](../reports/tables/test_predictions.csv)
* 📊 **Sample Verification Table (First 5 Test Students):**

| Student ID | Actual Final Score | Predicted Score ($\hat{y}$) | Error ($y - \hat{y}$) | Absolute Error |
| :---: | :---: | :---: | :---: | :---: |
| `STU_0201` | **$90.10$** | **$87.56$** | $+2.54$ | $2.54$ |
| `STU_0399` | **$75.70$** | **$76.65$** | $-0.95$ | $0.95$ |
| `STU_0035` | **$94.00$** | **$92.65$** | $+1.35$ | $1.35$ |
| `STU_0211` | **$89.20$** | **$88.98$** | $+0.22$ | $0.22$ |
| `STU_0096` | **$58.80$** | **$60.75$** | $-1.95$ | $1.95$ |

* 📊 **Error Distribution:** Mean error is $-0.198$ points (centered at zero); **$90\%$ of students** were predicted within $\pm 5.4$ points of their actual grades.

### 9B: Custom "What-If" Student Scenarios
* 📁 **File:** [`reports/tables/custom_predictions.csv`](../reports/tables/custom_predictions.csv)
* 📊 **Persona Test Cases:**
  * **Scenario 1 (High Attendance $98\%$, Low Midterm $45$):** Predicted Final = **$66.1$** *(Early warning flags student at risk)*.
  * **Scenario 2 (Low Attendance $55\%$, High Prior GPA $3.85$):** Predicted Final = **$83.3$** *(Self-study and foundational knowledge compensate)*.
  * **Scenario 3 (Cohort Median Student):** Predicted Final = **$78.0$**.

### 3. 🗣️ Exactly What to Say to Your Teacher:
> **English:**  
> *"For Requirement 9, we generated predictions for all 80 unseen test students. For example, student STU_0211 scored 89.2 and our model predicted 89.0, an error of just 0.22 points. Over 90% of predictions fall within plus or minus 5.4 points. We also tested custom personas, proving the model successfully flags at-risk students who struggle on midterms."*
>
> **ភាសាខ្មែរ (Khmer):**  
> *"សម្រាប់ចំណុចទី ៩ យើងបានទស្សន៍ទាយពិន្ទុសិស្សតេស្តទាំង ៨០ នាក់។ ឧទាហរណ៍ សិស្ស STU_0211 មានពិន្ទុជាក់ស្តែង ៨៩.២ ហើយម៉ូដែលទស្សន៍ទាយបាន ៨៩.០ (ខុសគ្នាត្រឹម ០.២២ ពិន្ទុ)។ ៩០% នៃការព្យាករណ៍ខុសគ្នាមិនលើសពី ៥.៤ ពិន្ទុឡើយ។ យើងក៏បានសាកល្បងសេណារីយ៉ូជាក់ស្តែង ដោយម៉ូដែលបានផ្តល់សញ្ញាអាសន្នលើសិស្សដែលខ្សោយពិន្ទុពាក់កណ្តាលឆមាស។"*

---

## Point 10: Explain the Results

Point 10 is structured across **four rigorous pillars**.

### 10A: The Fitted Mathematical Equation
* 📁 **Script Output:** [`scripts/run_part_b.py`](../scripts/run_part_b.py)
* 📊 **Fitted Unstandardized Formula:**
  $$\begin{aligned}
  \widehat{\text{final\_score}} = 1.77 &+ 0.3011 \times (\text{midterm\_score}) \\
  &+ 0.3968 \times (\text{study\_hours\_week}) \\
  &+ 4.3972 \times (\text{previous\_gpa}) \\
  &+ 0.1882 \times (\text{assignment\_avg}) \\
  &+ 0.1907 \times (\text{attendance\_pct})
  \end{aligned}$$

---

### 10B: Feature Importance Ranking (Standardized Beta Weights)
* 📁 **File:** [`reports/tables/coefficients.csv`](../reports/tables/coefficients.csv)
* 📊 **Standardized Beta Weights Table:**

| Rank | Feature Name | Standardized Beta ($\beta^*$) | Unstandardized Slope ($B$) | Practical Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| **#1** | **`midterm_score`** | **$+4.0621$** | $+0.301$ pts/pt | **Strongest determinant.** Summative mid-semester exam benchmark. |
| **#2** | **`study_hours_week`** | **$+2.1891$** | $+0.397$ pts/hr | **Highest return on effort.** Each extra study hr/week adds $\approx +0.40$ exam pts. |
| **#3** | **`previous_gpa`** | **$+2.0560$** | $+4.397$ pts/GPA pt | **Foundational baseline.** Each $1.0$ higher GPA adds $+4.40$ exam pts. |
| **#4** | **`assignment_avg`** | **$+2.0312$** | $+0.188$ pts/pt | **Formative effort.** Regular homework adds $+0.19$ pts per mark. |
| **#5** | **`attendance_pct`** | **$+1.9774$** | $+0.191$ pts/% | **Baseline exposure.** Classroom presence adds $+0.19$ pts per percent. |

---

### 10C: Residual Assumption Diagnostics
* 📁 **Diagnostic Figures:**
  * Residual distribution: [`reports/figures/fig09_residuals.png`](../reports/figures/fig09_residuals.png)
  * Actual vs. Predicted: [`reports/figures/fig10_actual_vs_pred.png`](../reports/figures/fig10_actual_vs_pred.png)
* 📊 **Gauss-Markov Verification:**
  1. **Mean Error:** $\mu_e = -0.198 \approx 0$ (unbiased predictions).
  2. **Normality:** Q-Q plot and error histogram follow a symmetric Gaussian bell curve.
  3. **Homoscedasticity:** Error spread is equal and constant across low, medium, and high score bands.

---

### 10D: Critical Academic Governance Disclaimer
* **Correlation $\neq$ Causation:** These coefficients represent predictive associations observed in this benchmark cohort. They do not constitute counterfactual guarantees (e.g., forcing an unmotivated student to sit at a desk will not automatically yield $+0.40$ points without quality study habits).
* **Synthetic Cohort Note:** The dataset is a calibrated synthetic benchmark ($N=398$); conclusions demonstrate methodology and should be validated with institutional data before real school deployment.

### 3. 🗣️ Exactly What to Say to Your Teacher:
> **English:**  
> *"To explain our results: our standardized coefficients show that midterm score (+4.06) is the strongest overall driver, while study hours (+2.19) provides the highest actionable return on effort—each additional weekly study hour contributes about +0.40 points to the final exam. GPA, assignments, and attendance provide steady positive support. Our residual plots confirm homoscedasticity and normality. Finally, we maintain scientific honesty: correlation does not prove causation, and these findings represent predictive associations."*
>
> **ភាសាខ្មែរ (Khmer):**  
> *"ដើម្បីពន្យល់លទ្ធផល៖ មេគុណស្តង់ដារបង្ហាញថា ពិន្ទុពាក់កណ្តាលឆមាស (+៤.០៦) ជះឥទ្ធិពលខ្លាំងជាងគេ ហើយម៉ោងសិក្សា (+២.១៩) ផ្តល់ផលចំណេញខ្ពស់បំផុតលើការខិតខំ ដោយរាល់ការបន្ថែមម៉ោងសិក្សា ១ ម៉ោង ជួយបន្ថែមពិន្ទុប្រហែល +០.៤០ ពិន្ទុ។ ក្រាហ្វ Residuals បញ្ជាក់ថាកំហុសមានលក្ខណៈធម្មតា និងគ្មានលម្អៀង។ ចុងក្រោយ យើងសូមបញ្ជាក់ថា ទំនាក់ទំនងស្ថិតិនេះ (Correlation) មិនមែនជាការសន្និដ្ឋានហេតុនិងផលដាច់ខាត (Causation) នោះឡើយ។"*

---

## Teacher Q&A Defense Quick-Reference

| Teacher Question | Your Fast, Defensible Answer |
| :--- | :--- |
| **"Why not use Pass/Fail classification?"** | *"Binary classification discards continuous grade variance. A student dropping from 85 to 65 needs intervention, but classification treats both as 'Pass'. Regression gives exact point forecasts."* |
| **"Why did Linear Regression beat Random Forest?"** | *"When the underlying phenomenon is linear additive, decision trees suffer from high variance and step-function errors. On 318 training rows, Random Forest overfitted ($R^2=0.98 \to 0.90$), whereas Linear Regression generalized better."* |
| **"Did feature scaling cause data leakage?"** | *"No. We used scikit-learn Pipelines. The scaler was fitted strictly on $X_{\text{train}}$ only and applied to transform $X_{\text{test}}$ without recalculation."* |
| **"How do you know features aren't redundant?"** | *"We ran VIF diagnostics. All features scored $< 4.3$, far below the collinearity risk threshold of $5.0$."* |
| **"Can you run the code live?"** | *"Yes! Running `python scripts/run_part_b.py` in terminal executes all 10 steps in under 3 seconds."* |
