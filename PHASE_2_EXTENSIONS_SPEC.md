# Phase 2 Specification: At-Risk Early Warning Engine, Model Drivers Visualizer & Interactive Web Application

> **Project:** Student Score Prediction (Topic 01)\
> **Status:** Specification Approved & Implementing\
> **Focus:** Translating predictive regression outputs into actionable pedagogical interventions, comparative feature diagnostics, and a production-grade web interface.

---

## 1. Objective & Scope

While Phase 1 established a statistically sound, leak-free regression pipeline ($R^2 = 0.9288$, Test RMSE = $3.393$), predicting scores alone does not solve the educational problem. Phase 2 extends the core system with three production-grade components:

1. **Early-Warning & Prescription Engine (`src/early_warning.py`):**
   - Triages students into three actionable risk categories based on predicted outcomes.
   - Computes quantitative prescription formulas (e.g. required extra study hours $\Delta h$ or attendance boost $\Delta a$) needed to elevate an at-risk student across the passing threshold ($70.0$ points).
   - Generates an automated cohort-wide risk audit: `reports/tables/cohort_risk_audit.csv`.

2. **Comparative Model Driver Visualizer (`src/importance_visualizer.py`):**
   - Produces a publication-ready comparative visualization (`fig11_feature_importance_comparison.png`) that juxtaposes parametric standardized $\beta$ coefficients with non-linear tree-based Gini importances.

3. **Interactive Teacher & Advisor Web Application (`app.py`):**
   - A standalone web interface built in Streamlit providing:
     - Individual Student Simulator (real-time sliders, confidence bounds, risk badges, prescription cards).
     - Batch Class Uploader (upload arbitrary class `.csv`, calculate predictions, download graded audit spreadsheet).
     - Model Performance & Analytics Dashboard.

4. **Integration & Regression Testing (`tests/test_early_warning.py`):**
   - Automated unit tests validating risk classification logic, prescription math, and edge case bounds ($0-100$).

---

## 2. Technical Specifications

### 2.1 Risk Tier Triage Definition

| Tier                              | Score Range               | Status Code     | Advisory Action Required                                         |
| --------------------------------- | ------------------------- | --------------- | ---------------------------------------------------------------- |
| **🟢 Low Risk (On Track)**        | $\hat{y} \ge 75.0$        | `LOW_RISK`      | No intervention needed; praise consistent performance.           |
| **🟡 Moderate Risk (Borderline)** | $60.0 \le \hat{y} < 75.0$ | `MODERATE_RISK` | Targeted homework support and formative feedback recommended.    |
| **🔴 High Risk (Critical)**       | $\hat{y} < 60.0$          | `HIGH_RISK`     | Immediate intervention, mandatory counseling, remedial sessions. |

### 2.2 Mathematical Prescription Engine

Let target safe passing score be $Y^* = 70.0$. For any student with $\hat{y} < Y^*$, the point gap is:
$\Delta y = Y^* - \hat{y}$

Using the unstandardized linear regression coefficients ($B_{\text{study}} \approx 0.397$ pts/hr, $B_{\text{attendance}} \approx 0.191$ pts/%):

1. **Study Hours Intervention:**
   $\Delta h = \frac{\Delta y}{B_{\text{study}}} \approx \frac{\Delta y}{0.397}$
2. **Attendance Boost Intervention:**
   $\Delta a = \min\left(100.0 - a_{\text{current}}, \frac{\Delta y}{B_{\text{attendance}}}\right)$
3. **Balanced Dual Prescription:**
   A feasible combination distributing effort across both self-study and class attendance without exceeding physical constraints ($h \le 50$, $a \le 100$).

---

## 3. Deliverables Checklist

- [x] Master Phase 2 Specification: `PHASE_2_EXTENSIONS_SPEC.md`
- [ ] Early Warning & Prescription Module: `src/early_warning.py`
- [ ] Cohort Risk Audit Dataset: `reports/tables/cohort_risk_audit.csv`
- [ ] Feature Importance Comparison Visualizer: `src/importance_visualizer.py`
- [ ] Comparative Plot: `reports/figures/fig11_feature_importance_comparison.png`
- [ ] Interactive Streamlit Web Application: `app.py`
- [ ] Unit Test Suite: `tests/test_early_warning.py`
- [ ] Test Verification: `pytest tests/` passing 100%
