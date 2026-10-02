# 00. Project Overview

## Problem

Predict a student's **final score** (0 to 100) using information known before the final exam.

Why it matters: if a student is likely to score low, a teacher can help early.

## Task type

Supervised regression. The target is a continuous number. Do not convert to pass/fail.

## Target and features

| Role | Column | Type |
|---|---|---|
| Target | `final_score` | float, 0 to 100 |
| Feature | `attendance_pct` | float, 0 to 100 |
| Feature | `study_hours_week` | float, hours per week |
| Feature | `assignment_avg` | float, 0 to 100 |
| Feature | `midterm_score` | float, 0 to 100 |
| Feature | `previous_gpa` | float, 0.0 to 4.0 |

Optional alternative target: `final_gpa` (0.0 to 4.0). Pick one target and keep it for the whole project. Default is `final_score`.

## Models

1. Multiple Linear Regression (baseline, interpretable)
2. Random Forest Regressor (non linear comparison)
3. Optional bonus: Ridge or Gradient Boosting

## In scope

- Dataset collection and documentation
- Cleaning, missing values, EDA, stats, 5+ charts
- Training and comparing 2+ models
- 80/20 split, metrics, model selection, predictions, explanation
- Final report and reproducible code

## Out of scope

- Web app or deployment
- Classification (pass/fail)
- Causal claims
- Personal data beyond what the dataset needs

## Assumptions

- Python 3.10 or newer
- Dataset has at least 100 rows. If fewer, use cross validation and say so in the report.
- Random seed is `42` everywhere.

## Deliverables

1. Cleaned dataset and raw dataset
2. Notebook(s) or scripts for Part A and Part B
3. Figures (5+)
4. Saved model files
5. Metrics table and predictions table
6. Final report (Markdown or PDF)
