# Student Score Prediction

A reproducible, leak-free machine learning data science pipeline that predicts continuous student final exam scores ($0$ to $100$) based on pre-final indicators (attendance, study hours, assignment averages, midterm score, and previous GPA).

Built strictly according to the multi-agent specification pack in [`student-score-prediction-spec/`](student-score-prediction-spec/) and verified against [`docs/05_ACCEPTANCE_CRITERIA.md`](student-score-prediction-spec/docs/05_ACCEPTANCE_CRITERIA.md).

---

## Performance Summary

| Model | Test MAE | Test RMSE | Test $R^2$ | 5-Fold CV $R^2$ | Status |
|---|---|---|---|---|---|
| **Multiple Linear Regression** | **2.627** | **3.393** | **0.9288** | **0.8815 $\pm$ 0.0176** | **Champion Model** |
| **Ridge Regression ($\alpha=1.0$)** | 2.627 | 3.394 | 0.9288 | 0.8816 $\pm$ 0.0176 | Regularized Baseline |
| **Random Forest Regressor** | 3.148 | 4.020 | 0.9000 | 0.8542 $\pm$ 0.0255 | Comparison Model |

*Selection Rationale:* Multiple Linear Regression achieved lower test RMSE ($3.393$ vs $4.020$) and demonstrated superior parsimony and interpretability.

---

## Repository Structure

```
Data_Analysis/
├── config.yaml                     # Single source of truth configuration
├── requirements.txt                # Pinned dependencies
├── pytest.ini                      # Pytest configuration
├── README.md                       # Main project landing documentation
├── PROJECT_OVERVIEW_CONTEXT.md     # Master context specification document
├── student-score-prediction-spec/  # Detailed specifications & agent briefs
├── data/
│   ├── raw/                        # Untouched raw dataset & data dictionary
│   │   ├── students_raw.csv
│   │   ├── DATA_SOURCE.md
│   │   └── DATA_DICTIONARY.md
│   ├── interim/                    # Cleaned, validated canonical data (398 rows)
│   │   └── students_clean.csv
│   └── processed/                  # Leak-free 80/20 train and test sets
│       ├── train.csv               # 318 samples (79.9%)
│       └── test.csv                # 80 samples (20.1%)
├── notebooks/
│   ├── 01_data_cleaning.ipynb      # Interactive data cleaning notebook
│   ├── 02_eda.ipynb                # Exploratory analysis notebook
│   └── 03_modeling.ipynb           # Model training and evaluation notebook
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration loader
│   ├── data_loader.py              # Ingestion and benchmark generator
│   ├── cleaning.py                 # V1-V7 validation & skewness imputation
│   ├── eda.py                      # Descriptive stats & 8 figures generator
│   ├── features.py                 # VIF diagnostics, split & train scaler
│   ├── train.py                    # Scikit-learn Pipeline model training
│   ├── evaluate.py                 # Metrics, 5-fold CV, selection & diagnostics
│   └── predict.py                  # Single-student & scenario inference
├── models/
│   ├── scaler.joblib               # StandardScaler (fit on train only)
│   ├── linear_regression.joblib    # Serialized Linear Regression Pipeline
│   ├── random_forest.joblib        # Serialized Random Forest Pipeline
│   ├── ridge.joblib                # Serialized Ridge Regression Pipeline
│   └── best_model.joblib           # Champion Model Pipeline
├── reports/
│   ├── figures/                    # 10 High-res figures (150 DPI)
│   │   ├── fig01_final_score_hist.png
│   │   ├── fig02_corr_heatmap.png
│   │   ├── fig03_midterm_vs_final.png
│   │   ├── fig04_attendance_box.png
│   │   ├── fig05_study_vs_final.png
│   │   ├── fig06_pairplot.png
│   │   ├── fig07_outliers.png
│   │   ├── fig08_gpa_vs_final.png
│   │   ├── fig09_residuals.png
│   │   └── fig10_actual_vs_pred.png
│   ├── tables/                     # Output statistics and prediction tables
│   │   ├── cleaning_log.csv
│   │   ├── missing_values.csv
│   │   ├── descriptive_stats.csv
│   │   ├── eda_findings.md
│   │   ├── vif_report.csv
│   │   ├── feature_selection_notes.md
│   │   ├── training_log.csv
│   │   ├── model_comparison.csv
│   │   ├── metrics.csv
│   │   ├── cv_results.csv
│   │   ├── selection_decision.md
│   │   ├── test_predictions.csv
│   │   ├── custom_predictions.csv
│   │   ├── coefficients.csv
│   │   └── feature_importance.csv
│   └── final_report.md             # 15-Section Comprehensive Technical Report
├── tests/
│   ├── test_cleaning.py            # Unit tests for schema & cleaning rules
│   └── test_pipeline.py            # End-to-end integration & anti-leakage tests
└── handoff/                        # Complete audit trail across all 10 agents
    ├── STATUS_LOG.md               # Final QA PASS status log
    ├── handoff_00_orchestrator.md
    ├── handoff_01_data_collector.md
    ├── handoff_02_data_cleaner.md
    ├── handoff_03_eda_analyst.md
    ├── handoff_04_visualization.md
    ├── handoff_05_feature_and_split.md
    ├── handoff_06_model_trainer.md
    ├── handoff_07_evaluator.md
    ├── handoff_08_report_writer.md
    └── handoff_09_qa_reviewer.md
```

---

## Quickstart & Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data validation and cleaning
python src/cleaning.py

# 3. Generate descriptive statistics and visualizations
python src/eda.py

# 4. Feature engineering and 80/20 train/test split
python src/features.py

# 5. Train all models
python src/train.py

# 6. Evaluate models, cross-validation, and scenario predictions
python src/evaluate.py
python src/predict.py

# 7. Execute test suite
python -m pytest -v tests/
```

---

## Verification & QA Sign-Off

- **Automated Tests:** 6/6 tests passing (`pytest tests/`).
- **Data Leakage Check:** Split occurred prior to scaling; scaler fit exclusively on $X_{\text{train}}$.
- **Anti-Overfitting:** $R^2_{\text{train}} - R^2_{\text{test}} = -0.0407$ ($< 0.15$ limit).
- **QA Result:** **PASS** (recorded in [`handoff/STATUS_LOG.md`](handoff/STATUS_LOG.md)).
