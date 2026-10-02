# 03. Part A: Data Analysis Spec

Maps to the 8 required steps. Each step lists inputs, outputs, and how to know it is done.

| # | Required step | Owner agent | Output file(s) |
|---|---|---|---|
| A1 | Define the problem | 01 Data Collector (with Orchestrator) | `docs/00_PROJECT_OVERVIEW.md`, report section 1 |
| A2 | Collect a dataset | 01 Data Collector | `data/raw/students_raw.csv`, `DATA_SOURCE.md`, `DATA_DICTIONARY.md` |
| A3 | Clean the data | 02 Data Cleaner | `data/interim/students_clean.csv`, `cleaning_log.csv` |
| A4 | Handle missing values | 02 Data Cleaner | `reports/tables/missing_values.csv` |
| A5 | Perform EDA | 03 EDA Analyst | `notebooks/02_eda.ipynb` |
| A6 | Descriptive statistics | 03 EDA Analyst | `reports/tables/descriptive_stats.csv` |
| A7 | At least 5 visualizations | 04 Visualization | `reports/figures/*.png` |
| A8 | Important patterns and relationships | 03 EDA Analyst | `reports/tables/eda_findings.md` |

## A1. Define the problem

Write one paragraph: who benefits, what is predicted, what data is used, what success looks like.

## A2. Collect a dataset

- Follow `02_DATA_SPEC.md`.
- Minimum 100 rows preferred.
- Save an untouched copy in `data/raw/`. Never edit it.

## A3 and A4. Clean and handle missing values

Order of work:

1. Load raw data
2. Rename columns to canonical names
3. Fix dtypes
4. Remove duplicates
5. Apply range checks (V3)
6. Build missingness table
7. Apply missing value policy
8. Save clean data and log

Done when: no missing values in features, no out of range values, row counts before and after are logged.

## A6. Descriptive statistics

For every numeric column: count, mean, median, std, min, 25%, 75%, max, skewness, kurtosis. Save as CSV. Add 2 to 3 sentences of reading in the notebook.

## A7. Required visualizations (minimum 5)

| # | Chart | File name | What it shows |
|---|---|---|---|
| 1 | Histogram + KDE of `final_score` | `fig01_final_score_hist.png` | Target distribution |
| 2 | Correlation heatmap (all numeric) | `fig02_corr_heatmap.png` | Linear relationships, multicollinearity |
| 3 | Scatter + regression line: `midterm_score` vs `final_score` | `fig03_midterm_vs_final.png` | Strongest expected link |
| 4 | Box plot of `final_score` by attendance group (low/medium/high) | `fig04_attendance_box.png` | Attendance effect |
| 5 | Scatter or hexbin: `study_hours_week` vs `final_score` | `fig05_study_vs_final.png` | Study hours link |
| 6 (extra) | Pair plot | `fig06_pairplot.png` | All pair relations |
| 7 (extra) | Box plots for outliers | `fig07_outliers.png` | Outlier check |
| 8 (extra) | `previous_gpa` vs `final_score` | `fig08_gpa_vs_final.png` | Prior performance link |

Chart rules: title, axis labels with units, readable font, saved at 150 dpi or higher, one idea per chart.

Attendance groups: low (<70), medium (70 to 89), high (90 and above). Change cutoffs only if the data needs it, and say so.

## A8. Patterns to investigate

Write answers in `reports/tables/eda_findings.md`:

1. Which feature has the highest correlation with `final_score`?
2. Does attendance matter more or less than study hours?
3. Are there students with high study hours but low scores? How many?
4. Are there outliers? Real or errors?
5. Which features are strongly correlated with each other (above 0.8)? This matters for linear regression.
6. Is `final_score` roughly normal? Any skew?
7. Any surprising result? Explain it carefully without claiming cause.

## Part A done when

- All 8 steps have output files
- At least 5 figures exist and are referenced in the notebook
- Findings file answers all 7 questions
