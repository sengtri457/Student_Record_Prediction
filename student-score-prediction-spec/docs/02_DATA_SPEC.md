# 02. Data Spec

## Canonical schema

All data must end up in this shape in `data/interim/students_clean.csv`.

| Column | Type | Allowed range | Missing allowed in raw | Notes |
|---|---|---|---|---|
| `student_id` | string/int | unique | no | Anonymous ID only. No names. |
| `attendance_pct` | float | 0 to 100 | yes | % of classes attended |
| `study_hours_week` | float | 0 to 80 | yes | Self reported hours per week |
| `assignment_avg` | float | 0 to 100 | yes | Average of assignment scores |
| `midterm_score` | float | 0 to 100 | yes | Midterm exam score |
| `previous_gpa` | float | 0.0 to 4.0 | yes | GPA from the earlier term |
| `final_score` | float | 0 to 100 | **no** | Target. Drop rows where it is missing. |

If a scale differs (for example GPA out of 5, or scores out of 20), convert to the ranges above and write the conversion in the data notes.

## Source options

Pick one. Record the choice in `data/raw/DATA_SOURCE.md`.

| Option | Pros | Cons |
|---|---|---|
| A. Own survey of classmates | Original, matches features exactly | Small (50 to 150 rows), self reported |
| B. UCI Student Performance | Real, well known, G1/G2/G3 grades | Needs column mapping, no assignment score |
| C. Kaggle student performance datasets | Easy to get, has study hours and attendance | Columns vary, check quality and license |
| D. Hybrid (survey plus public data) | More rows | Needs care, document how merged |
| E. Synthetic | Only for testing the pipeline | Patterns are fake. Do not use for final conclusions. |

If using E for testing, label every file `SYNTHETIC` and replace it before the final report.

### Suggested column mapping (verify against the real file)

| Canonical | UCI Student Performance | Kaggle style |
|---|---|---|
| `attendance_pct` | derive from `absences` (needs total classes) | `Attendance` |
| `study_hours_week` | `studytime` (it is a 1 to 4 band, not hours) | `Hours_Studied` |
| `assignment_avg` | not available | not always available |
| `midterm_score` | `G1` or `G2` (scaled to 0 to 100) | not always available |
| `previous_gpa` | not available | `Previous_Scores` (proxy) |
| `final_score` | `G3` (scaled to 0 to 100) | `Exam_Score` |

If a feature is missing in the dataset, say so in the report. Either drop it or use a documented proxy. Do not invent values.

## Data dictionary

The Data Collector agent must produce `data/raw/DATA_DICTIONARY.md` with: column name, meaning, unit, how collected, and any known problems.

## Validation rules

| Rule ID | Check | Action |
|---|---|---|
| V1 | Required columns present | Fail the pipeline |
| V2 | Duplicate `student_id` | Drop exact duplicates, review the rest |
| V3 | Value outside allowed range | Set to NaN, log count |
| V4 | Wrong dtype (text in numeric column) | Coerce to numeric, bad values become NaN |
| V5 | `final_score` missing | Drop row, log count |
| V6 | Row with more than 50% features missing | Drop row, log count |
| V7 | Inconsistent text/format | Standardize |

## Missing value policy

| Situation | Method |
|---|---|
| Column roughly symmetric (skew between -0.5 and 0.5) | Mean |
| Column skewed | Median |
| Less than 5% missing and random | Median/mean is fine |
| More than 30% missing in a column | Consider dropping the column, explain |
| Row mostly empty | Drop row |

Before filling, make a missingness table (`reports/tables/missing_values.csv`) with count and percent per column. After filling, confirm zero missing in features.

## Outlier policy

- Detect with IQR (1.5 x IQR) and look at box plots.
- Do not auto delete. Keep valid extremes (a student who truly got 98).
- Remove or cap only when the value is impossible or clearly an entry error.
- Log every change in `reports/tables/cleaning_log.csv`.

## Sensitive data

- No names, emails, phone numbers.
- Survey data needs consent. Say how you got it.
- Store only anonymous IDs.
