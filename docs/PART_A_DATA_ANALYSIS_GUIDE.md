# Part A – Data Analysis Guide

**Project:** Student Record Prediction (student academic performance analysis)\
**Scope:** Teacher's Part A only — eight data-analysis requirements\
**Repository:** [sengtri457/Student_Record_Prediction](https://github.com/sengtri457/Student_Record_Prediction)\
**Evidence:** Existing `master` branch data, executed notebooks, scripts, reports, and figures inspected on 27 October 2026.

> **What is verified?** The raw and cleaned CSV files, saved analysis tables, two notebook outputs, source code, and eight Part A image filenames were inspected. Numerical summaries and correlations in this guide were cross-checked against the cleaned dataset. Some narrative statements in older reports disagree with the underlying results; the discrepancies are listed at the end rather than repeated as facts. This is a data-analysis report, **not a model-training or prediction report**.

## Part A checklist

| Teacher requirement                    | Evidence in this repository                                                                                                                |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 1. Define the problem                  | This section; [`README.md`](README.md); [`config.yaml`](config.yaml)                                                                       |
| 2. Collect a dataset                   | [`students_raw.csv`](data/raw/students_raw.csv); [source](data/raw/DATA_SOURCE.md); [dictionary](data/raw/DATA_DICTIONARY.md)              |
| 3. Clean the data                      | [`01_data_cleaning.ipynb`](notebooks/01_data_cleaning.ipynb); [`src/cleaning.py`](src/cleaning.py); [log](reports/tables/cleaning_log.csv) |
| 4. Handle missing values               | [missing-values report](reports/tables/missing_values.csv); [cleaned CSV](data/interim/students_clean.csv)                                 |
| 5. Perform EDA                         | [`02_eda.ipynb`](notebooks/02_eda.ipynb); [`src/eda.py`](src/eda.py)                                                                       |
| 6. Calculate descriptive statistics    | [descriptive-statistics table](reports/tables/descriptive_stats.csv)                                                                       |
| 7. Create at least five visualizations | [eight existing figures](reports/figures/) — Figures 01–08 below                                                                           |
| 8. Identify patterns and relationships | [EDA findings](reports/tables/eda_findings.md), checked against [cleaned data](data/interim/students_clean.csv)                            |

## 1. Define the problem

Student academic performance can differ depending on class attendance, study time, assignment results, midterm examination scores, and previous academic achievement. Before drawing conclusions about performance, we need to understand the data: Are the records reliable? What does a typical student score look like? Which recorded variables are associated with the final examination score?

**Part A objective:** Explore and describe student performance with accurate data-cleaning steps, summary statistics, and clear visualizations. The findings can help a teacher discuss patterns in academic data, but they do not establish what _causes_ a student to perform well or poorly.

The outcome examined is **`final_score`**, a numerical final examination score from 0 to 100. The five other measured academic variables are `attendance_pct`, `study_hours_week`, `assignment_avg`, `midterm_score`, and `previous_gpa`. `student_id` identifies records; it is not a score or explanatory measure.

**Expected Part A outputs:** a validated student dataset, missing-value and cleaning records, descriptive statistics, at least five charts, and cautious interpretations of the observed relationships.

## 2. Collect a dataset

The original file is [`data/raw/students_raw.csv`](data/raw/students_raw.csv). The repository's [source documentation](data/raw/DATA_SOURCE.md) calls this an **"Academic Performance Benchmark (Option D: Hybrid / Realistically Calibrated Academic Cohort)"** and records a collection/generation date of **2 October 2026**. More importantly, [`src/data_loader.py`](src/data_loader.py) contains `generate_benchmark_raw_dataset()`, which creates the academic variables using random-number generation with a fixed seed and deliberately inserts duplicate and problematic records for cleaning practice. This is therefore a **synthetic benchmark dataset**, not verified real-world student observations. It contains anonymous-looking synthetic IDs rather than personal names.

**Verified original size:** **402 rows × 7 columns**. The generator creates 400 student rows and adds two duplicate-ID rows as cleaning test cases. After cleaning, **398 unique, complete rows × 7 columns** remain in [`data/interim/students_clean.csv`](data/interim/students_clean.csv).

The repository's [data dictionary](data/raw/DATA_DICTIONARY.md) defines these fields:

| Column             | Meaning                                              | Expected range    |
| ------------------ | ---------------------------------------------------- | ----------------- |
| `student_id`       | Anonymous record ID, such as `STU_0001`              | Unique identifier |
| `attendance_pct`   | Percentage of classes attended                       | 0–100%            |
| `study_hours_week` | Study hours outside class per week                   | 0–80 hours        |
| `assignment_avg`   | Average assignment and quiz score                    | 0–100 points      |
| `midterm_score`    | Midterm examination score                            | 0–100 points      |
| `previous_gpa`     | GPA from earlier academic periods                    | 0–4.0             |
| **`final_score`**  | **Final examination score (main variable analyzed)** | **0–100 points**  |

These variables describe a constructed academic cohort. Relationships in synthetic data are useful for demonstrating analysis methods, but should not be generalized to a real school without real, appropriately collected observations.

## 3. Clean the data

Cleaning means correcting or removing problems that could make the analysis misleading. The implemented procedure is in [`src/cleaning.py`](src/cleaning.py), and its execution is visible in [`notebooks/01_data_cleaning.ipynb`](notebooks/01_data_cleaning.ipynb). The [cleaning log](reports/tables/cleaning_log.csv) records the actions taken.

| Actual cleaning action                                                  | Verified result                                                                     | Why it matters                                                                  |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Check the seven required columns (V1)                                   | All present                                                                         | Prevents accidental analysis of an incomplete schema                            |
| Trim spaces from `student_id` (V7)                                      | Standardization applied to 402 records                                              | IDs with extra spaces should match the same record                              |
| Drop repeated `student_id`, keeping the first record (V2)               | **2 duplicate rows removed**: IDs `STU_0013` and `STU_0046`                         | Prevents counting the same student ID more than once                            |
| Convert numeric fields using `pd.to_numeric(..., errors='coerce')` (V4) | Numeric conversion applied across retained records                                  | Makes calculations consistent and converts unparseable values to missing values |
| Check permitted numeric ranges (V3)                                     | **1 invalid value**: `STU_0023` had `attendance_pct = 125.0`; replaced with missing | Attendance cannot exceed 100%                                                   |
| Remove rows with missing `final_score` (V5)                             | **2 rows removed**                                                                  | Final-score comparisons require a known final score                             |
| Remove rows missing over 50% of the five feature values (V6)            | **0 rows removed**                                                                  | Guards against highly incomplete records                                        |
| Fill remaining feature gaps (A4)                                        | Values imputed as described below                                                   | Makes the cleaned dataset complete for the Part A summaries                     |

**Row-count reconciliation:** 402 raw rows − 2 duplicate-ID rows − 2 rows missing `final_score` = **398 cleaned rows**. Replacing invalid or missing feature values does not itself remove rows.

The actual code uses the following patterns (short excerpt from [`src/cleaning.py`](src/cleaning.py)):

```Python
# Drop duplicate IDs; keep the first occurrence.
dup_mask = df.duplicated(subset=["student_id"], keep="first")
df = df[~dup_mask].copy()

# Convert columns into numeric values, or NaN if conversion fails.
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace values outside allowed ranges with NaN.
for col, (low, high) in ALLOWED_RANGES.items():
    oob_mask = (df[col] < low) | (df[col] > high)
    df.loc[oob_mask, col] = np.nan
```

The cleaned output is saved as [`data/interim/students_clean.csv`](data/interim/students_clean.csv). It retains all seven expected columns, has unique IDs, and contains no missing values. Importantly, the cleaning notebook's introductory explanation of labels **V1–V7** is not identical to the rule names in `src/cleaning.py`; the table above uses the **implemented code and cleaning-log labels**.

## 4. Handle missing values

A _missing value_ means that no usable number is recorded for a field. The source code first removes duplicate IDs, validates ranges, and removes records with missing final scores; **then** it counts missing values and chooses how to fill the remaining gaps. Therefore the missing-value report is a **pre-imputation report after earlier cleaning**, not the untouched raw-data count.

| Column             | Missing in raw CSV | Missing just before imputation | Actual action                                                     | Missing after cleaning |
| ------------------ | -----------------: | -----------------------------: | ----------------------------------------------------------------- | ---------------------: |
| `student_id`       |                  0 |                              0 | No change                                                         |                      0 |
| `attendance_pct`   |                  0 |                          **1** | Fill with mean **77.85**                                          |                      0 |
| `study_hours_week` |                  8 |                          **8** | Fill with mean **21.30**                                          |                      0 |
| `assignment_avg`   |                  6 |                          **6** | Fill with median **85.90**                                        |                      0 |
| `midterm_score`    |                  0 |                              0 | No imputation needed                                              |                      0 |
| `previous_gpa`     |                  5 |                          **5** | Fill with mean **3.01**                                           |                      0 |
| `final_score`      |              **2** |                              0 | Drop the two missing-target rows; do **not** invent a final score |                      0 |

**Why does attendance have one missing value after having zero in the raw CSV?** The invalid `125.0%` entry was converted to `NaN` during range validation. It became a missing value to be filled.

The code computes each feature's skewness (a measure of how uneven its distribution is). When absolute skewness is **at most 0.5**, it fills gaps with the **mean** (arithmetic average). Otherwise it fills gaps with the **median** (middle value), which is less affected by extreme observations. That is why `assignment_avg` uses its median: its pre-imputation skewness is approximately **−0.538**. The code rounds replacement values to two decimals.

This is a practical strategy for a small number of missing entries, not proof that the missing values were random. Its effect on any particular record should be considered when interpreting results. The executed notebook's final checks report **zero remaining nulls**, and the saved cleaned CSV confirms this.

Supporting files: [`reports/tables/missing_values.csv`](reports/tables/missing_values.csv), [`reports/tables/cleaning_log.csv`](reports/tables/cleaning_log.csv), and [`notebooks/01_data_cleaning.ipynb`](notebooks/01_data_cleaning.ipynb).

## 5. Perform exploratory data analysis (EDA)

_Exploratory Data Analysis_ means looking carefully at the data before drawing conclusions. EDA helps describe the typical values, see the spread of scores, identify unusual records, and understand whether two variables tend to move together.

The actual Part A workflow is:

1. Load the **398-row cleaned dataset** in [`notebooks/02_eda.ipynb`](notebooks/02_eda.ipynb).
2. Calculate summary statistics (means, medians, standard deviations, percentiles, skewness, and kurtosis).
3. Plot single-variable distributions for all six numerical academic fields.
4. Examine **Pearson correlations** (linear relationships) and **Spearman correlations** (rank-based relationships) in notebook heatmaps.
5. Check for optional categorical subgroup fields. The notebook reports that `parent_education`, `internet_access`, and `tutoring` **are not present** in the cleaned data, so those comparisons are **not performed**.
6. Generate the eight saved Part A figures using [`src/eda.py`](src/eda.py).
7. Summarize patterns in [`reports/tables/eda_findings.md`](reports/tables/eda_findings.md), while cross-checking the claims against the CSV where older narrative text differs.

The notebook creates additional **inline** plots, whereas `src/eda.py` saves the presentation-ready images under [`reports/figures/`](reports/figures/). The analyses show associations, not proof that attendance, time spent studying, or any other feature directly causes final-score differences.

## 6. Calculate descriptive statistics

The table below is taken from [`reports/tables/descriptive_stats.csv`](reports/tables/descriptive_stats.csv) and agrees with the cleaned-data values and the executed EDA notebook. All rows have **398 observations after cleaning**.

| Numerical variable         |   Count |      Mean |    Median | Standard deviation |   Minimum |    Maximum |
| -------------------------- | ------: | --------: | --------: | -----------------: | --------: | ---------: |
| `attendance_pct` (%)       |     398 |     77.85 |     78.25 |              10.40 |     45.40 |     100.00 |
| `study_hours_week` (hours) |     398 |     21.30 |     21.20 |               5.49 |      5.80 |      39.30 |
| `assignment_avg` (points)  |     398 |     85.06 |     85.90 |              11.07 |     42.10 |     100.00 |
| `midterm_score` (points)   |     398 |     79.21 |     79.85 |              13.71 |     25.50 |     100.00 |
| `previous_gpa` (0–4)       |     398 |      3.01 |      3.02 |               0.48 |      1.80 |       4.00 |
| **`final_score`** (points) | **398** | **78.11** | **79.40** |          **11.96** | **42.50** | **100.00** |

**How to read these statistics:**

- **Count** is the number of available entries.
- **Mean** is the sum of the values divided by the count.
- **Median** is the middle value after sorting; half of the observations fall on either side.
- **Standard deviation** shows how widely numbers vary around the mean; a larger number means more spread, on that variable's own scale.
- **Minimum** and **maximum** are the smallest and largest recorded values.

**Interpretation:** The cleaned cohort's average final score is **78.11 points**, and its median is **79.40 points**. Midterm scores vary more widely (standard deviation **13.71**) than final scores (**11.96**). Average weekly study time is **21.30 hours**. These descriptions apply to this _synthetic dataset_; they are not estimates for real students in a particular school.

The CSV also records quartiles, skewness, and kurtosis. For `final_score`, its first quartile is **69.53**, third quartile **87.05**, and skewness **−0.246**. This indicates a modestly left-skewed distribution; it does **not** independently establish that the data follow a normal distribution.

## 7. Create at least five visualizations

**Eight saved Part A image files exist**, so the teacher's minimum of five is already met. The exact paths below are verified against the repository. The descriptions refer to the current cleaned dataset; all results are **associations or distribution summaries**.

### Figure 1 — Distribution of final scores

- **Chart:** Histogram with a smooth kernel-density estimate (KDE) curve.
- **Variables:** `final_score` on the horizontal axis; number of students on the vertical axis.
- **Purpose:** See common score ranges, spread, and overall shape.
- **How to read it:** Taller bars indicate more records in that score range; the smooth curve shows the general shape.
- **Observed result:** Scores are centered around the high 70s (**mean 78.11**, **median 79.40**) with slight negative skew (**−0.246**). Actual scores range from **42.5 to 100**.
- **Figure:** [`reports/figures/fig01_final_score_hist.png`](reports/figures/fig01_final_score_hist.png).

![1.00](reports/figures/fig01_final_score_hist.png)

### Figure 2 — Correlation heatmap

- **Chart:** Annotated Pearson correlation heatmap.
- **Variables:** The six numerical academic fields, compared in pairs.
- **Purpose:** Find stronger or weaker linear relationships.
- **How to read it:** Coefficients near **+1** mean a strong positive linear association, near **0** mean little linear association, and near **−1** mean a strong negative linear association.
- **Observed result:** `midterm_score` has the strongest association with `final_score` (**r = 0.894**); all five explanatory fields have positive correlations with the final score.
- **Figure:** [`reports/figures/fig02_corr_heatmap.png`](reports/figures/fig02_corr_heatmap.png).

![1.00](reports/figures/fig02_corr_heatmap.png)

### Figure 3 — Midterm score versus final score

- **Chart:** Scatter plot with a fitted trend line.
- **Variables:** `midterm_score` (x-axis) and `final_score` (y-axis).
- **Purpose:** Check how two examination scores vary together.
- **How to read it:** Each dot represents a student record. An upward trend suggests higher values often appear together.
- **Observed result:** A strong positive linear relationship (**r = 0.894**) is present; a higher midterm score tends to accompany a higher final score in this synthetic cohort.
- **Figure:** [`reports/figures/fig03_midterm_vs_final.png`](reports/figures/fig03_midterm_vs_final.png).

![1.00](reports/figures/fig03_midterm_vs_final.png)

### Figure 4 — Final scores by attendance group

- **Chart:** Box plot with individual points, grouped by attendance.
- **Variables:** `attendance_pct` groups and `final_score`.
- **Purpose:** Compare typical final scores and score spread among different attendance levels.
- **How to read it:** The line inside each box is the median; the box covers the middle half of scores. Dots show observations.
- **Observed result:** With **low attendance (<70%)**, **99** records have a median final score of **64.4**. With **medium attendance (70% to <90%)**, **246** records have a median of **80.8**. With **high attendance (≥90%)**, **53** records have a median of **92.8**. These differences do not demonstrate causation.
- **Figure:** [`reports/figures/fig04_attendance_box.png`](reports/figures/fig04_attendance_box.png).

![1.00](reports/figures/fig04_attendance_box.png)

### Figure 5 — Study hours versus final score

- **Chart:** Scatter plot with a fitted trend line.
- **Variables:** `study_hours_week` (x-axis) and `final_score` (y-axis).
- **Purpose:** Check whether students who report more weekly study hours tend to have higher scores.
- **How to read it:** Each dot shows a student; the line summarizes the positive or negative direction.
- **Observed result:** A strong positive association (**r = 0.821**) appears in this synthetic cohort. It does **not** mean that adding a study hour guarantees a score increase.
- **Figure:** [`reports/figures/fig05_study_vs_final.png`](reports/figures/fig05_study_vs_final.png).

![1.00](reports/figures/fig05_study_vs_final.png)

### Figure 6 — Pairwise academic relationships

- **Chart:** Pair plot (scatter plots and individual-variable distributions).
- **Variables:** All six numerical academic fields.
- **Purpose:** Inspect multiple distributions and pairwise patterns together.
- **How to read it:** Each off-diagonal panel compares two variables; diagonal panels show each variable's distribution.
- **Observed result:** Several academic variables rise together. In particular, `assignment_avg` and `midterm_score` have a positive inter-feature correlation (**r = 0.803**).
- **Figure:** [`reports/figures/fig06_pairplot.png`](reports/figures/fig06_pairplot.png).

### Figure 7 — Possible outliers

- **Chart:** Box plots of four scores/percentages on a comparable 0–100 scale.
- **Variables:** `attendance_pct`, `assignment_avg`, `midterm_score`, and `final_score`.
- **Purpose:** Identify points that lie unusually far beyond the middle of a distribution.
- **How to read it:** Individual points past a box plot's whiskers are **statistical outlier flags**, not automatically data-entry errors.
- **Observed result:** Applying the usual **1.5 × interquartile range (IQR)** rule to the cleaned CSV flags **1 attendance**, **3 assignment**, **3 midterm**, and **1 final-score** observations (counted separately for each variable). The retained values are within the data dictionary's permitted ranges; they have not been removed merely for being unusual.
- **Figure:** [`reports/figures/fig07_outliers.png`](reports/figures/fig07_outliers.png).

### Figure 8 — Previous GPA versus final score

- **Chart:** Scatter plot with a fitted trend line.
- **Variables:** `previous_gpa` (x-axis) and `final_score` (y-axis).
- **Purpose:** Compare historical academic performance with current final examination scores.
- **How to read it:** An upward trend means larger GPAs tend to occur with higher final scores.
- **Observed result:** The relationship is positive and strong (**r = 0.844**) in this dataset.
- **Figure:** [`reports/figures/fig08_gpa_vs_final.png`](reports/figures/fig08_gpa_vs_final.png).

**Scope note:** Other figures in `reports/figures/` relate to modeling or prediction and are deliberately **not** part of this Part A guide.

## 8. Identify important patterns and relationships

The values below are **Pearson correlation coefficients calculated from the 398-row cleaned dataset** and rounded to three decimal places. Pearson's `r` measures the direction and strength of a **linear association**; it does not establish cause and effect.

| Academic variable  | Correlation with `final_score` | Interpretation                                 |
| ------------------ | -----------------------------: | ---------------------------------------------- |
| `midterm_score`    |                      **0.894** | Strongest observed positive linear association |
| `assignment_avg`   |                      **0.854** | Strong positive association                    |
| `previous_gpa`     |                      **0.844** | Strong positive association                    |
| `study_hours_week` |                      **0.821** | Strong positive association                    |
| `attendance_pct`   |                      **0.807** | Strong positive association                    |

**Key conclusions from this dataset:**

1. **Midterm score is the strongest single correlate of final score.** Students with higher midterm results tend to have higher final scores in this cohort (**r = 0.894**).
2. **Both attendance and study time are associated with final score.** Study hours (**0.821**) have a slightly higher correlation than attendance (**0.807**), but this small difference should not be overinterpreted as greater causal importance.
3. **Assignment scores and previous GPA are also strongly associated.** Their coefficients are **0.854** and **0.844**, respectively.
4. **There are zero records meeting one specific high-study/low-score definition.** Using study hours in the top quarter (**≥ 24.775 hours/week**) and final scores in the bottom quarter (**≤ 69.525 points**), the count is **0 of 398**. This is specific to the generated cohort; it is not evidence that studying more always prevents poor results.
5. **One pair of explanatory variables exceeds 0.80.** `assignment_avg` and `midterm_score` correlate at **r = 0.803**. This is worth noticing when interpreting individual relationships, but a high pairwise correlation by itself does not prove a statistical problem.
6. **The final-score distribution is mildly left-skewed.** Skewness is **−0.246** (kurtosis **−0.490**). There are some unusual values, but the observed final scores stay within the allowed **0–100** range.
7. **The generator itself creates related variables.** These relationships are expected in a simulated dataset designed with shared underlying factors. We cannot claim that any variable _causes_ higher real-world marks based on these observations.

For comparison with earlier authored interpretations, see [`reports/tables/eda_findings.md`](reports/tables/eda_findings.md). Where its narrative contradicts the data, the **checked CSV calculations above take precedence**.

## How to reproduce Part A

Run these commands **from the repository root** with a Python environment that has the dependencies from [`requirements.txt`](requirements.txt):

```Shell
pip install -r requirements.txt
python src/cleaning.py
python src/eda.py
```

These are real entry points in the project, not invented commands. `src/cleaning.py` regenerates the cleaned dataset and its logs from the original raw CSV; `src/eda.py` regenerates the descriptive statistics, eight Part A figures, and EDA findings. **They overwrite their own generated Part A outputs**, so copy any manually edited reports first if you wish to preserve them. Neither command requires executing Part B scripts.

To inspect the executed notebooks, after installing dependencies use:

```Shell
cd notebooks
jupyter notebook
```

Then open `01_data_cleaning.ipynb` or `02_eda.ipynb` in the browser. **The existing notebook code uses** **`../`** **paths**, so launch the notebook server from the `notebooks/` directory as shown. If you need to rerun the notebooks, use the same directory context.

**Where to find results:**

- Raw input: [`data/raw/students_raw.csv`](data/raw/students_raw.csv)
- Clean output: [`data/interim/students_clean.csv`](data/interim/students_clean.csv)
- Cleaning audit: [`reports/tables/cleaning_log.csv`](reports/tables/cleaning_log.csv)
- Missing values: [`reports/tables/missing_values.csv`](reports/tables/missing_values.csv)
- Descriptive table: [`reports/tables/descriptive_stats.csv`](reports/tables/descriptive_stats.csv)
- Interpretations: [`reports/tables/eda_findings.md`](reports/tables/eda_findings.md)
- Images: [`reports/figures/`](reports/figures/)

## Verification and remaining evidence issues

**Verified:** The repository contains all eight Part A figure filenames, the required cleaning and EDA notebooks, raw/clean CSVs, source documentation, descriptive table, and missing-value report. The executed notebooks show **402 → 398 rows** and the EDA statistics match the saved descriptive-statistics table. The underlying cleaned CSV contains **398 unique IDs**, zero missing values, and valid recorded numeric ranges.

**Issues in existing source documentation (not altered here):**

1. [`reports/final_report.md`](reports/final_report.md) presents conflicting correlation values compared with the actual cleaned dataset and [`reports/tables/eda_findings.md`](reports/tables/eda_findings.md). For example, some narrative passages give much smaller midterm/final correlations. Use **0.894** based on the cleaned CSV.
2. The summary markdown cell at the end of [`notebooks/02_eda.ipynb`](notebooks/02_eda.ipynb) states **0.864** for midterm/final and **0.771** for assignment/final; these do not match the checked values (**0.894** and **0.854**). Its executed statistics table is consistent with the CSV, but that prose cell should eventually be corrected.
3. [`reports/tables/eda_findings.md`](reports/tables/eda_findings.md) lists an inter-feature correlation of **0.803** and then incorrectly states all inter-feature correlations are **below 0.80**. In fact, **0.803 > 0.80**.
4. The introduction in [`notebooks/01_data_cleaning.ipynb`](notebooks/01_data_cleaning.ipynb) uses V1–V7 labels for different checks than those implemented in [`src/cleaning.py`](src/cleaning.py); the implementation and generated log are the better references for the performed operations.
5. The source document uses a "collection date" and privacy/compliance wording for the benchmark dataset. The executable generator establishes that the data are **synthetic**, but independent institutional collection or legal-compliance claims were not verified and should not be presented as facts.

**Limitations of this verification:** Existing notebook outputs and saved files were inspected and numeric calculations were checked against both CSVs. The environment available for this review did **not** run the project's Python scripts or re-render the images; do not represent that as a fresh successful execution. No existing data, model, notebook, source code, or report was changed as part of preparing this guide.

**Teacher-ready conclusion:** The current repository contains evidence for all **eight** Part A requirements. Before the presentation, use this guide's checked values and avoid quoting contradictory numbers from the older summary paragraphs.
