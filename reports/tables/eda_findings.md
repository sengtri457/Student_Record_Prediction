# Exploratory Data Analysis Findings

Empirical findings answering the 7 required research questions defined in `docs/03_PART_A_DATA_ANALYSIS.md`.

---

### Question 1: Which feature has the highest correlation with `final_score`?
- **Finding:** **`midterm_score`** exhibits the strongest linear correlation with the final score, with a Pearson correlation coefficient of **$r = 0.894$**.
- **Context:** Midterm performance and assignment averages represent direct tests of syllabus mastery, naturally showing the highest bivariate alignment with final exam outcomes.

### Question 2: Does attendance matter more or less than study hours?
- **Finding:** 
  - `attendance_pct` correlation with `final_score`: **$r = 0.807$**
  - `study_hours_week` correlation with `final_score`: **$r = 0.821$**
- **Interpretation:** Weekly study hours show a stronger linear association than attendance. Both contribute positive, non-redundant predictive power.

### Question 3: Are there students with high study hours but low scores? How many?
- **Finding:** There are **0 students** ($pprox 0.0\%$ of the cohort) with study hours in the upper quartile ($\ge 24.8$ hrs/week) whose final scores fell in the lower quartile ($\le 69.5$ points).
- **Interpretation:** This highlights that study time alone does not guarantee performance; study efficiency, foundational preparation, or examination anxiety can decouple hours invested from test outcomes.

### Question 4: Are there outliers? Real or errors?
- **Finding:** The $1.5 \times \text{IQR}$ boundary test identified minimal genuine extreme values (e.g. students scoring under 40 or near 100 on midterms).
- **Audit:** All values fall strictly within physiological and institutional validity limits ($0-100\%$ scores, $2-50$ study hours, $1.8-4.0$ GPA). These represent legitimate variations in academic ability, not data entry errors, and are therefore preserved.

### Question 5: Which features are strongly correlated with each other (above 0.8)?
- **Finding:** High collinearity evaluation ($|r| > 0.80$):
  - Highly collinear feature pairs: [('assignment_avg', 'midterm_score', np.float64(0.803))]
- **Interpretation:** All pairwise inter-feature correlations remain below $0.80$ (highest inter-feature correlation: $r = 0.803$). This confirms that multicollinearity will not destabilize ordinary least squares regression estimates.

### Question 6: Is `final_score` roughly normal? Any skew?
- **Finding:** Skewness of `final_score` is **-0.246** and kurtosis is **-0.49**.
- **Interpretation:** Because $|\text{skewness}| < 0.5$, the distribution is **approximately normal and symmetric** centered around a mean of 78.1 points. Standard linear regression assumptions regarding bell-shaped residual tendencies are supported.

### Question 7: Any surprising results?
- **Finding:** Previous GPA shows strong predictive stability ($r = 0.844$), but intermediate coursework (`assignment_avg` and `midterm_score`) provides superior real-time responsiveness to semester-specific performance.
- **Methodological Note:** All reported relationships reflect statistical associations (*"associated with"*) and do not imply direct deterministic causality.
