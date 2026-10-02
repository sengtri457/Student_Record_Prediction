# Feature Selection & Multicollinearity Assessment

- **Target Variable (B1):** `final_score` (continuous final examination score, 0-100 scale). Strictly excluded from input feature matrix.
- **Candidate Features (B2):** ['attendance_pct', 'study_hours_week', 'assignment_avg', 'midterm_score', 'previous_gpa']

### Variance Inflation Factor (VIF) Diagnostic:
| Feature | VIF Score | Multicollinearity Risk |
|---|---|---|
| `attendance_pct` | 2.939 | Low |
| `study_hours_week` | 2.828 | Low |
| `assignment_avg` | 3.907 | Low |
| `midterm_score` | 4.27 | Low |
| `previous_gpa` | 3.281 | Low |

### Assessment Conclusion:
All VIF values are well below the conservative threshold of 5.0 (and well below 10.0). No feature pairs exhibit pathological collinearity. All 5 features represent unique, domain-grounded facets of academic engagement and historical preparation:
1. `attendance_pct`: Measures classroom engagement and exposure to lecture instruction.
2. `study_hours_week`: Captures independent out-of-classroom effort.
3. `assignment_avg`: Assesses continuous coursework and formative homework mastery.
4. `midterm_score`: Benchmarks standardized mid-semester performance.
5. `previous_gpa`: Provides historical cumulative academic baseline.

All 5 features are retained in the modeling matrix without dimensionality reduction.
