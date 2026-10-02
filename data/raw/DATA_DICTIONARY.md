# Data Dictionary

Canonical schema specifications for student score prediction.

| Column Name | Data Type | Units / Scale | Permitted Range | Missing in Raw | Description |
|---|---|---|---|---|---|
| `student_id` | String | Identifier | Unique string (`STU_XXXX`) | No | Unique anonymous student identifier. |
| `attendance_pct` | Float | Percentage (%) | 0.0 to 100.0 | Yes (allowed) | Percentage of scheduled classroom lectures attended throughout the term. |
| `study_hours_week` | Float | Hours / Week | 0.0 to 80.0 | Yes (allowed) | Self-reported dedicated weekly academic study hours outside classroom sessions. |
| `assignment_avg` | Float | Points (0-100) | 0.0 to 100.0 | Yes (allowed) | Weighted cumulative arithmetic mean of all homework assignments and quizzes. |
| `midterm_score` | Float | Points (0-100) | 0.0 to 100.0 | Yes (allowed) | Standardized midterm examination score. |
| `previous_gpa` | Float | Grade Points | 0.0 to 4.0 | Yes (allowed) | Cumulative Grade Point Average from previous academic periods on a 4.0 scale. |
| `final_score` | Float | Points (0-100) | 0.0 to 100.0 | **No (Drop if missing)** | **Target Variable**: Final course examination score. |

## Notes on Range & Boundary Conditions
- Any values for `attendance_pct`, `assignment_avg`, `midterm_score`, or `final_score` below 0 or above 100 represent measurement or recording anomalies and must be flagged by validation rule V3.
- `study_hours_week` capped at 80 hours/week to represent physical physiological limits.
- `previous_gpa` strictly defined on a 4.0 standard academic scale.
