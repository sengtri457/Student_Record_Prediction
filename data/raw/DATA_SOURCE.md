# Data Source Documentation

## Dataset Provenance
- **Dataset File:** `data/raw/students_raw.csv`
- **Option Selected:** Academic Performance Benchmark (Option D: Hybrid / Realistically Calibrated Academic Cohort)
- **Collection Date:** 2026-10-02
- **Sample Size:** 402 records (including test cases for data cleaning validation)
- **Target Variable:** `final_score` (Continuous scale: 0 to 100)

## Privacy and Ethics Statement
- **Anonymization:** No Personally Identifiable Information (PII) such as student names, phone numbers, email addresses, or residential information is collected or stored.
- **Identifier:** Each student record is indexed using a synthetic alphanumeric identifier format (`STU_0001` to `STU_0400`).
- **Consent and Compliance:** Complies with FERPA guidelines for anonymous academic analytics modeling. Data represents anonymous performance tracking solely intended for early educational intervention.

## Scope of Observed Features
All observations capture behaviors and performance indicators measured *strictly prior* to the administration of the final examination:
- Historical academic ability (`previous_gpa`)
- Engagement and presence (`attendance_pct`)
- Extracurricular preparation effort (`study_hours_week`)
- Continuous coursework mastery (`assignment_avg`)
- Mid-term examination benchmark (`midterm_score`)
