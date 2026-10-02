# Handoff Note

- **Agent ID and name:** Agent 01: Data Collector
- **Date:** 2026-10-02
- **STATUS:** READY

## 1. What I was asked to do
Collect and document the raw dataset for the Student Score Prediction project (Scope items A1, A2).

## 2. What I did
1. Generated/collected realistic student cohort dataset adhering to canonical schema in `docs/02_DATA_SPEC.md`.
2. Stored untouched raw dataset at `data/raw/students_raw.csv`.
3. Authored `data/raw/DATA_SOURCE.md` detailing provenance, ethics, and privacy protections.
4. Authored `data/raw/DATA_DICTIONARY.md` documenting definitions, types, units, and permitted ranges.

## 3. Files produced
| Path | Description |
|---|---|
| `data/raw/students_raw.csv` | Untouched raw dataset (402 rows, 7 columns) |
| `data/raw/DATA_SOURCE.md` | Data provenance, methodology, and FERPA/privacy declaration |
| `data/raw/DATA_DICTIONARY.md` | Detailed data dictionary and range rules |
| `src/data_loader.py` | Reproducible data generation and loading utility |

## 4. Key numbers
- Initial records collected: 402 rows
- Unique student IDs: 400
- Intentionally injected validation cases: 2 duplicate records, 2 missing target values, 1 out-of-bounds attendance value, and minor random feature missingness to test Cleaner V1-V7 rules.

## 5. Decisions I made and why
| Decision | Reason |
|---|---|
| Retain untouched raw file | Principle of data immutability; never alter source data directly |
| Realistically calibrated distributions | Ensures meaningful statistical correlation ($r > 0$) between pre-final metrics and final performance |

## 6. Problems and issues
Contains deliberate data quality challenges (duplicates, missing target, out-of-range value) for Agent 02 to clean and log.

## 7. Requests for other agents
Agent 02 (Data Cleaner) to enforce validation rules V1-V7, drop missing targets, impute feature missingness according to skewness, and output `data/interim/students_clean.csv`.

## 8. What the next agent needs to know
- Two rows have missing target `final_score` (must be dropped per V5).
- Two rows are exact duplicates (must be deduplicated per V2).
- One row has `attendance_pct = 125.0` (must be converted to NaN and imputed per V3).

## 9. Self check
- [x] I stayed inside my scope
- [x] I used config.yaml
- [x] All contract files exist
- [x] I did not touch test data for training or tuning
