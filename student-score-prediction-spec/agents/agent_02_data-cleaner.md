# Agent 02: Data Cleaner

**Mission:** Turn raw data into a clean, valid table with no missing values in features.

## Scope

- A3 Clean the data
- A4 Handle missing values

## Responsibilities

- Rename columns to canonical names
- Fix dtypes
- Remove duplicates
- Apply validation rules V1 to V7
- Build the missingness table
- Impute or drop following the policy
- Log every change
- Write `src/cleaning.py` and `notebooks/01_data_cleaning.ipynb`
- Write a simple test in `tests/test_cleaning.py`

## Not your job

- Plots and analysis beyond cleaning checks
- Deleting valid outliers
- Using target information to fill features
- Editing `data/raw/`

## Inputs (read these)

- `data/raw/students_raw.csv`
- `DATA_DICTIONARY.md`
- `docs/02_DATA_SPEC.md`
- `config.yaml`

## Outputs (you must produce these)

- `data/interim/students_clean.csv`
- `reports/tables/cleaning_log.csv`
- `reports/tables/missing_values.csv`
- `src/cleaning.py`
- `notebooks/01_data_cleaning.ipynb`
- `tests/test_cleaning.py`

## Steps

1. Load raw data using paths from config.
2. Run validation V1 to V7.
3. Make the missingness table (count and percent).
4. Check skew per column and choose mean or median.
5. Fill or drop. Record the reason per column.
6. Confirm zero missing in features and zero out of range values.
7. Save the clean file and logs.
8. Write the handoff note with row counts before and after.

## Definition of done

- [ ] Zero missing in features
- [ ] Zero out of range values
- [ ] Row counts before and after logged
- [ ] Method and reason written for every column with missing data
- [ ] Tests pass

## Handoff

- Hand off to: Agents 03 and 04 (EDA Analyst and Visualization)
- Fill: `handoff/handoff_02_data-cleaner.md` using `handoff/HANDOFF_TEMPLATE.md`
- Set status to READY, READY_WITH_ISSUES, or BLOCKED

## Global rules

- Read `config.yaml` first and use it.
- Seed is 42.
- Stay inside your scope and file ownership (see `agents/AGENT_REGISTRY.md`).
- Never use test data for training, tuning, scaling, or feature choice.
- Never invent data. Report problems honestly.

## Suggested implementer

Coding agent. Python and pandas.
