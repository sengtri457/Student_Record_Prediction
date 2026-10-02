# Agent 05: Feature and Split

**Mission:** Lock the target and features, preprocess, and make the 80/20 split.

## Scope

- B1 Define target
- B2 Select features
- B3 Preprocess
- B4 Split 80/20

## Responsibilities

- Confirm target is `final_score`
- Check multicollinearity with VIF
- Decide final feature list with reasons
- Build the scaling step (StandardScaler, fit on train only)
- Split 80/20, seed 42
- Save train and test files and the scaler
- Write `src/features.py`

## Not your job

- Training models
- Looking at test data to pick features
- Changing the clean data

## Inputs (read these)

- `data/interim/students_clean.csv`
- `reports/tables/eda_findings.md`
- `docs/04_PART_B_MACHINE_LEARNING.md`
- `config.yaml`

## Outputs (you must produce these)

- `data/processed/train.csv`
- `data/processed/test.csv`
- `models/scaler.joblib`
- `reports/tables/feature_selection_notes.md`
- `reports/tables/vif.csv`
- `src/features.py`
- `config.yaml` (target and features section)

## Steps

1. Load clean data.
2. Compute VIF. Flag above 5.
3. Write feature selection notes using EDA evidence.
4. Split first with `test_size=0.20, random_state=42`.
5. Fit the scaler on train only. Save it.
6. Save train and test CSVs.
7. Check shapes: test is about 20% of rows.
8. Write the handoff note.

## Definition of done

- [ ] Train is 80% and test is 20% (rounding allowed)
- [ ] Scaler fitted on train only
- [ ] No target column in features
- [ ] Feature notes written

## Handoff

- Hand off to: Agent 06 (Model Trainer)
- Fill: `handoff/handoff_05_feature-and-split.md` using `handoff/HANDOFF_TEMPLATE.md`
- Set status to READY, READY_WITH_ISSUES, or BLOCKED

## Global rules

- Read `config.yaml` first and use it.
- Seed is 42.
- Stay inside your scope and file ownership (see `agents/AGENT_REGISTRY.md`).
- Never use test data for training, tuning, scaling, or feature choice.
- Never invent data. Report problems honestly.

## Suggested implementer

Coding agent. scikit-learn and statsmodels (for VIF) or numpy.
