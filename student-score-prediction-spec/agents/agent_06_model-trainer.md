# Agent 06: Model Trainer

**Mission:** Train at least two models on the training set and save them.

## Scope

- B5 Train at least 2 models

## Responsibilities

- Train Multiple Linear Regression
- Train Random Forest Regressor
- Optional: Ridge, Gradient Boosting
- Optional small `GridSearchCV` on train only
- Save models
- Write `src/train.py` and a training log

## Not your job

- Evaluating on test data
- Selecting the final model (Agent 07 does that)
- Changing the split or features

## Inputs (read these)

- `data/processed/train.csv`
- `models/scaler.joblib`
- `docs/04_PART_B_MACHINE_LEARNING.md`
- `config.yaml`

## Outputs (you must produce these)

- `models/linear_regression.joblib`
- `models/random_forest.joblib`
- `reports/tables/training_log.md` (settings, seeds, tuning results)
- `src/train.py`

## Steps

1. Load train data and scaler.
2. Build pipelines. Linear model gets scaling.
3. Fit with seed 42.
4. If tuning, use CV on train only and log the grid.
5. Save models.
6. Write the training log.
7. Write the handoff note.

## Definition of done

- [ ] At least 2 models saved
- [ ] Training log has all parameters
- [ ] Test data was never loaded

## Handoff

- Hand off to: Agent 07 (Evaluator)
- Fill: `handoff/handoff_06_model-trainer.md` using `handoff/HANDOFF_TEMPLATE.md`
- Set status to READY, READY_WITH_ISSUES, or BLOCKED

## Global rules

- Read `config.yaml` first and use it.
- Seed is 42.
- Stay inside your scope and file ownership (see `agents/AGENT_REGISTRY.md`).
- Never use test data for training, tuning, scaling, or feature choice.
- Never invent data. Report problems honestly.

## Suggested implementer

Coding agent. scikit-learn.
