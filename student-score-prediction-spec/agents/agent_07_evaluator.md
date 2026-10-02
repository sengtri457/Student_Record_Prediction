# Agent 07: Evaluator

**Mission:** Measure, compare, select, predict, and explain.

## Scope

- B6 Compare models
- B7 Evaluate performance
- B8 Select model
- B9 Make predictions
- B10 Explain results

## Responsibilities

- Compute MAE, RMSE, R2 on train and test
- Run 5-fold CV on train
- Check overfitting and residuals
- Build the comparison table
- Pick the model using the rule in `04_PART_B_MACHINE_LEARNING.md` and write the decision
- Save `best_model.joblib`
- Predict on test and on 3+ custom students
- Extract coefficients and feature importance
- Make fig09 and fig10, plus a feature importance chart
- Write `src/evaluate.py` and `src/predict.py`
- Run red flag checks from the acceptance doc

## Not your job

- Retraining after seeing test results to improve the score
- Editing train/test files
- Writing the final report

## Inputs (read these)

- `data/processed/test.csv`
- `models/*.joblib`
- `docs/04_PART_B_MACHINE_LEARNING.md`
- `docs/05_ACCEPTANCE_CRITERIA.md`
- `config.yaml`

## Outputs (you must produce these)

- `reports/tables/metrics.csv`
- `model_comparison.csv`
- `cv_results.csv`
- `selection_decision.md`
- `test_predictions.csv`
- `custom_predictions.csv`
- `coefficients.csv`
- `feature_importance.csv`
- `reports/figures/fig09_residuals.png`
- `fig10_actual_vs_pred.png`
- `fig11_feature_importance.png`
- `models/best_model.joblib`
- `src/evaluate.py`, `src/predict.py`

## Steps

1. Load models and data.
2. Score train and test. Save metrics.
3. Run CV. Save results.
4. Compare, apply selection rule, write decision.
5. Run red flag checks. If one triggers, stop and tell the Orchestrator.
6. Make predictions and charts.
7. Write a plain language explanation in the handoff note.
8. Write the handoff note with the key numbers.

## Definition of done

- [ ] All metrics on train and test exist
- [ ] Comparison table exists
- [ ] Decision file explains the choice
- [ ] 3 or more custom predictions exist
- [ ] Coefficients and importances saved
- [ ] No red flags left unexplained

## Handoff

- Hand off to: Agent 08 (Report Writer)
- Fill: `handoff/handoff_07_evaluator.md` using `handoff/HANDOFF_TEMPLATE.md`
- Set status to READY, READY_WITH_ISSUES, or BLOCKED

## Global rules

- Read `config.yaml` first and use it.
- Seed is 42.
- Stay inside your scope and file ownership (see `agents/AGENT_REGISTRY.md`).
- Never use test data for training, tuning, scaling, or feature choice.
- Never invent data. Report problems honestly.

## Suggested implementer

Coding agent with solid stats judgment.
