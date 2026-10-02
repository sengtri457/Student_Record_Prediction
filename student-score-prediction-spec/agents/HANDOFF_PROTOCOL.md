# Handoff Protocol

## Order of work

```
00 Orchestrator (start)
  -> 01 Data Collector
  -> 02 Data Cleaner
  -> 03 EDA Analyst  <-> 04 Visualization (run together)
  -> 05 Feature and Split
  -> 06 Model Trainer
  -> 07 Evaluator
  -> 08 Report Writer
  -> 09 QA Reviewer
  -> 00 Orchestrator (close)
```

Gate rule: the next agent starts only when the previous agent's handoff note says `STATUS: READY`.

## Contracts between agents

| From | To | Contract (files that must exist) |
|---|---|---|
| 01 | 02 | `data/raw/students_raw.csv`, `DATA_SOURCE.md`, `DATA_DICTIONARY.md` |
| 02 | 03, 04 | `data/interim/students_clean.csv`, `cleaning_log.csv`, `missing_values.csv`. Zero missing in features. |
| 03, 04 | 05 | `descriptive_stats.csv`, `eda_findings.md`, `fig01` to `fig05` minimum |
| 05 | 06 | `data/processed/train.csv`, `test.csv`, `models/scaler.joblib`, updated `config.yaml` |
| 06 | 07 | `models/linear_regression.joblib`, `random_forest.joblib`, training log |
| 07 | 08 | `metrics.csv`, `model_comparison.csv`, `cv_results.csv`, `selection_decision.md`, `test_predictions.csv`, `custom_predictions.csv`, `feature_importance.csv`, `coefficients.csv`, `fig09`, `fig10` |
| 08 | 09 | `reports/final_report.md` |
| 09 | 00 | QA result in `STATUS_LOG.md` (PASS or FAIL with list) |

## Handoff note rules

1. Each agent writes `handoff/handoff_<agent_id>_<name>.md` using `HANDOFF_TEMPLATE.md`.
2. Note must list: files produced, decisions made, problems found, open questions, and what the next agent needs to know.
3. Status is one of: `READY`, `READY_WITH_ISSUES`, `BLOCKED`.
4. `BLOCKED` goes back to the Orchestrator, who decides.
5. Do not hide problems. Missing features, weak data, odd results all go in the note.

## Failure handling

| Situation | Action |
|---|---|
| Previous contract file missing | Stop. Mark `BLOCKED`. Tell Orchestrator. |
| Agent finds a bug in upstream output | Write it in handoff. Do not fix upstream files. Orchestrator reassigns. |
| QA fails | Orchestrator sends the fail list to the owning agents. They fix and re-handoff. QA runs again. |
| Red flag from `05_ACCEPTANCE_CRITERIA.md` | Evaluator (07) stops and notifies Orchestrator before the report is written. |

## Rules for every agent

- Read `config.yaml` first. Use it. Do not hardcode.
- Seed is always 42.
- Never edit `data/raw/`.
- Never use test data to train, tune, scale, or select features.
- Never invent data. Never type numbers into the report by hand.
- Keep code simple and commented in plain language.
- Write short, direct sentences in docs.
