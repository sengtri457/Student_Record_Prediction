# Agent Registry

Ten agents. Each one owns a clear scope. An agent must not touch files outside its scope.

## Agent list

| ID | Agent | Scope in one line | Best implemented by |
|---|---|---|---|
| 00 | Orchestrator | Runs the order, checks handoffs, resolves conflicts | Planning/general agent with file access |
| 01 | Data Collector | Gets the dataset, documents source and dictionary | Research agent with web access, or a human for surveys |
| 02 | Data Cleaner | Cleaning, validation, missing values | Coding agent (Python, pandas) |
| 03 | EDA Analyst | Descriptive stats, patterns, findings | Coding agent with notebook skills |
| 04 | Visualization | The 5+ figures | Coding agent (matplotlib, seaborn) |
| 05 | Feature and Split | Feature choice, scaling, 80/20 split | Coding agent (scikit-learn) |
| 06 | Model Trainer | Train 2+ models, save them | Coding agent (scikit-learn) |
| 07 | Evaluator | Metrics, CV, comparison, selection, predictions, explanation | Coding agent with stats sense |
| 08 | Report Writer | Final report from saved tables and figures | Writing agent |
| 09 | QA Reviewer | Checks everything against acceptance criteria | Review agent, separate from the builders |

## Scope matrix

R = does the work. A = approves. C = consulted. I = informed.

| Scope item | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 Define problem | A | R | | | | | | | C | I |
| A2 Collect dataset | A | R | | | | | | | | I |
| A3 Clean data | A | C | R | | | | | | | I |
| A4 Missing values | A | | R | C | | | | | | I |
| A5 EDA | A | | | R | C | | | | | I |
| A6 Descriptive stats | A | | | R | | | | | | I |
| A7 Visualizations (5+) | A | | | C | R | | | | | I |
| A8 Patterns | A | | | R | | | | | C | I |
| B1 Target | A | | | | | R | | | | I |
| B2 Features | A | | | C | | R | | | | I |
| B3 Preprocess | A | | | | | R | | | | I |
| B4 Split 80/20 | A | | | | | R | | | | I |
| B5 Train 2+ models | A | | | | | | R | | | I |
| B6 Compare | A | | | | | | | R | | I |
| B7 Evaluate | A | | | | | | C | R | | I |
| B8 Select model | A | | | | | | | R | | I |
| B9 Predictions | A | | | | | | | R | | I |
| B10 Explain | A | | | | C | | | R | C | I |
| Report | A | | | C | C | | | C | R | I |
| Final sign off | A | | | | | | | | | R |

## File ownership (who may write what)

| Path | Owner |
|---|---|
| `data/raw/*` | 01 |
| `data/interim/*`, `reports/tables/missing_values.csv`, `cleaning_log.csv`, `src/cleaning.py` | 02 |
| `reports/tables/descriptive_stats.csv`, `eda_findings.md`, `notebooks/02_eda.ipynb`, `src/eda.py` | 03 |
| `reports/figures/fig01` to `fig08` | 04 |
| `data/processed/*`, `src/features.py`, `models/scaler.joblib`, `config.yaml` (features, target) | 05 |
| `src/train.py`, `models/linear_regression.joblib`, `random_forest.joblib` | 06 |
| `src/evaluate.py`, `src/predict.py`, `models/best_model.joblib`, metrics and prediction tables, `fig09`, `fig10` | 07 |
| `reports/final_report.md` | 08 |
| `handoff/STATUS_LOG.md` (sign off row), QA notes | 09 |
| `handoff/*` (own handoff note) | each agent |

If an agent needs a change in someone else's file, it writes a request in its handoff note. It does not edit the file.

## Capability needs

| Capability | Agents |
|---|---|
| Run Python and read/write files | 02, 03, 04, 05, 06, 07, 09 |
| Web access for finding datasets | 01 |
| Jupyter notebook writing | 03, 05, 06, 07 |
| Long form writing | 08 |
| Independent review (should not be the same instance that built things) | 09 |

## Merging agents (solo or small setups)

| If you only have | Merge like this |
|---|---|
| 1 coding agent + 1 human | Human does 01 and 08. Agent does 02 to 07 and 09 (separate session for 09). |
| 2 agents | Agent A: 01, 02, 03, 04. Agent B: 05, 06, 07, 08. Human does 09. |
| 3 agents | Data (01 to 04), ML (05 to 07), Docs and QA (08 to 09, but do 09 in a fresh session) |

Rule when merging: keep the handoff notes. One note per original agent, even if one worker did several.
