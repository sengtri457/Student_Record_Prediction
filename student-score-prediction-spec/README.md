# Student Score Prediction: Spec Pack

Topic 01. Predict a student's final score from attendance, study hours, assignment scores, midterm score, and previous GPA.

Models: Multiple Linear Regression (baseline) and Random Forest Regressor (comparison).

## How to use this pack

1. Read `docs/00_PROJECT_OVERVIEW.md` first.
2. Read `agents/AGENT_REGISTRY.md` to see which agent owns which scope.
3. Run agents in the order set in `agents/HANDOFF_PROTOCOL.md`.
4. Every agent fills a handoff note from `handoff/HANDOFF_TEMPLATE.md` when done.
5. The QA agent signs off using `docs/05_ACCEPTANCE_CRITERIA.md`.

## File map

| Path | What it is |
|---|---|
| `docs/00_PROJECT_OVERVIEW.md` | Problem, goal, scope, assumptions |
| `docs/01_ARCHITECTURE.md` | End to end pipeline, folders, environment, config |
| `docs/02_DATA_SPEC.md` | Data sources, schema, data dictionary, validation rules |
| `docs/03_PART_A_DATA_ANALYSIS.md` | Spec for the 8 required analysis steps |
| `docs/04_PART_B_MACHINE_LEARNING.md` | Spec for the 10 required ML steps |
| `docs/05_ACCEPTANCE_CRITERIA.md` | Pass/fail checklist for the whole project |
| `docs/06_REPORT_SPEC.md` | Final report structure |
| `docs/07_RISKS_AND_DECISIONS.md` | Known risks and decisions already made |
| `agents/AGENT_REGISTRY.md` | All agents, scope matrix, ownership |
| `agents/HANDOFF_PROTOCOL.md` | Order of work, contracts between agents |
| `agents/agent_XX_*.md` | One brief per agent |
| `handoff/HANDOFF_TEMPLATE.md` | Template each agent fills when finished |
| `handoff/STATUS_LOG.md` | Running progress table |

## Solo use

If you are one person (or one coding agent), you can merge agents. See "Merging agents" in `agents/AGENT_REGISTRY.md`.
