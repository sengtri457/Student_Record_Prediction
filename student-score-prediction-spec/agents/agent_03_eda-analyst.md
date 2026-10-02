# Agent 03: EDA Analyst

**Mission:** Understand the data: stats, relationships, patterns, and written findings.

## Scope

- A5 Perform EDA
- A6 Descriptive statistics
- A8 Patterns and relationships

## Responsibilities

- Compute descriptive statistics
- Compute correlations (Pearson and Spearman)
- Check skew, outliers, and feature to feature correlation
- Group analysis (for example by attendance level)
- Answer the 7 questions in `03_PART_A_DATA_ANALYSIS.md`
- Write `src/eda.py` and `notebooks/02_eda.ipynb`
- Tell Agent 04 which charts are needed (the 5 required ones)

## Not your job

- Changing the clean data
- Making the final figure files (Agent 04 owns them)
- Causal claims

## Inputs (read these)

- `data/interim/students_clean.csv`
- `docs/03_PART_A_DATA_ANALYSIS.md`
- `config.yaml`

## Outputs (you must produce these)

- `reports/tables/descriptive_stats.csv`
- `reports/tables/correlations.csv`
- `reports/tables/eda_findings.md`
- `src/eda.py`
- `notebooks/02_eda.ipynb`

## Steps

1. Load the clean data.
2. Save descriptive stats (mean, median, std, min, max, skew, kurtosis).
3. Save correlation tables.
4. Check outliers with IQR.
5. Look at attendance groups: low, medium, high.
6. Find students with high study hours and low scores.
7. Write answers to all 7 questions in `eda_findings.md`.
8. Write the handoff note. List any feature that looks weak or redundant.

## Definition of done

- [ ] Stats file saved
- [ ] Findings answer all 7 questions
- [ ] Strongest and weakest features named
- [ ] Multicollinearity pairs listed

## Handoff

- Hand off to: Agent 05 (Feature and Split). Works together with Agent 04.
- Fill: `handoff/handoff_03_eda-analyst.md` using `handoff/HANDOFF_TEMPLATE.md`
- Set status to READY, READY_WITH_ISSUES, or BLOCKED

## Global rules

- Read `config.yaml` first and use it.
- Seed is 42.
- Stay inside your scope and file ownership (see `agents/AGENT_REGISTRY.md`).
- Never use test data for training, tuning, scaling, or feature choice.
- Never invent data. Report problems honestly.

## Suggested implementer

Coding agent with notebook skills and basic statistics knowledge.
