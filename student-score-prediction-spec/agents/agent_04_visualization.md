# Agent 04: Visualization

**Mission:** Create the figures for Part A. At least 5.

## Scope

- A7 Visualizations (5+)

## Responsibilities

- Make fig01 to fig05 (required) and fig06 to fig08 (extra)
- Follow chart rules: title, axis labels with units, 150 dpi or higher
- Save PNGs to `reports/figures/`
- Write a one line caption for each figure in `reports/figures/CAPTIONS.md`

## Not your job

- Changing data
- Drawing conclusions that EDA Analyst did not support
- Model figures fig09 and fig10 (Agent 07 owns them)

## Inputs (read these)

- `data/interim/students_clean.csv`
- `docs/03_PART_A_DATA_ANALYSIS.md` (chart table)
- `reports/tables/eda_findings.md`
- `config.yaml`

## Outputs (you must produce these)

- `reports/figures/fig01_final_score_hist.png`
- `fig02_corr_heatmap.png`
- `fig03_midterm_vs_final.png`
- `fig04_attendance_box.png`
- `fig05_study_vs_final.png`
- Optional `fig06` to `fig08`
- `reports/figures/CAPTIONS.md`

## Steps

1. Read the chart table in the Part A spec.
2. Build each chart from the clean data with code.
3. Check labels, units, readability.
4. Save at 150 dpi or higher.
5. Write captions.
6. Write the handoff note with the list of files.

## Definition of done

- [ ] At least 5 figures exist with the exact file names
- [ ] Each has title and labeled axes
- [ ] Captions file exists

## Handoff

- Hand off to: Agent 05. Notify Agent 08 that figures are available.
- Fill: `handoff/handoff_04_visualization.md` using `handoff/HANDOFF_TEMPLATE.md`
- Set status to READY, READY_WITH_ISSUES, or BLOCKED

## Global rules

- Read `config.yaml` first and use it.
- Seed is 42.
- Stay inside your scope and file ownership (see `agents/AGENT_REGISTRY.md`).
- Never use test data for training, tuning, scaling, or feature choice.
- Never invent data. Report problems honestly.

## Suggested implementer

Coding agent. matplotlib and seaborn.
