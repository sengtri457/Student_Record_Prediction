# Agent 01: Data Collector

**Mission:** Get the dataset, store it untouched, and document where it came from.

## Scope

- A1 Define problem (writes it)
- A2 Collect dataset

## Responsibilities

- Pick a data source from `02_DATA_SPEC.md` options
- Download or build the survey and collect responses
- Map columns to canonical names where possible (note only, do not clean)
- Write the problem paragraph
- Save the raw file and docs

## Not your job

- Cleaning or imputing
- Editing values
- Inventing or generating data for final use
- Collecting names or contact details

## Inputs (read these)

- `docs/00_PROJECT_OVERVIEW.md`
- `docs/02_DATA_SPEC.md`

## Outputs (you must produce these)

- `data/raw/students_raw.csv`
- `data/raw/DATA_SOURCE.md` (source, link, license or consent, date, row count)
- `data/raw/DATA_DICTIONARY.md`
- Problem paragraph in handoff note

## Steps

1. Choose the source. Prefer real data.
2. Check license or consent.
3. Save the file as-is in `data/raw/`.
4. Count rows and columns.
5. List which canonical columns exist, which are missing, and which are proxies.
6. Write the data dictionary.
7. Write the handoff note. Be clear about missing features.

## Definition of done

- [ ] Raw file saved and not modified
- [ ] Source and license/consent documented
- [ ] Dictionary complete
- [ ] At least 100 rows, or the shortage is flagged
- [ ] Missing canonical features flagged

## Handoff

- Hand off to: Agent 02 (Data Cleaner)
- Fill: `handoff/handoff_01_data-collector.md` using `handoff/HANDOFF_TEMPLATE.md`
- Set status to READY, READY_WITH_ISSUES, or BLOCKED

## Global rules

- Read `config.yaml` first and use it.
- Seed is 42.
- Stay inside your scope and file ownership (see `agents/AGENT_REGISTRY.md`).
- Never use test data for training, tuning, scaling, or feature choice.
- Never invent data. Report problems honestly.

## Suggested implementer

Research agent with web access. A human is needed if running a survey.
