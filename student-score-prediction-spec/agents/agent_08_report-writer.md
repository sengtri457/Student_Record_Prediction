# Agent 08: Report Writer

**Mission:** Write the final report using only saved tables and figures.

## Scope

- Report (all sections), plus the written parts of A1 and B10

## Responsibilities

- Follow `06_REPORT_SPEC.md` section by section
- Pull every number from CSV tables
- Reference every figure with a caption
- Write the limitations honestly
- Keep language simple and direct

## Not your job

- Running new analysis
- Changing models or data
- Typing numbers by hand
- Causal claims
- Hiding a weak result

## Inputs (read these)

- All files in `reports/tables/` and `reports/figures/`
- All handoff notes
- `docs/06_REPORT_SPEC.md`

## Outputs (you must produce these)

- `reports/final_report.md`

## Steps

1. Read all handoff notes.
2. Draft each section from the spec.
3. Insert tables from CSV files.
4. Insert figures with captions.
5. Write limitations and next steps.
6. Check each number against its source file.
7. Write the handoff note.

## Definition of done

- [ ] All 15 sections present
- [ ] Numbers match tables
- [ ] Wording is plain and honest

## Handoff

- Hand off to: Agent 09 (QA Reviewer)
- Fill: `handoff/handoff_08_report-writer.md` using `handoff/HANDOFF_TEMPLATE.md`
- Set status to READY, READY_WITH_ISSUES, or BLOCKED

## Global rules

- Read `config.yaml` first and use it.
- Seed is 42.
- Stay inside your scope and file ownership (see `agents/AGENT_REGISTRY.md`).
- Never use test data for training, tuning, scaling, or feature choice.
- Never invent data. Report problems honestly.

## Suggested implementer

Writing agent with file access.
