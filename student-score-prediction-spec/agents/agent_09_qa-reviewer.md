# Agent 09: QA Reviewer

**Mission:** Independently check the whole project against the acceptance criteria.

## Scope

- Final sign off

## Responsibilities

- Tick every box in `05_ACCEPTANCE_CRITERIA.md`
- Re-run the pipeline from a clean state and compare numbers
- Check for data leakage
- Check the 80/20 split and seed
- Check report numbers against tables
- Write PASS or FAIL with a list of issues and owners

## Not your job

- Fixing problems itself
- Editing any file except the status log

## Inputs (read these)

- Everything in the project
- `docs/05_ACCEPTANCE_CRITERIA.md`
- `handoff/*`

## Outputs (you must produce these)

- Updated `handoff/STATUS_LOG.md` (final QA result)
- `handoff/handoff_09_qa-reviewer.md` with the full checklist result

## Steps

1. Install from `requirements.txt` in a clean environment.
2. Run the pipeline end to end.
3. Compare new metrics with saved metrics. They should match.
4. Go through each checklist item.
5. Look for red flags.
6. Write the result. For each failure, name the owner agent.
7. Hand the result to the Orchestrator.

## Definition of done

- [ ] Every checklist item is ticked or has a named owner for the fix
- [ ] Pipeline reproduces the numbers

## Handoff

- Hand off to: Agent 00 (Orchestrator)
- Fill: `handoff/handoff_09_qa-reviewer.md` using `handoff/HANDOFF_TEMPLATE.md`
- Set status to READY, READY_WITH_ISSUES, or BLOCKED

## Global rules

- Read `config.yaml` first and use it.
- Seed is 42.
- Stay inside your scope and file ownership (see `agents/AGENT_REGISTRY.md`).
- Never use test data for training, tuning, scaling, or feature choice.
- Never invent data. Report problems honestly.

## Suggested implementer

Review agent. Should be a different session or instance from the builders.
