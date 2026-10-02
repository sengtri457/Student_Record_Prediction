# 05. Acceptance Criteria

The QA agent (09) ticks every box. Any unchecked box means not done.

## Part A

- [ ] Problem defined in one clear paragraph
- [ ] Raw dataset saved, source and license/consent documented
- [ ] Data dictionary exists
- [ ] Cleaning log exists with before/after row counts
- [ ] Missing value table exists, method chosen per column with reason
- [ ] No missing values left in features
- [ ] No out of range values left
- [ ] Descriptive stats table exists (mean, median, std, min, max, skew)
- [ ] At least 5 required figures exist with titles and labels
- [ ] EDA findings answer all 7 questions in `03_PART_A_DATA_ANALYSIS.md`

## Part B

- [ ] Target defined and not in features
- [ ] Features selected with reasons, VIF checked
- [ ] Scaler fitted on train only
- [ ] Split is exactly 80/20, seed 42
- [ ] At least 2 models trained
- [ ] Metrics: MAE, RMSE, R2 on train and test
- [ ] Cross validation results exist
- [ ] Comparison table exists
- [ ] Model selected with written reason following the selection rule
- [ ] Test predictions table exists
- [ ] At least 3 custom predictions exist
- [ ] Coefficients and feature importance exist
- [ ] Results explained in plain language

## Engineering

- [ ] Whole pipeline runs from a clean clone: install, run, get same numbers
- [ ] All paths, seed, columns come from `config.yaml`
- [ ] No hardcoded absolute paths
- [ ] Tests pass (`pytest`)
- [ ] No data leakage (checked against `01_ARCHITECTURE.md`)

## Report

- [ ] All sections in `06_REPORT_SPEC.md` present
- [ ] Every number in the report matches the tables
- [ ] Limitations section is honest
- [ ] No causal claims

## Red flags (stop and investigate)

| Flag | Likely cause |
|---|---|
| Test R2 above 0.98 | Leakage, or a feature that is basically the target |
| Test R2 below 0 | Bug, wrong target, or no signal |
| Train R2 far above test R2 | Overfitting |
| Test score much better than CV | Lucky split |
| Different results each run | Missing seed |
