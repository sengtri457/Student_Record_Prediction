# 04. Part B: Machine Learning Spec

| # | Required step | Owner agent | Output file(s) |
|---|---|---|---|
| B1 | Define target variable | 05 Feature and Split | `config.yaml` (target) |
| B2 | Select relevant features | 05 Feature and Split | `config.yaml` (features), `reports/tables/feature_selection_notes.md` |
| B3 | Preprocess the data | 05 Feature and Split | `src/features.py`, scaler saved in `models/scaler.joblib` |
| B4 | Split 80/20 | 05 Feature and Split | `data/processed/train.csv`, `test.csv` |
| B5 | Train at least 2 models | 06 Model Trainer | `models/*.joblib` |
| B6 | Compare the models | 07 Evaluator | `reports/tables/model_comparison.csv` |
| B7 | Evaluate performance | 07 Evaluator | `reports/tables/metrics.csv`, `cv_results.csv` |
| B8 | Select model by metrics | 07 Evaluator | `models/best_model.joblib`, `selection_decision.md` |
| B9 | Make predictions | 07 Evaluator | `reports/tables/test_predictions.csv`, `custom_predictions.csv` |
| B10 | Explain results | 07 Evaluator + 08 Report Writer | `reports/tables/feature_importance.csv`, `coefficients.csv`, report section |

## B1. Target

`final_score`. Float. Never scaled as a feature, never used as input.

## B2. Feature selection

Start with the 5 features. Justify each using EDA (correlation, plot). Check multicollinearity with VIF (flag above 5, or 10 if relaxed). If two features are almost the same, keep one or explain why both stay. Do not add features that leak the target.

## B3. Preprocessing

- Numeric only, so no encoding needed.
- Scaling: `StandardScaler` for Linear Regression. Random Forest does not need it. Using the same scaled data for both keeps the pipeline simple. Say this in the notes.
- Fit the scaler on train only. Transform test with the train scaler.
- Best practice: use a scikit-learn `Pipeline` so scaling and model travel together.

## B4. Split

```python
train_test_split(X, y, test_size=0.20, random_state=42)
```

Train 80%, test 20%. Save both. Never touch test data during training or tuning.

## B5. Models

| Model | Setup |
|---|---|
| Multiple Linear Regression | `LinearRegression()` inside a Pipeline with scaler |
| Random Forest Regressor | `RandomForestRegressor(n_estimators=300, random_state=42)` |
| Optional | `Ridge`, `GradientBoostingRegressor` |

Optional tuning: `GridSearchCV` with 5 folds on the training set only. Keep grids small.

## B6 and B7. Evaluation

Metrics on both train and test:

| Metric | Meaning |
|---|---|
| MAE | Average error in score points |
| RMSE | Error that punishes big misses |
| R2 | Share of variance explained |

Also run 5-fold cross validation on the training set. Report mean and std of R2 and RMSE.

Overfitting check: if train R2 minus test R2 is above about 0.15, flag it and discuss.

Residual checks for Linear Regression: residual vs predicted plot, residual histogram. Save as `fig09_residuals.png` and `fig10_actual_vs_pred.png`.

`model_comparison.csv` columns: `model, train_mae, test_mae, train_rmse, test_rmse, train_r2, test_r2, cv_r2_mean, cv_r2_std`.

## B8. Model selection rule

1. Lowest test RMSE wins.
2. If RMSE differs by less than 2% between models, prefer the simpler one (Linear Regression).
3. Reject any model that clearly overfits.
4. Write the decision and reason in `selection_decision.md`.

## B9. Predictions

- `test_predictions.csv`: `student_id, actual, predicted, error`.
- `custom_predictions.csv`: at least 3 made up students, for example:
  - high attendance, low midterm
  - low attendance, high previous GPA
  - average on everything
- Chart: actual vs predicted scatter with a 45 degree line.

## B10. Explain results

- Linear Regression: table of coefficients (on scaled data), sign and size. Optionally show unscaled coefficients to say "one more study hour is linked to X points".
- Random Forest: feature importance bar chart. Optionally permutation importance for a fairer view.
- Compare both: do they agree on the top features?
- Plain language summary: what worked, what did not, how much error to expect (in points).
- Use "associated with", never "causes".

## Part B done when

- Both models trained, saved, evaluated on train and test
- Comparison table exists and a model is selected with written reasons
- Predictions and explanation files exist
- R2 above 0.98 triggers a leakage review before sign off
