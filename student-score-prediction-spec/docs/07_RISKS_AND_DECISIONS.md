# 07. Risks and Decisions

## Decisions already made

| ID | Decision | Reason |
|---|---|---|
| D1 | Target is `final_score`, regression | Matches the topic |
| D2 | Models: Linear Regression and Random Forest | Required, covers simple and flexible |
| D3 | Seed 42 | Repeatable |
| D4 | Split 80/20 | Required |
| D5 | Scale for both models | Simple pipeline |
| D6 | Select by lowest test RMSE, prefer simpler if within 2% | Clear rule, avoids picking complexity for no gain |
| D7 | No synthetic data in final results | Fake patterns give fake conclusions |

## Risks

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | Dataset too small | Unstable results | 5-fold CV, say it in limitations |
| R2 | Dataset lacks some features | Cannot match topic exactly | Document proxy or drop feature |
| R3 | Data leakage | Fake high scores | Split first, fit scaler on train, check red flags |
| R4 | Midterm dominates | Other features look useless | Also show results without midterm as an extra experiment |
| R5 | Multicollinearity | Unstable coefficients | VIF check, drop or combine features |
| R6 | Self reported study hours | Noise | Mention in limitations |
| R7 | Privacy | Ethics issue | Anonymous IDs only, consent |
| R8 | Random Forest overfits on small data | Poor test score | Limit depth, use CV |

## Optional extras (only after required work passes)

- Experiment: model without `midterm_score` to see what else predicts well
- Ridge regression
- Gradient Boosting
- Permutation importance
- Prediction intervals for linear regression
