# Handoff Note

- **Agent ID and name:** Agent 05: Feature and Split
- **Date:** 2026-10-02
- **STATUS:** READY

## 1. What I was asked to do
Define target variable, verify feature selection with VIF checks, fit feature scalers strictly on training data (anti-leakage), and execute an 80/20 train/test partition (Scope items B1, B2, B3, B4).

## 2. What I did
1. Extracted target variable `final_score` (continuous float scale: 0 to 100) and verified exclusion from feature set.
2. Conducted Variance Inflation Factor (VIF) collinearity diagnostics and documented results in `reports/tables/vif_report.csv` and `reports/tables/feature_selection_notes.md`.
3. Partitioned canonical cleaned data into an 80% training set (318 rows) and a 20% test set (80 rows) using fixed `random_state=42`.
4. Fit `StandardScaler` strictly on $X_{\text{train}}$ features and serialized the artifact to `models/scaler.joblib`.
5. Exported partitioned data to `data/processed/train.csv` and `data/processed/test.csv`.

## 3. Files produced
| Path | Description |
|---|---|
| `data/processed/train.csv` | Training split (318 rows, 80%) |
| `data/processed/test.csv` | Holdout testing split (80 rows, 20%) |
| `models/scaler.joblib` | Fitted StandardScaler (fit on train only) |
| `reports/tables/vif_report.csv` | VIF metrics table |
| `reports/tables/feature_selection_notes.md` | Feature rationale and collinearity audit |
| `src/features.py` | Standalone feature preprocessing & split module |

## 4. Key numbers
- Training set size: 318 samples (79.9%)
- Test set size: 80 samples (20.1%)
- VIF scores: All between 2.828 and 4.270 (all < 5.0, confirming Low collinearity)
- Seed: 42

## 5. Decisions I made and why
| Decision | Reason |
|---|---|
| Retained all 5 features | All VIFs < 5.0; each feature provides distinct academic signal |
| Scaler fit on train only | Strict anti-leakage compliance; prevents test distribution from contaminating model training |

## 6. Problems and issues
None.

## 7. Requests for other agents
Agent 06 (Model Trainer) to fit Multiple Linear Regression and Random Forest Regressor models on `data/processed/train.csv` using hyperparameters from `config.yaml`.

## 8. What the next agent needs to know
- Train on `data/processed/train.csv` only.
- Scaler is at `models/scaler.joblib`.
- Target column is `final_score`.

## 9. Self check
- [x] I stayed inside my scope
- [x] I used config.yaml
- [x] All contract files exist
- [x] I did not touch test data for training or tuning
