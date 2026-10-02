# Model Selection Decision

- **Selected Model:** **LINEAR_REGRESSION**
- **Artifact Path:** `models\best_model.joblib`
- **Selection Rule:** Lowest test RMSE, with a 2% tolerance favoring simpler baseline models (Multiple Linear Regression).

### Performance Metrics Summary:
| Metric | Multiple Linear Regression | Random Forest Regressor | Ridge Regression |
|---|---|---|---|
| **Test RMSE** | **3.393** | 4.02 | 3.394 |
| **Test MAE** | **2.627** | 3.148 | 2.627 |
| **Test $R^2$** | **0.9288** | 0.9 | 0.9288 |
| **CV $R^2$ (Mean $\pm$ Std)** | 0.8815 $\pm$ 0.0176 | 0.8542 $\pm$ 0.0255 | 0.8816 $\pm$ 0.0176 |
| **CV RMSE (Mean)** | 3.971 | 4.397 | 3.97 |

### Decision Rationale:
Multiple Linear Regression achieved lower test RMSE (3.393) compared to Random Forest (4.02).

Overfitting Verification:
- Linear Regression: $R^2_{\text{train}} - R^2_{\text{test}} = -0.0407$ ($< 0.15$ threshold: PASS)
- Random Forest: $R^2_{\text{train}} - R^2_{\text{test}} = 0.0809$
