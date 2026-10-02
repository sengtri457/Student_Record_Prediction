"""Model training module for Student Score Prediction.
Trains Multiple Linear Regression (baseline), Random Forest Regressor (comparison),
and Ridge Regression (regularized benchmark) inside scikit-learn Pipelines.
"""

import os
import yaml
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


def load_config(config_path: str = "config.yaml") -> dict:
    """Load config yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_models(
    train_path: str = "data/processed/train.csv",
    models_dir: str = "models",
    log_path: str = "reports/tables/training_log.csv"
) -> dict:
    """Trains multiple regression models on training data only and serializes them."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    config = load_config()
    target_col = config["target"]
    feature_cols = config["features"]
    seed = config["seed"]
    rf_params = config["models"]["random_forest"]

    train_df = pd.read_csv(train_path)
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    models = {}
    training_log = []

    # Model 1: Multiple Linear Regression (Pipeline with StandardScaler)
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_path = os.path.join(models_dir, "linear_regression.joblib")
    joblib.dump(lr_pipeline, lr_path)
    models["linear_regression"] = lr_pipeline
    training_log.append({
        "model_name": "Multiple Linear Regression",
        "type": "Parametric Baseline",
        "pipeline": "StandardScaler -> LinearRegression",
        "train_samples": len(X_train),
        "status": "Trained & Serialized",
        "artifact_path": lr_path
    })

    # Model 2: Random Forest Regressor (Pipeline with StandardScaler for uniform interface)
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=rf_params.get("n_estimators", 300),
            max_depth=rf_params.get("max_depth", None),
            random_state=rf_params.get("random_state", seed),
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_path = os.path.join(models_dir, "random_forest.joblib")
    joblib.dump(rf_pipeline, rf_path)
    models["random_forest"] = rf_pipeline
    training_log.append({
        "model_name": "Random Forest Regressor",
        "type": "Non-linear Ensemble",
        "pipeline": "StandardScaler -> RandomForestRegressor(n_estimators=300)",
        "train_samples": len(X_train),
        "status": "Trained & Serialized",
        "artifact_path": rf_path
    })

    # Model 3 (Bonus): Ridge Regression
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=seed))
    ])
    ridge_pipeline.fit(X_train, y_train)
    ridge_path = os.path.join(models_dir, "ridge.joblib")
    joblib.dump(ridge_pipeline, ridge_path)
    models["ridge"] = ridge_pipeline
    training_log.append({
        "model_name": "Ridge Regression",
        "type": "L2 Regularized Linear",
        "pipeline": "StandardScaler -> Ridge(alpha=1.0)",
        "train_samples": len(X_train),
        "status": "Trained & Serialized",
        "artifact_path": ridge_path
    })

    df_log = pd.DataFrame(training_log)
    df_log.to_csv(log_path, index=False)

    print("All models trained and serialized successfully:")
    for m in training_log:
        print(f"  - {m['model_name']} -> {m['artifact_path']}")
    print(f"Training log saved to: {log_path}")

    return models


if __name__ == "__main__":
    train_models()
