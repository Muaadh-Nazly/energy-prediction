"""
run_xgboost_subprocess.py

Standalone script that fits XGBoost and prints its test-set metrics as
JSON. Run as a subprocess (never imported into a process that also
imports torch) because XGBoost and PyTorch each bundle their own OpenMP
runtime, and loading both in one process segfaults on this machine.

Usage: python -m src.run_xgboost_subprocess (run from the project root)
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.data_preprocessing import scale_data


def main():
    train_df = pd.read_csv(
        "data/processed/train_engineered.csv", index_col=0, parse_dates=True
    )
    test_df = pd.read_csv(
        "data/processed/test_engineered.csv", index_col=0, parse_dates=True
    )
    with open("data/processed/selected_features.txt") as f:
        selected_features = f.read().splitlines()

    X_train_full, X_test, y_train_full, y_test, _, target_scaler = scale_data(
        train_df, test_df, selected_features, target_col="Appliances"
    )

    xgb_model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb_model.fit(X_train_full, y_train_full)
    xgb_preds_scaled = xgb_model.predict(X_test)

    xgb_preds = target_scaler.inverse_transform(
        xgb_preds_scaled.reshape(-1, 1)
    ).flatten()
    y_test_wh = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    nonzero_mask = y_test_wh != 0
    results = {
        "model": "XGBoost (flat, 32 features)",
        "MAE": round(float(mean_absolute_error(y_test_wh, xgb_preds)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test_wh, xgb_preds))), 4),
        "MAPE": round(
            float(
                np.mean(
                    np.abs(
                        (y_test_wh[nonzero_mask] - xgb_preds[nonzero_mask])
                        / y_test_wh[nonzero_mask]
                    )
                )
                * 100
            ),
            4,
        ),
        "R2": round(float(r2_score(y_test_wh, xgb_preds)), 4),
    }
    print(json.dumps(results))


if __name__ == "__main__":
    sys.exit(main())
