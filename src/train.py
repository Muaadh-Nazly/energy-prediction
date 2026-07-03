"""
train.py

Training and evaluation utilities shared across all 7 models. Keeps the
fit/evaluate/save logic in one place so notebooks 03-05 call the same
functions instead of duplicating metric computation per model.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_sklearn_model(model, X_train, y_train):
    """
    Fit a scikit-learn-compatible model on training data.

    Parameters
    ----------
    model : estimator
        Unfitted model from src.model.get_model().
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Scaled training target.

    Returns
    -------
    estimator
        The same model instance, fitted in place.
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, target_scaler=None, model_name="model"):
    """
    Evaluate a fitted model and compute MAE, RMSE, MAPE, and R2.

    If target_scaler is provided, predictions and y_test are inverse
    transformed back to original Wh units before computing metrics, so
    reported errors are interpretable (e.g. "MAE of 12 Wh") rather than
    being in the arbitrary 0-1 MinMax scale.

    Parameters
    ----------
    model : fitted estimator
        Model with a .predict() method.
    X_test : np.ndarray
        Scaled test features.
    y_test : np.ndarray
        Scaled test target.
    target_scaler : MinMaxScaler, optional
        Fitted target scaler from src.data_preprocessing.scale_data.
        If None, metrics are computed in scaled space instead.
    model_name : str, default "model"
        Label used in the returned results dict, for building comparison
        tables across multiple models.

    Returns
    -------
    dict
        {"model": model_name, "MAE": float, "RMSE": float,
         "MAPE": float, "R2": float}
    """
    y_pred = model.predict(X_test)

    y_test = np.asarray(y_test).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)

    if target_scaler is not None:
        y_test = target_scaler.inverse_transform(y_test)
        y_pred = target_scaler.inverse_transform(y_pred)

    y_test = y_test.flatten()
    y_pred = y_pred.flatten()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # MAPE guards against division by zero for any zero-Wh rows
    nonzero_mask = y_test != 0
    mape = (
        np.mean(
            np.abs((y_test[nonzero_mask] - y_pred[nonzero_mask]) / y_test[nonzero_mask])
        )
        * 100
        if nonzero_mask.any()
        else np.nan
    )

    return {
        "model": model_name,
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4),
        "R2": round(float(r2), 4),
    }


def compare_models(results_list):
    """
    Build a comparison table from a list of evaluate_model() outputs.

    Parameters
    ----------
    results_list : list of dict
        Each dict as returned by evaluate_model().

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by RMSE ascending (best model first).
    """
    df = pd.DataFrame(results_list)
    return df.sort_values("RMSE", ascending=True).reset_index(drop=True)


def save_sklearn_model(model, filepath):
    """
    Save a fitted scikit-learn model to disk using joblib.

    Parameters
    ----------
    model : fitted estimator
        Model to save.
    filepath : str
        Output path, conventionally ending in .pkl.

    Returns
    -------
    None
    """
    joblib.dump(model, filepath)


def load_sklearn_model(filepath):
    """
    Load a previously saved scikit-learn model.

    Parameters
    ----------
    filepath : str
        Path to a .pkl file saved by save_sklearn_model.

    Returns
    -------
    estimator
        The loaded fitted model.
    """
    return joblib.load(filepath)
