"""
feature_engineering.py

Feature engineering and feature selection functions for the Appliance
Energy Prediction project. Built on top of the time features already
produced by src/data_preprocessing.py::extract_time_features.

Design notes:
- Cyclical encoding is required for hour and day_of_week since raw integer
  encoding implies a false distance between e.g. hour 23 and hour 0.
- Rolling and lag features are computed on the target (Appliances) and are
  shifted so that no feature at time t uses information from time t itself
  (i.e. rolling windows are computed on the series shifted by 1 step first).
  Without this shift, a rolling mean over t-5..t would include the current
  target value and leak the answer directly into the input features.
- Lags of 1, 3, 6, 18, 36, 144 steps are retained per the ACF/PACF analysis:
  lag 1-2 covers the AR(1) structure seen in PACF, and lag 144 covers the
  daily seasonality confirmed by STL and the ACF U-shape.
- The weekend lunch peak interaction encodes an EDA finding that does not
  exist in the raw columns: weekends show an 11:00-12:00 consumption peak
  that weekdays do not, so week_status alone or hour alone cannot capture it.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


# ---------------------------------------------------------------------------
# Cyclical encoding
# ---------------------------------------------------------------------------


def add_cyclical_features(df, column, period):
    """
    Add sine and cosine encodings of a cyclical integer column.

    This ensures the model sees adjacent cycle values (e.g. hour 23 and
    hour 0) as close together, which a raw integer encoding cannot express.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing the column to encode.
    column : str
        Name of the cyclical column (e.g. "hour", "day_of_week").
    period : int
        Number of steps in one full cycle (24 for hour, 7 for day_of_week).

    Returns
    -------
    pd.DataFrame
        Copy of df with two new columns: f"{column}_sin" and f"{column}_cos".
    """
    df = df.copy()
    radians = 2 * np.pi * df[column] / period
    df[f"{column}_sin"] = np.sin(radians)
    df[f"{column}_cos"] = np.cos(radians)
    return df


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------


def add_rolling_features(df, column, windows, stats=("mean", "std")):
    """
    Add rolling window statistics for a column, shifted by 1 step so that
    the current row's own value is never included in its own feature
    (avoids target leakage when column is the prediction target).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset, assumed sorted in temporal order.
    column : str
        Column to compute rolling statistics on (e.g. "Appliances").
    windows : list of int
        Window sizes in number of rows, e.g. [6, 18, 36] for 1hr/3hr/6hr
        at 10-minute intervals.
    stats : tuple of str, default ("mean", "std")
        Which rolling statistics to compute. Supported: "mean", "std".

    Returns
    -------
    pd.DataFrame
        Copy of df with new columns named f"{column}_roll{window}_{stat}".
        Leading rows will contain NaN until the window is filled; these
        should be dropped after all engineering steps are complete.
    """
    df = df.copy()
    shifted = df[column].shift(1)
    for window in windows:
        rolled = shifted.rolling(window=window)
        if "mean" in stats:
            df[f"{column}_roll{window}_mean"] = rolled.mean()
        if "std" in stats:
            df[f"{column}_roll{window}_std"] = rolled.std()
    return df


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------


def add_lag_features(df, column, lags):
    """
    Add lagged versions of a column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset, assumed sorted in temporal order.
    column : str
        Column to lag (e.g. "Appliances").
    lags : list of int
        Lag steps to create, e.g. [1, 3, 6, 18, 36, 144] confirmed by
        ACF/PACF analysis.

    Returns
    -------
    pd.DataFrame
        Copy of df with new columns named f"{column}_lag{lag}". Leading
        rows will contain NaN for lag steps beyond the row index.
    """
    df = df.copy()
    for lag in lags:
        df[f"{column}_lag{lag}"] = df[column].shift(lag)
    return df


# ---------------------------------------------------------------------------
# Interaction / domain features
# ---------------------------------------------------------------------------


def add_interaction_features(df, indoor_temp_cols=None):
    """
    Add domain-driven interaction features identified during EDA.

    Adds:
    - heat_index_proxy: T_out * RH_out
    - indoor_outdoor_delta: mean of indoor temperature sensors minus T_out
    - weekend_lunch_peak: 1 if day_of_week is Saturday/Sunday AND hour is
      11 or 12, else 0. Uses day_of_week directly (0=Monday, 6=Sunday)
      rather than week_status, to avoid depending on whatever encoding
      week_status happens to use.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset that already contains T_out, RH_out, hour, and day_of_week
        (from extract_time_features), plus indoor temperature sensors.
    indoor_temp_cols : list of str, optional
        Indoor temperature columns to average for indoor_outdoor_delta.
        Defaults to T1-T9.

    Returns
    -------
    pd.DataFrame
        Copy of df with the three new interaction columns appended.
    """
    df = df.copy()
    if indoor_temp_cols is None:
        indoor_temp_cols = [f"T{i}" for i in range(1, 10)]

    df["heat_index_proxy"] = df["T_out"] * df["RH_out"]
    df["indoor_outdoor_delta"] = df[indoor_temp_cols].mean(axis=1) - df["T_out"]
    df["weekend_lunch_peak"] = (
        (df["day_of_week"].isin([5, 6])) & (df["hour"].isin([11, 12]))
    ).astype(int)
    return df


# ---------------------------------------------------------------------------
# Missing value handling after engineering
# ---------------------------------------------------------------------------


def drop_engineering_nans(df):
    """
    Drop leading rows containing NaN values introduced by rolling and lag
    feature creation (e.g. the first 144 rows once a lag-144 feature exists).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset after rolling/lag feature creation.

    Returns
    -------
    pd.DataFrame
        Dataset with NaN-containing rows removed.
    dict
        {"n_dropped": int, "pct_dropped": float}
    """
    n_before = len(df)
    df_clean = df.dropna()
    n_dropped = n_before - len(df_clean)
    pct_dropped = round((n_dropped / n_before) * 100, 4)
    return df_clean, {"n_dropped": n_dropped, "pct_dropped": pct_dropped}


# ---------------------------------------------------------------------------
# Feature selection - Stage 2: correlation redundancy
# ---------------------------------------------------------------------------


def select_features_correlation(df, feature_cols, target_col, threshold=0.95):
    """
    Stage 2 feature selection: remove redundant features that are highly
    correlated with each other, keeping whichever member of a correlated
    pair has the stronger correlation with the target.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing feature_cols and target_col.
    feature_cols : list of str
        Candidate feature columns (after Stage 1 zero-signal removal).
    target_col : str
        Target column used to break ties between correlated features.
    threshold : float, default 0.95
        Absolute correlation threshold above which two features are
        considered redundant.

    Returns
    -------
    list of str
        Retained feature columns.
    list of tuple
        Dropped pairs as (dropped_feature, kept_feature, correlation).
    """
    corr_matrix = df[feature_cols].corr().abs()
    target_corr = df[feature_cols + [target_col]].corr()[target_col].abs()

    to_drop = set()
    dropped_pairs = []
    cols = feature_cols

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_i, col_j = cols[i], cols[j]
            if col_i in to_drop or col_j in to_drop:
                continue
            corr_val = corr_matrix.loc[col_i, col_j]
            if corr_val > threshold:
                if target_corr[col_i] >= target_corr[col_j]:
                    weaker, stronger = col_j, col_i
                else:
                    weaker, stronger = col_i, col_j
                to_drop.add(weaker)
                dropped_pairs.append((weaker, stronger, round(float(corr_val), 4)))

    retained = [c for c in feature_cols if c not in to_drop]
    return retained, dropped_pairs


# ---------------------------------------------------------------------------
# Feature selection - Stage 3: Random Forest importance
# ---------------------------------------------------------------------------


def select_features_rf_importance(
    X_train, y_train, feature_names, top_n=None, threshold=None, random_state=42
):
    """
    Stage 3 feature selection: rank features by Random Forest impurity-based
    importance and select either the top N or those above a threshold.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like of shape (n_samples,)
        Training target vector.
    feature_names : list of str
        Names corresponding to columns of X_train.
    top_n : int, optional
        If provided, keep only the top N most important features.
    threshold : float, optional
        If provided (and top_n is None), keep features with importance
        greater than or equal to this value.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    list of str
        Retained feature names, ordered by descending importance.
    pd.DataFrame
        Full importance ranking for all input features.
    """
    rf = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, np.ravel(y_train))

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": rf.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    if top_n is not None:
        retained = importance_df["feature"].head(top_n).tolist()
    elif threshold is not None:
        retained = importance_df.loc[
            importance_df["importance"] >= threshold, "feature"
        ].tolist()
    else:
        retained = importance_df["feature"].tolist()

    return retained, importance_df


# ---------------------------------------------------------------------------
# Feature selection - Stage 4: permutation importance
# ---------------------------------------------------------------------------


def select_features_permutation(
    model, X_val, y_val, feature_names, n_repeats=10, random_state=42, threshold=0.0
):
    """
    Stage 4 feature selection: compute permutation importance on a held-out
    validation set using a fitted model, and select features whose removal
    causes a real drop in performance (as opposed to impurity-based
    importance, which can be biased toward high-cardinality features).

    Parameters
    ----------
    model : fitted estimator
        A fitted model (e.g. the Random Forest from Stage 3) implementing
        a scikit-learn-compatible predict/score interface.
    X_val : array-like of shape (n_samples, n_features)
        Validation feature matrix, not used for fitting model.
    y_val : array-like of shape (n_samples,)
        Validation target vector.
    feature_names : list of str
        Names corresponding to columns of X_val.
    n_repeats : int, default 10
        Number of times each feature is randomly shuffled.
    random_state : int, default 42
        Random seed for reproducibility.
    threshold : float, default 0.0
        Minimum mean importance to retain a feature. Default keeps any
        feature whose shuffling increases error at all.

    Returns
    -------
    list of str
        Retained feature names, ordered by descending importance.
    pd.DataFrame
        Full permutation importance ranking (mean and std) for all features.
    """
    result = permutation_importance(
        model,
        X_val,
        np.ravel(y_val),
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    retained = importance_df.loc[
        importance_df["importance_mean"] > threshold, "feature"
    ].tolist()

    return retained, importance_df


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------


def save_processed_data(df, filepath, decimals=4):
    """
    Round all numeric columns to a fixed number of decimal places and save
    to CSV, keeping processed file sizes and diffs consistent across steps.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to save.
    filepath : str
        Output CSV path.
    decimals : int, default 4
        Number of decimal places to round numeric columns to.

    Returns
    -------
    None
    """
    df_rounded = df.copy()
    numeric_cols = df_rounded.select_dtypes(include=[np.number]).columns
    df_rounded[numeric_cols] = df_rounded[numeric_cols].round(decimals)
    df_rounded.to_csv(filepath, index=True)


def add_lag_interaction_features(df):
    """
    Create interaction terms between Appliances_lag1 and time-of-day /
    weekday-weekend indicators, to let models capture whether short-term
    usage persistence differs across the day or by day type. This targets
    nonlinear structure that a linear model cannot represent without the
    interaction being explicitly supplied.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Appliances_lag1, hour_sin, hour_cos, week_status.

    Returns
    -------
    pd.DataFrame
        Copy of df with lag1_x_hour_sin, lag1_x_hour_cos, lag1_x_is_weekend added.
    """
    df = df.copy()
    df["lag1_x_hour_sin"] = df["Appliances_lag1"] * df["hour_sin"]
    df["lag1_x_hour_cos"] = df["Appliances_lag1"] * df["hour_cos"]
    df["is_weekend"] = (df["week_status"] == "Weekend").astype(int)
    df["lag1_x_is_weekend"] = df["Appliances_lag1"] * df["is_weekend"]
    return df


def add_indoor_climate_composites(df):
    """
    Create composite indoor temperature and humidity features by averaging
    across room sensors, excluding T6/RH_6 since EDA identified T6 as
    outdoor-adjacent rather than a true indoor reading. Reduces
    multicollinearity among T1-T9/RH_1-9 while preserving indoor climate
    signal as a single domain feature.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain T1-T5, T7-T9, RH_1-RH_5, RH_7-RH_9.

    Returns
    -------
    pd.DataFrame
        Copy of df with mean_indoor_temp and mean_indoor_rh added.
    """
    df = df.copy()
    indoor_temp_cols = ["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"]
    indoor_rh_cols = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"]
    df["mean_indoor_temp"] = df[indoor_temp_cols].mean(axis=1)
    df["mean_indoor_rh"] = df[indoor_rh_cols].mean(axis=1)
    return df
