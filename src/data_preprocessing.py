"""
data_preprocessing.py
----------------------
Core data loading, cleaning, and preprocessing functions for the
Appliance Energy Prediction project.

All functions are imported into Jupyter notebooks and the training
pipeline (train.py). This keeps the notebooks clean and the logic reusable.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_data(filepath):
    """
    Load the raw energy dataset and parse the datetime column.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Time-sorted DataFrame with a parsed datetime 'date' column.
    """
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def check_data_quality(df):
    """
    Check the dataset for missing values, duplicates, and noise columns.

    The UCI dataset includes rv1 and rv2 as intentional random noise
    control columns. This function flags them for removal.

    Parameters
    ----------
    df : pd.DataFrame
        Raw loaded DataFrame.

    Returns
    -------
    dict
        Summary report with missing counts, duplicate count,
        and rv column statistics.
    """
    missing = df.isnull().sum()
    duplicates = df.duplicated().sum()

    rv_identical = (df["rv1"] == df["rv2"]).all() if "rv1" in df.columns else None
    rv_corr = {
        col: round(df[col].corr(df["Appliances"]), 6)
        for col in ["rv1", "rv2"]
        if col in df.columns
    }

    return {
        "missing_values": missing[missing > 0].to_dict(),
        "duplicate_rows": int(duplicates),
        "rv_columns_identical": rv_identical,
        "rv_correlation_with_target": rv_corr,
        "shape": df.shape,
        "date_range": (str(df["date"].min()), str(df["date"].max())),
    }


def drop_noise_columns(df):
    """
    Drop the random noise columns rv1 and rv2.

    These columns are identical to each other and have near-zero
    correlation with the target variable. They were added to the UCI
    dataset as control variables to validate feature selection methods.
    Keeping them would add noise to model training.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing rv1 and rv2 columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with rv1 and rv2 removed.
    """
    cols_to_drop = [c for c in ["rv1", "rv2"] if c in df.columns]
    return df.drop(columns=cols_to_drop)


def clip_outliers_iqr(df, column="Appliances", multiplier=1.5):
    """
    Clip outliers in a column using the IQR (Interquartile Range) method.

    Clipping is used instead of row removal because deleting rows from a
    time series breaks temporal continuity. Any rolling averages or lag
    features computed after a deletion would produce NaN gaps or incorrect
    values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Target column to apply clipping on. Default is 'Appliances'.
    multiplier : float
        IQR fence multiplier. Default is 1.5 (standard Tukey fences).

    Returns
    -------
    pd.DataFrame
        DataFrame with the specified column clipped.
    dict
        Outlier statistics: Q1, Q3, IQR, fences, and outlier counts.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    upper_fence = Q3 + multiplier * IQR
    lower_fence = max(0, Q1 - multiplier * IQR)

    stats = {
        "Q1": round(Q1, 2),
        "Q3": round(Q3, 2),
        "IQR": round(IQR, 2),
        "upper_fence": round(upper_fence, 2),
        "lower_fence": round(lower_fence, 2),
        "outliers_high": int((df[column] > upper_fence).sum()),
        "outliers_low": int((df[column] < lower_fence).sum()),
    }

    df_out = df.copy()
    df_out[column] = df_out[column].clip(lower=lower_fence, upper=upper_fence)
    return df_out, stats


def extract_time_features(df):
    """
    Extract and engineer time-based features from the date column.

    The dataset is missing NSM, WeekStatus, and Day_of_week columns
    that are referenced in the assessment description. This function
    engineers all of them from the parsed datetime index.

    Engineered features:
    - hour        : Hour of day (0-23)
    - day_of_week : Day number (0=Monday, 6=Sunday)
    - month       : Month number (1-12)
    - week_status : Binary string - 'Weekday' or 'Weekend'
    - NSM         : Seconds elapsed since midnight (matches UCI description)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a parsed 'date' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with all time feature columns appended.
    """
    df = df.copy()
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week_status"] = df["day_of_week"].apply(
        lambda x: "Weekend" if x >= 5 else "Weekday"
    )
    df["NSM"] = (
        df["date"].dt.hour * 3600 + df["date"].dt.minute * 60 + df["date"].dt.second
    )
    return df


def temporal_train_test_split(df, test_size=0.2):
    """
    Split the time series into training and test sets without shuffling.

    Standard random train/test splits must never be used on time series
    data because they allow future data to leak into training. This
    function takes the first (1 - test_size) fraction as training and
    the remainder as the test set, preserving chronological order.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame sorted by date.
    test_size : float
        Proportion of data for testing. Default is 0.2 (20%).

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        train_df, test_df
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def get_feature_columns(df):
    """
    Return numeric feature column names, excluding the target and metadata.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after feature engineering is complete.

    Returns
    -------
    list
        Sorted list of numeric feature column names.
    """
    exclude = {"date", "Appliances", "week_status"}
    return [
        c
        for c in df.columns
        if c not in exclude
        and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]


def scale_data(train_df, test_df, feature_cols, target_col="Appliances"):
    """
    Scale features and target using separate scaling strategies.

    Strategy:
    - Input features  : StandardScaler
      Sensor readings (temperature, humidity) are approximately normally
      distributed. StandardScaler centers them at 0 with unit variance,
      which is more appropriate than MinMax for normally distributed data.

    - Target variable : MinMaxScaler
      Applied after outlier clipping to keep values in [0, 1].
      This matches the output neuron range and avoids scale mismatch
      during loss computation.

    Critical: Both scalers are fit ONLY on training data.
    Fitting on the full dataset would constitute data leakage, as the
    test set statistics would influence the transformation.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training set after temporal split.
    test_df : pd.DataFrame
        Test set after temporal split.
    feature_cols : list
        Input feature column names.
    target_col : str
        Target column name. Default is 'Appliances'.

    Returns
    -------
    X_train, X_test : np.ndarray
        Scaled input feature arrays.
    y_train, y_test : np.ndarray
        Scaled target arrays.
    feature_scaler : StandardScaler
        Fitted input scaler (needed to inverse transform for evaluation).
    target_scaler : MinMaxScaler
        Fitted target scaler (needed to inverse transform predictions).
    """
    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler()

    X_train = feature_scaler.fit_transform(train_df[feature_cols].values)
    X_test = feature_scaler.transform(test_df[feature_cols].values)

    y_train = target_scaler.fit_transform(
        train_df[target_col].values.reshape(-1, 1)
    ).flatten()
    y_test = target_scaler.transform(
        test_df[target_col].values.reshape(-1, 1)
    ).flatten()

    return X_train, X_test, y_train, y_test, feature_scaler, target_scaler
