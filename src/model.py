"""
model.py

Model construction functions for the Appliance Energy Prediction project.
Provides a single get_model() dispatcher so train.py and the notebooks can
request any of the 7 locked models through one consistent interface,
regardless of whether the underlying library is scikit-learn or TensorFlow.

Implemented now: Linear Regression, Random Forest (Phase 3 baselines).
Placeholders: LSTM, GRU, CNN-LSTM, TCN, CNN-LSTM+Attention (Phase 4) -
these raise NotImplementedError with a note on what will fill them in.
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def build_linear_regression(**kwargs):
    """
    Build a Linear Regression baseline model.

    Parameters
    ----------
    **kwargs
        Passed directly to sklearn.linear_model.LinearRegression.

    Returns
    -------
    LinearRegression
        Unfitted model instance.
    """
    return LinearRegression(**kwargs)


def build_random_forest(n_estimators=200, max_depth=None, random_state=42, **kwargs):
    """
    Build a Random Forest baseline model.

    Parameters
    ----------
    n_estimators : int, default 200
        Number of trees.
    max_depth : int, optional
        Maximum tree depth. None means nodes expand until pure or
        min_samples_split is reached.
    random_state : int, default 42
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to RandomForestRegressor.

    Returns
    -------
    RandomForestRegressor
        Unfitted model instance.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        **kwargs,
    )


def build_lstm(input_shape, **kwargs):
    """
    Build the LSTM model (Phase 4).

    Parameters
    ----------
    input_shape : tuple
        (sequence_length, n_features), set once the sequence window
        hyperparameter sweep [24, 72, 144] is finalized.

    Returns
    -------
    Not yet implemented. Will return a compiled tf.keras.Model with
    SpatialDropout1D, Adam optimizer, and Cosine Annealing LR schedule,
    per the locked blueprint.
    """
    raise NotImplementedError("build_lstm will be implemented in Phase 4.")


def build_gru(input_shape, **kwargs):
    """
    Build the GRU model (Phase 4).

    Parameters
    ----------
    input_shape : tuple
        (sequence_length, n_features).

    Returns
    -------
    Not yet implemented. See build_lstm docstring for shared design notes.
    """
    raise NotImplementedError("build_gru will be implemented in Phase 4.")


def build_cnn_lstm(input_shape, **kwargs):
    """
    Build the CNN-LSTM hybrid model (Phase 4).

    Parameters
    ----------
    input_shape : tuple
        (sequence_length, n_features).

    Returns
    -------
    Not yet implemented. Conv1D feature extraction feeding into LSTM
    temporal layers.
    """
    raise NotImplementedError("build_cnn_lstm will be implemented in Phase 4.")


def build_tcn(input_shape, **kwargs):
    """
    Build the Temporal Convolutional Network model (Phase 4, initiative addition).

    Parameters
    ----------
    input_shape : tuple
        (sequence_length, n_features).

    Returns
    -------
    Not yet implemented. Dilated causal convolutions with residual blocks.
    """
    raise NotImplementedError("build_tcn will be implemented in Phase 4.")


def build_cnn_lstm_attention(input_shape, **kwargs):
    """
    Build the CNN-LSTM + Attention model (Phase 4, primary star model).

    Parameters
    ----------
    input_shape : tuple
        (sequence_length, n_features).

    Returns
    -------
    Not yet implemented. Attention weights from this model will be
    visualized in notebook 05 per the locked blueprint.
    """
    raise NotImplementedError(
        "build_cnn_lstm_attention will be implemented in Phase 4."
    )


MODEL_REGISTRY = {
    "linear_regression": build_linear_regression,
    "random_forest": build_random_forest,
    "lstm": build_lstm,
    "gru": build_gru,
    "cnn_lstm": build_cnn_lstm,
    "tcn": build_tcn,
    "cnn_lstm_attention": build_cnn_lstm_attention,
}


def get_model(name, **kwargs):
    """
    Single dispatcher for all 7 locked model architectures.

    Parameters
    ----------
    name : str
        One of: "linear_regression", "random_forest", "lstm", "gru",
        "cnn_lstm", "tcn", "cnn_lstm_attention".
    **kwargs
        Forwarded to the corresponding build_* function.

    Returns
    -------
    Unfitted model instance (sklearn estimator or, from Phase 4 onward,
    a compiled tf.keras.Model).

    Raises
    ------
    ValueError
        If name is not a recognized model key.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name '{name}'. Must be one of {list(MODEL_REGISTRY.keys())}."
        )
    return MODEL_REGISTRY[name](**kwargs)
