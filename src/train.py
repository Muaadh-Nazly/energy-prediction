"""
train.py

Training and evaluation utilities shared across all 7 models. Keeps the
fit/evaluate/save logic in one place so notebooks 03-05 call the same
functions instead of duplicating metric computation per model.
"""

from src.model import get_model
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed=42):
    """
    Set random seeds across random, numpy, and torch (CPU/CUDA) so that
    weight initialization and DataLoader shuffling are reproducible given
    the same seed. Called at the start of train_torch_model.

    Parameters
    ----------
    seed : int, default 42
        Seed value applied to random, numpy, and torch RNGs.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def get_device():
    """
    Auto-detect the best available device: CUDA GPU, Apple Silicon MPS,
    or CPU as the final fallback.

    Returns
    -------
    str
        "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_sequences(X, y, window_size):
    """
    Convert a 2D scaled feature matrix into 3D sequences for recurrent
    and convolutional sequence models.

    For each output index i, X_seq[i] holds rows [i, i+window_size) and
    y_seq[i] holds the target at row i+window_size - i.e. window_size past
    rows are used to predict the next row's target, preserving causality.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Scaled feature matrix, temporally ordered.
    y : np.ndarray of shape (n_samples,)
        Scaled target vector, temporally ordered, aligned with X.
    window_size : int
        Number of past time steps per sequence (e.g. 24, 72, 144).

    Returns
    -------
    np.ndarray of shape (n_samples - window_size, window_size, n_features)
        Sequence input array.
    np.ndarray of shape (n_samples - window_size,)
        Target array aligned with each sequence.
    """
    n_samples = X.shape[0] - window_size
    n_features = X.shape[1]
    X_seq = np.zeros((n_samples, window_size, n_features), dtype=np.float32)
    y_seq = np.zeros((n_samples,), dtype=np.float32)
    for i in range(n_samples):
        X_seq[i] = X[i : i + window_size]
        y_seq[i] = y[i + window_size]
    return X_seq, y_seq


class SequenceDataset(Dataset):
    """PyTorch Dataset wrapping pre-windowed sequence arrays."""

    def __init__(self, X_seq, y_seq):
        """
        Parameters
        ----------
        X_seq : np.ndarray of shape (n_samples, window_size, n_features)
        y_seq : np.ndarray of shape (n_samples,)
        """
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y_seq, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# PyTorch training loop (Phase 4)
# ---------------------------------------------------------------------------


def train_torch_model(
    model,
    X_train_seq,
    y_train_seq,
    X_val_seq,
    y_val_seq,
    epochs=100,
    batch_size=64,
    lr=1e-3,
    patience=10,
    loss_fn=None,
    device=None,
    verbose=True,
    seed=42,
):
    """
    Train a PyTorch sequence model with Adam, Cosine Annealing LR
    schedule, and early stopping on validation loss - per the locked
    blueprint. Restores the best-validation-loss weights before returning.

    Parameters
    ----------
    model : nn.Module
        Unfitted model from get_model().
    X_train_seq, y_train_seq : np.ndarray
        Windowed training sequences and targets, from create_sequences.
    X_val_seq, y_val_seq : np.ndarray
        Windowed validation sequences and targets, used for early
        stopping and LR scheduling context.
    epochs : int, default 100
        Maximum training epochs.
    batch_size : int, default 64
        DataLoader batch size.
    lr : float, default 1e-3
        Initial learning rate for Adam; annealed via CosineAnnealingLR.
    patience : int, default 10
        Epochs without validation improvement before stopping early.
    loss_fn : callable, optional
        Loss function. Defaults to nn.HuberLoss(), justified by the
        target's skewness (3.39) - Huber is less sensitive to the large
        residuals produced by high-consumption outlier rows than MSE.
        Pass nn.MSELoss() explicitly to run the MSE comparison experiment.
    device : str, optional
        "cuda" or "cpu". Defaults to auto-detect.
    verbose : bool, default True
        Whether to print progress every 5 epochs.
    seed : int, default 42
        Random seed applied to numpy/torch RNGs and the training
        DataLoader's shuffle order, for reproducible runs.

    Returns
    -------
    nn.Module
        The trained model, with best validation-loss weights restored.
    dict
        {"train_loss": [...], "val_loss": [...]} per-epoch history, for
        plotting learning curves in the notebook.
    """
    set_seed(seed)
    device = device or get_device()
    model = model.to(device)
    loss_fn = loss_fn or nn.HuberLoss()

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        SequenceDataset(X_train_seq, y_train_seq),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        SequenceDataset(X_val_seq, y_val_seq), batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_losses.append(loss_fn(preds, yb).item())

        scheduler.step()
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(
                f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch+1} (best val_loss: {best_val_loss:.4f})"
                    )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_torch_model(
    model,
    X_test_seq,
    y_test_seq,
    target_scaler=None,
    model_name="model",
    device=None,
    batch_size=128,
):
    """
    Evaluate a trained PyTorch model and compute MAE, RMSE, MAPE, R2 -
    same metric set and inverse-transform logic as evaluate_model() for
    the sklearn baselines, so both are directly comparable.

    Parameters
    ----------
    model : trained nn.Module
        Model returned by train_torch_model.
    X_test_seq, y_test_seq : np.ndarray
        Windowed test sequences and targets.
    target_scaler : MinMaxScaler, optional
        Fitted target scaler. If provided, predictions and targets are
        inverse transformed to original Wh units before computing metrics.
    model_name : str, default "model"
        Label used in the returned results dict.
    device : str, optional
        "cuda" or "cpu". Defaults to auto-detect.
    batch_size : int, default 128
        Batch size used for inference.

    Returns
    -------
    dict
        {"model": model_name, "MAE": float, "RMSE": float,
         "MAPE": float, "R2": float}
    """
    device = device or get_device()
    model = model.to(device)
    model.eval()

    loader = DataLoader(
        SequenceDataset(X_test_seq, y_test_seq), batch_size=batch_size, shuffle=False
    )
    preds_list = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            preds_list.append(model(xb).cpu().numpy())
    y_pred = np.concatenate(preds_list)
    y_true = np.asarray(y_test_seq)

    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    if target_scaler is not None:
        y_true = target_scaler.inverse_transform(y_true)
        y_pred = target_scaler.inverse_transform(y_pred)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    nonzero_mask = y_true != 0
    mape = (
        np.mean(
            np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])
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


# ---------------------------------------------------------------------------
# Sequence window sweep (sweep-then-finalize approach)
# ---------------------------------------------------------------------------


def run_window_sweep(
    model_name,
    X_train,
    y_train,
    X_val,
    y_val,
    n_features,
    windows=(24, 72, 144),
    sweep_epochs=15,
    sweep_patience=3,
    batch_size=64,
    model_kwargs=None,
    device=None,
    verbose=True,
    seed=42,
):
    """
    Lightweight sweep: trains one architecture at each candidate window
    size with a reduced epoch budget and aggressive early stopping, to
    identify the best window for that architecture before committing to
    a full training run. This is the sweep phase of the two-stage
    sweep-then-finalize approach - full training with full epochs and
    full patience happens separately, only at the winning window.

    Parameters
    ----------
    model_name : str
        One of "lstm", "gru", "cnn_lstm", "tcn", "cnn_lstm_attention".
    X_train, y_train : np.ndarray
        Scaled training features and target (2D and 1D, not yet windowed).
    X_val, y_val : np.ndarray
        Scaled validation features and target.
    n_features : int
        Number of input features, needed to construct the model.
    windows : tuple of int, default (24, 72, 144)
        Candidate sequence window sizes to sweep over.
    sweep_epochs : int, default 15
        Reduced epoch budget used only during the sweep phase.
    sweep_patience : int, default 3
        Reduced early stopping patience used only during the sweep phase.
    batch_size : int, default 64
        DataLoader batch size.
    model_kwargs : dict, optional
        Extra keyword arguments forwarded to the model constructor.
    device : str, optional
        "cuda" or "cpu". Defaults to auto-detect.
    verbose : bool, default True
        Whether to print per-window results.

    Returns
    -------
    int
        The window size with the lowest best validation loss.
    dict
        {window: best_val_loss} for every window tried, for the report.
    """
    model_kwargs = model_kwargs or {}
    results = {}
    for window in windows:
        X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, window)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, window)
        model = get_model(model_name, n_features=n_features, **model_kwargs)
        _, history = train_torch_model(
            model,
            X_tr_seq,
            y_tr_seq,
            X_val_seq,
            y_val_seq,
            epochs=sweep_epochs,
            patience=sweep_patience,
            batch_size=batch_size,
            device=device,
            verbose=False,
            seed=42,
        )
        best_val = min(history["val_loss"])
        results[window] = round(best_val, 6)
        if verbose:
            print(f"  window={window}: best val_loss={best_val:.6f}")

    best_window = min(results, key=results.get)
    if verbose:
        print(f"Best window for {model_name}: {best_window}")
    return best_window, results
