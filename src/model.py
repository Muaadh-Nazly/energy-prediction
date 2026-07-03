"""
model.py

Model construction functions for the Appliance Energy Prediction project,
using PyTorch for the deep learning architectures (LSTM, GRU, CNN-LSTM,
TCN, CNN-LSTM+Attention) and scikit-learn for the two baselines.

get_model() is the single dispatcher used across train.py and all
notebooks, so callers don't need to know which library backs a given
architecture.
"""

import torch
import torch.nn as nn
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
        Maximum tree depth. None means nodes expand until pure.
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


class SpatialDropout1D(nn.Module):
    """
    Spatial dropout for sequence inputs of shape (batch, seq_len, features).

    Standard nn.Dropout zeroes individual values independently at every
    time step, which lets a recurrent layer patch over a dropped value
    using neighboring time steps - weakening the regularization effect.
    SpatialDropout1D instead drops entire feature channels consistently
    across the whole sequence, which is the locked design choice for all
    RNN-based models in this project. Implemented via nn.Dropout2d by
    treating features as channels.
    """

    def __init__(self, p=0.2):
        """
        Parameters
        ----------
        p : float, default 0.2
            Fraction of feature channels to zero out during training.
        """
        super().__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, seq_len, features)

        Returns
        -------
        torch.Tensor of shape (batch, seq_len, features)
        """
        x = x.permute(0, 2, 1).unsqueeze(3)  # (batch, features, seq_len, 1)
        x = self.dropout(x)
        x = x.squeeze(3).permute(0, 2, 1)  # back to (batch, seq_len, features)
        return x


class LSTMModel(nn.Module):
    """LSTM model with SpatialDropout1D on the input sequence."""

    def __init__(self, n_features, hidden_size=64, num_layers=1, dropout=0.2):
        """
        Parameters
        ----------
        n_features : int
            Number of input features per time step.
        hidden_size : int, default 64
            LSTM hidden state size.
        num_layers : int, default 1
            Number of stacked LSTM layers.
        dropout : float, default 0.2
            SpatialDropout1D rate applied to the input sequence.
        """
        super().__init__()
        self.spatial_dropout = SpatialDropout1D(dropout)
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, seq_len, n_features)

        Returns
        -------
        torch.Tensor of shape (batch,)
            Predicted scaled Appliances value for each sequence.
        """
        x = self.spatial_dropout(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)


class GRUModel(nn.Module):
    """GRU model with SpatialDropout1D on the input sequence."""

    def __init__(self, n_features, hidden_size=64, num_layers=1, dropout=0.2):
        """
        Parameters
        ----------
        n_features : int
            Number of input features per time step.
        hidden_size : int, default 64
            GRU hidden state size.
        num_layers : int, default 1
            Number of stacked GRU layers.
        dropout : float, default 0.2
            SpatialDropout1D rate applied to the input sequence.
        """
        super().__init__()
        self.spatial_dropout = SpatialDropout1D(dropout)
        self.gru = nn.GRU(n_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, seq_len, n_features)

        Returns
        -------
        torch.Tensor of shape (batch,)
        """
        x = self.spatial_dropout(x)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)


class CNNLSTMModel(nn.Module):
    """
    Conv1D feature extractor feeding into an LSTM temporal layer.
    The convolution learns local patterns across nearby time steps before
    the LSTM models longer-range temporal dependencies.
    """

    def __init__(
        self, n_features, cnn_channels=32, kernel_size=3, hidden_size=64, dropout=0.2
    ):
        """
        Parameters
        ----------
        n_features : int
            Number of input features per time step.
        cnn_channels : int, default 32
            Number of Conv1D output channels.
        kernel_size : int, default 3
            Conv1D kernel width.
        hidden_size : int, default 64
            LSTM hidden state size.
        dropout : float, default 0.2
            SpatialDropout1D rate applied after the convolution.
        """
        super().__init__()
        self.conv = nn.Conv1d(
            n_features, cnn_channels, kernel_size, padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.spatial_dropout = SpatialDropout1D(dropout)
        self.lstm = nn.LSTM(cnn_channels, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, seq_len, n_features)

        Returns
        -------
        torch.Tensor of shape (batch,)
        """
        x = x.permute(0, 2, 1)  # (batch, features, seq_len) for Conv1d
        x = self.relu(self.conv(x))
        x = x.permute(0, 2, 1)  # back to (batch, seq_len, channels)
        x = self.spatial_dropout(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)


class Chomp1d(nn.Module):
    """Removes trailing padding added by causal (dilated) convolutions."""

    def __init__(self, chomp_size):
        """
        Parameters
        ----------
        chomp_size : int
            Number of trailing time steps to remove.
        """
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, channels, seq_len)

        Returns
        -------
        torch.Tensor with chomp_size trailing steps removed.
        """
        return x[:, :, : -self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """
    A single TCN residual block: two dilated causal convolutions with
    ReLU, dropout, and a residual connection (with a 1x1 conv to match
    channel dimensions when needed).
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        """
        Parameters
        ----------
        n_inputs : int
            Input channel count.
        n_outputs : int
            Output channel count.
        kernel_size : int
            Convolution kernel width.
        dilation : int
            Dilation factor for this block.
        dropout : float, default 0.2
            Dropout rate after each convolution.
        """
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, n_inputs, seq_len)

        Returns
        -------
        torch.Tensor of shape (batch, n_outputs, seq_len)
        """
        out = self.dropout1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network: stacked dilated causal convolutions
    with residual connections. Dilation doubles at each layer (1, 2, 4, ...)
    so the effective receptive field grows exponentially with depth,
    letting the model see far back in the sequence without needing a
    proportionally deep stack.
    """

    def __init__(
        self, n_features, num_channels=(32, 32, 32), kernel_size=3, dropout=0.2
    ):
        """
        Parameters
        ----------
        n_features : int
            Number of input features per time step.
        num_channels : tuple of int, default (32, 32, 32)
            Output channels per TemporalBlock layer.
        kernel_size : int, default 3
            Convolution kernel width for every block.
        dropout : float, default 0.2
            Dropout rate applied inside every block.
        """
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2**i
            in_ch = n_features if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, seq_len, n_features)

        Returns
        -------
        torch.Tensor of shape (batch,)
        """
        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = out[:, :, -1]  # last time step
        return self.fc(out).squeeze(-1)


class Attention(nn.Module):
    """
    Simple additive attention over LSTM outputs at every time step,
    producing a single context vector and per-step attention weights
    for visualization in notebook 05.
    """

    def __init__(self, hidden_size):
        """
        Parameters
        ----------
        hidden_size : int
            Hidden size of the LSTM outputs being attended over.
        """
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        """
        Parameters
        ----------
        lstm_out : torch.Tensor of shape (batch, seq_len, hidden_size)

        Returns
        -------
        torch.Tensor of shape (batch, hidden_size)
            Weighted context vector.
        torch.Tensor of shape (batch, seq_len)
            Attention weights, summing to 1 across seq_len for each row.
        """
        scores = self.attn(lstm_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
        return context, weights


class CNNLSTMAttentionModel(nn.Module):
    """
    Conv1D feature extractor + LSTM + attention over the LSTM's per-step
    outputs. This is the project's primary model - attention weights are
    extracted and visualized in notebook 05 to interpret which time steps
    the model relies on most for each prediction.
    """

    def __init__(
        self, n_features, cnn_channels=32, kernel_size=3, hidden_size=64, dropout=0.2
    ):
        """
        Parameters
        ----------
        n_features : int
            Number of input features per time step.
        cnn_channels : int, default 32
            Number of Conv1D output channels.
        kernel_size : int, default 3
            Conv1D kernel width.
        hidden_size : int, default 64
            LSTM hidden state size.
        dropout : float, default 0.2
            SpatialDropout1D rate applied after the convolution.
        """
        super().__init__()
        self.conv = nn.Conv1d(
            n_features, cnn_channels, kernel_size, padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.spatial_dropout = SpatialDropout1D(dropout)
        self.lstm = nn.LSTM(cnn_channels, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, return_attention=False):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, seq_len, n_features)
        return_attention : bool, default False
            If True, also return the per-step attention weights.

        Returns
        -------
        torch.Tensor of shape (batch,)
            Predicted scaled Appliances value.
        torch.Tensor of shape (batch, seq_len), optional
            Attention weights, only returned if return_attention=True.
        """
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv(x))
        x = x.permute(0, 2, 1)
        x = self.spatial_dropout(x)
        lstm_out, _ = self.lstm(x)
        context, weights = self.attention(lstm_out)
        out = self.fc(context).squeeze(-1)
        if return_attention:
            return out, weights
        return out


def get_model(name, n_features=None, **kwargs):
    """
    Single dispatcher for all 7 locked model architectures.

    Parameters
    ----------
    name : str
        One of: "linear_regression", "random_forest", "lstm", "gru",
        "cnn_lstm", "tcn", "cnn_lstm_attention".
    n_features : int, optional
        Required for the 5 deep learning models - number of input
        features per time step. Not used for the two sklearn baselines.
    **kwargs
        Forwarded to the corresponding model constructor.

    Returns
    -------
    Unfitted model instance - an sklearn estimator for the two baselines,
    or an nn.Module (not yet moved to device or trained) for the 5 deep
    learning architectures.

    Raises
    ------
    ValueError
        If name is not recognized, or if n_features is missing for a
        deep learning model.
    """
    if name == "linear_regression":
        return build_linear_regression(**kwargs)
    if name == "random_forest":
        return build_random_forest(**kwargs)
    if name in ("lstm", "gru", "cnn_lstm", "tcn", "cnn_lstm_attention"):
        if n_features is None:
            raise ValueError(f"n_features is required to build '{name}'.")
        if name == "lstm":
            return LSTMModel(n_features, **kwargs)
        if name == "gru":
            return GRUModel(n_features, **kwargs)
        if name == "cnn_lstm":
            return CNNLSTMModel(n_features, **kwargs)
        if name == "tcn":
            return TCNModel(n_features, **kwargs)
        if name == "cnn_lstm_attention":
            return CNNLSTMAttentionModel(n_features, **kwargs)
    raise ValueError(f"Unknown model name '{name}'.")
