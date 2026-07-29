import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from typing import Tuple
import numpy as np

from datasets import BaseDataset


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        feature_cols: list[str] = None,
        target_col: str = None,
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.to(self.device)

    def _create_sequences(self, df: pd.DataFrame, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert pandas DataFrame to sequence tensors.

        Args:
            df (pd.DataFrame): Input dataframe.
            seq_length (int): Length of the sequences to create.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Feature and target tensors.
        """
        xs, ys = [], []

        features = df[self.feature_cols].values
        targets = df[self.target_col].values

        for i in range(len(df) - seq_length):
            xs.append(features[i: i + seq_length])
            ys.append([targets[i + seq_length]])  # Enclosing in [] keeps shape as (N, 1)

        X_tensor = torch.tensor(np.array(xs), dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(np.array(ys), dtype=torch.float32, device=self.device)

        return X_tensor, y_tensor

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.lstm.num_layers, x.size(
                0), self.lstm.hidden_size, device=self.device)
            c0 = torch.zeros(self.lstm.num_layers, x.size(
                0), self.lstm.hidden_size, device=self.device)

        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last time step
        return out, h_n, c_n

    def fit_and_evaluate(self, df_train, df_test, seq_length, lr=0.01, epochs=100, batch_size=32) -> float:
        """Train model and return evaluation metric.

        Args:
            df_train (pd.DataFrame): Training dataframe.
            df_test (pd.DataFrame): Test dataframe.
            seq_length (int): Length of the sequences to create.
            lr (float): Learning rate.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each mini-batch.

        Returns:
                float: The validation loss (MSE).
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Basic mini-batch training loop
        X_train, y_train = self._create_sequences(df_train, seq_length)
        X_test, y_test = self._create_sequences(df_test, seq_length)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs, _, _ = self(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        # Evaluate on validation set
        self.eval()
        with torch.no_grad():
            val_preds, _, _ = self(X_test)
            val_loss = criterion(val_preds, y_test).item()

        # Return the MSE loss
        return val_loss
