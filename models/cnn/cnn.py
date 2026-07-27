import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple


class CNNModel(nn.Module):
    """1D-CNN Architecture for time-series regression."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_filters: int = 64,
        kernel_size: int = 3,
        num_conv_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size

        layers = []
        in_channels = input_size

        # Stack 1D Convolutional blocks
        for _ in range(num_conv_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    padding="same",  # Preserves sequence length dimension
                )
            )
            layers.append(nn.BatchNorm1d(num_filters))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_channels = num_filters

        self.conv_network = nn.Sequential(*layers)

        # Global Average Pooling compresses temporal dimension safely regardless of seq_length
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, seq_length, num_features)
        # Conv1d expects: (batch_size, num_features, seq_length)
        x = x.transpose(1, 2)

        x = self.conv_network(x)
        x = self.global_pool(x)  # Shape: (batch_size, num_filters, 1)
        x = torch.flatten(x, 1)   # Shape: (batch_size, num_filters)
        out = self.fc(x)          # Shape: (batch_size, output_size)

        return out

    def fit_and_evaluate(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        lr: float,
        epochs: int,
        batch_size: int,
        device: str = "cpu",
    ) -> float:
        """Train on train split and return validation MSE loss."""
        self.to(device)
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_val, y_val = X_val.to(device), y_val.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            self.train()
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        self.eval()
        with torch.no_grad():
            val_preds = self(X_val)
            val_loss = criterion(val_preds, y_val).item()

        return val_loss
