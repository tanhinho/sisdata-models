import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List

from datasets.base_dataset import BaseDataset
from models.base_model import BaseModel


class TCNPyFuncWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = TCNModel(None)
        self.model.load_state_dict(torch.load(context.artifacts["weights"]))

    def predict(self, model_input):
        return self.model(model_input)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels,
                                    1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels: List[int], kernel_size=2, dropout=0.0):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                          dilation=dilation_size, padding=padding, dropout=dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(BaseModel):
    NAME = "tcn"

    def __init__(
        self,
        dataset: BaseDataset,
        num_inputs: int,
        is_optimizing: bool = False,
        hidden_size: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.0,
        seq_length: int = 32,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        forecast_horizon: int = 3,
    ):
        super().__init__(dataset=dataset, is_optimizing=is_optimizing, forecast_horizon=forecast_horizon)

        self.seq_length = seq_length
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        num_channels = [hidden_size] * num_layers

        self.tcn = TemporalConvNet(num_inputs, num_channels,
                                   kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(hidden_size, self.forecast_horizon)
        self.to(self.device)

    def forward(self, x, *_):
        # x: (batch, seq_len, features) -> TCN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        # take last time-step features
        last = out[:, :, -1]
        out = self.fc(last)
        return out, None, None

    def _fit_and_evaluate_impl(self, df_train, df_test) -> tuple[float, dict]:
        criterion_mse = nn.MSELoss()
        criterion_mae = nn.L1Loss()

        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        X_train, y_train = self.dataset.create_sequences(
            df_train, self.seq_length, self.forecast_horizon, self.device
        )
        X_test, y_test = self.dataset.create_sequences(
            df_test, self.seq_length, self.forecast_horizon, self.device
        )

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.epochs):
            self.train()
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs, _, _ = self(x_batch)
                loss = criterion_mse(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Evaluate on validation set
            self.eval()
            with torch.no_grad():
                val_preds_scaled, _, _ = self(X_test)

                # Scaled MSE
                val_mse = criterion_mse(val_preds_scaled, y_test).item()

                # Real Physical Gram MAE using L1Loss
                val_preds_g = torch.tensor(
                    self.dataset.unscale_target(val_preds_scaled), device=self.device
                )
                val_targets_g = torch.tensor(
                    self.dataset.unscale_target(y_test), device=self.device
                )
                val_mae = criterion_mae(val_preds_g, val_targets_g).item()

                ss_res = torch.sum((val_targets_g - val_preds_g) ** 2)
                ss_tot = torch.sum((val_targets_g - torch.mean(val_targets_g)) ** 2)
                val_r2 = (1 - (ss_res / (ss_tot + 1e-8))).item()

            train_mse = loss.item()
            mlflow.log_metrics(
                {
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "val_mae_grams": val_mae,
                    "val_r2": val_r2,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch+1:03d}/{self.epochs:03d} | "
                f"Train MSE: {train_mse:.4f} | "
                f"Val MSE: {val_mse:.4f} | "
                f"Val MAE (grams): {val_mae:.4f} | "
                f"Val R²: {val_r2:.4f}"
            )

        params = {
            "seq_length": self.seq_length,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "hidden_size": self.tcn.network[-1].net[0].in_channels if hasattr(self.tcn.network[-1], 'net') else None,
            "num_layers": len(self.tcn.network),
        }
        return val_mse, params
