import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from datasets.base_dataset import BaseDataset
from models.base_model import BaseModel


class LSTMModel(BaseModel):
    NAME = "lstm"

    def __init__(
        self,
        dataset: BaseDataset,
        is_optimizing: bool = False,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        seq_length: int = 32,
        lr: float = 0.01,
        epochs: int = 100,
        batch_size: int = 32,
        forecast_horizon: int = 3,
    ):
        super().__init__(dataset=dataset, is_optimizing=is_optimizing, forecast_horizon=forecast_horizon)

        self.seq_length = seq_length
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        input_size = len(self.dataset.FEATURE_COLS)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, self.forecast_horizon)
        self.to(self.device)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.lstm.num_layers, x.size(
                0), self.lstm.hidden_size, device=self.device)
            c0 = torch.zeros(self.lstm.num_layers, x.size(
                0), self.lstm.hidden_size, device=self.device)

        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last time step
        return out, h_n, c_n

    def _fit_and_evaluate_impl(self, df_train, df_test) -> tuple[float, dict]:
        criterion_mse = nn.MSELoss()
        criterion_mae = nn.L1Loss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # Basic mini-batch training loop
        X_train, y_train = self.dataset.create_sequences(
            df_train,
            self.seq_length,
            self.forecast_horizon,
            self.device,
        )

        X_test, y_test = self.dataset.create_sequences(
            df_test,
            self.seq_length,
            self.forecast_horizon,
            self.device,
        )

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

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

        # Return the MSE loss and the parameters used for training
        params = {
            "seq_length": self.seq_length,
            "learning_rate": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "hidden_size": self.lstm.hidden_size,
            "num_layers": self.lstm.num_layers,
            "dropout": self.lstm.dropout if self.lstm.num_layers > 1 else 0.0,
        }
        return val_mse, params
