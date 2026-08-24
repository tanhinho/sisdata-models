import math
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple

from datasets.base_dataset import BaseDataset
from models.base_model import BaseModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return x


class TransformerModel(BaseModel):
    NAME = "transformer_decoder"

    def __init__(
        self,
        dataset: BaseDataset,
        is_optimizing: bool = False,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
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
        self.num_layers = num_layers

        self.d_model = d_model
        input_size = len(self.dataset.FEATURE_COLS)

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_multiplier * d_model,
            dropout=dropout,
            batch_first=False,
        )

        # map decoder outputs (per time-step) to a single scalar prediction
        self.output_proj = nn.Linear(d_model, 1)
        self.to(self.device)

    def forward(self, x, *_):
        # x: (batch, seq_len, features)
        batch = x.size(0)
        src = self.input_proj(x)  # (batch, seq_len, d_model)
        src = src.permute(1, 0, 2)  # (seq_len, batch, d_model)
        src = self.pos_enc(src)

        # prepare tgt as zeros for forecast horizon
        tgt_len = self.forecast_horizon
        tgt = torch.zeros(tgt_len, batch, self.d_model, device=self.device)
        tgt = self.pos_enc(tgt)

        memory = self.transformer.encoder(src)
        out = self.transformer.decoder(tgt, memory)
        # out: (tgt_len, batch, d_model)
        out = self.output_proj(out)  # (tgt_len, batch, 1)
        out = out.squeeze(-1).permute(1, 0)  # (batch, tgt_len)
        return out, None, None

    def _fit_and_evaluate_impl(self, df_train, df_test) -> tuple[float, dict]:
        criterion_mse = nn.MSELoss()
        criterion_mae = nn.L1Loss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        X_train, y_train = self.dataset.create_sequences(
            df_train, self.seq_length, self.forecast_horizon, self.device)
        X_test, y_test = self.dataset.create_sequences(
            df_test, self.seq_length, self.forecast_horizon, self.device)

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

            train_mse = loss.item()
            mlflow.log_metrics(
                {
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "val_mae_grams": val_mae
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch+1:03d}/{self.epochs:03d} | "
                f"Train MSE: {train_mse:.4f} | "
                f"Val MSE: {val_mse:.4f} | "
                f"Val MAE (grams): {val_mae:.4f}"
            )

        params = {
            "seq_length": self.seq_length,
            "learning_rate": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
        }
        return val_mse, params
