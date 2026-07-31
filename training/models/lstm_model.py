import torch
import torch.nn as nn
import torch.optim as optim

from datasets.base_dataset import BaseDataset
from models.base_model import BaseModel
from .utils import create_sequences


class LSTMModel(BaseModel):
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
    ):
        super().__init__(dataset=dataset, is_optimizing=is_optimizing)

        self.seq_length = seq_length
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        input_size = len(self.dataset.FEATURE_COLS)
        output_size = 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)
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
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # Basic mini-batch training loop
        X_train, y_train = create_sequences(
            df_train,
            self.seq_length,
            self.dataset.FEATURE_COLS,
            self.dataset.TARGET_COL,
            self.device,
        )
        X_test, y_test = create_sequences(
            df_test,
            self.seq_length,
            self.dataset.FEATURE_COLS,
            self.dataset.TARGET_COL,
            self.device,
        )

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.train()
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs, _, _ = self(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}")

        # Evaluate on validation set
        self.eval()
        with torch.no_grad():
            val_preds, _, _ = self(X_test)
            val_loss = criterion(val_preds, y_test).item()

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
        return val_loss, params
