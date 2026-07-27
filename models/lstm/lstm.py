import torch
import torch.nn as nn
import torch.optim as optim


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
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

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.lstm.num_layers, x.size(
                0), self.lstm.hidden_size, device=self.device)
            c0 = torch.zeros(self.lstm.num_layers, x.size(
                0), self.lstm.hidden_size, device=self.device)

        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last time step
        return out, h_n, c_n

    def fit_and_evaluate(self, X_train, y_train, X_val, y_val, lr=0.01, epochs=100, batch_size=32) -> float:
        """Train model and return evaluation metric for Optuna.

        Args:
            X_train (torch.Tensor): Training features.
            y_train (torch.Tensor): Training targets.
            X_val (torch.Tensor): Validation features.
            y_val (torch.Tensor): Validation targets.
            lr (float): Learning rate.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each mini-batch.

        Returns:
                float: The validation loss (MSE).
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Basic mini-batch training loop
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
            val_preds, _, _ = self(X_val)
            val_loss = criterion(val_preds, y_val).item()

        # Return the MSE loss for Optuna to minimize
        return val_loss
