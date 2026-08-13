import numpy as np
from typing import Dict, Tuple, Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from datasets.base_dataset import BaseDataset
from models.base_model import BaseModel


class RandomForestModel(BaseModel):
    NAME = "random_forest"

    def __init__(
        self,
        dataset: BaseDataset,
        is_optimizing: bool = False,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        seq_length: int = 32,
        forecast_horizon: int = 3,
    ):
        super().__init__(dataset=dataset, is_optimizing=is_optimizing, forecast_horizon=forecast_horizon)

        self.seq_length = seq_length
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        self.model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth)

    def forward(self, x):
        # x: (batch, seq_len, features)
        # RandomForest predicts on flattened sequence
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x_flat = x.reshape(x.shape[0], -1)
        preds = self.model.predict(x_flat)
        return preds, None, None

    def _fit_and_evaluate_impl(self, df_train, df_test) -> Tuple[float, Dict[str, object]]:
        # Prepare data using dataset helper
        X_train, y_train = self.dataset.create_sequences(
            df_train, self.seq_length, self.forecast_horizon, self.device
        )
        X_test, y_test = self.dataset.create_sequences(
            df_test, self.seq_length, self.forecast_horizon, self.device
        )

        X_train_np = X_train.cpu().numpy().reshape(X_train.shape[0], -1)
        y_train_np = y_train.cpu().numpy()
        X_test_np = X_test.cpu().numpy().reshape(X_test.shape[0], -1)
        y_test_np = y_test.cpu().numpy()

        # Fit sklearn model
        self.model.fit(X_train_np, y_train_np)

        preds = self.model.predict(X_test_np)
        val_loss = float(mean_squared_error(y_test_np, preds))

        params = {
            "seq_length": self.seq_length,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
        }

        return val_loss, params
