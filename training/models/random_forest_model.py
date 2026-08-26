import mlflow
import numpy as np
from typing import Dict, Tuple, Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

        # Reshape tensors to 2D for sklearn
        X_train_np = X_train.cpu().numpy().reshape(X_train.shape[0], -1)
        y_train_np = y_train.cpu().numpy().reshape(y_train.shape[0], -1)
        X_test_np = X_test.cpu().numpy().reshape(X_test.shape[0], -1)
        y_test_np = y_test.cpu().numpy().reshape(y_test.shape[0], -1)

        # Fit sklearn model
        self.model.fit(X_train_np, y_train_np)

        # Predict and calculate MSE and unscaled MAE
        preds_scaled = self.model.predict(X_test_np)
        val_mse = float(mean_squared_error(y_test_np, preds_scaled))

        preds_g = self.dataset.unscale_target(preds_scaled)
        targets_g = self.dataset.unscale_target(y_test_np)
        val_mae = float(mean_absolute_error(targets_g, preds_g))
        ss_res = np.sum((targets_g - preds_g) ** 2)
        ss_tot = np.sum((targets_g - np.mean(targets_g)) ** 2)
        val_r2 = float(1 - (ss_res / (ss_tot + 1e-8)))
        mlflow.log_metrics({
            "val_mse": val_mse,
            "val_mae_grams": val_mae,
            "val_r2": val_r2
        })

        print(
            f"RF Evaluation | Scaled MSE: {val_mse:.4f} | Gram MAE: {val_mae:.2f}g | R²: {val_r2:.4f}")
        params = {
            "seq_length": self.seq_length,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
        }

        return val_mse, params
