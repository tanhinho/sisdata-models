import mlflow
import numpy as np
from typing import Dict, Tuple, Optional

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from datasets.base_dataset import BaseDataset
from models.base_model import BaseModel


class XGBoostModel(BaseModel):
    NAME = "xgboost"

    def __init__(
        self,
        dataset: BaseDataset,
        is_optimizing: bool = False,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        seq_length: int = 32,
        forecast_horizon: int = 3,
    ):
        super().__init__(
            dataset=dataset,
            is_optimizing=is_optimizing,
            forecast_horizon=forecast_horizon,
        )

        self.seq_length = seq_length
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        # Base XGBoost Regressor
        base_xgb = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=42,
            n_jobs=-1,
        )

        # Wrap in MultiOutputRegressor to support multi-horizon targets (forecast_horizon > 1)
        if self.forecast_horizon > 1:
            self.model = MultiOutputRegressor(base_xgb)
        else:
            self.model = base_xgb

    def forward(self, x):
        # x: (batch, seq_len, features)
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x_flat = x.reshape(x.shape[0], -1)
        preds = self.model.predict(x_flat)
        return preds, None, None

    def _fit_and_evaluate_impl(
        self, df_train, df_test
    ) -> Tuple[float, Dict[str, object]]:
        # Prepare data using dataset helper
        X_train, y_train = self.dataset.create_sequences(
            df_train, self.seq_length, self.forecast_horizon, self.device
        )
        X_test, y_test = self.dataset.create_sequences(
            df_test, self.seq_length, self.forecast_horizon, self.device
        )

        # Reshape inputs to 2D matrices: (n_samples, seq_length * n_features)
        X_train_np = X_train.cpu().numpy().reshape(X_train.shape[0], -1)
        X_test_np = X_test.cpu().numpy().reshape(X_test.shape[0], -1)

        # Reshape targets to 2D matrices safely: (n_samples, forecast_horizon)
        y_train_np = y_train.cpu().numpy().reshape(y_train.shape[0], -1)
        y_test_np = y_test.cpu().numpy().reshape(y_test.shape[0], -1)

        # Fit XGBoost model
        # For single horizon targets, flatten y to 1D to prevent XGBoost warnings
        if self.forecast_horizon == 1:
            self.model.fit(X_train_np, y_train_np.ravel())
        else:
            self.model.fit(X_train_np, y_train_np)

        # Predict
        preds_scaled = self.model.predict(X_test_np)

        # Ensure prediction shape matches (n_samples, forecast_horizon)
        if preds_scaled.ndim == 1:
            preds_scaled = preds_scaled.reshape(-1, 1)

        val_mse = float(mean_squared_error(y_test_np, preds_scaled))

        preds_g = self.dataset.unscale_target(preds_scaled)
        targets_g = self.dataset.unscale_target(y_test_np)

        val_mae = float(mean_absolute_error(targets_g, preds_g))
        ss_res = np.sum((targets_g - preds_g) ** 2)
        ss_tot = np.sum((targets_g - np.mean(targets_g)) ** 2)
        val_r2 = float(1 - (ss_res / (ss_tot + 1e-8)))

        mlflow.log_metrics(
            {
                "val_mse": val_mse,
                "val_mae_grams": val_mae,
                "val_r2": val_r2,
            }
        )

        print(
            f"XGB Evaluation | Scaled MSE: {val_mse:.4f} | Gram MAE: {val_mae:.2f}g | R²: {val_r2:.4f}"
        )
        params = {
            "seq_length": self.seq_length,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
        }

        return val_mse, params
