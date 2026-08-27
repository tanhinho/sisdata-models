from typing import Any
import optuna

from models.tcn_model import TCNModel
from optimizers.base_optimizer import BaseOptimizer


class TCNOptimizer(BaseOptimizer):
    """Optuna optimizer for TCNModel."""
    MODEL = TCNModel

    def _objective_impl(self, trial: optuna.Trial) -> float:
        hidden_size = trial.suggest_int("hidden_size", 16, 256, log=True)
        num_layers = trial.suggest_int("num_layers", 1, 5)
        kernel_size = trial.suggest_int("kernel_size", 2, 5)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        seq_length = trial.suggest_int("seq_length", 5, 15)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", 10, 50)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        model = self.MODEL(
            dataset=self.dataset,
            num_inputs=len(self.dataset.FEATURE_COLS),
            forecast_horizon=self.forecast_horizon,
            is_optimizing=True,
            hidden_size=hidden_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            seq_length=seq_length,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
        )

        val_loss = model.fit_and_evaluate(run_name=f"optuna_trial_{trial.number}")
        return val_loss
