from typing import Any, Dict, Tuple
import optuna

from models.lstm_model import LSTMModel
from optimizers.base_optimizer import BaseOptimizer


class LSTMOptimizer(BaseOptimizer):
    """Optuna optimizer that finds the best hyperparameters for an LSTM model."""
    MODEL = LSTMModel

    def _objective_impl(self, trial: optuna.Trial) -> Tuple[float, Dict[str, Any]]:
        # Sample hyperparameters
        hidden_size = trial.suggest_int("hidden_size", 32, 256, log=True)
        num_layers = trial.suggest_int("num_layers", 1, 5)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        seq_length = trial.suggest_int("seq_length", 5, 15)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", 10, 50)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        # Create model instance
        model = self.MODEL(
            dataset=self.dataset,
            forecast_horizon=self.forecast_horizon,
            is_optimizing=True,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            seq_length=seq_length,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
        )

        # Train and evaluate
        val_loss = model.fit_and_evaluate(run_name=f"optuna_trial_{trial.number}")

        return val_loss
