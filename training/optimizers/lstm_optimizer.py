from typing import Any, Dict, Tuple
import optuna
from models import LSTMModel
from optimizers import BaseOptimizer


class LSTMOptimizer(BaseOptimizer):
    """Optuna optimizer that finds the best hyperparameters for an LSTM model."""

    def _objective_impl(self, trial: optuna.Trial) -> Tuple[float, Dict[str, Any]]:
        # Sample hyperparameters
        seq_length = trial.suggest_int("sequence_length", 8, 64)
        hidden_size = trial.suggest_int("hidden_size", 32, 256, log=True)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        epochs = trial.suggest_int("epochs", 10, 50)

        input_size = self.dataset.df_train[self.dataset.FEATURE_COLS].shape[1]

        # Create model instance
        model = LSTMModel(
            dataset=self.dataset,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            seq_length=seq_length,
            lr=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
        )

        # Train and evaluate
        val_loss = model.fit_and_evaluate(
            run_name=f"trial_{trial.number}",
            df_train=self.dataset.df_train,
            df_test=self.dataset.df_val,
            seq_length=seq_length,
            lr=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
        )

        params = {
            "sequence_length": seq_length,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs
        }

        return val_loss, params
