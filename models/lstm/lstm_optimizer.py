from typing import Any, Dict, Optional, Tuple
import optuna
import torch
from lstm import LSTMModel
import pandas as pd
import numpy as np


class LSTMOptimizer:
    """Optuna optimizer that generates dynamic sequence lengths per trial."""

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        input_size: int,
        feature_cols: list[str],
        target_col: str,
        n_trials: int = 15,
        seed: Optional[int] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.df_train = df_train
        self.df_val = df_val
        self.input_size = input_size
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.n_trials = n_trials
        self.seed = seed
        self.study: Optional[optuna.Study] = None

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna to minimize validation loss.

        Args:
            trial (optuna.Trial): The trial object for hyperparameter suggestions.

        Returns:
            float: The validation loss (MSE) to minimize.
        """
        # Sample hyperparameters
        seq_length = trial.suggest_int("sequence_length", 8, 64)
        hidden_size = trial.suggest_int("hidden_size", 32, 256, log=True)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        epochs = trial.suggest_int("epochs", 10, 50)

        # Create model instance
        model = LSTMModel(
            input_size=self.input_size,
            output_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
        )

        # Train and evaluate
        val_loss = model.fit_and_evaluate(
            df_train=self.df_train,
            df_test=self.df_val,
            seq_length=seq_length,
            lr=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
        )

        return val_loss

    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter search minimizing validation loss.

        Returns:
            Dict[str, Any]: Dictionary containing best parameters, best loss and study.
            The keys are `best_params`, `best_loss`, and `study`.
        """
        sampler = optuna.samplers.TPESampler(seed=self.seed) if self.seed else None
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self.study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=True)

        return {
            "best_params": self.study.best_params,
            "best_loss": self.study.best_value,
            "study": self.study,
        }
