from typing import Any, List
import optuna

from models.transformer_model import TransformerModel
from optimizers.base_optimizer import BaseOptimizer


class TransformerOptimizer(BaseOptimizer):
    """Optuna optimizer for TransformerModel (decoder-style)."""
    MODEL = TransformerModel

    def _objective_impl(self, trial: optuna.Trial) -> float:
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        # choose nhead that divides d_model
        possible_nheads: List[int] = [h for h in [1, 2, 4, 8] if d_model % h == 0]
        nhead = trial.suggest_categorical("nhead", possible_nheads)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        ff_multiplier = trial.suggest_categorical("ff_multiplier", [2, 4, 8])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        seq_length = trial.suggest_int("seq_length", 5, 15)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", 10, 50)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        model = self.MODEL(
            dataset=self.dataset,
            forecast_horizon=self.forecast_horizon,
            is_optimizing=True,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
            seq_length=seq_length,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
        )

        val_loss = model.fit_and_evaluate(run_name=f"optuna_trial_{trial.number}")
        return val_loss
