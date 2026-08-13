from typing import Any
import optuna

from models.random_forest_model import RandomForestModel
from optimizers.base_optimizer import BaseOptimizer


class RandomForestOptimizer(BaseOptimizer):
    """Optuna optimizer for RandomForestModel."""
    MODEL = RandomForestModel

    def _objective_impl(self, trial: optuna.Trial) -> float:
        # hyperparameter search space
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 30)
        seq_length = trial.suggest_int("seq_length", 5, 15)

        model = self.MODEL(
            dataset=self.dataset,
            forecast_horizon=self.forecast_horizon,
            is_optimizing=True,
            n_estimators=n_estimators,
            max_depth=max_depth,
            seq_length=seq_length,
        )

        val_loss = model.fit_and_evaluate(run_name=f"optuna_trial_{trial.number}")
        return val_loss
