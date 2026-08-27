from typing import Any, Dict, Tuple
import optuna

from models.xgboost_model import XGBoostModel
from optimizers.base_optimizer import BaseOptimizer


class XGBoostOptimizer(BaseOptimizer):
    """Optuna optimizer that finds the best hyperparameters for an XGBoost model."""
    MODEL = XGBoostModel

    def _objective_impl(self, trial: optuna.Trial) -> Tuple[float, Dict[str, Any]]:
        # Sample hyperparameters tailored for XGBoost
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
        max_depth = trial.suggest_int("max_depth", 3, 12)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        seq_length = trial.suggest_int("seq_length", 5, 15)

        # Create model instance
        model = self.MODEL(
            dataset=self.dataset,
            forecast_horizon=self.forecast_horizon,
            is_optimizing=True,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            seq_length=seq_length,
        )

        # Train and evaluate
        val_loss = model.fit_and_evaluate(run_name=f"optuna_trial_{trial.number}")

        return val_loss
