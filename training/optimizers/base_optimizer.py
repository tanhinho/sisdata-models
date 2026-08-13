from abc import ABC
from typing import Any, Optional
import optuna
import torch

from datasets.base_dataset import BaseDataset


class BaseOptimizer(ABC):
    """Abstract base class for hyperparameter optuna optimizers.

    Attributes:
        dataset (BaseDataset): The dataset object containing training, validation, and test data.
        n_trials (int): The number of optimization trials.
        seed (Optional[int]): The random seed for reproducibility.
    """
    MODEL = None  # Child classes must override this with the model class to optimize.

    def __init__(
        self,
        dataset: BaseDataset,
        forecast_horizon: int = 3,
        n_trials: int = 30,
        seed: Optional[int] = None,
    ):
        self.forecast_horizon = forecast_horizon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.n_trials = n_trials
        self.seed = seed
        self.study: Optional[optuna.Study] = None

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna to optimize. 

        Args:
            trial (optuna.Trial): The trial object for hyperparameter suggestions.

        Returns:
            float: The validation loss (MSE) to minimize.
        """
        val_loss = self._objective_impl(trial)
        return val_loss

    def _objective_impl(self, trial: optuna.Trial) -> Any:
        """Implementation of the objective function.
        Child classes should implement this method with the specific logic for training and evaluating the model.

        Args:
            trial (optuna.Trial): The trial object for hyperparameter suggestions.

        Returns:
            Any: The result of the objective function.
        """
        pass

    def optimize(self) -> optuna.Study:
        """Run hyperparameter search minimizing validation loss.

        Returns:
            Study: The Optuna study object containing the optimization results.
        """
        sampler = optuna.samplers.TPESampler(seed=self.seed) if self.seed else None
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self.study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=True)

        return self.study
