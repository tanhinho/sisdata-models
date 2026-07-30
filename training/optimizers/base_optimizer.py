from abc import ABC
from typing import Any, Dict, Optional
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
        n_trials: int = 15,
        seed: Optional[int] = None,
    ):
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
        pass

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
