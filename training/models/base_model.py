from abc import ABC, abstractmethod

import mlflow
from datasets import BaseDataset

import torch
import torch.nn as nn

from optimizers import BaseOptimizer


class BaseModel(nn.Module, ABC):
    """Abstract base class for models.
    Child classes should implement the `forward` and `_fit_and_evaluate_impl` methods.

    Attributes:
        dataset (BaseDataset): The dataset object containing training, validation, and test data.
        is_optimizing (bool): Flag indicating if the model is being optimized (True) or trained normally (False).
    """

    # Default optimizer for the model. Child classes must override this.
    OPTIMIZER: BaseOptimizer = None

    def __init__(
        self,
        dataset: BaseDataset,
        is_optimizing: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.is_optimizing = is_optimizing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def forward(self, x):
        """Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        pass

    def fit_and_evaluate(self, run_name: str = "fit_and_evaluate") -> float:
        """Train model and return evaluation metric.
        Args:
            **kwargs: Additional arguments for training and evaluation.
        Returns:
            float: The evaluation metric (e.g., validation loss).
        """
        with mlflow.start_run(run_name=run_name, nested=True):
            df_test = self.dataset.df_val if self.is_optimizing else self.dataset.df_test

            loss, params = self._fit_and_evaluate_impl(self.dataset.df_train, df_test)
            mlflow.log_params(params)
            mlflow.log_metrics({"validation_loss": loss})
            return loss

    @abstractmethod
    def _fit_and_evaluate_impl(self, df_train, df_test) -> tuple[float, dict]:
        """Implementation of the fit and evaluate method.
        Child classes should implement this method to define the training and evaluation logic.

        Args:
            df_train (pd.DataFrame): Training data.
            df_test (pd.DataFrame): Test data.

        Returns:
            tuple[float, dict]: A tuple containing the evaluation metric (e.g., validation loss) and a dictionary of parameters to log.
        """
        pass
