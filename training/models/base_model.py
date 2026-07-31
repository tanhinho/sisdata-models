from abc import ABC, abstractmethod
import os
import mlflow
import torch
import torch.nn as nn

from datasets.base_dataset import BaseDataset

COMMIT_SHA = os.getenv('COMMIT_SHA', 'local-dev')


class BaseModel(nn.Module, ABC):
    """Abstract base class for models.
    Child classes should implement the `forward` and `_fit_and_evaluate_impl` methods.

    Attributes:
        NAME (str): A descriptive name for the model.
        dataset (BaseDataset): The dataset object containing training, validation, and test data.
        is_optimizing (bool): Flag indicating if the model is being optimized (True) or trained normally (False).
    """
    NAME = None  # Child classes should override this with a descriptive name.

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
            float: The evaluation metric (e.g., loss).
        """
        with mlflow.start_run(run_name=run_name, nested=True):
            run_type = "optuna_trial" if self.is_optimizing else "final_model"
            mlflow.set_tags({
                "dataset": self.dataset.NAME,
                "model": self.NAME,
                "run_type": run_type,
                "sha": COMMIT_SHA,
            })

            df_test = self.dataset.df_val if self.is_optimizing else self.dataset.df_test

            loss, params = self._fit_and_evaluate_impl(self.dataset.df_train, df_test)
            mlflow.log_params(params)
            mlflow.log_metrics({"loss": loss})
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
