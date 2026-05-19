from abc import ABC, abstractmethod
from typing import Tuple, Any
import pandas as pd
from sklearn.pipeline import Pipeline


class BasePreprocessor(ABC):
    """Abstract base class for dataset preprocessors."""

    def __init__(self):
        self.preprocessor: Pipeline = None
        self.data: pd.DataFrame = None

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load the raw dataset.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        pass

    @abstractmethod
    def build_preprocessor(self) -> Pipeline:
        """Build the preprocessing pipeline.

        Returns:
            Pipeline: A sklearn Pipeline with preprocessing steps.
        """
        pass

    def fit(self, X: pd.DataFrame) -> "BasePreprocessor":
        """Fit the preprocessor on the data.

        Args:
            X: Training data.

        Returns:
            Self for method chaining.
        """
        self.preprocessor.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to data.

        Args:
            X: Data to preprocess.

        Returns:
            pd.DataFrame: Preprocessed data.
        """
        return self.preprocessor.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Data to fit and transform.

        Returns:
            pd.DataFrame: Fitted and transformed data.
        """
        return self.preprocessor.fit_transform(X)

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
        """Handle missing values.

        Args:
            df: DataFrame with potential missing values.
            strategy: "drop" or "mean".

        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        match strategy:
            case "drop":
                return df.dropna()
            case "mean":
                return df.fillna(df.mean())
            case "median":
                return df.fillna(df.median())
            case "mode":
                # Mode can return multiple values, so we take the first one
                return df.fillna(df.mode().iloc[0])
            case _:
                raise ValueError(f"Unknown strategy: {strategy}")
        return df
