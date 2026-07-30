from abc import ABC
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler


class BaseDataset(ABC):
    """Abstract base class for datasets.
    Child classes should only define the FILEPATH, FEATURE_COLS, and TARGET_COL class attributes.

    Attributes:
        FILEPATH (str): The path to the dataset CSV file.
        FEATURE_COLS (list[str]): The list of feature column names.
        TARGET_COL (str): The name of the target column.
    """
    FILEPATH: str = ""
    FEATURE_COLS: list[str] = []
    TARGET_COL: str = ""

    def __init__(self):
        self.df_train, self.df_val, self.df_test = self._load_and_preprocess()

    def _load_and_preprocess(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads, sorts, splits, and scales the dataset.

        Args:
            train_ratio (float): Proportion of the dataset to include in the train split.
            val_ratio (float): Proportion of the dataset to include in the validation split.

        The test split will be the remaining portion of the dataset after train and validation splits.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The train, validation, and test DataFrames.
        """
        # Load and sort
        df = pd.read_csv(self.FILEPATH)
        # The dataset only has 56 nan rows, so might as well just drop them for simplicity.
        # If there were more, we could consider other strategies.
        df = df.dropna()
        df["created_at"] = pd.to_datetime(df["created_at"].str.replace(" CET", "", regex=False))

        df = df.sort_values("created_at").reset_index(drop=True)

        # Sequential split
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        df_train = df.iloc[:train_end].copy()
        df_val = df.iloc[train_end:val_end].copy()
        df_test = df.iloc[val_end:].copy()

        # Fit scaler ONLY on train set to prevent data leakage
        scaler = StandardScaler()
        df_train[self.FEATURE_COLS] = scaler.fit_transform(df_train[self.FEATURE_COLS])
        df_val[self.FEATURE_COLS] = scaler.transform(df_val[self.FEATURE_COLS])
        df_test[self.FEATURE_COLS] = scaler.transform(df_test[self.FEATURE_COLS])

        return df_train, df_val, df_test
