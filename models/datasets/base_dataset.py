from abc import ABC
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler


class BaseDataset(ABC):
    FILEPATH: str
    FEATURE_COLS: list[str]
    TARGET_COL: str

    @classmethod
    def load_and_preprocess(
        cls,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and preprocess the dataset."""
        """Loads, sorts, splits, and scales the dataset cleanly in one function."""
        # Load and sort
        df = pd.read_csv(cls.FILEPATH)
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
        df_train[cls.FEATURE_COLS] = scaler.fit_transform(
            df_train[cls.FEATURE_COLS])

        df_val[cls.FEATURE_COLS] = scaler.transform(
            df_val[cls.FEATURE_COLS])
        df_test[cls.FEATURE_COLS] = scaler.transform(
            df_test[cls.FEATURE_COLS])

        return df_train, df_val, df_test
