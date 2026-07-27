from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


class DatasetAPreprocessor:
    """Class to handle loading and preprocessing of Dataset A."""
    FEATURE_COLS = [
        "Temperature (C)",
        "Turbidity(NTU)",
        "Dissolved Oxygen(g/ml)",
        "PH",
        "Ammonia(g/ml)",
        "Nitrate(g/ml)",
        "Population",
        "Fish_Length(cm)",
    ]

    TARGET_COL = "Fish_Weight(g)"

    @staticmethod
    def load_and_preprocess(
        filepath: str = "data/IoTpond1.csv",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
        """Loads, sorts, splits, and scales the dataset cleanly in one function."""
        # Load and sort
        df = pd.read_csv(filepath)
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
        df_train[DatasetAPreprocessor.FEATURE_COLS] = scaler.fit_transform(
            df_train[DatasetAPreprocessor.FEATURE_COLS])

        df_val[DatasetAPreprocessor.FEATURE_COLS] = scaler.transform(
            df_val[DatasetAPreprocessor.FEATURE_COLS])
        df_test[DatasetAPreprocessor.FEATURE_COLS] = scaler.transform(
            df_test[DatasetAPreprocessor.FEATURE_COLS])

        return df_train, df_val, df_test
