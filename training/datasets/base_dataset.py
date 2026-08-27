from abc import ABC, abstractmethod
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt


class BaseDataset(ABC):
    """Abstract base class for time-series datasets.

    Child classes should define:
        FILEPATH
        TIMESTAMP_COL
        FEATURE_COLS
        TARGET_COL
        NAME
        SCALE_TARGET

    The dataset handles:
        - loading and cleaning
        - daily aggregation
        - chronological train/validation/test splitting
        - feature scaling
        - sequence creation
    """

    FILEPATH: str = ""
    FEATURE_COLS: list[str] = []
    TIMESTAMP_COL: str = ""
    TARGET_COL: str = ""
    NAME: str = None
    SCALE_TARGET: bool = True  # Whether to scale the target values or not.

    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        df = pd.read_csv(self.FILEPATH)
        df = self._remove_unused_columns(df)
        df = self._preprocess(df)
        df_train, df_val, df_test = self._split_data(df, train_ratio, val_ratio)
        # df_train, df_val, df_test = self._select_features(df_train, df_val, df_test)
        df_train, df_val, df_test = self._scale_data(df_train, df_val, df_test)
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

    def _remove_unused_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns that are not in FEATURE_COLS, TARGET_COL, or TIMESTAMP_COL.

        Args:
            df (pd.DataFrame): The DataFrame to clean.

        Returns:
                pd.DataFrame: The cleaned DataFrame.
        """
        allowed_cols = set(self.FEATURE_COLS + [self.TARGET_COL, self.TIMESTAMP_COL])
        return df.loc[:, df.columns.intersection(allowed_cols)]

    def _scale_data(self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
        """Scales the features and, optionally, the target values.

        This method fits a StandardScaler on the training set and transforms the training, validation, and test sets accordingly.
        The scaled values are stored in the respective DataFrames (df_train, df_val, df_test),
        and the scalers are stored as attributes (feature_scaler for features, target_scaler for the target).
        If SCALE_TARGET is set to False, the target values will not be scaled, and target_scaler will be set to None.

        Args:
            df_train (pd.DataFrame): The training set.
            df_val (pd.DataFrame): The validation set.
            df_test (pd.DataFrame): The test set.
        """
        # Scale features (fit ONLY on train set)
        self.feature_scaler = StandardScaler()
        df_train[self.FEATURE_COLS] = self.feature_scaler.fit_transform(
            df_train[self.FEATURE_COLS]
        )
        df_val[self.FEATURE_COLS] = self.feature_scaler.transform(
            df_val[self.FEATURE_COLS]
        )
        df_test[self.FEATURE_COLS] = self.feature_scaler.transform(
            df_test[self.FEATURE_COLS]
        )

        if not self.SCALE_TARGET:
            self.target_scaler = None
            return df_train, df_val, df_test

        # Scale the target (fit ONLY on train set)
        self.target_scaler = StandardScaler()
        df_train[self.TARGET_COL] = self.target_scaler.fit_transform(
            df_train[[self.TARGET_COL]]
        )
        df_val[self.TARGET_COL] = self.target_scaler.transform(
            df_val[[self.TARGET_COL]]
        )
        df_test[self.TARGET_COL] = self.target_scaler.transform(
            df_test[[self.TARGET_COL]]
        )

        return df_train, df_val, df_test

    def _split_data(
        self, df: pd.DataFrame, train_ratio: float, val_ratio: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs a single chronological split across the whole dataset."""
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        df_train = df.iloc[:train_end].copy()
        df_val = df.iloc[train_end:val_end].copy()
        df_test = df.iloc[val_end:].copy()

        return df_train, df_val, df_test

    def unscale_target(self, scaled_target: np.ndarray) -> np.ndarray:
        """Unscale the target values back to their original scale.

        Args:
            scaled_target (np.ndarray): The scaled target values to unscale.

        Returns:
            np.ndarray: The unscaled target values.
        """
        if not self.SCALE_TARGET:
            return scaled_target  # No scaling was applied, return as is.

        if isinstance(scaled_target, torch.Tensor):
            scaled_target = scaled_target.detach().cpu().numpy()

        shape = scaled_target.shape
        unscaled = self.target_scaler.inverse_transform(scaled_target.reshape(-1, 1))
        return unscaled.reshape(shape)

    def _select_features(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        X_train = df_train[self.FEATURE_COLS]
        y_train = df_train[self.TARGET_COL]

        X_val = df_val[self.FEATURE_COLS]
        y_val = df_val[self.TARGET_COL]

        # Train LightGBM ONLY on training data
        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

        print(f"Training LightGBM for feature importance on {self.NAME}...")

        model.fit(X_train, y_train)

        print("Training complete. Calculating permutation importance...")

        # Calculate permutation importance on training data
        result = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )

        importance = pd.DataFrame(
            {
                "importance": result.importances_mean,
                "std": result.importances_std,
            },
            index=self.FEATURE_COLS,
        ).sort_values("importance")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.barh(
            importance.index,
            importance["importance"],
            xerr=importance["std"],
        )

        ax.set_xlabel("Permutation Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance - LightGBM")
        fig.tight_layout()

        # Save instead of plt.show()
        plot_path = f"feature_importance_{self.NAME}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Feature importance plot saved to: {plot_path}")

        print("\nFeature importance:")
        print(importance)

        # Select features above threshold
        threshold = 0.01

        selected_features = importance[
            importance["importance"] > threshold
        ].index.tolist()

        print("\nSelected features:")
        print(selected_features)

        # Update FEATURE_COLS
        self.FEATURE_COLS = selected_features

        # Keep only selected features + target + date
        columns = selected_features + [
            self.TARGET_COL,
            "date",
        ]

        df_train = df_train[columns].copy()
        df_val = df_val[columns].copy()
        df_test = df_test[columns].copy()

        return df_train, df_val, df_test

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        forecast_horizon: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create sequences of input features and target values for model training.

        Args:
            df (pd.DataFrame): The DataFrame containing the dataset to create sequences from.
            sequence_length (int): The length of each sequence.
            forecast_horizon (int): The number of time steps to forecast.
            device (torch.device, optional): The device to move the tensors to. Defaults to torch.device("cpu").

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input feature sequences and target value sequences as PyTorch tensors.
        """
        df = df.sort_values("date").reset_index(drop=True)

        features = df[self.FEATURE_COLS].values
        targets = df[self.TARGET_COL].values

        xs, ys = [], []
        total_window = sequence_length + forecast_horizon

        # Slide window step-by-step
        for i in range(len(df) - total_window + 1):
            x_win = features[i: i + sequence_length]
            y_win = targets[i + sequence_length: i + total_window]

            xs.append(x_win)
            ys.append(y_win)

        if not xs:
            raise ValueError("No sequences created. Check dataset size or window parameters.")

        X = torch.tensor(np.array(xs), dtype=torch.float32, device=device)
        y = torch.tensor(np.array(ys), dtype=torch.float32, device=device)

        return X, y

    @abstractmethod
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataset.

        This method should be implemented by child classes to handle dataset-specific preprocessing
        steps, such as filtering, renaming columns, or handling missing values.

        Args:
            df: Raw DataFrame loaded from the dataset's CSV file.

        Returns:
            pd.DataFrame: Preprocessed DataFrame ready for further processing.
        """
        pass
