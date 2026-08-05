from abc import ABC
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch


class BaseDataset(ABC):
    """Abstract base class for time-series datasets.

    Child classes should only define:
        FILEPATH
        TIMESTAMP_COL
        FEATURE_COLS
        TARGET_COL
        NAME

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

    # Columns that should be averaged when aggregating to daily data.
    MEAN_COLS: list[str] = []

    # Columns for which the last observation of the day should be used.
    LAST_COLS: list[str] = []

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ):
        (
            self.df_train,
            self.df_val,
            self.df_test,
            self.scaler,
        ) = self._load_and_preprocess(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

    def _load_and_preprocess(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        StandardScaler,
    ]:
        """Load, aggregate, split and scale the dataset.

        Args:
            train_ratio: Proportion of data to use for training.
            val_ratio: Proportion of data to use for validation.

            The test ratio is implicitly 1 - train_ratio - val_ratio.
        Returns:
            Tuple of (train_df, val_df, test_df, scaler).
        """
        df = pd.read_csv(self.FILEPATH).dropna(subset=[self.TIMESTAMP_COL])

        # Clean timestamps
        timestamp_series = (
            df[self.TIMESTAMP_COL]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+(UTC|CET|CEST|GMT)$", "", regex=True)
            .str.replace(r"\s+[+-]\d{2}:\d{2}$", "", regex=True)
            .str.replace("T", " ", regex=False)
        )
        df[self.TIMESTAMP_COL] = pd.to_datetime(timestamp_series, errors="coerce")
        df = df.dropna(subset=[self.TIMESTAMP_COL]).sort_values(self.TIMESTAMP_COL)

        # Daily aggregation (including pond column)
        df["date"] = df[self.TIMESTAMP_COL].dt.floor("D")

        # Group by pond and date for mean/last aggregations
        daily_mean = df.groupby(["pond", "date"])[self.MEAN_COLS].mean()
        daily_last = df.groupby(["pond", "date"])[self.LAST_COLS].last()
        df = pd.concat([daily_mean, daily_last], axis=1).reset_index()

        # --- PROCESS EACH POND SEPARATELY ---
        processed_ponds = []
        all_cols = list(set(self.FEATURE_COLS + [self.TARGET_COL]))

        for pond_id, group in df.groupby("pond"):
            group = group.sort_values("date")

            # Reindex full date range for THIS specific pond
            full_date_range = pd.date_range(
                start=group["date"].min(),
                end=group["date"].max(),
                freq="D",
            )
            group = group.set_index("date").reindex(full_date_range)

            # Interpolate short gaps within this pond only
            group[all_cols] = group[all_cols].interpolate(method="linear", limit=3)
            group[all_cols] = group[all_cols].ffill().bfill()

            group["pond"] = pond_id
            group = group.reset_index(names="date")
            processed_ponds.append(group)

        df_processed = pd.concat(processed_ponds, ignore_index=True)

        # --- CHRONOLOGICAL SPLIT PER POND ---
        train_dfs, val_dfs, test_dfs = [], [], []

        for pond_id, group in df_processed.groupby("pond"):
            group = group.sort_values("date").reset_index(drop=True)
            n = len(group)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train_dfs.append(group.iloc[:train_end])
            val_dfs.append(group.iloc[train_end:val_end])
            test_dfs.append(group.iloc[val_end:])

        df_train = pd.concat(train_dfs, ignore_index=True)
        df_val = pd.concat(val_dfs, ignore_index=True)
        df_test = pd.concat(test_dfs, ignore_index=True)

        # Scale features
        scaler = StandardScaler()
        df_train[self.FEATURE_COLS] = scaler.fit_transform(df_train[self.FEATURE_COLS])
        df_val[self.FEATURE_COLS] = scaler.transform(df_val[self.FEATURE_COLS])
        df_test[self.FEATURE_COLS] = scaler.transform(df_test[self.FEATURE_COLS])

        return df_train, df_val, df_test, scaler

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        forecast_horizon: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create consecutive daily sequences for time-series forecasting.

        Args:
            df: DataFrame to create sequences from.
            sequence_length: Number of past days used as input.
            forecast_horizon: Number of future days to predict.
            device: Torch device.

        Returns:
            Tuple of (X, y) tensors, where X has shape (num_sequences, sequence_length, num_features)
            and y has shape (num_sequences, forecast_horizon).
        """
        xs, ys = [], []
        total_window = sequence_length + forecast_horizon
        one_day = np.timedelta64(1, "D")

        # Group by pond so sequence windows NEVER mix across ponds
        for pond_id, group in df.groupby("pond"):
            group = group.sort_values("date").reset_index(drop=True)

            features = group[self.FEATURE_COLS].values
            targets = group[self.TARGET_COL].values
            dates = pd.to_datetime(group["date"]).values

            for i in range(len(group) - total_window + 1):
                input_start = i
                input_end = i + sequence_length
                forecast_end = input_end + forecast_horizon

                input_features = features[input_start:input_end]
                forecast_targets = targets[input_end:forecast_end]
                window_dates = dates[input_start:forecast_end]

                # 1. Skip if window contains NaNs
                if np.isnan(input_features).any() or np.isnan(forecast_targets).any():
                    continue

                # 2. Strict consecutive days check inside this pond
                date_diffs = np.diff(window_dates)
                if not np.all(date_diffs == one_day):
                    continue

                xs.append(input_features)
                ys.append(forecast_targets)

        if len(xs) == 0:
            raise ValueError("No valid sequences created! Check window sizes or data gaps.")

        X = torch.tensor(np.asarray(xs), dtype=torch.float32, device=device)
        y = torch.tensor(np.asarray(ys), dtype=torch.float32, device=device)

        return X, y
