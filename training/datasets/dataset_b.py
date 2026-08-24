from datasets.base_dataset import BaseDataset

from typing import Tuple
import numpy as np
import pandas as pd
import torch


class DatasetB(BaseDataset):
    """Class to handle loading and preprocessing of Dataset B."""
    FILEPATH = "train-data/dataset_a/IoTpond1.csv"

    FEATURE_COLS = [
        "Temperature(C)",
        "Turbidity(NTU)",
        "Dissolved Oxygen(g/ml)",
        "PH",
        "Ammonia(g/ml)",
        "Nitrate(g/ml)",
    ]

    TARGET_COL = "Fish_Weight(g)"

    NAME = "dataset_b"

    TIMESTAMP_COL = "created_at"

    MEAN_COLS = [
        "Temperature(C)",
        "Turbidity(NTU)",
        "Dissolved Oxygen(g/ml)",
        "PH",
        "Ammonia(g/ml)",
        "Nitrate(g/ml)",
    ]

    LAST_COLS = [
        "Fish_Weight(g)",
    ]

    SCALE_TARGET = True

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Strip trailing 3-4 letter timezone code (e.g., " CET")
        regex = r"\s+[A-Z]{3,4}$"
        clean_timestamps = df[self.TIMESTAMP_COL].astype(str).str.replace(regex, "", regex=True)
        df[self.TIMESTAMP_COL] = pd.to_datetime(clean_timestamps)
        df = df.dropna(subset=[self.TIMESTAMP_COL]).sort_values(self.TIMESTAMP_COL)

        # Set time index to easily group by daily frequency ("D")
        df = df.set_index(self.TIMESTAMP_COL)

        # Convert impossible physical values to NaN
        df.loc[df["Ammonia(g/ml)"] > 10.0, "Ammonia(g/ml)"] = np.nan
        df.loc[df["Ammonia(g/ml)"] < 0.0, "Ammonia(g/ml)"] = np.nan

        # Daily Aggregation
        means = df[self.MEAN_COLS].resample("D").mean()
        lasts = df[self.LAST_COLS].resample("D").last()

        daily = pd.concat([means, lasts], axis=1)

        # Linearly interpolate fish growth (TARGET_COL) across 15-day gaps
        daily[self.TARGET_COL] = daily[self.TARGET_COL].interpolate(method="linear")

        # Forward/backward fill small sensor gaps
        daily[self.FEATURE_COLS] = daily[self.FEATURE_COLS].ffill().bfill()

        daily = daily.reset_index().rename(columns={self.TIMESTAMP_COL: "date"})

        daily = daily.dropna(subset=self.FEATURE_COLS + [self.TARGET_COL])

        # Save csv for debugging purposes
        daily.to_csv("dataset_b.csv", index=False)

        return daily

    def _aggregate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Loads 20-second data, aggregates daily, and interpolates target gaps."""

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

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        forecast_horizon: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def unscale_target(self, y_scaled: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(y_scaled, torch.Tensor):
            y_scaled = y_scaled.detach().cpu().numpy()

        shape = y_scaled.shape
        unscaled = self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1))
        return unscaled.reshape(shape)
