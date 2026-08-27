from datasets.base_dataset import BaseDataset

from typing import Tuple
import numpy as np
import pandas as pd
import torch


class DatasetB(BaseDataset):
    """Class to handle loading and preprocessing of Dataset B."""
    FILEPATH = "train-data/IoTpond1.csv"

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
        df = df[self.MEAN_COLS].resample("D").mean()

        # SGR / exponential interpolation: interpolate in log-space, then exponentiate.
        # This is equivalent to assuming constant instantaneous growth rate g
        # between each pair of real measurements: W_t = W1 * exp(g*(t - t1)).
        # Read the README.md for more details on this method\
        log_target = np.log(df[self.TARGET_COL])
        log_target_interp = log_target.interpolate(method="time")
        df[self.TARGET_COL] = np.exp(log_target_interp)

        # Forward/backward fill small sensor gaps
        df[self.FEATURE_COLS] = df[self.FEATURE_COLS].ffill().bfill()

        df = df.reset_index().rename(columns={self.TIMESTAMP_COL: "date"})

        df = df.dropna(subset=self.FEATURE_COLS + [self.TARGET_COL])

        return df
