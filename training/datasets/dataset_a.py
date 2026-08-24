from datasets.base_dataset import BaseDataset

from typing import Tuple
import numpy as np
import pandas as pd
import torch


class DatasetA(BaseDataset):
    """Class to handle loading and preprocessing of Dataset A."""
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

    NAME = "dataset_a"

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

        # Identify rows where the target value changes (new readings)
        is_new_reading = df[self.TARGET_COL].ne(df[self.TARGET_COL].shift())
        is_new_reading.iloc[0] = True  # always keep the very first row

        # Create a sparse version of the target column, keeping only new readings
        target_sparse = df[self.TARGET_COL].where(is_new_reading)

        # Find the index of the absolute last new reading
        # is_new_reading[::-1].idxmax() reverses the series and finds the first True
        last_reading_idx = is_new_reading[::-1].idxmax()

        # Truncate the dataframe up to and including the last true reading
        df = df.loc[:last_reading_idx]
        is_new_reading = is_new_reading.loc[:last_reading_idx]

        # SGR / exponential interpolation: interpolate in log-space, then exponentiate.
        # This is equivalent to assuming constant instantaneous growth rate g
        # between each pair of real measurements: W_t = W1 * exp(g*(t - t1)).
        # Read the README.md for more details on this method\
        log_target = np.log(target_sparse)
        log_target_interp = log_target.interpolate(method="time")
        df[self.TARGET_COL] = np.exp(log_target_interp)

        # Forward/backward fill small sensor gaps
        df[self.FEATURE_COLS] = df[self.FEATURE_COLS].ffill().bfill()

        df = df.reset_index().rename(columns={self.TIMESTAMP_COL: "date"})

        # Drop any remaining NaNs at boundaries
        df = df.dropna(subset=self.FEATURE_COLS + [self.TARGET_COL])

        return df
