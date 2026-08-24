from datasets.base_dataset import BaseDataset

import numpy as np
import pandas as pd


class DatasetC(BaseDataset):
    """Class to handle loading and preprocessing of Dataset C."""
    FILEPATH = "train-data/Data_Model_IoTMLCQ_2024.csv"

    FEATURE_COLS = [
        "Survival Rate (%)",
        "Disease Occurrence (Cases)",
        "Temperature (°C)",
        "Dissolved Oxygen (mg/L)",
        "pH",
        "Turbidity (NTU)",
        "Oxygenation Interventions",
        "Corrective Interventions",
        "Average Temperature (°C)",
        "Oxygenation Automatic",
        "Thermal Risk Index",
        "Low Oxygen Alert",
        "Health Status",
    ]

    TARGET_COL = "Average Fish Weight (g)"

    NAME = "dataset_c"

    TIMESTAMP_COL = "Datetime"

    SCALE_TARGET = True

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Strip trailing 3-4 letter timezone code (e.g., " CET")
        regex = r"\s+[A-Z]{3,4}$"
        clean_timestamps = df[self.TIMESTAMP_COL].astype(str).str.replace(regex, "", regex=True)
        df[self.TIMESTAMP_COL] = pd.to_datetime(clean_timestamps)
        df = df.dropna(subset=[self.TIMESTAMP_COL]).sort_values(self.TIMESTAMP_COL)

        # Set time index to easily group by daily frequency ("D")
        df = df.set_index(self.TIMESTAMP_COL)

        # Handle categorical columns
        df = self._handle_categorical(df)

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

    def _handle_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to numerical values."""
        df["Oxygenation Automatic"] = df["Oxygenation Automatic"].map({"Yes": 1, "No": 0})
        df["Thermal Risk Index"] = df["Thermal Risk Index"].map({"Normal": 0, "High": 1})
        df["Low Oxygen Alert"] = df["Low Oxygen Alert"].map({"Safe": 1})
        df["Health Status"] = df["Health Status"].map({"Stable": 0, "At Risk": 1})
        return df
