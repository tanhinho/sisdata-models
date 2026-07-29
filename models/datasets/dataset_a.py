from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datasets import BaseDataset


class DatasetA(BaseDataset):
    """Class to handle loading and preprocessing of Dataset A."""
    FILEPATH = "data/IoTpond1.csv"

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
