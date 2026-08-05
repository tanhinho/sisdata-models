from datasets.base_dataset import BaseDataset


class DatasetA(BaseDataset):
    """Class to handle loading and preprocessing of Dataset A."""
    FILEPATH = "data/dataset_a.csv"

    FEATURE_COLS = [
        "Temperature(C)",
        "Turbidity(NTU)",
        "Dissolved Oxygen(g/ml)",
        "PH",
        "Nitrate(g/ml)",
        "Population",
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
        "Population",
        "Fish_Weight(g)",
    ]
