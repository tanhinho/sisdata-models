# models/preprocess/dataset_a.py
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from .base import BasePreprocessor


class DatasetAPreprocessor(BasePreprocessor):
    """Preprocessor for Dataset A."""

    numeric_cols = [
        "Average Fish Weight (g)", "Survival Rate (%)",
        "Disease Occurrence (Cases)", "Temperature (°C)",
        "Dissolved Oxygen (mg/L)", "pH", "Turbidity (NTU)",
        "Corrective Interventions"
    ]

    categorical_cols = [
        "Oxygenation Automatic", "Oxygenation Interventions",
        "Thermal Risk Index", "Low Oxygen Alert", "Health Status"
    ]

    def __init__(self):
        super().__init__()
        self.preprocessor = self.build_preprocessor()

    def load_data(self) -> pd.DataFrame:
        """Load data from CSV."""
        return pd.read_csv("path/to/dataset_a.csv")

    def build_preprocessor(self) -> Pipeline:
        """Build preprocessing pipeline for Dataset A."""
        numeric_features = ["age", "weight", "temperature"]
        categorical_features = ["species", "location"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(drop="first"), categorical_features),
            ]
        )

        return Pipeline(steps=[
            ("preprocessor", preprocessor),
        ])
