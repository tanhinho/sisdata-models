from typing import Tuple
import numpy as np
import pandas as pd
import torch


def create_sequences(
    df: pd.DataFrame,
        seq_length: int,
        feature_cols: list[str],
        target_col: str,
        device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert pandas DataFrame to sequence tensors by creating overlapping sequences of features
    and corresponding targets.

        Args:
            df (pd.DataFrame): Input dataframe.
            seq_length (int): Length of the sequences to create.
            feature_cols (list[str]): List of feature column names.
            target_col (str): Name of the target column.
            device (torch.device): Device to move the tensors to.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Feature and target tensors.
        """
    xs, ys = [], []

    features = df[feature_cols].values
    targets = df[target_col].values

    for i in range(len(df) - seq_length):
        xs.append(features[i: i + seq_length])
        ys.append([targets[i + seq_length]])  # Enclosing in [] keeps shape as (N, 1)

    X_tensor = torch.tensor(np.array(xs), dtype=torch.float32, device=device)
    y_tensor = torch.tensor(np.array(ys), dtype=torch.float32, device=device)

    return X_tensor, y_tensor
