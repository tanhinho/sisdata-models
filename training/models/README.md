# Models Module

This folder contains the model architectures used in the training pipeline.

## Purpose

- Define reusable model classes that share a common interface.
- Encapsulate training/evaluation logic per architecture.
- Report metrics and parameters to MLflow through the shared base class workflow.

## Files

- `base_model.py`: abstract base class for all models.
- `lstm_model.py`: LSTM sequence model.
- `tcn_model.py`: Temporal Convolutional Network (TCN) model.
- `transformer_model.py`: Transformer-based sequence model.
- `random_forest_model.py`: Random Forest regressor over flattened sequence windows.

## Shared Interface

All model classes inherit from `BaseModel` and must override:

- `forward(...)`: inference logic for the architecture.
- `_fit_and_evaluate_impl(df_train, df_test)`: required architecture-specific training and validation/evaluation routine.

The shared `fit_and_evaluate(...)` method in `BaseModel`:

- opens a nested MLflow run.
- sets run tags (`dataset`, `model`, `run_type`, `sha`).
- logs parameters and loss metric.

## Input/Output Shape Expectations

For sequence models (LSTM, TCN, Transformer):

- Input `X`: `(batch, seq_len, num_features)`
- Target `y`: `(batch, forecast_horizon)`

For `RandomForestModel`:

- sequence windows are flattened internally before fitting/prediction.

## How Models Are Used

1. An optimizer class chooses hyperparameters for one model architecture.
2. The training pipeline instantiates the selected model with those hyperparameters.
3. The model trains/evaluates via `_fit_and_evaluate_impl(...)`.
4. The parent training flow logs final artifacts (including model state dict and scaler), then registers the version in MLflow.

## Notes

- `BaseModel.FORECAST_HORIZON` must be changed to match the desired forecast horizon.
- Device selection is automatic (`cuda` if available, otherwise `cpu`).
