# Preprocessing Module

This module prepares the data for modeling. It includes data cleaning, feature engineering, and transformations to ensure the data is in the best format for training machine learning models.

Because the datasets come from different fish species and sources, the preprocessing steps may vary. The sections below describe the preprocessing techniques used for each dataset in this project.

## Where Raw Training Data Must Be Placed

Current training code expects Dataset A at:

- `train-data/dataset_a.csv` (repository root)

This is important for Docker Compose because the training service bind-mounts:

- local `./train-data` -> container `/app/train-data`

So inside the container, the dataset is resolved as `/app/train-data/dataset_a.csv`, matching the dataset loader path `train-data/dataset_a.csv` relative to `/app`.

If your raw source files are stored elsewhere (for example under `data/`), copy or sync the needed file to `train-data/` before running training, or update both:

1. The dataset file path in code.
2. The Compose bind mount path.

## Datasets

### Dataset A

- **Source**: [Sensor Based Aquaponics Fish Pond Datasets](https://www.kaggle.com/datasets/e81da8b7666dc7af41cdc3aa5ef96c5547e4f412598a030f40d444550965e34f)
- **Original layout**: The source dataset is organized as separate pond files (originally 12 files, one per pond).
- **Project merge strategy**: Pond files are merged into a single training table (`dataset_a.csv`) and a `pond` column is added/preserved so records can be sorted by `(pond, timestamp)`.

- **Features**:
  - created_at: Date and time of the sensor reading.
  - Temperature (C): Water temperature in degrees Celsius.
  - Turbidity(NTU): Water turbidity measured in Nephelometric Turbidity Units (NTU).
  - Dissolved Oxygen(g/ml): Dissolved oxygen concentration in grams per milliliter.
  - PH: Water pH values.
  - Ammonia(g/ml): Ammonia concentration in grams per milliliter.
  - Nitrate(g/ml): Nitrate concentration in grams per milliliter.
  - Population: Number of fish in the pond.
  - Fish_Length(cm): Average length of the fish in centimeters.
  - Fish_Weight(g): Average weight of the fish in grams.
- **Target Variable**: Fish_Weight(g)

### Dataset A Processing Pipeline

- Load `dataset_a.csv` and parse `created_at` into timestamps.
- Sort and process records per pond timeline.
- Aggregate to daily records grouped by `(pond, date)` using:
  - mean aggregation for sensor columns.
  - last-value aggregation for selected state/target columns.
- Reindex each pond independently to a full daily date range.
- Interpolate short gaps within each pond only, then forward/backfill within pond boundaries.
- Perform chronological train/validation/test splits independently per pond, then concatenate splits.
- Fit `StandardScaler` on training feature columns and apply transforms to validation/test.
- Build sliding windows for forecasting.

### Critical Temporal Constraint

Sliding windows must strictly operate within a single pond. A sequence can never mix observations from different ponds, even if timestamps are consecutive.
