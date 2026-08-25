# Preprocessing Module

This module prepares the data for modeling. It includes data cleaning, feature engineering, and transformations to ensure the data is in the best format for training machine learning models.

Because the datasets come from different fish species and sources, the preprocessing steps may vary. The sections below describe the preprocessing techniques used for each dataset in this project.

## Where Raw Training Data Must Be Placed

Current training code expects data at:

- `train-data` (repository root)

This is important for Docker Compose because the training service bind-mounts:

- local `./train-data` -> container `/app/train-data`

If your raw source files are stored elsewhere (for example under `data/`), copy or sync the needed file to `train-data/` before running training, or update both:

1. The dataset file path in code.
2. The Compose bind mount path.

## Datasets

### Dataset A

- **Source**: [Sensor Based Aquaponics Fish Pond Datasets](https://www.kaggle.com/datasets/e81da8b7666dc7af41cdc3aa5ef96c5547e4f412598a030f40d444550965e34f)
- **Original layout**: The source dataset is organized as separate pond files (originally 12 files, one per pond). We use only pond 1.

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

- Load `IoTPond1.csv` and parse `created_at` into timestamps.
- Sort the dataframe by timestamp, removing where timestamp is missing.
- Remove the `Population` column, as it has a constant value for all rows.
- Convert impossible physical values for `Ammonia(g/ml)` to NaN.
- Interpolate the fish weight (TARGET_COL) using the Specific Growth Rate (SGR) equation.
- Forward/backward fill sensor gaps.
- Perform chronological train/validation/test splits independently per pond, then concatenate splits.
- Fit `StandardScaler` on training feature columns and apply transforms to validation/test.
- Build sliding windows for forecasting.

### Dataset B

- **Source**: [Sensor Based Aquaponics Fish Pond Datasets](https://www.kaggle.com/datasets/e81da8b7666dc7af41cdc3aa5ef96c5547e4f412598a030f40d444550965e34f)
- **Original layout**: The source dataset is organized as separate pond files (originally 12 files, one per pond). We use only pond 1.

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

### Dataset B Processing Pipeline

- Load `IoTPond1.csv` and parse `created_at` into timestamps.
- Sort the dataframe by timestamp, removing where timestamp is missing.
- Remove the `Population` column, as it has a constant value for all rows.
- Convert impossible physical values for `Ammonia(g/ml)` to NaN.
- Aggregate sensor readings to daily averages, then interpolate the fish weight (TARGET_COL) using the Specific Growth Rate (SGR) equation.
- Interpolate the fish weight (TARGET_COL) using the Specific Growth Rate (SGR) equation.
- Forward/backward fill sensor gaps.
- Perform chronological train/validation/test splits independently per pond, then concatenate splits.
- Fit `StandardScaler` on training feature columns and apply transforms to validation/test.
- Build sliding windows for forecasting.

### Dataset C

- **Source**: [IoT Monitoring of Water Quality and Tilapia](https://www.kaggle.com/datasets/jocelyndumlao/iot-monitoring-of-water-quality-and-tilapia)
- **Original layout**: The source dataset is a single XLSX file with hourly readings from a tilapia pond. It is converted to CSV for this project.

- **Features**:
  - Datetime: Date and time of each reading.
  - Month: Data collection month (January to June).
  - Average Fish Weight (g): Average weight of the tilapia fish in grams.
  - Survival Rate (%): Percentage of fish survival during the monitoring period.
  - Disease Occurrence (Cases): Number of disease cases observed.
  - Temperature (°C): Water temperature readings.
  - Dissolved Oxygen (mg/L): Levels of dissolved oxygen in the water.
  - pH: Water pH values.
  - Turbidity (NTU): Water turbidity measured in Nephelometric Turbidity Units (NTU).
  - Oxygenation Automatic: Indicates if automatic oxygenation was applied (Yes/No).
  - Oxygenation Interventions: Oxygenation interventions applied (Yes/No).
  - Corrective Interventions: Number of corrective measures taken.
  - Thermal Risk Index: Indicates if the thermal risk is "High" or "Normal."
  - Low Oxygen Alert: Shows "Critical" if DO levels are below 5 mg/L, otherwise "Safe."
  - Health Status: Fish health status, showing "At Risk" or "Stable" based on thermal and oxygen risk alerts.

Some additional features were present in the dataset but were not mentioned in the Kaggle description. These features appeared to contain redundant and duplicate information, such as the month in which the measurements were collected, or otherwise irrelevant information. Therefore, they were excluded from the final dataset.

### Dataset C Processing Pipeline

- Load `train-data/Data_Model_IoTMLCQ_2024.csv` and parse `Datetime` into timestamps.
- Sort the dataframe by timestamp, removing where timestamp is missing.
- Remove the `Low Oxygen Alert` column, as it has a constant value for all rows.
- Interpolate the fish weight (TARGET_COL) using the Specific Growth Rate (SGR) equation.
- Forward/backward fill sensor gaps.
- Perform chronological train/validation/test splits independently per pond, then concatenate splits.
- Fit `StandardScaler` on training feature columns and apply transforms to validation/test.
- Build sliding windows for forecasting.

### Dataset D

- **Source**: [IoT Monitoring of Water Quality and Tilapia](https://www.kaggle.com/datasets/jocelyndumlao/iot-monitoring-of-water-quality-and-tilapia)
- **Original layout**: The source dataset is a single XLSX file with hourly readings from a tilapia pond. It is converted to CSV for this project.

- **Features**:
  - Datetime: Date and time of each reading.
  - Month: Data collection month (January to June).
  - Average Fish Weight (g): Average weight of the tilapia fish in grams.
  - Survival Rate (%): Percentage of fish survival during the monitoring period.
  - Disease Occurrence (Cases): Number of disease cases observed.
  - Temperature (°C): Water temperature readings.
  - Dissolved Oxygen (mg/L): Levels of dissolved oxygen in the water.
  - pH: Water pH values.
  - Turbidity (NTU): Water turbidity measured in Nephelometric Turbidity Units (NTU).
  - Oxygenation Automatic: Indicates if automatic oxygenation was applied (Yes/No).
  - Oxygenation Interventions: Oxygenation interventions applied (Yes/No).
  - Corrective Interventions: Number of corrective measures taken.
  - Thermal Risk Index: Indicates if the thermal risk is "High" or "Normal."
  - Low Oxygen Alert: Shows "Critical" if DO levels are below 5 mg/L, otherwise "Safe."
  - Health Status: Fish health status, showing "At Risk" or "Stable" based on thermal and oxygen risk alerts.

Some additional features were present in the dataset but were not mentioned in the Kaggle description. These features appeared to contain redundant and duplicate information, such as the month in which the measurements were collected, or otherwise irrelevant information. Therefore, they were excluded from the final dataset.

### Dataset D Processing Pipeline

- Load `train-data/Data_Model_IoTMLCQ_2024.csv` and parse `Datetime` into timestamps.
- Sort the dataframe by timestamp, removing where timestamp is missing.
- Remove the `Low Oxygen Alert` column, as it has a constant value for all rows.
- Aggregate sensor readings to daily averages, then interpolate the fish weight (TARGET_COL) using the Specific Growth Rate (SGR) equation.
- Interpolate the fish weight (TARGET_COL) using the Specific Growth Rate (SGR) equation.
- Forward/backward fill sensor gaps.
- Perform chronological train/validation/test splits independently per pond, then concatenate splits.
- Fit `StandardScaler` on training feature columns and apply transforms to validation/test.
- Build sliding windows for forecasting.

### Critical Temporal Constraint

Sliding windows must strictly operate within a single pond. A sequence can never mix observations from different ponds, even if timestamps are consecutive.
