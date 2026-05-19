# Preprocessing Module

This module prepares the data for modeling. It includes data cleaning, feature engineering, and transformations to ensure the data is in the best format for training machine learning models.

Because the datasets come from different fish species and sources, the preprocessing steps may vary. The sections below describe the preprocessing techniques used for each dataset in this project.

## Datasets

### Dataset A

- **Source**: [IoT Monitoring of Water Quality and Tilapia](https://www.kaggle.com/datasets/jocelyndumlao/iot-monitoring-of-water-quality-and-tilapia/data)
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
- **Preprocessing**: [scaling, encoding, etc.]
