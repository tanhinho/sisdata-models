# Tests Directory Structure

This directory contains all tests for the sisdata-models project, organized by component/feature for better maintainability.

## Directory Structure

```
tests/
├── __init__.py                          # Package marker
├── conftest.py                          # Shared fixtures and configuration
├── pytest.ini                           # Pytest configuration
├── lstm/                                # LSTM model tests
│   ├── __init__.py
│   ├── test_lstm_model.py               # Unit tests for LSTM model
│   ├── test_parameter_optimization.py   # Unit tests for parameter optimization
│   └── test_parameter_optimization_integration.py  # Integration tests
├── api/                                 # API serving tests
│   ├── __init__.py
│   └── test_api.py
├── training/                            # Model training tests
│   ├── __init__.py
│   └── test_train.py
├── e2e/                                 # End-to-end tests
│   ├── __init__.py
│   └── test_e2e.py
└── fixtures/                            # Shared test utilities
    └── __init__.py
```

## Test Categories

### LSTM Tests (`lstm/`)

- **Unit Tests** (`test_parameter_optimization.py`): Tests for the `EvolutionaryLSTMOptimizer` class
  - Initialization and configuration
  - Random value generators
  - Individual representation and conversion
  - Fitness evaluation
  - Genetic operators (crossover, mutation)
  - Optimization process
  - Edge cases and error handling

- **Integration Tests** (`test_parameter_optimization_integration.py`): Tests with real model training
  - Real LSTM model training with synthetic data
  - Custom fitness functions
  - Convergence properties
  - Noisy fitness functions

### API Tests (`api/`)

- Tests for the serving API endpoints

### Training Tests (`training/`)

- Tests for model training pipelines

### End-to-End Tests (`e2e/`)

- Complete workflow tests

## Running Tests

### Run all tests

```bash
pytest
```

### Run specific test directory

```bash
pytest tests/lstm/
```

### Run specific test file

```bash
pytest tests/lstm/test_parameter_optimization.py
```

### Run specific test class

```bash
pytest tests/lstm/test_parameter_optimization.py::TestEvolutionaryLSTMOptimizerInitialization
```

### Run specific test

```bash
pytest tests/lstm/test_parameter_optimization.py::TestEvolutionaryLSTMOptimizerInitialization::test_basic_initialization
```

### Run with markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run LSTM tests
pytest -m lstm

# Exclude slow tests
pytest -m "not slow"
```

### Run with verbose output

```bash
pytest -v
```

### Run with coverage report

```bash
pytest --cov=models --cov=serving
```

## Test Configuration

The `conftest.py` file provides:

- Automatic path setup for importing project modules
- Random seed fixture for reproducibility
- Synthetic data fixtures for time series tests
- Pytest marker configuration

The `pytest.ini` file configures:

- Test discovery patterns
- Output formatting
- Custom markers
- Minimum Python version requirement

## Adding New Tests

When adding new test files:

1. Create them in the appropriate subdirectory (`lstm/`, `api/`, `training/`, or `e2e/`)
2. Name the file `test_*.py`
3. Use the fixtures provided in `conftest.py`
4. Add appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)

Example:

```python
import pytest
from models.lstm import LSTMModel

@pytest.mark.lstm
@pytest.mark.unit
class TestLSTMModel:
    def test_initialization(self):
        model = LSTMModel(input_size=1, hidden_size=64, output_size=1)
        assert model is not None
```

## Dependencies

Tests require:

- pytest
- numpy
- torch
- deap (for parameter optimization)

Install test dependencies:

```bash
pip install pytest numpy torch deap
```
