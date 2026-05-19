"""
Shared pytest configuration and fixtures for all tests.
"""

import pytest
import numpy as np


@pytest.fixture
def synthetic_timeseries():
    """Create synthetic time series data for testing."""
    np.random.seed(42)

    # Generate simple sine wave with noise
    t = np.linspace(0, 10, 100)
    data = np.sin(t) + 0.1 * np.random.randn(100)

    return {
        "data": data,
        "t": t,
        "length": len(data),
    }


@pytest.fixture
def synthetic_sequences(synthetic_timeseries):
    """Create sequences from synthetic time series."""
    data = synthetic_timeseries["data"]
    time_lag = 10

    X = []
    y = []
    for i in range(len(data) - time_lag):
        X.append(data[i: i + time_lag])
        y.append(data[i + time_lag])

    return {
        "X": np.array(X),
        "y": np.array(y),
        "time_lag": time_lag,
        "X_train": np.array(X[: int(0.8 * len(X))]),
        "y_train": np.array(y[: int(0.8 * len(y))]),
        "X_val": np.array(X[int(0.8 * len(X)):]),
        "y_val": np.array(y[int(0.8 * len(y)):]),
    }


# Pytest configuration
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
