"""Tests for shared utility functions."""

import pytest
import numpy as np
from models.utils import (
    random_int,
    random_float,
    random_log_uniform,
    random_batch_size,
)


@pytest.mark.unit
@pytest.mark.utils
class TestRandomGenerators:
    """Test utility random value generators."""

    def test_random_int(self):
        """Test random integer generation."""
        for _ in range(100):
            value = random_int(10, 20)
            assert isinstance(value, (int, np.integer))
            assert 10 <= value <= 20

    def test_random_float(self):
        """Test random float generation."""
        for _ in range(100):
            value = random_float(0.0, 1.0)
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0

    def test_random_log_uniform(self):
        """Test log-uniform random generation."""
        min_val, max_val = 1e-5, 1e-2
        for _ in range(100):
            value = random_log_uniform(min_val, max_val)
            assert isinstance(value, float)
            assert min_val <= value <= max_val

    def test_random_batch_size(self):
        """Test batch size generation (power of 2)."""
        min_val, max_val = 16, 128
        for _ in range(100):
            value = random_batch_size(min_val, max_val)
            assert isinstance(value, (int, np.integer))
            assert min_val <= value <= max_val
            # Check if it's a power of 2
            assert (value & (value - 1)) == 0, f"{value} is not a power of 2"


@pytest.mark.unit
@pytest.mark.utils
class TestRandomGeneratorsSeed:
    """Test reproducibility with seeds."""

    def test_random_int_reproducible(self):
        """Test that same seed produces same random ints."""
        np.random.seed(42)
        values1 = [random_int(1, 100) for _ in range(10)]

        np.random.seed(42)
        values2 = [random_int(1, 100) for _ in range(10)]

        assert values1 == values2

    def test_random_float_reproducible(self):
        """Test that same seed produces same random floats."""
        np.random.seed(42)
        values1 = [random_float(0.0, 1.0) for _ in range(10)]

        np.random.seed(42)
        values2 = [random_float(0.0, 1.0) for _ in range(10)]

        assert np.allclose(values1, values2)

    def test_random_log_uniform_reproducible(self):
        """Test that same seed produces same log-uniform values."""
        np.random.seed(42)
        values1 = [random_log_uniform(1e-5, 1e-2) for _ in range(10)]

        np.random.seed(42)
        values2 = [random_log_uniform(1e-5, 1e-2) for _ in range(10)]

        assert np.allclose(values1, values2)

    def test_random_batch_size_reproducible(self):
        """Test that same seed produces same batch sizes."""
        np.random.seed(42)
        values1 = [random_batch_size(16, 128) for _ in range(10)]

        np.random.seed(42)
        values2 = [random_batch_size(16, 128) for _ in range(10)]

        assert values1 == values2
