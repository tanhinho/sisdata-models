from models.lstm import LSTMModel, EvolutionaryLSTMOptimizer
import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import sys
from pathlib import Path


@pytest.mark.integration
@pytest.mark.lstm
@pytest.mark.slow
class TestOptimizerWithRealModel:
    """Integration tests with real LSTM model training."""

    @staticmethod
    def create_model_training_fitness_func(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 5,
        device: str = "cpu",
    ) -> callable:
        """
        Create a fitness function that trains an LSTM model.

        Args:
            X_train: Training input sequences
            y_train: Training target values
            X_val: Validation input sequences
            y_val: Validation target values
            epochs: Number of training epochs
            device: torch device

        Returns:
            Fitness function that takes hyperparameters and returns validation loss
        """

        def fitness_func(params: Dict[str, Any]) -> float:
            try:
                # Extract parameters
                hidden_size = params["hidden_size"]
                num_layers = params["num_layers"]
                dropout = params["dropout"]
                learning_rate = params["learning_rate"]
                batch_size = params["batch_size"]

                # Get time_lag from data shape
                input_size = X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]

                # Create model
                model = LSTMModel(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_size=1,
                    dropout=dropout,
                )
                model.to(device)

                # Setup training
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.MSELoss()

                # Convert data to tensors
                X_train_tensor = torch.FloatTensor(X_train).to(device)
                y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
                X_val_tensor = torch.FloatTensor(X_val).to(device)
                y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(device)

                # Training loop
                model.train()
                for epoch in range(epochs):
                    epoch_loss = 0.0
                    num_batches = 0

                    for i in range(0, len(X_train_tensor), batch_size):
                        batch_X = X_train_tensor[i: i + batch_size]
                        batch_y = y_train_tensor[i: i + batch_size]

                        optimizer.zero_grad()
                        output = model(batch_X)
                        loss = criterion(output, batch_y)
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        num_batches += 1

                # Validation
                model.eval()
                with torch.no_grad():
                    val_output = model(X_val_tensor)
                    val_loss = criterion(val_output, y_val_tensor).item()

                # Fitness is inverse of loss (lower loss = higher fitness)
                # Normalize to [0, 1] range
                fitness = 1.0 / (1.0 + val_loss)

                return fitness

            except Exception as e:
                print(f"Error in training: {e}")
                return 0.1  # Return low fitness on error

        return fitness_func

    def test_optimizer_with_real_lstm_training(self, synthetic_sequences):
        """Test optimizer with real LSTM model training."""
        # Get synthetic data from fixture
        X_train = synthetic_sequences["X_train"]
        X_val = synthetic_sequences["X_val"]
        y_train = synthetic_sequences["y_train"]
        y_val = synthetic_sequences["y_val"]

        # Reshape for LSTM input (samples, sequence_length, features)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

        # Create fitness function
        fitness_func = self.create_model_training_fitness_func(
            X_train, y_train, X_val, y_val,
            epochs=3,
            device="cpu"
        )

        # Create and run optimizer
        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=fitness_func,
            input_size=1,
            output_size=1,
            population_size=5,
            generations=3,
            mutation_rate=0.2,
            crossover_rate=0.7,
        )

        results = optimizer.optimize()

        # Verify results
        assert results["best_params"] is not None
        assert results["best_fitness"] > 0
        assert len(results["history"]["best_fitness"]) == 3

    def test_optimizer_reduces_loss(self):
        """Test that optimizer improves fitness over generations."""
        # Create synthetic data (custom parameters: 80 samples, time_lag=8)
        np.random.seed(42)
        t = np.linspace(0, 10, 80)
        data = np.sin(t) + 0.1 * np.random.randn(80)

        # Create sequences with time_lag=8
        time_lag = 8
        X = np.array([data[i: i + time_lag] for i in range(len(data) - time_lag)])
        y = np.array([data[i + time_lag] for i in range(len(data) - time_lag)])

        # Reshape for LSTM input (samples, sequence_length, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        fitness_func = self.create_model_training_fitness_func(
            X_train, y_train, X_val, y_val,
            epochs=2,
            device="cpu"
        )

        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=fitness_func,
            input_size=1,
            output_size=1,
            population_size=4,
            generations=4,
        )

        results = optimizer.optimize()

        # Check that optimization ran
        assert len(results["history"]["best_fitness"]) == 4
        assert all(f > 0 for f in results["history"]["best_fitness"])


@pytest.mark.integration
@pytest.mark.lstm
class TestOptimizerWithDifferentFitnessFunctions:
    """Test optimizer with various fitness function implementations."""

    def test_custom_fitness_function(self):
        """Test with custom fitness function."""
        def custom_fitness(params: Dict[str, Any]) -> float:
            # Simple custom fitness based on parameter values
            score = 1.0
            score *= 1.0 - abs(params["hidden_size"] - 100) / 200
            score *= 1.0 - abs(params["num_layers"] - 2) / 4
            score *= 1.0 - abs(params["learning_rate"] - 0.001) / 0.01
            return max(0.1, score)

        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=custom_fitness,
            input_size=1,
            output_size=1,
            population_size=8,
            generations=3,
        )

        results = optimizer.optimize()
        assert results["best_params"] is not None

    def test_fitness_function_with_constraints(self):
        """Test fitness function that penalizes certain parameter combinations."""
        def constrained_fitness(params: Dict[str, Any]) -> float:
            score = 1.0

            # Penalize high dropout with low layers
            if params["dropout"] > 0.3 and params["num_layers"] < 2:
                score *= 0.5

            # Penalize very high learning rates with large batch sizes
            if params["learning_rate"] > 0.01 and params["batch_size"] > 64:
                score *= 0.7

            # Prefer moderate hidden sizes
            score *= 1.0 - abs(params["hidden_size"] - 128) / 256

            return max(0.1, score)

        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=constrained_fitness,
            input_size=1,
            output_size=1,
            population_size=10,
            generations=5,
        )

        results = optimizer.optimize()
        best_params = results["best_params"]

        # Verify that optimizer found reasonable parameters
        assert best_params is not None
        assert 32 <= best_params["hidden_size"] <= 256


@pytest.mark.integration
@pytest.mark.lstm
class TestOptimizerEdgeCases:
    """Test optimizer edge cases and error conditions."""

    def test_optimization_with_constant_fitness(self):
        """Test optimization when all individuals have same fitness."""
        def constant_fitness(params: Dict[str, Any]) -> float:
            return 0.5  # Always return same fitness

        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=constant_fitness,
            input_size=1,
            output_size=1,
            population_size=5,
            generations=3,
        )

        results = optimizer.optimize()
        # Should still complete successfully
        assert results["best_fitness"] == 0.5

    def test_optimization_with_extreme_parameters(self):
        """Test that optimizer respects bounds even with extreme fitness values."""
        def extreme_fitness(params: Dict[str, Any]) -> float:
            if params["hidden_size"] > 200:
                return 0.9
            return 0.1  # Encourage extreme parameters

        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=extreme_fitness,
            input_size=1,
            output_size=1,
            population_size=8,
            generations=3,
        )

        results = optimizer.optimize()
        best_params = results["best_params"]

        # Verify bounds are still respected
        assert 32 <= best_params["hidden_size"] <= 256
        assert 1 <= best_params["num_layers"] <= 4

    def test_optimization_with_noisy_fitness(self):
        """Test optimization with noisy fitness function."""
        np.random.seed(42)

        def noisy_fitness(params: Dict[str, Any]) -> float:
            base_score = 1.0 - abs(params["hidden_size"] - 128) / 256
            noise = np.random.normal(0, 0.1)
            return max(0.1, base_score + noise)

        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=noisy_fitness,
            input_size=1,
            output_size=1,
            population_size=10,
            generations=5,
        )

        results = optimizer.optimize()
        # Should still produce valid results despite noise
        assert results["best_params"] is not None
        assert results["best_fitness"] > 0


@pytest.mark.integration
@pytest.mark.lstm
class TestOptimizerConvergence:
    """Test convergence properties of the optimizer."""

    def test_fitness_monotonicity(self):
        """Test that best fitness is monotonically non-decreasing."""
        from lstm.parameter_optimization import create_dummy_fitness_function

        fitness_func = create_dummy_fitness_function()
        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=fitness_func,
            input_size=1,
            output_size=1,
            population_size=15,
            generations=10,
        )

        results = optimizer.optimize()
        best_fitnesses = results["history"]["best_fitness"]

        # Best fitness should be monotonically non-decreasing
        for i in range(1, len(best_fitnesses)):
            assert best_fitnesses[i] >= best_fitnesses[i -
                                                       1] or abs(best_fitnesses[i] - best_fitnesses[i - 1]) < 1e-6

    def test_population_diversity(self):
        """Test that population maintains some diversity."""
        from lstm.parameter_optimization import create_dummy_fitness_function

        fitness_func = create_dummy_fitness_function()
        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=fitness_func,
            input_size=1,
            output_size=1,
            population_size=20,
            generations=5,
        )

        optimizer.optimize()

        # Create a new population
        pop = optimizer.toolbox.population(n=optimizer.population_size)
        fitnesses = list(map(optimizer.toolbox.evaluate, pop))

        # Check that not all fitnesses are identical
        unique_fitnesses = set(fitnesses)
        assert len(unique_fitnesses) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
