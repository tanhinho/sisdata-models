from models.lstm import EvolutionaryLSTMOptimizer
import pytest
import numpy as np
import torch
from typing import Dict, Any
import sys
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "models"))


@pytest.fixture
def create_dummy_fitness_function():
    """
    Create a dummy fitness function for testing.
    In practice, replace this with actual model training and validation.
    """
    def fitness_func(params: Dict[str, Any]) -> float:
        # Dummy fitness: penalize extremes, reward balanced parameters
        hidden_size = params["hidden_size"]
        num_layers = params["num_layers"]
        dropout = params["dropout"]
        time_lag = params["time_lag"]
        learning_rate = params["learning_rate"]
        batch_size = params["batch_size"]

        # Simple scoring heuristic
        score = 1.0

        # Prefer moderate hidden size
        score *= 1.0 - abs(hidden_size - 128) / 256

        # Prefer 1-2 layers (fewer parameters)
        score *= 1.0 if 1 <= num_layers <= 2 else 0.7

        # Moderate dropout helps (up to 0.3)
        score *= 1.0 - abs(dropout - 0.2) * 2

        # Reasonable time lag (10-30 is good)
        score *= 1.0 - abs(time_lag - 20) / 50

        # Learning rate around 1e-3 is good
        score *= 1.0 - abs(np.log(learning_rate) - np.log(1e-3)) / 5

        # Batch size around 32 is good
        score *= 1.0 - abs(batch_size - 32) / 64

        return max(0.1, score)  # Ensure positive fitness

    return fitness_func


@pytest.mark.unit
@pytest.mark.lstm
class TestEvolutionaryLSTMOptimizerInitialization:
    """Test initialization of EvolutionaryLSTMOptimizer."""

    def test_basic_initialization(self):
        """Test basic initialization with default parameters."""
        fitness_func = create_dummy_fitness_function()
        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=fitness_func,
            input_size=1,
            output_size=1,
        )

        assert optimizer.input_size == 1
        assert optimizer.output_size == 1
        assert optimizer.population_size == 20
        assert optimizer.generations == 10
        assert optimizer.mutation_rate == 0.2
        assert optimizer.crossover_rate == 0.7
        assert optimizer.device == "cpu"

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        fitness_func = create_dummy_fitness_function()
        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=fitness_func,
            input_size=5,
            output_size=2,
            population_size=50,
            generations=20,
            mutation_rate=0.3,
            crossover_rate=0.8,
            device="cuda",
        )

        assert optimizer.input_size == 5
        assert optimizer.output_size == 2
        assert optimizer.population_size == 50
        assert optimizer.generations == 20
        assert optimizer.mutation_rate == 0.3
        assert optimizer.crossover_rate == 0.8
        assert optimizer.device == "cuda"

    def test_param_bounds(self):
        """Test that parameter bounds are properly set."""
        fitness_func = create_dummy_fitness_function()
        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=fitness_func,
            input_size=1,
            output_size=1,
        )

        expected_keys = {
            "hidden_size",
            "num_layers",
            "dropout",
            "time_lag",
            "learning_rate",
            "batch_size",
        }
        assert set(optimizer.param_bounds.keys()) == expected_keys

        # Verify bounds are tuples with min < max
        for param, (min_val, max_val) in optimizer.param_bounds.items():
            assert min_val < max_val, f"Invalid bounds for {param}"


@pytest.mark.unit
@pytest.mark.lstm
class TestIndividualRepresentation:
    """Test individual creation and conversion."""

    def test_individual_creation(self):
        """Test individual creation."""
        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=create_dummy_fitness_function(),
            input_size=1,
            output_size=1,
        )

        individual = optimizer._init_individual()
        assert len(individual) == 6  # 6 hyperparameters
        assert hasattr(individual, "fitness")

    def test_individual_to_params(self):
        """Test conversion of individual to parameters dict."""
        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=create_dummy_fitness_function(),
            input_size=1,
            output_size=1,
        )

        individual = optimizer._init_individual()
        params = optimizer._individual_to_params(individual)

        expected_keys = {
            "hidden_size",
            "num_layers",
            "dropout",
            "time_lag",
            "learning_rate",
            "batch_size",
        }
        assert set(params.keys()) == expected_keys

        # Verify types
        assert isinstance(params["hidden_size"], int)
        assert isinstance(params["num_layers"], int)
        assert isinstance(params["dropout"], float)
        assert isinstance(params["time_lag"], int)
        assert isinstance(params["learning_rate"], float)
        assert isinstance(params["batch_size"], int)

    def test_individual_params_within_bounds(self):
        """Test that individual parameters are within bounds."""
        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=create_dummy_fitness_function(),
            input_size=1,
            output_size=1,
        )

        for _ in range(100):
            individual = optimizer._init_individual()
            params = optimizer._individual_to_params(individual)

            assert optimizer.param_bounds["hidden_size"][0] <= params["hidden_size"] <= optimizer.param_bounds["hidden_size"][1]
            assert optimizer.param_bounds["num_layers"][0] <= params["num_layers"] <= optimizer.param_bounds["num_layers"][1]
            assert optimizer.param_bounds["dropout"][0] <= params["dropout"] <= optimizer.param_bounds["dropout"][1]
            assert optimizer.param_bounds["time_lag"][0] <= params["time_lag"] <= optimizer.param_bounds["time_lag"][1]
            assert optimizer.param_bounds["learning_rate"][0] <= params["learning_rate"] <= optimizer.param_bounds["learning_rate"][1]
            assert optimizer.param_bounds["batch_size"][0] <= params["batch_size"] <= optimizer.param_bounds["batch_size"][1]


@pytest.mark.unit
@pytest.mark.lstm
class TestFitnessEvaluation:
    """Test fitness evaluation."""

    def test_evaluate_individual(self):
        """Test individual evaluation."""
        fitness_func = create_dummy_fitness_function()
        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=fitness_func,
            input_size=1,
            output_size=1,
        )

        individual = optimizer._init_individual()
        fitness = optimizer._evaluate_individual(individual)

        assert isinstance(fitness, tuple)
        assert len(fitness) == 1
        assert isinstance(fitness[0], (float, np.floating))
        assert 0.0 <= fitness[0] <= 1.0  # Dummy function returns values in this range

    def test_evaluate_individual_error_handling(self):
        """Test that evaluation handles errors gracefully."""
        def bad_fitness_func(params):
            raise ValueError("Intentional error")

        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=bad_fitness_func,
            input_size=1,
            output_size=1,
        )

        individual = optimizer._init_individual()
        fitness = optimizer._evaluate_individual(individual)

        # Should return worst fitness (0.0) on error
        assert fitness == (0.0,)


@pytest.mark.unit
@pytest.mark.lstm
class TestGeneticOperators:
    """Test crossover and mutation operations."""

    def test_crossover(self):
        """Test crossover operation."""
        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=create_dummy_fitness_function(),
            input_size=1,
            output_size=1,
            crossover_rate=1.0,  # Force crossover
        )

        ind1 = optimizer._init_individual()
        ind2 = optimizer._init_individual()

        original_ind1 = ind1.copy()
        original_ind2 = ind2.copy()

        ind1_result, ind2_result = optimizer._custom_crossover(ind1, ind2)

        # Check that fitness values are deleted
        assert not ind1_result.fitness.valid
        assert not ind2_result.fitness.valid

        # Check that individuals exist
        assert len(ind1_result) == 6
        assert len(ind2_result) == 6

    def test_mutation(self):
        """Test mutation operation."""
        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=create_dummy_fitness_function(),
            input_size=1,
            output_size=1,
            mutation_rate=1.0,  # Force mutation
        )

        individual = optimizer._init_individual()
        original = individual.copy()

        mutated = optimizer._custom_mutation(individual)

        assert len(mutated) == 1
        assert not mutated[0].fitness.valid

        # At least one gene should have mutated
        assert not np.allclose(mutated[0], original)

    def test_mutation_respects_bounds(self):
        """Test that mutation respects parameter bounds."""
        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=create_dummy_fitness_function(),
            input_size=1,
            output_size=1,
            mutation_rate=1.0,
        )

        for _ in range(50):
            individual = optimizer._init_individual()
            mutated = optimizer._custom_mutation(individual)
            params = optimizer._individual_to_params(mutated[0])

            # Verify bounds
            assert optimizer.param_bounds["hidden_size"][0] <= params["hidden_size"] <= optimizer.param_bounds["hidden_size"][1]
            assert optimizer.param_bounds["num_layers"][0] <= params["num_layers"] <= optimizer.param_bounds["num_layers"][1]
            assert optimizer.param_bounds["dropout"][0] <= params["dropout"] <= optimizer.param_bounds["dropout"][1]
            assert optimizer.param_bounds["time_lag"][0] <= params["time_lag"] <= optimizer.param_bounds["time_lag"][1]
            assert optimizer.param_bounds["learning_rate"][0] <= params["learning_rate"] <= optimizer.param_bounds["learning_rate"][1]
            assert optimizer.param_bounds["batch_size"][0] <= params["batch_size"] <= optimizer.param_bounds["batch_size"][1]


@pytest.mark.unit
@pytest.mark.lstm
class TestOptimization:
    """Test the optimization process."""

    def test_optimize(self):
        """Test running the optimization."""
        fitness_func = create_dummy_fitness_function()
        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=fitness_func,
            input_size=1,
            output_size=1,
            population_size=10,
            generations=5,
        )

        results = optimizer.optimize()

        assert "best_params" in results
        assert "best_fitness" in results
        assert "history" in results

        best_params = results["best_params"]
        assert isinstance(best_params, dict)
        assert set(best_params.keys()) == {
            "hidden_size",
            "num_layers",
            "dropout",
            "time_lag",
            "learning_rate",
            "batch_size",
        }

    def test_optimization_history(self):
        """Test that optimization tracks history."""
        fitness_func = create_dummy_fitness_function()
        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=fitness_func,
            input_size=1,
            output_size=1,
            population_size=10,
            generations=5,
        )

        results = optimizer.optimize()
        history = results["history"]

        assert len(history["best_fitness"]) == 5  # 5 generations
        assert len(history["avg_fitness"]) == 5
        assert history["best_individual"] is not None
        assert history["best_params"] is not None

        # Best fitness should generally improve or stay the same
        for i in range(1, len(history["best_fitness"])):
            assert history["best_fitness"][i] >= 0

    def test_get_best_model_config(self):
        """Test getting best model configuration."""
        fitness_func = create_dummy_fitness_function()
        optimizer = EvolutionaryLSTMOptimizer(
            fitness_func=fitness_func,
            input_size=3,
            output_size=1,
            population_size=10,
            generations=5,
        )

        # Should raise error before optimization
        with pytest.raises(ValueError):
            optimizer.get_best_model_config()

        # Should work after optimization
        optimizer.optimize()
        config = optimizer.get_best_model_config()

        assert config["input_size"] == 3
        assert config["output_size"] == 1
        assert "hidden_size" in config
        assert "num_layers" in config
        assert "dropout" in config
        assert "time_lag" in config
        assert "learning_rate" in config
        assert "batch_size" in config


@pytest.mark.unit
@pytest.mark.lstm
class TestRandomSeed:
    """Test reproducibility with random seed."""

    def test_reproducibility(self):
        """Test that using same seed produces same results."""
        def run_optimization(seed):
            np.random.seed(seed)
            torch.manual_seed(seed)

            fitness_func = create_dummy_fitness_function()
            optimizer = EvolutionaryLSTMOptimizer(
                fitness_func=fitness_func,
                input_size=1,
                output_size=1,
                population_size=10,
                generations=5,
            )

            return optimizer.optimize()

        # Run twice with same seed
        results1 = run_optimization(42)
        results2 = run_optimization(42)

        # Results should be identical
        assert results1["best_fitness"] == results2["best_fitness"]
        assert results1["best_params"] == results2["best_params"]
