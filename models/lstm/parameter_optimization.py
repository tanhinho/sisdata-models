import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deap import base, creator, tools, algorithms
from typing import Tuple, Dict, Any, Callable
from lstm import LSTMModel
from models.utils import random_int, random_float, random_log_uniform, random_batch_size


class EvolutionaryLSTMOptimizer:
    """
    Evolutionary algorithm-based hyperparameter optimizer for LSTM models.
    Uses genetic algorithms to find optimal hyperparameters including time lag.
    """

    def __init__(
        self,
        fitness_func: Callable,
        input_size: int,
        output_size: int,
        device: str = "cpu",
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
    ):
        """
        Initialize the evolutionary optimizer.

        Args:
            fitness_func: Function that takes hyperparameters dict and returns fitness score
            input_size: Input feature dimension (after applying time lag)
            output_size: Output dimension
            device: torch device ('cpu' or 'cuda')
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.fitness_func = fitness_func
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Define hyperparameter bounds (min, max)
        self.param_bounds = {
            "hidden_size": (32, 256),  # Number of LSTM hidden units
            "num_layers": (1, 4),  # Number of LSTM layers
            "dropout": (0.0, 0.5),  # Dropout rate
            "time_lag": (3, 50),  # Sequence length / lookback window
            "learning_rate": (1e-5, 1e-2),  # Learning rate (log scale)
            "batch_size": (16, 128),  # Batch size (power of 2)
        }

        self.history = {
            "best_fitness": [],
            "avg_fitness": [],
            "best_individual": None,
            "best_params": None,
        }

        self._setup_deap()

    def _setup_deap(self):
        """Setup DEAP creator and toolbox for genetic algorithm."""
        # Clear any previous definitions (in case multiple optimizer instances are created)
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual

        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize fitness
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Attribute generators
        self.toolbox.register(
            "hidden_size",
            random_int,
            self.param_bounds["hidden_size"][0],
            self.param_bounds["hidden_size"][1],
        )
        self.toolbox.register(
            "num_layers",
            random_int,
            self.param_bounds["num_layers"][0],
            self.param_bounds["num_layers"][1],
        )
        self.toolbox.register(
            "dropout",
            random_float,
            self.param_bounds["dropout"][0],
            self.param_bounds["dropout"][1],
        )
        self.toolbox.register(
            "time_lag",
            random_int,
            self.param_bounds["time_lag"][0],
            self.param_bounds["time_lag"][1],
        )
        self.toolbox.register(
            "learning_rate",
            random_log_uniform,
            self.param_bounds["learning_rate"][0],
            self.param_bounds["learning_rate"][1],
        )
        self.toolbox.register(
            "batch_size",
            random_batch_size,
            self.param_bounds["batch_size"][0],
            self.param_bounds["batch_size"][1],
        )

        # Individual and population creation
        self.toolbox.register(
            "individual",
            self._init_individual,
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register(
            "mate", self._custom_crossover
        )  # Custom crossover
        self.toolbox.register(
            "mutate", self._custom_mutation
        )  # Custom mutation

        # Selection
        self.toolbox.register("select", tools.selBest)

    def _init_individual(self) -> "creator.Individual":
        """Create an individual with random hyperparameters."""
        ind = creator.Individual(
            [
                self.toolbox.hidden_size(),
                self.toolbox.num_layers(),
                self.toolbox.dropout(),
                self.toolbox.time_lag(),
                self.toolbox.learning_rate(),
                self.toolbox.batch_size(),
            ]
        )
        return ind

    def _individual_to_params(self, individual: "creator.Individual") -> Dict[str, Any]:
        """Convert individual chromosome to hyperparameter dictionary."""
        return {
            "hidden_size": int(individual[0]),
            "num_layers": int(individual[1]),
            "dropout": float(individual[2]),
            "time_lag": int(individual[3]),
            "learning_rate": float(individual[4]),
            "batch_size": int(individual[5]),
        }

    def _evaluate_individual(self, individual: "creator.Individual") -> Tuple[float,]:
        """Evaluate fitness of an individual."""
        params = self._individual_to_params(individual)
        try:
            fitness = self.fitness_func(params)
            return (fitness,)
        except Exception as e:
            print(f"Error evaluating individual {individual}: {e}")
            return (0.0,)  # Return worst fitness on error

    def _custom_crossover(
        self, ind1: "creator.Individual", ind2: "creator.Individual"
    ) -> Tuple["creator.Individual", "creator.Individual"]:
        """
        Custom crossover that respects parameter constraints.
        Uses uniform crossover for discrete parameters.
        """
        for i in range(len(ind1)):
            if np.random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]

        del ind1.fitness.values
        del ind2.fitness.values
        return ind1, ind2

    def _custom_mutation(self, individual: "creator.Individual") -> ("creator.Individual",):
        """
        Custom mutation that respects parameter bounds.
        Different mutation strategies for different parameter types.
        """
        bounds = [
            self.param_bounds["hidden_size"],
            self.param_bounds["num_layers"],
            self.param_bounds["dropout"],
            self.param_bounds["time_lag"],
            self.param_bounds["learning_rate"],
            self.param_bounds["batch_size"],
        ]

        for i in range(len(individual)):
            if not np.random.random() < self.mutation_rate:
                continue
            if i == 0:  # hidden_size
                individual[i] = random_int(bounds[i][0], bounds[i][1])
            elif i == 1:  # num_layers
                individual[i] = random_int(bounds[i][0], bounds[i][1])
            elif i == 2:  # dropout
                # Gaussian mutation for continuous values
                individual[i] = np.clip(
                    individual[i] + np.random.normal(0, 0.1),
                    bounds[i][0],
                    bounds[i][1],
                )
            elif i == 3:  # time_lag
                individual[i] = random_int(bounds[i][0], bounds[i][1])
            elif i == 4:  # learning_rate
                # Log-scale Gaussian mutation
                log_lr = np.log(individual[i])
                new_log_lr = log_lr + np.random.normal(0, 0.5)
                individual[i] = np.exp(
                    np.clip(new_log_lr, np.log(bounds[i][0]), np.log(bounds[i][1]))
                )
            elif i == 5:  # batch_size
                individual[i] = random_batch_size(bounds[i][0], bounds[i][1])

        del individual.fitness.values
        return (individual,)

    def optimize(self) -> Dict[str, Any]:
        """
        Run the evolutionary algorithm.

        Returns:
            Dictionary containing best parameters and optimization history
        """
        print(f"Starting evolutionary optimization...")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.generations}\n")

        # Create initial population
        pop = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Run genetic algorithm
        for gen in range(self.generations):
            # Select best individuals
            offspring = self.toolbox.select(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # Apply crossover
            for i in range(1, len(offspring), 2):
                if np.random.random() < self.crossover_rate:
                    self.toolbox.mate(offspring[i - 1], offspring[i])

            # Apply mutation
            for i in range(len(offspring)):
                if np.random.random() < self.mutation_rate:
                    self.toolbox.mutate(offspring[i])

            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population
            pop[:] = offspring

            # Record statistics
            fits = [ind.fitness.values[0] for ind in pop]
            best_fitness = max(fits)
            avg_fitness = np.mean(fits)

            self.history["best_fitness"].append(best_fitness)
            self.history["avg_fitness"].append(avg_fitness)

            best_ind = self.toolbox.select(pop, 1)[0]
            self.history["best_individual"] = best_ind
            self.history["best_params"] = self._individual_to_params(best_ind)

            print(
                f"Generation {gen + 1:3d} | "
                f"Best: {best_fitness:.4f} | "
                f"Avg: {avg_fitness:.4f} | "
                f"Params: {self.history['best_params']}"
            )

        print("\nOptimization completed!")
        return {
            "best_params": self.history["best_params"],
            "best_fitness": self.history["best_fitness"][-1],
            "history": self.history,
        }

    def get_best_model_config(self) -> Dict[str, Any]:
        """Get the best model configuration found."""
        if self.history["best_params"] is None:
            raise ValueError("Must run optimize() first")

        params = self.history["best_params"]
        return {
            "input_size": self.input_size,
            "hidden_size": params["hidden_size"],
            "num_layers": params["num_layers"],
            "output_size": self.output_size,
            "dropout": params["dropout"],
            "time_lag": params["time_lag"],
            "learning_rate": params["learning_rate"],
            "batch_size": params["batch_size"],
        }
