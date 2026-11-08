"""
Ant Colony Optimization (ACO) Core Engine - Mathematical Implementation of Ant System (AS).
Author: El Kachabi Badr-eddine
GitHub: https://github.com/el-kachabi
"""

from dataclasses import dataclass, field
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from src.tsp import City, Tour, TSPInstance


@dataclass
class IterationRecord:
    """Detailed telemetry captured at each iteration of the colony."""
    iteration: int
    best_length: float
    mean_length: float
    worst_length: float
    best_path: List[int]
    pheromone_min: float
    pheromone_max: float
    pheromone_mean: float
    entropy: float  # Measure of colony convergence / pheromone dispersion


@dataclass
class StepDecision:
    """Record of an individual ant's decision for step-by-step visual simulation."""
    ant_id: int
    current_city: int
    candidate_cities: List[int]
    probabilities: List[float]
    chosen_city: int
    pheromone_value: float
    heuristic_value: float


@dataclass
class ACOResult:
    """Complete summary and history returned by an ACO solver run."""
    algorithm_name: str
    instance_name: str
    best_tour: Tour
    convergence_history: List[IterationRecord]
    final_pheromone_matrix: np.ndarray
    execution_time_sec: float
    hyperparameters: Dict[str, any]
    step_records: Optional[List[StepDecision]] = None

    def get_convergence_dataframe(self):
        """Converts convergence history to a pandas DataFrame for charting."""
        import pandas as pd
        records = [
            {
                "Iteration": r.iteration,
                "Best Length": r.best_length,
                "Mean Length": r.mean_length,
                "Worst Length": r.worst_length,
                "Pheromone Mean": r.pheromone_mean,
                "Entropy": r.entropy,
            }
            for r in self.convergence_history
        ]
        return pd.DataFrame(records)


class AntColonyOptimizer:
    """
    Base class for Ant Colony Optimization metaheuristics on the TSP.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 3.0,
        rho: float = 0.5,
        q: float = 100.0,
        num_ants: int = 20,
        max_iter: int = 60,
        initial_pheromone: Optional[float] = None,
        seed: Optional[int] = 42,
    ):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        if not (0 < rho < 1):
            raise ValueError(f"rho must be in ]0, 1[, got {rho}")
        if q <= 0:
            raise ValueError(f"q must be > 0, got {q}")
        if num_ants < 2:
            raise ValueError(f"num_ants must be >= 2, got {num_ants}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.q = float(q)
        self.num_ants = int(num_ants)
        self.max_iter = int(max_iter)
        self.initial_pheromone = initial_pheromone
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def initialize_pheromone(self, instance: TSPInstance) -> np.ndarray:
        """
        Initializes the pheromone matrix tau_0.
        Default heuristic: tau_0 = m / L_nn or fallback to 1.0.
        """
        n = instance.dimension
        if self.initial_pheromone is not None:
            tau0 = float(self.initial_pheromone)
        else:
            # Estimate L_nn via nearest neighbor heuristic from city 0
            visited = [0]
            unvisited = set(range(1, n))
            current = 0
            approx_len = 0.0
            while unvisited:
                next_city = min(unvisited, key=lambda c: instance.distance_matrix[current, c])
                approx_len += float(instance.distance_matrix[current, next_city])
                visited.append(next_city)
                unvisited.remove(next_city)
                current = next_city
            approx_len += float(instance.distance_matrix[current, visited[0]])
            # tau_0 formula standard from Dorigo & Stützle: tau_0 = m / approx_len
            tau0 = max(1e-4, float(self.num_ants / approx_len))

        pheromone = np.full((n, n), tau0, dtype=np.float64)
        np.fill_diagonal(pheromone, 0.0)
        return pheromone

    def compute_transition_probabilities(
        self,
        current_city: int,
        unvisited: List[int],
        pheromone: np.ndarray,
        heuristic: np.ndarray,
    ) -> np.ndarray:
        """
        Computes transition probability vector P_{ij}^k:
        P_{ij}^k = (tau_{ij}^alpha * eta_{ij}^beta) / sum_l(tau_{il}^alpha * eta_{il}^beta)
        """
        tau_vals = pheromone[current_city, unvisited] ** self.alpha
        eta_vals = heuristic[current_city, unvisited] ** self.beta
        weights = tau_vals * eta_vals

        sum_weights = np.sum(weights)
        if sum_weights <= 1e-15 or np.isnan(sum_weights):
            # Uniform fallback if probabilities degenerate
            probs = np.full(len(unvisited), 1.0 / len(unvisited), dtype=np.float64)
        else:
            probs = weights / sum_weights

        return probs

    def calculate_entropy(self, pheromone: np.ndarray) -> float:
        """
        Calculates normalized Shannon entropy of the pheromone distribution.
        H = - sum(p * log(p)). Higher = dispersed/exploratory, Lower = concentrated/converged.
        """
        n = pheromone.shape[0]
        row_sums = np.sum(pheromone, axis=1, keepdims=True)
        # Avoid division by zero on diagonal
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        prob_matrix = pheromone / row_sums

        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.where(prob_matrix > 1e-12, np.log(prob_matrix), 0.0)
            entropy_per_row = -np.sum(prob_matrix * log_p, axis=1)

        mean_entropy = float(np.mean(entropy_per_row))
        max_possible_entropy = np.log(max(2, n - 1))
        normalized_entropy = mean_entropy / max_possible_entropy if max_possible_entropy > 0 else 0.0
        return float(np.clip(normalized_entropy, 0.0, 1.0))

    def solve(
        self,
        instance: TSPInstance,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        record_steps: bool = False,
    ) -> ACOResult:
        """Abstract execution method to be overridden by concrete algorithms."""
        raise NotImplementedError


class AntSystem(AntColonyOptimizer):
    """
    Standard Ant System (AS) implementation (Dorigo et al., 1996).
    Global update deposited by all m ants inversely proportional to tour length.
    """

    def solve(
        self,
        instance: TSPInstance,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        record_steps: bool = False,
    ) -> ACOResult:
        start_time = time.perf_counter()
        n = instance.dimension
        pheromone = self.initialize_pheromone(instance)
        heuristic = instance.heuristic_matrix

        best_global_tour: Optional[Tour] = None
        convergence_history: List[IterationRecord] = []
        step_records: List[StepDecision] = [] if record_steps else None

        for iteration in range(1, self.max_iter + 1):
            iteration_tours: List[Tour] = []

            # 1. Distribute initial cities for each ant
            # If num_ants >= n, distribute cyclically, else random choice
            initial_cities = [
                i % n if self.num_ants >= n else int(self.rng.integers(0, n))
                for i in range(self.num_ants)
            ]

            # 2. Tour construction for all m ants
            for ant_id in range(self.num_ants):
                current_city = initial_cities[ant_id]
                path = [current_city]
                unvisited = set(range(n))
                unvisited.remove(current_city)

                while unvisited:
                    unvisited_list = list(unvisited)
                    probs = self.compute_transition_probabilities(
                        current_city=current_city,
                        unvisited=unvisited_list,
                        pheromone=pheromone,
                        heuristic=heuristic,
                    )

                    # Roulette wheel selection
                    chosen_idx = int(self.rng.choice(len(unvisited_list), p=probs))
                    next_city = unvisited_list[chosen_idx]

                    if record_steps and iteration == 1 and ant_id == 0:
                        step_records.append(
                            StepDecision(
                                ant_id=ant_id,
                                current_city=current_city,
                                candidate_cities=unvisited_list,
                                probabilities=probs.tolist(),
                                chosen_city=next_city,
                                pheromone_value=float(pheromone[current_city, next_city]),
                                heuristic_value=float(heuristic[current_city, next_city]),
                            )
                        )

                    path.append(next_city)
                    unvisited.remove(next_city)
                    current_city = next_city

                tour = instance.create_tour(path)
                iteration_tours.append(tour)

            # 3. Analyze iteration performance
            lengths = [t.length for t in iteration_tours]
            best_iter_tour = min(iteration_tours, key=lambda t: t.length)
            worst_len = max(lengths)
            mean_len = float(np.mean(lengths))

            if best_global_tour is None or best_iter_tour.length < best_global_tour.length:
                best_global_tour = best_iter_tour

            # 4. Evaporation: tau_ij <- (1 - rho) * tau_ij
            pheromone = (1.0 - self.rho) * pheromone

            # 5. Global Pheromone Deposit by all m ants: Delta_tau_ij = sum(Q / L_k)
            delta_pheromone = np.zeros((n, n), dtype=np.float64)
            for tour in iteration_tours:
                deposit = self.q / tour.length
                path = tour.path
                for i in range(n):
                    u = path[i]
                    v = path[(i + 1) % n]
                    delta_pheromone[u, v] += deposit
                    delta_pheromone[v, u] += deposit  # Symmetric

            pheromone += delta_pheromone
            np.fill_diagonal(pheromone, 0.0)

            # Minimum pheromone floor to avoid numerical starvation
            pheromone = np.maximum(pheromone, 1e-6)

            # 6. Capture telemetry
            entropy_val = self.calculate_entropy(pheromone)
            record = IterationRecord(
                iteration=iteration,
                best_length=float(best_global_tour.length),
                mean_length=mean_len,
                worst_length=worst_len,
                best_path=list(best_global_tour.path),
                pheromone_min=float(np.min(pheromone[pheromone > 0])),
                pheromone_max=float(np.max(pheromone)),
                pheromone_mean=float(np.mean(pheromone)),
                entropy=entropy_val,
            )
            convergence_history.append(record)

            if progress_callback:
                progress_callback(iteration, self.max_iter, float(best_global_tour.length))

        elapsed_sec = time.perf_counter() - start_time

        return ACOResult(
            algorithm_name="Ant System (AS)",
            instance_name=instance.name,
            best_tour=best_global_tour,
            convergence_history=convergence_history,
            final_pheromone_matrix=pheromone,
            execution_time_sec=elapsed_sec,
            hyperparameters={
                "alpha": self.alpha,
                "beta": self.beta,
                "rho": self.rho,
                "q": self.q,
                "num_ants": self.num_ants,
                "max_iter": self.max_iter,
                "seed": self.seed,
            },
            step_records=step_records,
        )
