"""
Advanced ACO Variants: Elitist Ant System (EAS) and Ant Colony System (ACS).
Mathematical models and specialized state transition rules.
"""

from dataclasses import dataclass
import time
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from src.aco import ACOResult, AntColonyOptimizer, IterationRecord, StepDecision
from src.tsp import Tour, TSPInstance


class ElitistAntSystem(AntColonyOptimizer):
    """
    Elitist Ant System (EAS) - Dorigo, Maniezzo, Colorni (1996).
    Reinforces the best-so-far tour with an additional weight: e * (Q / L_best).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 3.0,
        rho: float = 0.5,
        q: float = 100.0,
        num_ants: int = 20,
        max_iter: int = 60,
        elitist_weight: float = 5.0,
        initial_pheromone: Optional[float] = None,
        seed: Optional[int] = 42,
    ):
        super().__init__(
            alpha=alpha,
            beta=beta,
            rho=rho,
            q=q,
            num_ants=num_ants,
            max_iter=max_iter,
            initial_pheromone=initial_pheromone,
            seed=seed,
        )
        if elitist_weight < 0:
            raise ValueError(f"elitist_weight must be >= 0, got {elitist_weight}")
        self.elitist_weight = float(elitist_weight)

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
            initial_cities = [
                i % n if self.num_ants >= n else int(self.rng.integers(0, n))
                for i in range(self.num_ants)
            ]

            # 1. Build tours for all m ants
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

            # 2. Evaluate performance
            lengths = [t.length for t in iteration_tours]
            best_iter_tour = min(iteration_tours, key=lambda t: t.length)
            worst_len = max(lengths)
            mean_len = float(np.mean(lengths))

            if best_global_tour is None or best_iter_tour.length < best_global_tour.length:
                best_global_tour = best_iter_tour

            # 3. Evaporation: tau_ij <- (1 - rho) * tau_ij
            pheromone = (1.0 - self.rho) * pheromone

            # 4. Standard Deposit by all m ants
            delta_pheromone = np.zeros((n, n), dtype=np.float64)
            for tour in iteration_tours:
                deposit = self.q / tour.length
                path = tour.path
                for i in range(n):
                    u = path[i]
                    v = path[(i + 1) % n]
                    delta_pheromone[u, v] += deposit
                    delta_pheromone[v, u] += deposit

            # 5. ELITIST DEPOSIT: extra reinforcement of best-so-far tour
            if self.elitist_weight > 0 and best_global_tour is not None:
                elitist_deposit = self.elitist_weight * (self.q / best_global_tour.length)
                best_path = best_global_tour.path
                for i in range(n):
                    u = best_path[i]
                    v = best_path[(i + 1) % n]
                    delta_pheromone[u, v] += elitist_deposit
                    delta_pheromone[v, u] += elitist_deposit

            pheromone += delta_pheromone
            np.fill_diagonal(pheromone, 0.0)
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
            algorithm_name="Elitist Ant System (EAS)",
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
                "elitist_weight": self.elitist_weight,
                "seed": self.seed,
            },
            step_records=step_records,
        )


class AntColonySystem(AntColonyOptimizer):
    """
    Ant Colony System (ACS) - Dorigo & Gambardella (1997).
    Features:
    1. Pseudo-random proportional rule with threshold q0 (direct exploitation vs roulette).
    2. Local pheromone update during tour construction (promotes search diversification).
    3. Global pheromone update applied only to the best-so-far tour.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 3.0,
        rho: float = 0.1,
        q: float = 100.0,
        num_ants: int = 15,
        max_iter: int = 60,
        q0: float = 0.90,
        xi: float = 0.10,
        initial_pheromone: Optional[float] = None,
        seed: Optional[int] = 42,
    ):
        super().__init__(
            alpha=alpha,
            beta=beta,
            rho=rho,
            q=q,
            num_ants=num_ants,
            max_iter=max_iter,
            initial_pheromone=initial_pheromone,
            seed=seed,
        )
        if not (0 <= q0 <= 1):
            raise ValueError(f"q0 must be in [0, 1], got {q0}")
        if not (0 < xi < 1):
            raise ValueError(f"xi must be in ]0, 1[, got {xi}")
        self.q0 = float(q0)
        self.xi = float(xi)

    def initialize_pheromone(self, instance: TSPInstance) -> Tuple[np.ndarray, float]:
        """
        In ACS, tau_0 = 1 / (n * L_nn). Returns pheromone matrix and tau_0 scalar.
        """
        n = instance.dimension
        if self.initial_pheromone is not None:
            tau0 = float(self.initial_pheromone)
        else:
            # Nearest neighbor from city 0
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
            tau0 = max(1e-6, 1.0 / (n * approx_len))

        pheromone = np.full((n, n), tau0, dtype=np.float64)
        np.fill_diagonal(pheromone, 0.0)
        return pheromone, tau0

    def solve(
        self,
        instance: TSPInstance,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        record_steps: bool = False,
    ) -> ACOResult:
        start_time = time.perf_counter()
        n = instance.dimension
        pheromone, tau0 = self.initialize_pheromone(instance)
        heuristic = instance.heuristic_matrix

        best_global_tour: Optional[Tour] = None
        convergence_history: List[IterationRecord] = []
        step_records: List[StepDecision] = [] if record_steps else None

        for iteration in range(1, self.max_iter + 1):
            iteration_tours: List[Tour] = []
            initial_cities = [
                i % n if self.num_ants >= n else int(self.rng.integers(0, n))
                for i in range(self.num_ants)
            ]

            # 1. Ants build tours in parallel with LOCAL pheromone updates
            for ant_id in range(self.num_ants):
                current_city = initial_cities[ant_id]
                path = [current_city]
                unvisited = set(range(n))
                unvisited.remove(current_city)

                while unvisited:
                    unvisited_list = list(unvisited)

                    # Attractiveness: tau^alpha * eta^beta
                    tau_vals = pheromone[current_city, unvisited_list] ** self.alpha
                    eta_vals = heuristic[current_city, unvisited_list] ** self.beta
                    scores = tau_vals * eta_vals

                    # Pseudo-Random Proportional Rule
                    q_rand = float(self.rng.uniform(0.0, 1.0))
                    if q_rand <= self.q0:
                        # Exploitation: direct argmax
                        best_candidate_idx = int(np.argmax(scores))
                        next_city = unvisited_list[best_candidate_idx]
                        probs = [1.0 if i == best_candidate_idx else 0.0 for i in range(len(unvisited_list))]
                    else:
                        # Exploration: roulette wheel
                        sum_scores = np.sum(scores)
                        if sum_scores <= 1e-15:
                            probs = np.full(len(unvisited_list), 1.0 / len(unvisited_list))
                        else:
                            probs = (scores / sum_scores).tolist()
                        chosen_idx = int(self.rng.choice(len(unvisited_list), p=probs))
                        next_city = unvisited_list[chosen_idx]

                    if record_steps and iteration == 1 and ant_id == 0:
                        step_records.append(
                            StepDecision(
                                ant_id=ant_id,
                                current_city=current_city,
                                candidate_cities=unvisited_list,
                                probabilities=probs,
                                chosen_city=next_city,
                                pheromone_value=float(pheromone[current_city, next_city]),
                                heuristic_value=float(heuristic[current_city, next_city]),
                            )
                        )

                    # LOCAL PHEROMONE UPDATE: tau_ij <- (1 - xi)*tau_ij + xi*tau0
                    pheromone[current_city, next_city] = (1.0 - self.xi) * pheromone[current_city, next_city] + self.xi * tau0
                    pheromone[next_city, current_city] = pheromone[current_city, next_city]

                    path.append(next_city)
                    unvisited.remove(next_city)
                    current_city = next_city

                # Return to start city: local update on closing edge
                start_c = path[0]
                end_c = path[-1]
                pheromone[end_c, start_c] = (1.0 - self.xi) * pheromone[end_c, start_c] + self.xi * tau0
                pheromone[start_c, end_c] = pheromone[end_c, start_c]

                tour = instance.create_tour(path)
                iteration_tours.append(tour)

            # 2. Iteration evaluation
            lengths = [t.length for t in iteration_tours]
            best_iter_tour = min(iteration_tours, key=lambda t: t.length)
            worst_len = max(lengths)
            mean_len = float(np.mean(lengths))

            if best_global_tour is None or best_iter_tour.length < best_global_tour.length:
                best_global_tour = best_iter_tour

            # 3. GLOBAL PHEROMONE UPDATE: applied ONLY to best-so-far tour
            # tau_ij <- (1 - rho)*tau_ij + rho * (Q / L_best)
            best_path = best_global_tour.path
            deposit_best = self.q / best_global_tour.length
            for i in range(n):
                u = best_path[i]
                v = best_path[(i + 1) % n]
                pheromone[u, v] = (1.0 - self.rho) * pheromone[u, v] + self.rho * deposit_best
                pheromone[v, u] = pheromone[u, v]

            np.fill_diagonal(pheromone, 0.0)
            pheromone = np.maximum(pheromone, 1e-6)

            # 4. Telemetry
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
            algorithm_name="Ant Colony System (ACS)",
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
                "q0": self.q0,
                "xi": self.xi,
                "seed": self.seed,
            },
            step_records=step_records,
        )
