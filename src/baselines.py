"""
Baseline Solvers for TSP Benchmarking: Nearest Neighbor, 2-Opt, and Exact Brute Force.
Reference algorithms for empirical comparison and validation.
"""

from dataclasses import dataclass
import itertools
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.tsp import Tour, TSPInstance


@dataclass
class BaselineResult:
    """Standardized output structure for baseline solvers."""
    solver_name: str
    instance_name: str
    best_tour: Tour
    execution_time_sec: float
    iterations_or_evaluations: int
    metadata: Dict[str, any]


class NearestNeighborSolver:
    """
    Greedy Nearest Neighbor heuristic (Glouton).
    Can start from a specific city or evaluate all starting cities (Multi-Start).
    """

    def __init__(self, multi_start: bool = True):
        self.multi_start = multi_start

    def solve(self, instance: TSPInstance) -> BaselineResult:
        start_time = time.perf_counter()
        n = instance.dimension
        dist_mat = instance.distance_matrix

        best_tour: Optional[Tour] = None
        evals = 0

        start_nodes = range(n) if self.multi_start else [0]

        for start in start_nodes:
            evals += 1
            visited = [start]
            unvisited = set(range(n))
            unvisited.remove(start)
            current = start

            while unvisited:
                # Find closest unvisited neighbor
                next_city = min(unvisited, key=lambda c: dist_mat[current, c])
                visited.append(next_city)
                unvisited.remove(next_city)
                current = next_city

            tour = instance.create_tour(visited)
            if best_tour is None or tour.length < best_tour.length:
                best_tour = tour

        elapsed = time.perf_counter() - start_time
        return BaselineResult(
            solver_name=f"Nearest Neighbor {'(Multi-Start)' if self.multi_start else '(Single)'}",
            instance_name=instance.name,
            best_tour=best_tour,
            execution_time_sec=elapsed,
            iterations_or_evaluations=evals,
            metadata={"multi_start": self.multi_start},
        )


class TwoOptSolver:
    """
    2-Opt Local Search heuristic.
    Repeatedly untangles crossing edges by reversing sub-segments until local optimality.
    """

    def __init__(self, max_passes: int = 500):
        self.max_passes = max_passes

    def solve(self, instance: TSPInstance, initial_path: Optional[List[int]] = None) -> BaselineResult:
        start_time = time.perf_counter()
        n = instance.dimension
        dist_mat = instance.distance_matrix

        # If no initial path provided, initialize with Nearest Neighbor
        if initial_path is None:
            nn = NearestNeighborSolver(multi_start=False).solve(instance)
            path = list(nn.best_tour.path)
        else:
            path = list(initial_path)

        improved = True
        passes = 0
        swaps = 0

        while improved and passes < self.max_passes:
            improved = False
            passes += 1

            for i in range(n - 1):
                for j in range(i + 2, n if i > 0 else n - 1):
                    # Current edges: (i, i+1) and (j, (j+1)%n)
                    u = path[i]
                    v = path[i + 1]
                    w = path[j]
                    z = path[(j + 1) % n]

                    current_dist = dist_mat[u, v] + dist_mat[w, z]
                    new_dist = dist_mat[u, w] + dist_mat[v, z]

                    if new_dist < current_dist - 1e-9:
                        # Reverse subsegment between i+1 and j
                        path[i + 1 : j + 1] = reversed(path[i + 1 : j + 1])
                        improved = True
                        swaps += 1
                        break
                if improved:
                    break

        best_tour = instance.create_tour(path)
        elapsed = time.perf_counter() - start_time

        return BaselineResult(
            solver_name="2-Opt Local Search",
            instance_name=instance.name,
            best_tour=best_tour,
            execution_time_sec=elapsed,
            iterations_or_evaluations=swaps,
            metadata={"passes": passes, "swaps_applied": swaps},
        )


class BruteForceSolver:
    """
    Exact Exhaustive Solver for small TSP instances (n <= 10).
    Evaluates all (n-1)! / 2 unique Hamiltonian cycles to guarantee global optimality.
    """

    def __init__(self, max_cities: int = 10):
        self.max_cities = max_cities

    def solve(self, instance: TSPInstance) -> BaselineResult:
        n = instance.dimension
        if n > self.max_cities:
            raise ValueError(
                f"Brute Force exact solver is limited to n <= {self.max_cities} cities (requested {n})."
            )

        start_time = time.perf_counter()
        dist_mat = instance.distance_matrix

        best_path: Optional[List[int]] = None
        best_length = float("inf")
        evals = 0

        # Fix node 0 as start to eliminate rotational symmetry: (n-1)! permutations
        remaining_nodes = list(range(1, n))

        for perm in itertools.permutations(remaining_nodes):
            # To break reversal symmetry, only evaluate permutations where perm[0] < perm[-1]
            if perm[0] > perm[-1]:
                continue

            evals += 1
            path = [0] + list(perm)

            # Compute closed tour distance
            cur_len = 0.0
            for i in range(n):
                cur_len += float(dist_mat[path[i], path[(i + 1) % n]])

            if cur_len < best_length:
                best_length = cur_len
                best_path = path

        best_tour = instance.create_tour(best_path)
        elapsed = time.perf_counter() - start_time

        return BaselineResult(
            solver_name="Exact Brute Force",
            instance_name=instance.name,
            best_tour=best_tour,
            execution_time_sec=elapsed,
            iterations_or_evaluations=evals,
            metadata={"permutations_evaluated": evals, "is_exact": True},
        )
