"""
Unit tests for Ant System (AS) algorithmic engine, probabilities, and pheromone dynamics.
Author: El Kachabi Badr-eddine
GitHub: https://github.com/el-kachabi
"""

import numpy as np
import pytest
from src.aco import AntColonyOptimizer, AntSystem
from src.tsp import TSPInstance


def test_transition_probabilities_sum_to_one():
    instance = TSPInstance.load_preset("Micro_5_Villes")
    optimizer = AntSystem(alpha=1.0, beta=3.0, rho=0.5, num_ants=5, max_iter=10, seed=42)
    pheromone = optimizer.initialize_pheromone(instance)

    unvisited = [1, 2, 3, 4]
    probs = optimizer.compute_transition_probabilities(
        current_city=0,
        unvisited=unvisited,
        pheromone=pheromone,
        heuristic=instance.heuristic_matrix,
    )

    assert len(probs) == len(unvisited)
    assert pytest.approx(float(np.sum(probs)), abs=1e-6) == 1.0
    assert np.all(probs >= 0.0)


def test_ant_system_solve_micro_instance():
    instance = TSPInstance.load_preset("Micro_5_Villes")
    optimizer = AntSystem(
        alpha=1.0,
        beta=3.0,
        rho=0.4,
        q=100.0,
        num_ants=15,
        max_iter=30,
        seed=42,
    )

    result = optimizer.solve(instance, record_steps=True)

    # Valid tour returned
    assert result.best_tour.is_valid is True
    assert len(result.best_tour.path) == 5
    assert result.execution_time_sec > 0.0
    assert len(result.convergence_history) == 30

    # Best tour distance matches known exact optimum for Micro_5_Villes (264.8528)
    assert pytest.approx(result.best_tour.length, rel=1e-3) == 264.8528

    # Step decisions recorded
    assert result.step_records is not None
    assert len(result.step_records) > 0

    # Convergence history monotonic for best_length
    best_lengths = [r.best_length for r in result.convergence_history]
    for i in range(len(best_lengths) - 1):
        assert best_lengths[i + 1] <= best_lengths[i] + 1e-6


def test_reproducibility():
    instance = TSPInstance.load_preset("Micro_5_Villes")
    opt1 = AntSystem(alpha=1.0, beta=2.0, rho=0.5, num_ants=10, max_iter=15, seed=123)
    res1 = opt1.solve(instance)

    opt2 = AntSystem(alpha=1.0, beta=2.0, rho=0.5, num_ants=10, max_iter=15, seed=123)
    res2 = opt2.solve(instance)

    assert pytest.approx(res1.best_tour.length) == res2.best_tour.length
    assert res1.best_tour.path == res2.best_tour.path
