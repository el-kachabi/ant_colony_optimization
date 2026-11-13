"""
Unit tests for ACO variants (EAS, ACS) and baseline solvers (NN, 2-Opt, Brute Force).
Checks exact optimality certification and heuristic bounds.
"""

import pytest
from src.aco import AntSystem
from src.baselines import BruteForceSolver, NearestNeighborSolver, TwoOptSolver
from src.tsp import TSPInstance
from src.variants import AntColonySystem, ElitistAntSystem


def test_brute_force_exact_solution_5_cities():
    instance = TSPInstance.load_preset("Micro_5_Villes")
    solver = BruteForceSolver(max_cities=10)
    result = solver.solve(instance)

    assert result.best_tour.is_valid is True
    # The true mathematical optimum of Micro_5_Villes is 264.8528
    assert pytest.approx(result.best_tour.length, rel=1e-4) == 264.8528
    assert result.iterations_or_evaluations == 12  # (5-1)! / 2 = 12


def test_nearest_neighbor_and_two_opt():
    instance = TSPInstance.load_preset("Micro_5_Villes")
    nn = NearestNeighborSolver(multi_start=True).solve(instance)
    assert nn.best_tour.is_valid is True

    two_opt = TwoOptSolver().solve(instance, initial_path=nn.best_tour.path)
    assert two_opt.best_tour.is_valid is True
    # 2-Opt should improve or equal Nearest Neighbor
    assert two_opt.best_tour.length <= nn.best_tour.length + 1e-6


def test_elitist_ant_system():
    instance = TSPInstance.load_preset("Micro_5_Villes")
    eas = ElitistAntSystem(
        alpha=1.0,
        beta=3.0,
        rho=0.4,
        q=100.0,
        num_ants=15,
        max_iter=25,
        elitist_weight=5.0,
        seed=42,
    )
    result = eas.solve(instance)

    assert result.best_tour.is_valid is True
    assert pytest.approx(result.best_tour.length, rel=1e-3) == 264.8528
    assert len(result.convergence_history) == 25


def test_ant_colony_system():
    instance = TSPInstance.load_preset("Micro_5_Villes")
    acs = AntColonySystem(
        alpha=1.0,
        beta=3.0,
        rho=0.1,
        q=100.0,
        num_ants=10,
        max_iter=25,
        q0=0.85,
        xi=0.1,
        seed=42,
    )
    result = acs.solve(instance)

    assert result.best_tour.is_valid is True
    assert pytest.approx(result.best_tour.length, rel=1e-3) == 264.8528
    assert len(result.convergence_history) == 25


def test_all_solvers_bound_against_exact_optimum():
    """
    Mathematical consistency check: no heuristic should ever find
    a tour strictly shorter than the exact exhaustive brute force optimum.
    """
    instance = TSPInstance.load_preset("Micro_5_Villes")

    exact_res = BruteForceSolver().solve(instance)
    opt_len = exact_res.best_tour.length

    as_res = AntSystem(seed=42, max_iter=20).solve(instance)
    eas_res = ElitistAntSystem(seed=42, max_iter=20).solve(instance)
    acs_res = AntColonySystem(seed=42, max_iter=20).solve(instance)
    nn_res = NearestNeighborSolver().solve(instance)
    two_opt_res = TwoOptSolver().solve(instance)

    tolerance = 1e-6
    assert as_res.best_tour.length >= opt_len - tolerance
    assert eas_res.best_tour.length >= opt_len - tolerance
    assert acs_res.best_tour.length >= opt_len - tolerance
    assert nn_res.best_tour.length >= opt_len - tolerance
    assert two_opt_res.best_tour.length >= opt_len - tolerance
