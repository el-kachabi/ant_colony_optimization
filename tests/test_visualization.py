"""
Unit tests for Plotly visualization generation.
Verifies all interactive figures, maps, and matrices build properly.
"""

import numpy as np
import pytest
from src.aco import AntSystem
from src.tsp import TSPInstance
from src.visualization import (
    plot_colony_entropy_and_pheromones,
    plot_convergence_curves,
    plot_pheromone_network,
    plot_sensitivity_heatmap,
    plot_step_probabilities_barchart,
    plot_step_simulation_view,
    plot_tour,
    plot_tsp_instance,
)


def test_plotly_chart_creations():
    instance = TSPInstance.load_preset("Micro_5_Villes")
    assert plot_tsp_instance(instance) is not None

    aco = AntSystem(alpha=1.0, beta=3.0, rho=0.4, num_ants=10, max_iter=15, seed=42)
    res = aco.solve(instance, record_steps=True)

    assert plot_tour(instance, res.best_tour) is not None
    assert plot_pheromone_network(instance, res.final_pheromone_matrix, best_tour=res.best_tour) is not None
    assert plot_convergence_curves(res.convergence_history, optimal_known=instance.optimal_known) is not None
    assert plot_colony_entropy_and_pheromones(res.convergence_history) is not None

    assert res.step_records is not None and len(res.step_records) > 0
    step0 = res.step_records[0]
    assert plot_step_simulation_view(instance, step0, [step0.current_city], res.final_pheromone_matrix) is not None
    assert plot_step_probabilities_barchart(step0, instance) is not None

    x_vals = [0.5, 1.0, 2.0]
    y_vals = [1.0, 2.0, 3.0]
    z_mat = np.array([[280.0, 275.0, 270.0], [275.0, 268.0, 265.0], [270.0, 266.0, 264.8]])
    assert plot_sensitivity_heatmap("Alpha", x_vals, "Beta", y_vals, z_mat) is not None
