"""
Unit tests for TSP problem modeling, distance matrix, and tour validation.
Verifies Euclidean metric invariants, tour closure, and preset integrity.
"""

import numpy as np
import pytest
from src.tsp import City, Tour, TSPInstance


def test_city_distance():
    c1 = City(id=0, name="A", x=0.0, y=0.0)
    c2 = City(id=1, name="B", x=3.0, y=4.0)
    assert pytest.approx(c1.distance_to(c2), rel=1e-5) == 5.0
    assert pytest.approx(c2.distance_to(c1), rel=1e-5) == 5.0


def test_tsp_instance_properties():
    coords = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0), (0.0, 4.0)]
    instance = TSPInstance.from_coordinates(coords, name="Rectangle_4")

    assert instance.dimension == 4
    # Symmetric distance matrix
    assert np.allclose(instance.distance_matrix, instance.distance_matrix.T)
    # Diagonal is zero
    assert np.all(np.diag(instance.distance_matrix) == 0.0)
    # Heuristic matrix diagonal is zero
    assert np.all(np.diag(instance.heuristic_matrix) == 0.0)
    # Check specific distance
    assert pytest.approx(instance.distance_matrix[0, 1]) == 3.0
    assert pytest.approx(instance.distance_matrix[1, 2]) == 4.0
    assert pytest.approx(instance.distance_matrix[0, 2]) == 5.0
    # Heuristic visibility is 1 / distance
    assert pytest.approx(instance.heuristic_matrix[0, 1]) == 1.0 / 3.0
    assert pytest.approx(instance.heuristic_matrix[0, 2]) == 1.0 / 5.0


def test_tour_validation_and_length():
    coords = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0), (0.0, 4.0)]
    instance = TSPInstance.from_coordinates(coords, name="Rectangle_4")

    # Valid Hamiltonian cycle
    valid_path = [0, 1, 2, 3]
    tour = instance.create_tour(valid_path)
    assert tour.is_valid is True
    # Perimeter of 3x4 rectangle = 3 + 4 + 3 + 4 = 14.0
    assert pytest.approx(tour.length) == 14.0

    # Invalid: duplicate city
    invalid_path_dup = [0, 1, 1, 3]
    is_valid, msg = instance.validate_tour(invalid_path_dup)
    assert is_valid is False
    assert "doublons" in msg

    # Invalid: missing city
    invalid_path_len = [0, 1, 2]
    is_valid, msg = instance.validate_tour(invalid_path_len)
    assert is_valid is False


def test_preset_loading():
    instance = TSPInstance.load_preset("Micro_5_Villes")
    assert instance.dimension == 5
    assert instance.optimal_known is not None
    assert instance.distance_matrix.shape == (5, 5)
