"""
Configuration and settings for the Ant Colony Optimization (ACO) & TSP Platform.
Open-source research and educational environment.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# ==============================================================================
# PROJECT METADATA & BRANDING
# ==============================================================================
APP_TITLE = "Ant Colony Optimization & TSP Platform"
APP_SUBTITLE = "Métaheuristique Bio-Inspirée, Modélisation Mathématique & Résolution Interactive"
PROJECT_NAME = "ACO & TSP Lab"
VERSION = "1.0.0"

# ==============================================================================
# DEFAULT ALGORITHM HYPERPARAMETERS
# ==============================================================================
DEFAULT_ALPHA = 1.0              # Importance of pheromone trail (exploitation / memory)
DEFAULT_BETA = 3.0               # Importance of heuristic visibility (greedy exploration)
DEFAULT_RHO = 0.5                # Pheromone evaporation rate (0 < rho < 1)
DEFAULT_Q = 100.0                # Total pheromone quantity constant
DEFAULT_NUM_ANTS = 20            # Number of artificial ants in colony
DEFAULT_MAX_ITER = 60            # Maximum iterations
DEFAULT_SEED = 42                # Reproducibility seed

# Specific to Elitist Ant System (EAS)
DEFAULT_ELITIST_WEIGHT = 5.0     # Extra pheromone multiplier 'e' for best-so-far tour

# Specific to Ant Colony System (ACS)
DEFAULT_Q0 = 0.9                 # Pseudo-random proportional rule threshold
DEFAULT_XI = 0.1                 # Local pheromone evaporation rate
DEFAULT_INITIAL_PHEROMONE = 1.0  # Tau_0 fallback

# Bounds for interactive UI sliders
PARAM_BOUNDS = {
    "alpha": {"min": 0.0, "max": 5.0, "step": 0.1, "default": 1.0},
    "beta": {"min": 0.0, "max": 10.0, "step": 0.2, "default": 3.0},
    "rho": {"min": 0.01, "max": 0.99, "step": 0.01, "default": 0.50},
    "q": {"min": 10.0, "max": 1000.0, "step": 10.0, "default": 100.0},
    "num_ants": {"min": 2, "max": 100, "step": 1, "default": 20},
    "max_iter": {"min": 10, "max": 300, "step": 5, "default": 60},
    "elitist_weight": {"min": 1.0, "max": 20.0, "step": 1.0, "default": 5.0},
    "q0": {"min": 0.0, "max": 1.0, "step": 0.05, "default": 0.90},
    "xi": {"min": 0.01, "max": 0.50, "step": 0.01, "default": 0.10},
}

# ==============================================================================
# DESIGN SYSTEM & COLOR PALETTES (Dark Modern Theme)
# ==============================================================================
THEME_COLORS = {
    "bg_dark": "#0B0F19",
    "card_bg": "#131B2E",
    "card_border": "#1E293B",
    "text_main": "#F8FAFC",
    "text_muted": "#94A3B8",
    "primary_cyan": "#00F2FE",
    "secondary_blue": "#4FACFE",
    "accent_orange": "#FF9900",
    "accent_purple": "#8B5CF6",
    "success_green": "#10B981",
    "danger_red": "#EF4444",
    "pheromone_high": "#FF9900",
    "pheromone_low": "#1E293B",
    "best_tour": "#00F2FE",
}

# Plotly theme layout defaults
PLOTLY_LAYOUT_DEFAULTS = {
    "paper_bgcolor": "rgba(11, 15, 25, 0)",
    "plot_bgcolor": "rgba(19, 27, 46, 0.4)",
    "font": {"family": "Inter, Roboto, sans-serif", "color": "#E2E8F0"},
    "xaxis": {"gridcolor": "#1E293B", "zerolinecolor": "#334155"},
    "yaxis": {"gridcolor": "#1E293B", "zerolinecolor": "#334155"},
    "margin": {"l": 40, "r": 40, "t": 50, "b": 40},
}

# ==============================================================================
# BUILT-IN TSP CITY PRESETS (Normalized to 100x100 space)
# ==============================================================================
PRESET_INSTANCES: Dict[str, Dict[str, any]] = {
    "Micro_5_Villes": {
        "name": "Micro Instance (5 Villes - Cas Didactique)",
        "description": "Idéal pour calculs manuels pas-à-pas et vérification exacte",
        "coords": [
            (20.0, 20.0),
            (20.0, 80.0),
            (80.0, 80.0),
            (80.0, 20.0),
            (50.0, 50.0),
        ],
        "labels": ["Ville 0", "Ville 1", "Ville 2", "Ville 3", "Ville 4"],
        "optimal_known": 264.85,
    },
    "Small_10_Villes": {
        "name": "Petite Instance (10 Villes - Cercle Déformé)",
        "description": "Parfait pour valider le solveur Brute Force exact vs ACO",
        "coords": [
            (50.0, 85.0),
            (73.5, 77.0),
            (88.0, 58.0),
            (88.0, 35.0),
            (73.5, 16.0),
            (50.0, 8.0),
            (26.5, 16.0),
            (12.0, 35.0),
            (12.0, 58.0),
            (26.5, 77.0),
        ],
        "labels": [f"V{i}" for i in range(10)],
        "optimal_known": 240.90,
    },
    "Medium_20_Villes": {
        "name": "Moyenne Instance (20 Villes - Cluster & Périphérie)",
        "description": "Échelle intermédiaire illustrant le compromis exploration/exploitation",
        "coords": [
            (15.0, 25.0), (22.0, 80.0), (45.0, 60.0), (55.0, 90.0),
            (80.0, 75.0), (85.0, 40.0), (70.0, 20.0), (35.0, 15.0),
            (50.0, 45.0), (30.0, 50.0), (65.0, 70.0), (40.0, 30.0),
            (10.0, 55.0), (75.0, 85.0), (90.0, 15.0), (60.0, 10.0),
            (25.0, 35.0), (80.0, 55.0), (35.0, 75.0), (55.0, 25.0),
        ],
        "labels": [f"C{i}" for i in range(20)],
        "optimal_known": None,
    },
    "Standard_30_Villes": {
        "name": "Standard Instance (30 Villes - Carte Aléatoire Reproductible)",
        "description": "Instance de référence pour le benchmark des variantes AS, EAS, ACS",
        "coords": [
            (18.0, 24.0), (28.0, 15.0), (36.0, 32.0), (45.0, 21.0),
            (58.0, 35.0), (62.0, 18.0), (75.0, 28.0), (84.0, 12.0),
            (89.0, 38.0), (80.0, 48.0), (92.0, 65.0), (81.0, 78.0),
            (68.0, 62.0), (72.0, 88.0), (54.0, 75.0), (48.0, 92.0),
            (35.0, 80.0), (22.0, 85.0), (12.0, 70.0), (25.0, 60.0),
            (15.0, 45.0), (32.0, 48.0), (42.0, 58.0), (52.0, 45.0),
            (65.0, 40.0), (70.0, 52.0), (40.0, 70.0), (28.0, 72.0),
            (38.0, 38.0), (58.0, 58.0),
        ],
        "labels": [f"N{i}" for i in range(30)],
        "optimal_known": None,
    },
    "Benchmark_eil51": {
        "name": "Benchmark TSPLIB eil51 (51 Villes de référence)",
        "description": "Standard international TSPLIB - Optimum connu : 426",
        "coords": [
            (37.0, 52.0), (49.0, 49.0), (52.0, 64.0), (20.0, 26.0), (40.0, 30.0),
            (21.0, 47.0), (17.0, 63.0), (31.0, 62.0), (52.0, 33.0), (51.0, 21.0),
            (42.0, 41.0), (31.0, 32.0), (5.0, 25.0), (12.0, 42.0), (36.0, 16.0),
            (52.0, 41.0), (27.0, 23.0), (17.0, 33.0), (13.0, 13.0), (57.0, 58.0),
            (62.0, 42.0), (42.0, 57.0), (16.0, 57.0), (8.0, 52.0), (7.0, 38.0),
            (27.0, 68.0), (30.0, 48.0), (43.0, 67.0), (58.0, 48.0), (58.0, 27.0),
            (37.0, 69.0), (38.0, 46.0), (46.0, 10.0), (61.0, 33.0), (62.0, 63.0),
            (63.0, 69.0), (32.0, 22.0), (45.0, 35.0), (59.0, 15.0), (5.0, 6.0),
            (10.0, 17.0), (21.0, 10.0), (5.0, 64.0), (30.0, 15.0), (39.0, 10.0),
            (32.0, 39.0), (25.0, 32.0), (25.0, 55.0), (48.0, 28.0), (56.0, 37.0),
            (30.0, 40.0),
        ],
        "labels": [f"EIL_{i+1}" for i in range(51)],
        "optimal_known": 426.0,
    }
}
