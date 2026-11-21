"""
Moteur Mathématique et Algorithmique Unifié - Ant Colony Optimization (ACO) & TSP.
Version pédagogique simplifiée : Modélisation TSP, Ant System (AS) et méthodes de référence.
"""

from dataclasses import dataclass
import itertools
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

# ==============================================================================
# 1. INSTANCES TSP PRÉDÉFINIES (Coordonnées normalisées 100x100)
# ==============================================================================
PRESET_INSTANCES: Dict[str, Dict] = {
    "Micro_5_Villes": {
        "name": "Micro Instance (5 Villes - Cas Didactique)",
        "coords": [(20.0, 20.0), (20.0, 80.0), (80.0, 80.0), (80.0, 20.0), (50.0, 50.0)],
        "optimal_known": 264.85,
    },
    "Small_10_Villes": {
        "name": "Petite Instance (10 Villes - Validation Exacte)",
        "coords": [
            (50.0, 85.0), (73.5, 77.0), (88.0, 58.0), (88.0, 35.0), (73.5, 16.0),
            (50.0, 8.0), (26.5, 16.0), (12.0, 35.0), (12.0, 58.0), (26.5, 77.0),
        ],
        "optimal_known": 240.90,
    },
    "Medium_20_Villes": {
        "name": "Moyenne Instance (20 Villes - Démonstration ACO)",
        "coords": [
            (15.0, 25.0), (22.0, 80.0), (45.0, 60.0), (55.0, 90.0), (80.0, 75.0),
            (85.0, 40.0), (70.0, 20.0), (35.0, 15.0), (50.0, 45.0), (30.0, 50.0),
            (65.0, 70.0), (40.0, 30.0), (10.0, 55.0), (75.0, 85.0), (90.0, 15.0),
            (60.0, 10.0), (25.0, 35.0), (80.0, 55.0), (35.0, 75.0), (55.0, 25.0),
        ],
        "optimal_known": 374.51,
    },
}


# ==============================================================================
# 2. MODÉLISATION DU PROBLÈME DU VOYAGEUR DE COMMERCE (TSP)
# ==============================================================================
@dataclass(frozen=True)
class City:
    """Représente une ville dans le plan euclidien 2D."""
    id: int
    name: str
    x: float
    y: float


@dataclass
class TSPInstance:
    """Instance complète du TSP : villes, matrice des distances et visibilité heuristique."""
    name: str
    cities: List[City]
    distance_matrix: np.ndarray
    heuristic_matrix: np.ndarray
    optimal_known: Optional[float] = None

    @property
    def dimension(self) -> int:
        return len(self.cities)

    def tour_length(self, path: List[int]) -> float:
        """Calcule la distance totale fermée d'un cycle hamiltonien : L(pi)."""
        n = len(path)
        return float(sum(self.distance_matrix[path[i], path[(i + 1) % n]] for i in range(n)))

    @classmethod
    def from_coordinates(
        cls,
        coords: List[Tuple[float, float]],
        name: str = "TSP Personnalisé",
        optimal_known: Optional[float] = None,
    ) -> "TSPInstance":
        """Construit les matrices de distance d_ij et de visibilité eta_ij = 1 / d_ij."""
        cities = [City(i, f"Ville {i}", float(x), float(y)) for i, (x, y) in enumerate(coords)]
        pts = np.array(coords, dtype=np.float64)
        # Calcul vectorisé des distances euclidiennes
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        with np.errstate(divide="ignore"):
            heuristic = np.where(dist_matrix > 0, 1.0 / dist_matrix, 0.0)
        np.fill_diagonal(heuristic, 0.0)

        return cls(name, cities, dist_matrix, heuristic, optimal_known)

    @classmethod
    def load_preset(cls, key: str) -> "TSPInstance":
        data = PRESET_INSTANCES[key]
        return cls.from_coordinates(data["coords"], data["name"], data.get("optimal_known"))

    @classmethod
    def generate_random(cls, num_cities: int = 15, seed: int = 42) -> "TSPInstance":
        rng = np.random.default_rng(seed)
        coords = [(float(x), float(y)) for x, y in rng.uniform(5.0, 95.0, size=(num_cities, 2))]
        return cls.from_coordinates(coords, f"Carte Aléatoire ({num_cities} Villes)")


# ==============================================================================
# 3. ALGORITHME ANT COLONY OPTIMIZATION (ANT SYSTEM)
# ==============================================================================
@dataclass
class ACOResult:
    """Résultat synthétique d'une exécution ACO."""
    best_path: List[int]
    best_length: float
    history_best: List[float]
    history_mean: List[float]
    pheromone_matrix: np.ndarray
    execution_time_ms: float


class AntSystem:
    """
    Algorithme Ant System (Dorigo et al., 1996) avec option de renforcement élitiste.
    Implémente les 3 mécanismes clés : Transition probabiliste, Évaporation et Dépôt.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 3.0,
        rho: float = 0.5,
        q: float = 100.0,
        num_ants: int = 20,
        max_iter: int = 50,
        elitist_weight: float = 0.0,
        seed: int = 42,
    ):
        self.alpha = float(alpha)          # Poids des phéromones (exploitation)
        self.beta = float(beta)            # Poids de la visibilité (exploration gloutonne)
        self.rho = float(rho)              # Taux d'évaporation (0 < rho < 1)
        self.q = float(q)                  # Constante de dépôt Q
        self.num_ants = int(num_ants)      # Taille de la colonie (m)
        self.max_iter = int(max_iter)      # Nombre d'itérations (T)
        self.elitist_weight = float(elitist_weight)
        self.rng = np.random.default_rng(seed)

    def compute_probabilities(
        self, current: int, unvisited: List[int], pheromone: np.ndarray, heuristic: np.ndarray
    ) -> np.ndarray:
        """
        Règle proportionnelle de transition :
        P_ij = (tau_ij^alpha * eta_ij^beta) / sum(tau_il^alpha * eta_il^beta)
        """
        weights = (pheromone[current, unvisited] ** self.alpha) * (heuristic[current, unvisited] ** self.beta)
        total = np.sum(weights)
        if total <= 1e-15:
            return np.full(len(unvisited), 1.0 / len(unvisited))
        return weights / total

    def solve(self, instance: TSPInstance) -> ACOResult:
        """Exécute la simulation complète de la colonie de fourmis sur l'instance TSP."""
        start_t = time.perf_counter()
        n = instance.dimension
        pheromone = np.full((n, n), 1.0, dtype=np.float64)
        np.fill_diagonal(pheromone, 0.0)

        best_path: List[int] = []
        best_length = float("inf")
        history_best, history_mean = [], []

        for _ in range(self.max_iter):
            all_paths, all_lengths = [], []

            # ÉTAPE 1 : Construction d'une tournée complète par chaque fourmi
            for ant in range(self.num_ants):
                current = ant % n
                path = [current]
                unvisited = list(set(range(n)) - {current})

                while unvisited:
                    probs = self.compute_probabilities(current, unvisited, pheromone, instance.heuristic_matrix)
                    next_city = int(self.rng.choice(unvisited, p=probs))
                    path.append(next_city)
                    unvisited.remove(next_city)
                    current = next_city

                length = instance.tour_length(path)
                all_paths.append(path)
                all_lengths.append(length)

                if length < best_length:
                    best_length = length
                    best_path = list(path)

            # ÉTAPE 2 : Évaporation des phéromones -> tau_ij <- (1 - rho) * tau_ij
            pheromone *= (1.0 - self.rho)

            # ÉTAPE 3 : Dépôt de phéromones par chaque fourmi -> Delta_tau = Q / L_k
            for path, length in zip(all_paths, all_lengths):
                deposit = self.q / length
                for i in range(n):
                    u, v = path[i], path[(i + 1) % n]
                    pheromone[u, v] += deposit
                    pheromone[v, u] += deposit

            # Option Élitiste : Renforcement supplémentaire du meilleur chemin global
            if self.elitist_weight > 0 and best_path:
                elite_deposit = self.elitist_weight * (self.q / best_length)
                for i in range(n):
                    u, v = best_path[i], best_path[(i + 1) % n]
                    pheromone[u, v] += elite_deposit
                    pheromone[v, u] += elite_deposit

            pheromone = np.maximum(pheromone, 1e-6)
            np.fill_diagonal(pheromone, 0.0)

            history_best.append(best_length)
            history_mean.append(float(np.mean(all_lengths)))

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return ACOResult(best_path, best_length, history_best, history_mean, pheromone, elapsed_ms)


# ==============================================================================
# 4. ALGORITHMES DE RÉFÉRENCE (POUR COMPARAISON PÉDAGOGIQUE)
# ==============================================================================
def solve_nearest_neighbor(instance: TSPInstance) -> Tuple[List[int], float, float]:
    """Heuristique gloutonne du Plus Proche Voisin (choisit toujours la ville la plus proche)."""
    start_t = time.perf_counter()
    n = instance.dimension
    path = [0]
    unvisited = set(range(1, n))

    while unvisited:
        curr = path[-1]
        next_city = min(unvisited, key=lambda city: instance.distance_matrix[curr, city])
        path.append(next_city)
        unvisited.remove(next_city)

    length = instance.tour_length(path)
    return path, length, (time.perf_counter() - start_t) * 1000.0


def solve_brute_force(instance: TSPInstance) -> Tuple[List[int], float, float]:
    """Recherche exhaustive exacte évaluant toutes les permutations (réservé à N <= 10)."""
    start_t = time.perf_counter()
    n = instance.dimension
    best_path, best_len = [], float("inf")

    for perm in itertools.permutations(range(1, n)):
        candidate = [0] + list(perm)
        length = instance.tour_length(candidate)
        if length < best_len:
            best_len = length
            best_path = candidate

    return best_path, best_len, (time.perf_counter() - start_t) * 1000.0
