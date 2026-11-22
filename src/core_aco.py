"""
Moteur Mathématique et Algorithmique Unifié - Ant Colony Optimization (ACO) & TSP.
Version pédagogique simplifiée : Modélisation TSP, Ant System (AS) et méthodes de référence.
Chaque classe et méthode est commentée pas-à-pas pour faciliter l'explication du code.
"""

from dataclasses import dataclass
import itertools
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

# ==============================================================================
# 1. INSTANCES TSP PRÉDÉFINIES (Coordonnées normalisées dans un plan 100 x 100)
# ==============================================================================
# Ces 3 instances servent de banc d'essai reproductible :
# - 5 villes  : permet de vérifier le calcul à la main et de comparer avec Brute Force (12 cycles).
# - 10 villes : limite haute de la recherche exacte Brute Force (181 440 cycles).
# - 20 villes : démontre la supériorité de l'ACO (374.51) face à l'algorithme Glouton (390.29).
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
# Décorateur @dataclass(frozen=True) :
# - @dataclass génère automatiquement le constructeur __init__ et l'affichage __repr__.
# - L'option frozen=True rend l'objet immuable (ses coordonnées x, y ne peuvent plus être modifiées par erreur).
@dataclass(frozen=True)
class City:
    """Représente une ville (sommet du graphe) dans le plan euclidien 2D."""
    id: int      # Identifiant unique de la ville (0 à n-1)
    name: str    # Nom lisible (ex: "Ville 0")
    x: float     # Coordonnée horizontale X
    y: float     # Coordonnée verticale Y


# Décorateur @dataclass :
# Génère automatiquement __init__(self, name, cities, distance_matrix, ...) à partir des attributs typés.
@dataclass
class TSPInstance:
    """Instance complète du TSP : villes, matrice des distances D et visibilité heuristique eta."""
    name: str                        # Nom de l'instance
    cities: List[City]               # Liste ordonnée des n villes
    distance_matrix: np.ndarray      # Matrice carrée (n x n) des distances euclidiennes d_ij
    heuristic_matrix: np.ndarray     # Matrice carrée (n x n) de visibilité heuristique eta_ij = 1 / d_ij
    optimal_known: Optional[float] = None  # Longueur du cycle optimal de référence (si connue)

    # Décorateur @property :
    # Transforme une méthode en attribut en lecture seule (accessible via instance.dimension sans parenthèses).
    @property
    def dimension(self) -> int:
        """Retourne le nombre total de villes (n) du problème."""
        return len(self.cities)

    def tour_length(self, path: List[int]) -> float:
        """
        Calcule la distance totale fermée L(pi) d'un cycle hamiltonien.
        L'opérateur modulo '(i + 1) % n' permet de relier automatiquement la dernière ville à la ville de départ.
        """
        n = len(path)
        return float(sum(self.distance_matrix[path[i], path[(i + 1) % n]] for i in range(n)))

    # Décorateur @classmethod :
    # Définit une méthode de fabrique (factory) liée à la classe (cls) et non à une instance existante :
    # permet de construire un objet directement via TSPInstance.from_coordinates(...).
    @classmethod
    def from_coordinates(
        cls,
        coords: List[Tuple[float, float]],
        name: str = "TSP Personnalisé",
        optimal_known: Optional[float] = None,
    ) -> "TSPInstance":
        """Construit les matrices de distance d_ij et de visibilité eta_ij = 1 / d_ij de manière vectorisée."""
        # 1. Création de la liste des objets City
        cities = [City(i, f"Ville {i}", float(x), float(y)) for i, (x, y) in enumerate(coords)]
        pts = np.array(coords, dtype=np.float64)

        # 2. Calcul vectorisé NumPy (broadcasting 3D) de toutes les distances euclidiennes sans boucle Python :
        # pts[:, np.newaxis, :] est de forme (n, 1, 2) et pts[np.newaxis, :, :] est de forme (1, n, 2).
        # Leur soustraction produit toutes les différences (x_i - x_j, y_i - y_j) pour chaque paire de villes (i, j).
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        # 3. Calcul de la visibilité heuristique : eta_ij = 1 / d_ij (attrait glouton des villes proches).
        # Sur la diagonale (i == j), la distance est nulle : on fixe eta_ii = 0.0 pour éviter toute division par zéro.
        with np.errstate(divide="ignore"):
            heuristic = np.where(dist_matrix > 0, 1.0 / dist_matrix, 0.0)
        np.fill_diagonal(heuristic, 0.0)

        return cls(name, cities, dist_matrix, heuristic, optimal_known)

    # Décorateur @classmethod permettant d'instancier directement un problème étalonné par sa clé.
    @classmethod
    def load_preset(cls, key: str) -> "TSPInstance":
        """Charge l'une des instances de référence définies dans PRESET_INSTANCES."""
        data = PRESET_INSTANCES[key]
        return cls.from_coordinates(data["coords"], data["name"], data.get("optimal_known"))

    # Décorateur @classmethod permettant de générer une nouvelle carte aléatoire reproductible.
    @classmethod
    def generate_random(cls, num_cities: int = 15, seed: int = 42) -> "TSPInstance":
        """Génère aléatoirement num_cities villes dans un carré [5, 95] x [5, 95]."""
        rng = np.random.default_rng(seed)
        coords = [(float(x), float(y)) for x, y in rng.uniform(5.0, 95.0, size=(num_cities, 2))]
        return cls.from_coordinates(coords, f"Carte Aléatoire ({num_cities} Villes)")


# ==============================================================================
# 3. ALGORITHME ANT COLONY OPTIMIZATION (ANT SYSTEM)
# ==============================================================================
# Décorateur @dataclass : structure proprement les résultats retournés par le solveur ACO.
@dataclass
class ACOResult:
    """Résultat synthétique d'une exécution ACO."""
    best_path: List[int]            # Meilleur cycle hamiltonien trouvé [v_0, v_1, ..., v_{n-1}]
    best_length: float              # Distance minimale associée L_best
    history_best: List[float]       # Historique de la meilleure distance au fil des itérations
    history_mean: List[float]       # Historique de la distance moyenne de la colonie par itération
    pheromone_matrix: np.ndarray    # Matrice finale des phéromones tau_ij
    execution_time_ms: float        # Temps de calcul total en millisecondes


class AntSystem:
    """
    Algorithme Ant System (Dorigo et al., 1996) avec option de renforcement élitiste (EAS).
    Implémente les 3 mécanismes clés : Transition probabiliste, Évaporation et Dépôt de phéromones.
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
        self.alpha = float(alpha)                  # Poids des phéromones tau (exploitation de la mémoire collective)
        self.beta = float(beta)                    # Poids de la visibilité eta = 1/d (exploration gloutonne locale)
        self.rho = float(rho)                      # Taux d'évaporation des phéromones (0 < rho < 1)
        self.q = float(q)                          # Constante de dépôt de phéromone Q
        self.num_ants = int(num_ants)              # Nombre de fourmis dans la colonie (m)
        self.max_iter = int(max_iter)              # Nombre d'itérations / générations (T)
        self.elitist_weight = float(elitist_weight) # Poids élitiste e (0 = Ant System standard, >0 = Elitist Ant System)
        self.rng = np.random.default_rng(seed)     # Générateur aléatoire fixé pour la reproductibilité

    def compute_probabilities(
        self, current: int, unvisited: List[int], pheromone: np.ndarray, heuristic: np.ndarray
    ) -> np.ndarray:
        """
        Calcule la probabilité de transition P_ij de la ville 'current' vers chaque ville candidate 'unvisited' :
        P_ij = (tau_ij^alpha * eta_ij^beta) / sum(tau_il^alpha * eta_il^beta)
        """
        # Numérateur : produit de l'intensité des phéromones (puissance alpha) et de la proximité (puissance beta)
        weights = (pheromone[current, unvisited] ** self.alpha) * (heuristic[current, unvisited] ** self.beta)
        total = np.sum(weights)
        # Sécurité numérique : si la somme est quasi-nulle, on retourne une probabilité uniforme équiprobable
        if total <= 1e-15:
            return np.full(len(unvisited), 1.0 / len(unvisited))
        # Normalisation pour que la somme des probabilités soit exactement égale à 1.0
        return weights / total

    def solve(self, instance: TSPInstance) -> ACOResult:
        """Exécute la simulation complète de la colonie de fourmis sur l'instance TSP."""
        start_t = time.perf_counter()
        n = instance.dimension

        # Initialisation uniforme des phéromones à tau_0 = 1.0 sur toutes les arêtes (0 sur la diagonale)
        pheromone = np.full((n, n), 1.0, dtype=np.float64)
        np.fill_diagonal(pheromone, 0.0)

        best_path: List[int] = []
        best_length = float("inf")
        history_best, history_mean = [], []

        # Boucle principale sur les T générations
        for _ in range(self.max_iter):
            all_paths, all_lengths = [], []

            # ------------------------------------------------------------------
            # ÉTAPE 1 : Construction d'une tournée complète par chaque fourmi
            # ------------------------------------------------------------------
            for ant in range(self.num_ants):
                # Répartition des fourmis sur les différentes villes de départ
                current = ant % n
                path = [current]
                unvisited = list(set(range(n)) - {current})

                # La fourmi visite les villes restantes une par une par tirage à la roulette pondérée
                while unvisited:
                    probs = self.compute_probabilities(current, unvisited, pheromone, instance.heuristic_matrix)
                    next_city = int(self.rng.choice(unvisited, p=probs))
                    path.append(next_city)
                    unvisited.remove(next_city)
                    current = next_city

                # Calcul du coût (longueur totale) de la tournée construite par la fourmi
                length = instance.tour_length(path)
                all_paths.append(path)
                all_lengths.append(length)

                # Mise à jour du record global si la fourmi a trouvé un meilleur cycle
                if length < best_length:
                    best_length = length
                    best_path = list(path)

            # ------------------------------------------------------------------
            # ÉTAPE 2 : Évaporation des phéromones -> tau_ij <- (1 - rho) * tau_ij
            # Permet d'oublier progressivement les mauvaises pistes explorées au début.
            # ------------------------------------------------------------------
            pheromone *= (1.0 - self.rho)

            # ------------------------------------------------------------------
            # ÉTAPE 3 : Dépôt de phéromones par chaque fourmi -> Delta_tau = Q / L_k
            # Plus le cycle L_k est court, plus la quantité déposée sur ses arêtes est grande.
            # ------------------------------------------------------------------
            for path, length in zip(all_paths, all_lengths):
                deposit = self.q / length
                for i in range(n):
                    u, v = path[i], path[(i + 1) % n]
                    # Graphe symétrique non orienté : renforcement de (u, v) et (v, u)
                    pheromone[u, v] += deposit
                    pheromone[v, u] += deposit

            # Option Élitiste (EAS) : renforcement supplémentaire du meilleur chemin global connu
            if self.elitist_weight > 0 and best_path:
                elite_deposit = self.elitist_weight * (self.q / best_length)
                for i in range(n):
                    u, v = best_path[i], best_path[(i + 1) % n]
                    pheromone[u, v] += elite_deposit
                    pheromone[v, u] += elite_deposit

            # Maintien d'un seuil minimal (1e-6) pour éviter l'annulation numérique totale d'une piste
            pheromone = np.maximum(pheromone, 1e-6)
            np.fill_diagonal(pheromone, 0.0)

            # Enregistrement des statistiques de convergence de la génération
            history_best.append(best_length)
            history_mean.append(float(np.mean(all_lengths)))

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return ACOResult(best_path, best_length, history_best, history_mean, pheromone, elapsed_ms)


# ==============================================================================
# 4. ALGORITHMES DE RÉFÉRENCE (POUR COMPARAISON PÉDAGOGIQUE)
# ==============================================================================
def solve_nearest_neighbor(instance: TSPInstance) -> Tuple[List[int], float, float]:
    """
    Heuristique gloutonne du Plus Proche Voisin :
    Part de la Ville 0 et choisit toujours la ville non visitée la plus proche.
    Très rapide O(n²), mais myope (subit souvent de longs détours en fin de parcours).
    """
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
    """
    Recherche exhaustive exacte évaluant toutes les permutations (réservé à N <= 10).
    Garantit l'optimum mathématique absolu en O(n!).
    """
    start_t = time.perf_counter()
    n = instance.dimension
    best_path, best_len = [], float("inf")

    # On fixe la ville 0 en tête et on parcourt toutes les permutations des villes 1 à n-1
    for perm in itertools.permutations(range(1, n)):
        # Élimination des cycles miroirs (parcourir le même cycle à l'envers donne la même distance)
        if perm[0] > perm[-1]:
            continue
        candidate = [0] + list(perm)
        length = instance.tour_length(candidate)
        if length < best_len:
            best_len = length
            best_path = candidate

    return best_path, best_len, (time.perf_counter() - start_t) * 1000.0
