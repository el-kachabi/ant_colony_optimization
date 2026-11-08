"""
Mathematical modeling and instance generation for the Traveling Salesperson Problem (TSP).
Author: El Kachabi Badr-eddine
GitHub: https://github.com/el-kachabi
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np


@dataclass(frozen=True)
class City:
    """Represents a city (node) in 2D Euclidean space."""
    id: int
    name: str
    x: float
    y: float

    def distance_to(self, other: "City") -> float:
        """Euclidean distance to another city."""
        return float(np.hypot(self.x - other.x, self.y - other.y))


@dataclass
class Tour:
    """Represents a closed Hamiltonian cycle visiting all cities."""
    path: List[int]
    length: float
    is_valid: bool = True

    def __len__(self) -> int:
        return len(self.path)

    def to_coordinates(self, cities: List[City]) -> Tuple[List[float], List[float]]:
        """Returns closed coordinates (x, y) including return to starting city."""
        xs = [cities[idx].x for idx in self.path] + [cities[self.path[0]].x]
        ys = [cities[idx].y for idx in self.path] + [cities[self.path[0]].y]
        return xs, ys


@dataclass
class TSPInstance:
    """
    Complete TSP problem instance containing cities, distance matrix,
    heuristic visibility matrix, and known optimal reference length.
    """
    name: str
    cities: List[City]
    dimension: int
    distance_matrix: np.ndarray
    heuristic_matrix: np.ndarray
    optimal_known: Optional[float] = None
    description: str = ""

    def __post_init__(self):
        # Validate matrix shapes
        n = self.dimension
        if self.distance_matrix.shape != (n, n):
            raise ValueError(f"Distance matrix shape {self.distance_matrix.shape} != ({n}, {n})")
        if self.heuristic_matrix.shape != (n, n):
            raise ValueError(f"Heuristic matrix shape {self.heuristic_matrix.shape} != ({n}, {n})")

    def compute_tour_length(self, path: List[int]) -> float:
        """
        Computes total closed length of a tour: L = sum(d[i, i+1]) + d[n-1, 0].
        """
        n = len(path)
        if n != self.dimension:
            raise ValueError(f"Tour length {n} does not match instance dimension {self.dimension}")
        length = 0.0
        for i in range(n):
            u = path[i]
            v = path[(i + 1) % n]
            length += float(self.distance_matrix[u, v])
        return length

    def validate_tour(self, path: List[int]) -> Tuple[bool, str]:
        """
        Validates if a sequence of city indices forms a valid Hamiltonian cycle.
        """
        n = self.dimension
        if len(path) != n:
            return False, f"La tournée contient {len(path)} villes au lieu de {n}."
        unique_nodes = set(path)
        if len(unique_nodes) != n:
            return False, f"La tournée contient des doublons (nombre d'uniques : {len(unique_nodes)})."
        if any(idx < 0 or idx >= n for idx in path):
            return False, "Certains indices de villes sont hors limites."
        return True, "Tournée hamiltonienne valide."

    def create_tour(self, path: List[int]) -> Tour:
        """Constructs and validates a Tour object."""
        is_valid, _ = self.validate_tour(path)
        length = self.compute_tour_length(path) if is_valid else float("inf")
        return Tour(path=list(path), length=length, is_valid=is_valid)

    @classmethod
    def from_coordinates(
        cls,
        coords: List[Tuple[float, float]],
        name: str = "Custom_TSP",
        labels: Optional[List[str]] = None,
        optimal_known: Optional[float] = None,
        description: str = "",
    ) -> "TSPInstance":
        """Builds a TSP instance from a list of (x, y) coordinates."""
        dimension = len(coords)
        if dimension < 3:
            raise ValueError("A TSP instance requires at least 3 cities.")

        cities = []
        for i, (x, y) in enumerate(coords):
            lbl = labels[i] if labels and i < len(labels) else f"City_{i}"
            cities.append(City(id=i, name=lbl, x=float(x), y=float(y)))

        # Vectorized Euclidean distance matrix computation
        coords_arr = np.array(coords, dtype=np.float64)  # shape (n, 2)
        diff = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]  # shape (n, n, 2)
        distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Heuristic visibility matrix: eta_ij = 1 / d_ij (with 0 on diagonal)
        with np.errstate(divide="ignore", invalid="ignore"):
            heuristic_matrix = np.where(distance_matrix > 0, 1.0 / distance_matrix, 0.0)
        np.fill_diagonal(heuristic_matrix, 0.0)

        return cls(
            name=name,
            cities=cities,
            dimension=dimension,
            distance_matrix=distance_matrix,
            heuristic_matrix=heuristic_matrix,
            optimal_known=optimal_known,
            description=description,
        )

    @classmethod
    def generate_random(
        cls,
        num_cities: int = 20,
        width: float = 100.0,
        height: float = 100.0,
        seed: Optional[int] = 42,
        name: str = "Random_TSP",
    ) -> "TSPInstance":
        """Generates a random TSP instance with coordinates in [0, width] x [0, height]."""
        rng = np.random.default_rng(seed)
        xs = rng.uniform(5.0, width - 5.0, size=num_cities)
        ys = rng.uniform(5.0, height - 5.0, size=num_cities)
        coords = list(zip(xs, ys))
        labels = [f"City_{i}" for i in range(num_cities)]
        desc = f"Génération aléatoire uniforme de {num_cities} villes (graine={seed})"
        return cls.from_coordinates(
            coords=coords,
            name=name,
            labels=labels,
            description=desc,
        )

    @classmethod
    def generate_circular(
        cls,
        num_cities: int = 12,
        radius: float = 40.0,
        center: Tuple[float, float] = (50.0, 50.0),
        noise_std: float = 0.0,
        seed: Optional[int] = 42,
        name: str = "Circular_TSP",
    ) -> "TSPInstance":
        """
        Generates cities along a circle with optional Gaussian radial noise.
        When noise_std=0, the optimal tour is known analytically (the perimeter polygon).
        """
        rng = np.random.default_rng(seed)
        angles = np.linspace(0, 2 * np.pi, num_cities, endpoint=False)
        r = radius + (rng.normal(0, noise_std, size=num_cities) if noise_std > 0 else 0)
        xs = center[0] + r * np.cos(angles)
        ys = center[1] + r * np.sin(angles)
        coords = list(zip(xs, ys))
        labels = [f"C_{i}" for i in range(num_cities)]

        # Calculate perimeter if noise is 0
        chord_length = 2 * radius * np.sin(np.pi / num_cities)
        optimal_known = float(num_cities * chord_length) if noise_std == 0.0 else None

        desc = f"Instance circulaire régulière ({num_cities} villes, rayon={radius})"
        return cls.from_coordinates(
            coords=coords,
            name=name,
            labels=labels,
            optimal_known=optimal_known,
            description=desc,
        )

    @classmethod
    def generate_grid(
        cls,
        rows: int = 4,
        cols: int = 5,
        spacing: float = 20.0,
        offset: Tuple[float, float] = (15.0, 15.0),
        name: str = "Grid_TSP",
    ) -> "TSPInstance":
        """Generates cities arranged on a regular Cartesian grid."""
        coords = []
        labels = []
        idx = 0
        for r in range(rows):
            for c in range(cols):
                x = offset[0] + c * spacing
                y = offset[1] + r * spacing
                coords.append((x, y))
                labels.append(f"G_{r}_{c}")
                idx += 1
        desc = f"Grille régulière de {rows}x{cols} ({rows*cols} villes, espacement={spacing})"
        return cls.from_coordinates(
            coords=coords,
            name=name,
            labels=labels,
            description=desc,
        )

    @classmethod
    def from_tsplib_file(cls, filepath: Union[str, Path]) -> "TSPInstance":
        """
        Parses a standard TSPLIB (.tsp) EUC_2D file.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"TSPLIB file not found at: {path}")

        name = path.stem
        coords = []
        labels = []
        dimension = 0
        in_coord_section = False
        optimal_known = None

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.upper() == "EOF":
                    break
                if line_str.startswith("NAME"):
                    parts = line_str.split(":")
                    if len(parts) > 1:
                        name = parts[1].strip()
                elif line_str.startswith("DIMENSION"):
                    dimension = int(line_str.split(":")[1].strip())
                elif line_str.startswith("COMMENT") and "optimal" in line_str.lower():
                    # Attempt to extract known optimal from comment if present
                    try:
                        for token in line_str.replace(":", " ").replace("-", " ").split():
                            if token.replace(".", "").isdigit():
                                optimal_known = float(token)
                                break
                    except Exception:
                        pass
                elif line_str.startswith("NODE_COORD_SECTION"):
                    in_coord_section = True
                    continue

                if in_coord_section:
                    parts = line_str.split()
                    if len(parts) >= 3:
                        node_id = parts[0]
                        x = float(parts[1])
                        y = float(parts[2])
                        coords.append((x, y))
                        labels.append(f"N_{node_id}")

        return cls.from_coordinates(
            coords=coords,
            name=name,
            labels=labels,
            optimal_known=optimal_known,
            description=f"Instance standard TSPLIB {name} ({len(coords)} villes)",
        )

    @classmethod
    def load_preset(cls, preset_key: str, presets_path: Optional[Union[str, Path]] = None) -> "TSPInstance":
        """
        Loads a preset instance from data/presets.json or config/settings.py.
        """
        if presets_path is None:
            presets_path = Path(__file__).resolve().parent.parent / "data" / "presets.json"

        p_path = Path(presets_path)
        if p_path.exists():
            with open(p_path, "r", encoding="utf-8") as f:
                presets = json.load(f)
            if preset_key in presets:
                entry = presets[preset_key]
                coords = [tuple(c) for c in entry["coords"]]
                labels = entry.get("labels", [f"V_{i}" for i in range(len(coords))])
                return cls.from_coordinates(
                    coords=coords,
                    name=entry.get("name", preset_key),
                    labels=labels,
                    optimal_known=entry.get("optimal_known"),
                    description=entry.get("description", ""),
                )

        raise KeyError(f"Preset '{preset_key}' non trouvé dans {presets_path}")
