# Spécifications Techniques et Architecture Logicielle
## Projet : Optimisation par Colonie de Fourmis (Ant Colony Optimization – ACO)
**Auteur :** El Kachabi Badr-eddine  
**Profil GitHub :** [https://github.com/el-kachabi](https://github.com/el-kachabi)  
**Dépôt :** [ant_colony_optimization](https://github.com/el-kachabi/ant_colony_optimization)  

---

## 1. Vision et Objectifs du Système

Le projet `ant_colony_optimization` est une suite logicielle et scientifique dédiée à l'apprentissage interactif, à la modélisation mathématique et à la résolution du **Problème du Voyageur de Commerce (Traveling Salesperson Problem - TSP)** au moyen de métaheuristiques bio-inspirées par colonies de fourmis.

### Piliers d'ingénierie :
1. **Intégrité Mathématique Absolue :** Tous les calculs, probabilités, dépôts de phéromones et longueurs de parcours sont calculés en temps réel par des algorithmes réels (aucune courbe simulée ou maquettée).
2. **Modularité Découplée :** Séparation stricte entre le moteur algorithmique pur (`src/`), la configuration (`config/`), les jeux de données (`data/`), les tests unitaires (`tests/`) et l'interface utilisateur interactive (`pages/`).
3. **Reproductibilité Scientifique :** Contrôle systématique des graines pseudo-aléatoires (`seed`) pour garantir la comparaison équitable entre variantes.
4. **Expérience Utilisateur Interactive Haut de Gamme :** Interface moderne sous Streamlit combinée à des visualisations graphiques réactives sous Plotly.

---

## 2. Fondements Mathématiques Formels

### 2.1 Modélisation du TSP
Soit un ensemble de $n$ villes $V = \{0, 1, \dots, n-1\}$ repérées par leurs coordonnées cartésiennes $(x_i, y_i) \in \mathbb{R}^2$.
Le réseau est représenté par un graphe complet symétrique non orienté $G = (V, E)$.
La matrice des distances euclidiennes $D = (d_{ij})_{n \times n}$ est définie par :
$$d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}, \quad \forall i, j \in V$$
avec $d_{ii} = 0$ et $d_{ij} = d_{ji}$.

L'objectif est de trouver une permutation $\pi = (\pi_0, \pi_1, \dots, \pi_{n-1})$ de $V$ (cycle hamiltonien) minimisant la longueur totale :
$$L(\pi) = \sum_{i=0}^{n-2} d_{\pi_i, \pi_{i+1}} + d_{\pi_{n-1}, \pi_0}$$

### 2.2 Visibilité Heuristique
L'attractivité locale immédiate d'un arc $(i, j)$ est inversement proportionnelle à sa distance :
$$\eta_{ij} = \frac{1}{d_{ij}} \quad (\text{avec } \eta_{ii} = 0)$$

### 2.3 Règle de Déplacement Proportionnelle (Ant System)
À chaque étape de construction, une fourmi $k$ située au nœud courant $i$ choisit la ville suivante $j$ parmi ses villes non visitées $N_i^k$ avec la probabilité :
$$P_{ij}^k = \begin{cases} \dfrac{[\tau_{ij}]^\alpha \cdot [\eta_{ij}]^\beta}{\displaystyle\sum_{l \in N_i^k} [\tau_{il}]^\alpha \cdot [\eta_{il}]^\beta} & \text{si } j \in N_i^k \\ 0 & \text{sinon} \end{cases}$$
* $\alpha \ge 0$ contrôle l'influence de la phéromone $\tau_{ij}$ (mémoire collective, exploitation).
* $\beta \ge 0$ contrôle l'influence de la visibilité $\eta_{ij}$ (heuristique gloutonne, exploration).

### 2.4 Évaporation et Dépôt des Phéromones
À la fin de chaque génération (lorsque les $m$ fourmis ont chacune complété leur tournée) :
$$\tau_{ij}(t+1) = (1 - \rho)\,\tau_{ij}(t) + \sum_{k=1}^m \Delta \tau_{ij}^k(t)$$
où $\rho \in ]0, 1[$ est le taux d'évaporation, et :
$$\Delta \tau_{ij}^k(t) = \begin{cases} \dfrac{Q}{L_k} & \text{si l'arête } (i, j) \text{ est dans la tournée } k \\ 0 & \text{sinon} \end{cases}$$

### 2.5 Variantes Implémentées
1. **Ant System (AS) :** Algorithme standard historique (Dorigo, 1996).
2. **Elitist Ant System (EAS) :** Ajoute un renforcement additionnel $e \cdot \frac{Q}{L_{best}}$ sur chaque arête de la meilleure tournée globale observée depuis le début :
   $$\tau_{ij} \leftarrow \tau_{ij} + e \cdot \frac{Q}{L_{best}} \quad \forall (i, j) \in T_{best}$$
3. **Ant Colony System (ACS) :**
   - Règle de transition pseudo-aléatoire proportionnelle : si un nombre aléatoire $q \le q_0$, la fourmi choisit de manière déterministe le meilleur voisin $\arg\max_{l \in N_i^k} (\tau_{il} \cdot \eta_{il}^\beta)$ ; sinon, elle applique la roulette probabiliste standard.
   - Mise à jour locale : $\tau_{ij} \leftarrow (1 - \xi)\tau_{ij} + \xi \tau_0$ immédiatement après chaque déplacement.
   - Mise à jour globale : appliquée uniquement à la meilleure tournée globale.

---

## 3. Modèles de Données (Contrats d'Interface)

Les structures de données sont standardisées sous forme de `dataclasses` Python typées et immutables :

```python
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np

@dataclass(frozen=True)
class City:
    """Représentation d'un sommet / ville du TSP."""
    id: int
    name: str
    x: float
    y: float

@dataclass
class TSPInstance:
    """Instance complète du problème du voyageur de commerce."""
    name: str
    cities: List[City]
    dimension: int
    distance_matrix: np.ndarray
    heuristic_matrix: np.ndarray
    optimal_known: Optional[float] = None

@dataclass
class Tour:
    """Représentation d'une tournée hamiltonienne."""
    path: List[int]             # Indices ordonnés des villes [v0, v1, ..., v_{n-1}]
    length: float               # Distance totale fermée
    is_valid: bool = True       # Vérification du cycle hamiltonien

@dataclass
class IterationRecord:
    """Métriques collectées à chaque itération de la colonie."""
    iteration: int
    best_length: float
    mean_length: float
    worst_length: float
    best_path: List[int]
    pheromone_min: float
    pheromone_max: float
    pheromone_mean: float

@dataclass
class ACOResult:
    """Résultat complet d'une exécution ACO."""
    algorithm_name: str
    instance_name: str
    best_tour: Tour
    convergence_history: List[IterationRecord]
    final_pheromone_matrix: np.ndarray
    execution_time_sec: float
    hyperparameters: Dict[str, any]
```

---

## 4. Organisation des Modules Logiciels

```
ant_colony_optimization/
├── config/
│   ├── __init__.py
│   └── settings.py          # Constantes, presets, palettes et hyperparamètres par défaut
├── data/
│   ├── benchmarks/          # Fichiers TSPLIB (.tsp)
│   └── presets.json         # Instances didactiques préconfigurées (5, 10, 20, 30 villes)
├── docs/
│   ├── architecture.md      # Ce document de spécification
│   └── theory.md            # Manuel scientifique exhaustif et calculs numériques détaillés
├── pages/
│   ├── 1_Dashboard.py       # Vue d'ensemble, métriques globales et graphe du réseau
│   ├── 2_Theorie_ACO.py     # Cours interactif, formules LaTeX et calculatrice manuelle
│   ├── 3_Simulation.py      # Simulateur pas-à-pas de déplacement des fourmis
│   ├── 4_Resolution_TSP.py  # Solveur interactif complet multi-instances
│   ├── 5_Experiences.py     # Laboratoire de sensibilité (alpha, beta, rho, etc.)
│   ├── 6_Comparaison.py     # Banc d'essai comparatif multi-algorithmes
│   ├── 7_Applications.py    # Études de cas concrets (logistique, télécom, circuits)
│   └── 8_A_propos.py        # Crédits auteur, architecture et perspectives
├── src/
│   ├── __init__.py
│   ├── tsp.py               # Génération d'instances, distances, vérification de cycle
│   ├── aco.py               # Moteur Ant System (AS)
│   ├── variants.py          # Moteurs Elitist Ant System (EAS) et Ant Colony System (ACS)
│   ├── baselines.py         # Solveurs de référence : Glouton, 2-Opt, Brute Force exact
│   └── visualization.py     # Fonctions de rendu Plotly interactif
├── tests/
│   ├── __init__.py
│   ├── test_tsp.py          # Tests unitaires du modèle TSP
│   ├── test_aco.py          # Tests unitaires du moteur ACO & phéromones
│   └── test_variants.py     # Tests unitaires des variantes et baselines
├── app.py                   # Point d'entrée de l'application Streamlit
├── requirements.txt         # Dépendances pip validées
└── .gitignore               # Exclusion des fichiers confidentiels et caches
```

---

## 5. Gestion de l'État dans Streamlit (`st.session_state`)

Pour garantir la fluidité de navigation entre les 8 pages interactives sans recalculs redondants, les variables d'état partagées suivantes sont définies :

| Clé `st.session_state` | Type | Rôle |
| :--- | :--- | :--- |
| `tsp_instance` | `TSPInstance` | Instance courante active sélectionnée ou générée |
| `aco_params` | `dict` | Dictionnaire des hyperparamètres ($\alpha, \beta, \rho, Q, m, T$) |
| `last_aco_result` | `ACOResult` | Dernier résultat complet d'optimisation calculé |
| `simulation_step` | `int` | Étape courante dans le simulateur pas-à-pas |
| `benchmark_results` | `dict` | Résultats consolidés du tournoi comparatif d'algorithmes |
| `random_seed` | `int` | Graine de reproductibilité active |

---

## 6. Protocole de Validation & Tests Unitaires

La suite de tests automatisée `pytest` valide les propriétés mathématiques et algorithmiques suivantes :
1. **Distance Matrix :** Vérification de la nullité diagonale ($d_{ii} = 0$), de la stricte positivité ($d_{ij} > 0 \;\forall i \neq j$), et de la symétrie ($d_{ij} = d_{ji}$).
2. **Cycle Hamiltonien :** Vérification que chaque sommet apparaît exactement une fois et que la longueur totale est correctement sommée avec retour au point de départ.
3. **Distribution de Probabilités :** Vérification que $\sum_{j \in N_i^k} P_{ij}^k = 1.0 \pm 10^{-6}$ pour toute étape de décision.
4. **Conservation & Évaporation :** Vérification numérique que $\tau_{ij}(t+1) = (1 - \rho)\tau_{ij}(t)$ lorsqu'aucune fourmi n'emprunte l'arête $(i, j)$.
5. **Exactitude sur Petites Instances :** Comparaison des solutions ACO avec la solution exacte exhaustive calculée par le solveur Brute Force sur l'instance de 5 et 10 villes (erreur relative $< 1\%$).

---
*Ce document sert de contrat de développement pour toutes les phases d'implémentation subséquentes.*
